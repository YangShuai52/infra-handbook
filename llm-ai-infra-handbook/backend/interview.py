"""AI 面试官模拟（v0.3）。当前实现 B1 + B2 + B3。

- 会话存内存 dict（重启即失，单用户可接受）
- 出题强约束（system prompt）：仅围绕给定模块的知识卡片出题、每个问题必须
  引用卡片 id、同一话题最多追问 3 层、输出严格 JSON
- 答题流（SSE）：流式输出评价，末行以 @@DECISION@@ 协议携带 JSON 决策
  （评分 0~5 / followup / next / end）；评分映射到 P0 的 quality 四档写入刷题数据
- 无 Key 降级：追问链自测模式 —— 解析卡片「面试追问」小节成题，用户自评写回
- 练习报告：逐题得分 + 按卡片/模块聚类的薄弱点 + 推荐复习卡片（可一键进今日队列）

对 app 暴露：
    start(module, cards)             创建会话并出题（有 Key 走 LLM，无 Key 走自测）
    answer_stream(sid, ans, cards)   答题流（SSE 异步生成器，仅 LLM 模式）
    self_answer(sid, quality, cards) 自测模式：给当前题打分并推进（仅自测模式）
    report(sid, cards)               练习报告
    get_session(sid)                 取会话
"""
from __future__ import annotations

import json
import re
import time
import uuid

import generator
import study

_MAX_ROUNDS = 10
_MAX_DEPTH = 3          # 同一话题最多追问层数
_CTX_CHARS_PER_CARD = 1500  # 每张卡片注入上下文的正文字符数
_SELFTEST_HEADING = "面试追问"   # 自测模式抽取题目的章节
_SELFTEST_PER_CARD = 3           # 每张卡片最多取几条追问

_SYSTEM_PROMPT = """你是一名严格的 LLM 算法面试官，正在对候选人做「八股」模拟面试。
规则：
1. 只能围绕【给定知识卡片】的内容出题与追问；卡片未覆盖的内容一律不问。
2. 你提出的每个问题必须标注它依据的卡片 id（只能从给定列表中选择）。
3. 同一话题最多连续追问 3 层（"为什么 → 再为什么"），到层后必须换话题。
4. 问题要具体、可回答，避免"谈谈你的理解"这类空泛问法。
5. 输出必须是严格的 JSON 对象（不要用 markdown 代码块包裹），字段：
   {"question": "你的问题", "card_id": "依据的卡片id", "topic": "话题关键词"}
"""

_DECISION_PROMPT = """

评分与决策协议（每轮回复必须遵守）：
1. 先用 2~4 句话评价候选人刚才的回答：指出答对了什么、错在哪里、缺口是什么（面向候选人，不要提到评分数字）。
2. 然后在最后一行单独输出决策，格式严格为（前面不允许有任何其他字符）：
@@DECISION@@{"score": <0~5整数>, "action": "<followup|next|end>", "question": "<追问或下一题，action 为 followup/next 时必填>", "card_id": "<该问题依据的卡片id，action 为 followup/next 时必填>", "topic": "<话题关键词>"}
3. score 含义：0=完全错误 1=基本不会 2=只记得零星 3=方向对但有明显错误 4=正确但有瑕疵 5=完整且准确。
4. action 含义：followup=就当前话题继续追问；next=换一道新题；end=面试结束。
5. 当前追问深度已达 3 层时，禁止选 followup；轮次已达上限时，禁止选 next。
"""

_DECISION_TAG = "@@DECISION@@"

# 会话存储：{sid: {module, valid_ids, max_rounds, round, history, finished, created_at}}
sessions: dict[str, dict] = {}


def get_session(sid: str) -> dict | None:
    return sessions.get(sid)


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def _map_score(score: int | None) -> int:
    """模型 0~5 分 → P0 四档 quality {1,3,4,5}。"""
    if score is None or score <= 2:
        return 1
    return score if score in (3, 4, 5) else 5


def _build_user_prompt(module: str, module_cards: list[dict]) -> str:
    parts = [
        f"面试模块：{module}\n以下是该模块的全部知识卡片（出题范围，共 {len(module_cards)} 张）：\n"
    ]
    for c in module_cards:
        parts.append(
            f"--- 卡片 id：{c['id']} ｜ 标题：{c['title']} ｜ 频率：{c.get('freq', '')} ---\n"
            f"摘要：{c.get('summary', '')}\n{c['body'][:_CTX_CHARS_PER_CARD]}\n"
        )
    parts.append(
        "请从以上卡片中选择一个考察点，出第 1 轮第 1 题（不是追问）。"
        "严格按 system 约定的 JSON 格式输出。"
    )
    return "\n".join(parts)


def _extract_json(text: str) -> dict | None:
    """从模型输出中提取 JSON 对象（容忍 ```json 围栏与前后废话）。"""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _validate_question(q: dict | None, valid_ids: list[str]) -> bool:
    """问题合法性校验：有内容且 card_id 可溯源到给定卡片。"""
    if not isinstance(q, dict):
        return False
    question = q.get("question")
    if not isinstance(question, str) or len(question.strip()) < 8:
        return False
    return q.get("card_id") in valid_ids


async def start(module: str, cards: list[dict]) -> dict:
    """创建面试会话并生成第一题。

    返回 {ok, session_id, degraded, question?[, error]}。
    degraded=True 时 question 为空，前端应提示降级（追问链自测模式在 B3 提供）。
    """
    module_cards = [c for c in cards if c.get("module") == module]
    if not module_cards:
        return {"ok": False, "error": f"模块 {module} 不存在或没有卡片"}

    sid = uuid.uuid4().hex[:12]
    valid_ids = [c["id"] for c in module_cards]
    sessions[sid] = {
        "module": module,
        "valid_ids": valid_ids,
        "max_rounds": _MAX_ROUNDS,
        "round": 0,
        "history": [],
        "created_at": time.time(),
    }

    sessions[sid]["mode"] = "selftest" if not generator.is_llm_enabled() else "llm"
    sessions[sid]["selftest"] = []
    sessions[sid]["index"] = 0

    if not generator.is_llm_enabled():
        # B3：追问链自测模式 —— 直接拿卡片「面试追问」小节当题，用户自评写回
        questions = _build_selftest(module_cards, limit=_MAX_ROUNDS)
        if not questions:
            return {"ok": True, "session_id": sid, "degraded": True, "mode": "selftest",
                    "question": None, "message": "该模块没有可用于自测的追问内容"}
        sessions[sid]["selftest"] = questions
        sessions[sid]["round"] = 1
        return {"ok": True, "session_id": sid, "degraded": True, "mode": "selftest",
                "round": 1, "total": len(questions),
                "question": {"question": questions[0]["question"],
                             "card_id": questions[0]["card_id"],
                             "is_followup": False, "depth": 1,
                             "index": 1, "total": len(questions)}}

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(module, module_cards)},
    ]

    last_err = "模型输出不符合约定格式"
    for _ in range(2):  # 格式不合法时重试一次
        try:
            raw = await generator.chat_complete(messages)
        except Exception as e:  # noqa: BLE001
            return {"ok": True, "session_id": sid, "degraded": True, "question": None,
                    "error": f"生成失败：{e}"}
        q = _extract_json(raw)
        if _validate_question(q, valid_ids):
            q["is_followup"] = False
            q["depth"] = 1
            sessions[sid]["round"] = 1
            sessions[sid]["history"].append(
                {"role": "interviewer", "question": q["question"],
                 "card_id": q["card_id"], "depth": 1}
            )
            return {"ok": True, "session_id": sid, "degraded": False,
                    "round": 1, "question": q}
    return {"ok": True, "session_id": sid, "degraded": True, "question": None,
            "error": last_err}


# ================= B3：无 Key 自测模式 + 练习报告 =================

def _selftest_questions(card: dict) -> list[str]:
    """从卡片的「面试追问」小节抽取问题（知识库卡片均带该小节）。"""
    for sec in card.get("sections", []):
        if _SELFTEST_HEADING in sec.get("heading", ""):
            out = []
            for line in sec.get("content", "").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 知识库同时存在两种追问格式：
                #   **Q：问题？** / A：参考答案
                #   1. 问题？
                # 只抽取问题行，不能把 Q/A 格式里的答案误当成下一道题。
                qa_match = re.match(r"^\*\*Q[：:]\s*(.*?)\*\*$|^Q[：:]\s*(.*)$", line)
                if qa_match:
                    q = next(group for group in qa_match.groups() if group is not None).strip()
                elif re.match(r"^\d+[.、)]\s*", line):
                    q = re.sub(r"^\d+[.、)]\s*", "", line).strip()
                elif re.match(r"^[-*]\s+", line):
                    q = re.sub(r"^[-*]\s+", "", line).strip()
                else:
                    continue
                if len(q) >= 8:
                    out.append(q)
            if out:
                return out
    # 兜底：没有追问小节时用卡片标题做一道复述题
    return [f"请复述并推导「{card.get('title', card.get('id', ''))}」的核心结论"]


def _build_selftest(module_cards: list[dict], limit: int = _MAX_ROUNDS) -> list[dict]:
    out = []
    for c in module_cards:
        for q in _selftest_questions(c)[:_SELFTEST_PER_CARD]:
            out.append({"card_id": c["id"], "question": q})
            if len(out) >= limit:
                return out
    return out


def self_answer(sid: str, quality: int, cards: list[dict]) -> dict:
    """自测模式：给当前题打分并推进到下一题。

    quality 为 P0 四档 {1,3,4,5}；自测模式下 quality 同时作为报告用的 score。
    """
    sess = sessions.get(sid)
    if not sess:
        return {"ok": False, "error": "会话不存在"}
    if sess.get("mode") != "selftest":
        return {"ok": False, "error": "当前会话不是自测模式（已配置 LLM Key 时请用 /api/interview/answer）"}
    queue = sess.get("selftest") or []
    if sess.get("finished") or sess["index"] >= len(queue):
        return {"ok": False, "error": "自测已结束"}
    if quality not in (1, 3, 4, 5):
        return {"ok": False, "error": "quality 必须是 [1, 3, 4, 5] 之一"}

    item = queue[sess["index"]]
    already = any(
        h["role"] == "score" and h["card_id"] == item["card_id"] for h in sess["history"]
    )
    try:
        review = (study.log_review(item["card_id"], quality) if already
                  else study.answer(item["card_id"], quality))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"评分写回失败：{e}"}

    sess["history"].append({"role": "score", "card_id": item["card_id"],
                            "question": item["question"], "answer": "",
                            "score": quality, "quality": quality, "depth": 1})
    sess["index"] += 1
    done = sess["index"] >= len(queue)
    if done:
        sess["finished"] = True
    return {
        "ok": True, "mode": "selftest", "done": done, "review": review,
        "progress": {"current": sess["index"], "total": len(queue)},
        "next_question": (None if done else
                          {**queue[sess["index"]], "is_followup": False, "depth": 1,
                           "index": sess["index"] + 1, "total": len(queue)}),
    }


def report(sid: str, cards: list[dict]) -> dict | None:
    """练习报告：逐题得分 + 按卡片/模块聚类的薄弱点 + 推荐复习卡片。

    每条薄弱点都带 card_id / card_title，前端可点回卡片原文（可解释性）。
    """
    sess = sessions.get(sid)
    if not sess:
        return None
    by_id = {c["id"]: c for c in cards}

    scores = [h for h in sess["history"] if h["role"] == "score"]
    questions = []
    for s in scores:
        card = by_id.get(s["card_id"])
        questions.append({
            "question": s["question"], "card_id": s["card_id"],
            "card_title": card["title"] if card else s["card_id"],
            "module": card["module"] if card else "",
            "score": s.get("score"), "quality": s.get("quality"),
            "depth": s.get("depth", 1), "answer": s.get("answer", ""),
        })

    graded = [q for q in questions if isinstance(q["score"], int)]
    avg = round(sum(q["score"] for q in graded) / len(graded), 2) if graded else None
    distribution = {str(k): sum(1 for q in graded if q["score"] == k) for k in range(6)}

    # 按卡片聚合
    per_card: dict[str, dict] = {}
    for q in graded:
        e = per_card.setdefault(q["card_id"], {
            "card_id": q["card_id"], "card_title": q["card_title"], "module": q["module"],
            "count": 0, "scores": [],
        })
        e["count"] += 1
        e["scores"].append(q["score"])
    by_card = []
    for e in per_card.values():
        by_card.append({
            "card_id": e["card_id"], "card_title": e["card_title"], "module": e["module"],
            "count": e["count"],
            "avg_score": round(sum(e["scores"]) / len(e["scores"]), 2),
            "min_score": min(e["scores"]),
        })
    by_card.sort(key=lambda x: (x["avg_score"], x["min_score"]))

    # 按模块聚合
    per_module: dict[str, list[int]] = {}
    for q in graded:
        per_module.setdefault(q["module"], []).append(q["score"])
    by_module = [
        {"module": m, "count": len(v), "avg_score": round(sum(v) / len(v), 2)}
        for m, v in sorted(per_module.items())
    ]

    weak_points = [c for c in by_card if c["avg_score"] < 3 or c["min_score"] <= 2]
    suggestions = [
        {"card_id": c["card_id"], "title": c["card_title"], "module": c["module"],
         "avg_score": c["avg_score"], "count": c["count"],
         "reason": f"本场平均 {c['avg_score']} 分（{c['count']} 次作答）"}
        for c in weak_points[:5]
    ]

    return {
        "session_id": sid, "module": sess["module"], "mode": sess.get("mode", "llm"),
        "round": sess["round"], "max_rounds": sess["max_rounds"],
        "finished": bool(sess.get("finished")),
        "summary": {"total_questions": len(questions), "graded": len(graded),
                    "avg_score": avg, "distribution": distribution},
        "questions": questions,
        "by_card": by_card, "by_module": by_module,
        "weak_points": weak_points, "suggestions": suggestions,
    }


# ================= B2：答题流（SSE） =================

def _build_eval_messages(sess: dict, cur: dict, user_answer: str, cards: list[dict]) -> list[dict]:
    """构造「评价 + 决策」一轮的消息序列。cur 为当前问题条目。"""
    by_id = {c["id"]: c for c in cards}
    card = by_id.get(cur["card_id"])
    card_ctx = ""
    if card:
        card_ctx = (f"当前问题依据的卡片 id：{card['id']} ｜ 标题：{card['title']}\n"
                    f"卡片内容：\n{card.get('summary', '')}\n{card['body'][:2000]}\n")

    # 对话历史（压缩为问/答对，最多取最近 8 条）
    convo = []
    for h in sess["history"][-8:]:
        if h["role"] == "interviewer":
            convo.append(f"【面试官】（深度{h.get('depth', 1)}）{h['question']}")
        elif h["role"] == "candidate":
            convo.append(f"【候选人】{h['answer'][:500]}")
    convo.append(f"【候选人】（本次需要评价的回答）{user_answer}")

    other_ids = [i for i in sess["valid_ids"] if i != cur["card_id"]]
    user = (
        f"{card_ctx}\n"
        f"可选卡片 id 全集：{sess['valid_ids']}\n"
        f"其他可出题卡片：{other_ids}\n\n"
        f"当前轮次：第 {sess['round']}/{sess['max_rounds']} 题，当前追问深度：{cur.get('depth', 1)}/3\n"
        f"对话历史：\n" + "\n".join(convo) + "\n\n"
        "请评价候选人的回答，并按协议在最后一行输出 @@DECISION@@ 决策。"
        "若候选人回答质量高可追更深一层或换新题；若已考察充分可选 next；"
        f"若轮次/深度到达上限必须选 end（深度 {cur.get('depth', 1)}>=3 禁 followup）。"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT + _DECISION_PROMPT},
        {"role": "user", "content": user},
    ]


async def answer_stream(sid: str, user_answer: str, cards: list[dict]):
    """答题流：流式 yield SSE 事件（评价 token → decision → done）。

    事件格式（与 generator.stream_answer 一致）：
        data: {"type": "token", "text": "..."}\n\n
        data: {"type": "decision", "data": {...}}\n\n
        data: {"type": "error", "message": "..."}\n\n
        data: {"type": "done"}\n\n
    """
    sess = sessions.get(sid)
    if not sess:
        yield _sse({"type": "error", "message": "会话不存在"})
        yield _sse({"type": "done"})
        return
    if not sess["history"]:
        # 降级会话（start 时未生成问题，B3 提供追问链自测模式）
        yield _sse({"type": "error", "message": "未配置 LLM_API_KEY，追问链自测模式将在 B3 提供"})
        yield _sse({"type": "done"})
        return
    if sess.get("finished"):
        yield _sse({"type": "error", "message": "会话已结束"})
        yield _sse({"type": "done"})
        return
    if not generator.is_llm_enabled():
        yield _sse({"type": "error", "message": "未配置 LLM_API_KEY，追问链自测模式将在 B3 提供"})
        yield _sse({"type": "done"})
        return

    cur = sess["history"][-1]
    sess["history"].append({"role": "candidate", "answer": user_answer})
    messages = _build_eval_messages(sess, cur, user_answer, cards)

    # ---- 流式读取：评价部分即时转发，@@DECISION@@ 之后缓冲 ----
    commentary_done = False
    buf = ""
    decision_raw: list[str] = []
    try:
        async for delta in generator.chat_stream(messages):
            if commentary_done:
                decision_raw.append(delta)
                continue
            buf += delta
            idx = buf.find(_DECISION_TAG)
            if idx >= 0:
                head = buf[:idx]
                if head:
                    yield _sse({"type": "token", "text": head})
                decision_raw.append(buf[idx + len(_DECISION_TAG):])
                buf = ""
                commentary_done = True
                continue
            # 扣住可能是 TAG 前缀的尾部，避免协议标记被拆进评价文本
            hold = 0
            for k in range(min(len(_DECISION_TAG) - 1, len(buf)), 0, -1):
                if buf.endswith(_DECISION_TAG[:k]):
                    hold = k
                    break
            emit = buf[:-hold] if hold else buf
            if emit:
                yield _sse({"type": "token", "text": emit})
            buf = buf[-hold:] if hold else ""
        if not commentary_done and buf:
            yield _sse({"type": "token", "text": buf})  # 模型没输出决策标记：全文当评价
    except Exception as e:  # noqa: BLE001
        yield _sse({"type": "error", "message": f"生成异常：{e}"})
        yield _sse({"type": "done"})
        return

    # ---- 解析决策 ----
    d = _extract_json("".join(decision_raw)) if commentary_done else None
    note = None
    score = d.get("score") if d else None
    if not isinstance(score, int) or not (0 <= score <= 5):
        score = None
    action = d.get("action") if d else None
    if action not in ("followup", "next", "end"):
        action, note = "end", "模型决策缺失或格式异常，面试提前结束"

    # ---- 评分写回刷题数据（quality 四档互通）----
    # 同一张卡片在本场已评过分时只记流水、不推进间隔：否则一次面试的多次回答会把
    # SM-2 间隔层层放大（实测可到数年后），卡片等于退出复习队列。
    quality = _map_score(score)
    already_scored = any(
        h["role"] == "score" and h["card_id"] == cur["card_id"] for h in sess["history"]
    )
    review = None
    try:
        review = (study.log_review(cur["card_id"], quality) if already_scored
                  else study.answer(cur["card_id"], quality))
    except Exception as e:  # noqa: BLE001
        note = f"评分写回失败：{e}"

    # 记录本题成绩（B3 报告数据源）
    sess["history"].append({
        "role": "score", "card_id": cur["card_id"], "question": cur["question"],
        "answer": user_answer, "score": score, "quality": quality, "depth": cur.get("depth", 1),
    })

    question_out = None
    if action == "followup":
        new_depth = cur.get("depth", 1) + 1
        q = d.get("question") if d else None
        cid = d.get("card_id") if d else None
        if new_depth > _MAX_DEPTH:
            action, note = "next", "已达追问深度上限，转为换题"
        elif not (isinstance(q, str) and q.strip()) or cid not in sess["valid_ids"]:
            action, note = "end", "追问内容缺失或卡片 id 不可溯源，面试提前结束"
        else:
            question_out = {"question": q, "card_id": cid, "topic": d.get("topic", ""),
                            "is_followup": True, "depth": new_depth}
            sess["history"].append({"role": "interviewer", "question": q, "card_id": cid,
                                    "depth": new_depth})

    if action == "next":
        if sess["round"] >= sess["max_rounds"]:
            action, note = "end", "已达轮次上限，面试结束"
        else:
            q = d.get("question") if d else None
            cid = d.get("card_id") if d else None
            if not (isinstance(q, str) and q.strip()) or cid not in sess["valid_ids"]:
                action, note = "end", "新题内容缺失或卡片 id 不可溯源，面试提前结束"
            else:
                sess["round"] += 1
                question_out = {"question": q, "card_id": cid, "topic": d.get("topic", ""),
                                "is_followup": False, "depth": 1}
                sess["history"].append({"role": "interviewer", "question": q, "card_id": cid,
                                        "depth": 1})

    finished = False
    if action == "end":
        finished = True
        sess["finished"] = True

    yield _sse({"type": "decision", "data": {
        "score": score, "quality": quality, "action": action,
        "round": sess["round"], "max_rounds": sess["max_rounds"],
        "question": question_out, "finished": finished,
        "review": review, "note": note,
    }})
    yield _sse({"type": "done"})
