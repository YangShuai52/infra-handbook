"""LLM & AI Infra Handbook —— FastAPI 后端。

启动：
    cd backend
    uvicorn app:app --host 0.0.0.0 --port 8000
或直接： python app.py  （内部用 uvicorn）

可选环境变量（启用 RAG 生成）：
    LLM_API_KEY / LLM_BASE_URL / LLM_MODEL / LLM_TEMPERATURE
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import markdown
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from kb_lib import load_cards, load_modules, get_card_by_id
from search import build_index, search, snippet
from generator import is_llm_enabled, stream_answer
import models_archive as ma
import study
import interview

HERE = Path(__file__).resolve().parent
FRONTEND = HERE / "frontend"
KB_ROOT = HERE / "kb"

app = FastAPI(title="LLM & AI Infra Handbook", version="0.4.0")

# ---- 启动时构建索引 ----
print("[boot] 加载知识库…")
CARDS = load_cards()
MODULES = load_modules()
INDEX = build_index(CARDS)
study.init_db(CARDS)
print(f"[boot] 已加载 {len(CARDS)} 张卡片 / {len(MODULES)} 个模块")

_MD = markdown.Markdown(extensions=["fenced_code", "tables", "sane_lists", "toc"])


def render_md(text: str) -> str:
    _MD.reset()
    return _MD.convert(text)


# ---------------- 知识体系 API ----------------

@app.get("/api/status")
def api_status():
    return {
        "llm_enabled": is_llm_enabled(),
        "kb_cards": len(CARDS),
        "modules": len(MODULES),
    }


@app.get("/api/modules")
def api_modules():
    cards_by_module = {}
    for c in CARDS:
        cards_by_module.setdefault(c["module"], []).append(c)
    out = []
    for m in MODULES:
        cards = cards_by_module.get(m["module"], [])
        out.append({
            "id": m["id"],
            "title": m["title"],
            "module": m["module"],
            "order": m["order"],
            "desc": m["desc"],
            "card_count": len(cards),
            "cards": [
                {"id": c["id"], "title": c["title"], "freq": c["freq"],
                 "difficulty": c["diff_label"], "tags": c["tags"],
                 "summary": c["summary"]}
                for c in cards
            ],
        })
    return out


@app.get("/api/cards")
def api_cards(module: str | None = Query(None), q: str | None = Query(None), top_k: int = 50):
    if q:
        hits = search(INDEX, q, top_k=top_k)
        return [{
            "id": h["card"]["id"], "title": h["card"]["title"],
            "module": h["card"]["module"], "score": h["score"],
            "summary": h["card"]["summary"], "tags": h["card"]["tags"],
            "snippet": snippet(h["card"], q),
        } for h in hits]
    if module:
        cards = [c for c in CARDS if c["module"] == module]
    else:
        cards = CARDS
    return [{
        "id": c["id"], "title": c["title"], "module": c["module"],
        "freq": c["freq"], "difficulty": c["diff_label"], "tags": c["tags"],
        "summary": c["summary"],
    } for c in cards]


@app.get("/api/card/{card_id}")
def api_card(card_id: str):
    card = get_card_by_id(card_id, CARDS)
    if not card:
        raise HTTPException(status_code=404, detail="卡片不存在")
    # 把章节渲染为 HTML（含目录感）
    sections_html = []
    for sec in card["sections"]:
        sections_html.append(render_md(sec["content"]))
    return {
        "id": card["id"],
        "title": card["title"],
        "module": card["module"],
        "tags": card["tags"],
        "freq": card["freq"],
        "difficulty": card["diff_label"],
        "updated": card["updated"],
        "summary": card["summary"],
        "source": card["source"],
        "questions": card["questions"],
        "html": "".join(sections_html),
        "headings": [s["heading"] for s in card["sections"] if s["heading"]],
    }


# ---------------- 问答 API ----------------

@app.post("/api/ask")
async def api_ask(payload: dict):
    query = (payload or {}).get("query", "")
    top_k = int((payload or {}).get("top_k", 6))
    if not query or not query.strip():
        return JSONResponse(status_code=400, content={"error": "query 不能为空"})
    hits = search(INDEX, query, top_k=max(top_k, 6))
    # 预生成 snippet
    for h in hits:
        h["snippet"] = snippet(h["card"], query)

    if is_llm_enabled():
        return StreamingResponse(
            stream_answer(query, hits),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    # 无 Key 降级：直接返回检索卡片
    return {
        "mode": "retrieval",
        "query": query,
        "hits": [{
            "id": h["card"]["id"], "title": h["card"]["title"],
            "module": h["card"]["module"], "score": h["score"],
            "summary": h["card"]["summary"], "tags": h["card"]["tags"],
            "snippet": h["snippet"],
        } for h in hits],
    }


# ---------------- 模型档案 API ----------------

@app.get("/api/models/seeds")
def api_model_seeds():
    return ma.load_local_seeds()


@app.get("/api/models/trending")
async def api_model_trending(limit: int = 12):
    return await ma.fetch_trending(limit=limit)


@app.get("/api/models/card")
async def api_model_card(model_id: str):
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id 必填")
    return await ma.fetch_model_card(model_id)


@app.get("/api/arxiv")
async def api_arxiv(query: str = "cat:cs.CL AND (large language model)", max: int = 10):
    return await ma.fetch_arxiv(query=query, max_results=max)


# ---------------- 刷题 API ----------------
# 注意：以下路由必须在静态目录挂载到 "/" 之前注册（同 /healthz）

@app.get("/api/study/queue")
def api_study_queue(limit: int = 20, module: str | None = Query(None)):
    return study.queue(CARDS, limit=limit, module=module)


@app.get("/api/study/quiz/{card_id}")
def api_study_quiz(card_id: str):
    view = study.quiz_view(card_id, CARDS)
    if not view:
        raise HTTPException(status_code=404, detail="卡片不存在")
    return view


@app.post("/api/study/answer")
def api_study_answer(payload: dict):
    body = payload or {}
    card_id = body.get("card_id", "")
    if not card_id or not get_card_by_id(card_id, CARDS):
        raise HTTPException(status_code=404, detail="卡片不存在")
    try:
        return study.answer(card_id, body.get("quality"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/study/stats")
def api_study_stats():
    return study.stats(CARDS)


@app.get("/api/study/wrong-book")
def api_study_wrong_book(module: str | None = Query(None)):
    return study.wrong_book(CARDS, module=module)


# ---------------- 面试模拟 API ----------------

@app.post("/api/interview/start")
async def api_interview_start(payload: dict):
    module = (payload or {}).get("module", "")
    if not module:
        raise HTTPException(status_code=400, detail="module 必填")
    result = await interview.start(module, CARDS)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "模块不存在"))
    return result


@app.post("/api/interview/answer")
async def api_interview_answer(payload: dict):
    body = payload or {}
    sid = body.get("session_id", "")
    answer = body.get("answer", "")
    if not interview.get_session(sid):
        raise HTTPException(status_code=404, detail="会话不存在")
    if not answer.strip():
        raise HTTPException(status_code=400, detail="answer 不能为空")
    return StreamingResponse(
        interview.answer_stream(sid, answer, CARDS),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/interview/self-answer")
def api_interview_self_answer(payload: dict):
    body = payload or {}
    sid = body.get("session_id", "")
    if not interview.get_session(sid):
        raise HTTPException(status_code=404, detail="会话不存在")
    res = interview.self_answer(sid, body.get("quality"), CARDS)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "提交失败"))
    return res


@app.get("/api/interview/report")
def api_interview_report(session_id: str):
    rep = interview.report(session_id, CARDS)
    if rep is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return rep


@app.post("/api/interview/enqueue")
def api_interview_enqueue(payload: dict):
    """把报告推荐的薄弱卡片提前到今日复习队列（也可直接传 card_ids）。"""
    body = payload or {}
    card_ids = body.get("card_ids")
    if not card_ids:
        rep = interview.report(body.get("session_id", ""), CARDS)
        if rep is None:
            raise HTTPException(status_code=404, detail="会话不存在且未指定 card_ids")
        card_ids = [s["card_id"] for s in rep["suggestions"]]
    return study.enqueue_today(card_ids)


# ---------------- 静态前端 ----------------

# 注意：/healthz 必须在静态目录挂载到 "/" 之前注册，否则会被 StaticFiles 拦截
@app.get("/healthz")
def healthz():
    return {"ok": True}


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
