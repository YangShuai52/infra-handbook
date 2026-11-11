"""刷题与间隔重复（SM-2）数据层。

存储：SQLite 单文件 backend/data/study.db，标准库 sqlite3，无外部依赖。
单用户设计；将来多用户只需给两张表各加 user_id 列。

本模块包含（A1 数据层 + A2 SM-2 调度）：
    init_db(cards)   幂等建表 + 孤儿卡片清理
    answer           记录自评并推进 SM-2 调度
    queue            今日复习队列（到期 + 新卡补齐）
    quiz_view        刷题视图（不含正文，避免瞄到答案）
    stats            学习仪表盘
    wrong_book       错题本

表结构（与 ROADMAP 3.2 一致）：
    review_card  卡片复习状态，card_id 对应 kb 卡片 id，首次刷题懒创建
    review_log   复习流水，用于统计与错题本
"""
from __future__ import annotations

import os
import random
import sqlite3
from datetime import datetime, date, timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "data", "study.db")

_DDL = """
CREATE TABLE IF NOT EXISTS review_card (
    card_id       TEXT PRIMARY KEY,
    ease          REAL NOT NULL DEFAULT 2.5,
    interval_days REAL NOT NULL DEFAULT 0,
    reps          INTEGER NOT NULL DEFAULT 0,
    lapses        INTEGER NOT NULL DEFAULT 0,
    due_at        TEXT NOT NULL,
    last_review   TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id         TEXT NOT NULL,
    reviewed_at     TEXT NOT NULL,
    quality         INTEGER NOT NULL,
    interval_before REAL NOT NULL,
    interval_after  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_log_card ON review_log(card_id);
CREATE INDEX IF NOT EXISTS idx_review_log_time ON review_log(reviewed_at);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return date.today().isoformat()


def init_db(cards: list[dict] | None = None) -> None:
    """幂等建表；传入 cards 时清理孤儿卡片。

    孤儿卡片：review_card 中 card_id 不在当前知识库里的记录（卡片被删除/改名）。
    处理策略：删除其 review_card 状态行（不再进队列/统计）；
    review_log 保留作为历史流水，不做删除（统计查询以 review_card 为主表连接，自然排除）。
    """
    conn = _connect()
    try:
        conn.executescript(_DDL)
        if cards is not None:
            valid = {c["id"] for c in cards}
            rows = conn.execute("SELECT card_id FROM review_card").fetchall()
            orphans = [r["card_id"] for r in rows if r["card_id"] not in valid]
            if orphans:
                conn.executemany(
                    "DELETE FROM review_card WHERE card_id = ?",
                    [(cid,) for cid in orphans],
                )
                print(f"[study] 清理孤儿卡片 {len(orphans)} 张: {orphans[:5]}{'…' if len(orphans) > 5 else ''}")
        conn.commit()
    finally:
        conn.close()


# ================= SM-2 调度（A2） =================

_VALID_QUALITY = {1, 3, 4, 5}
_EASE_MIN = 1.3
_EASE_MAX = 2.8
_MASTERED_INTERVAL = 21  # interval 达到此天数视为「已掌握」


def answer(card_id: str, quality: int) -> dict:
    """记录一次自评并推进 SM-2 调度。

    quality 合法值 {1, 3, 4, 5}（对应前端四档按钮），非法值抛 ValueError。
    首次刷卡懒创建状态行。返回 {"next_interval_days": int, "due_at": iso_date}。
    """
    if quality not in _VALID_QUALITY:
        raise ValueError(f"quality 必须是 {sorted(_VALID_QUALITY)} 之一，收到 {quality!r}")

    conn = _connect()
    try:
        row = conn.execute(
            "SELECT ease, interval_days, reps, lapses FROM review_card WHERE card_id = ?",
            (card_id,),
        ).fetchone()
        ease = row["ease"] if row else 2.5
        interval_before = row["interval_days"] if row else 0.0
        reps = row["reps"] if row else 0
        lapses = row["lapses"] if row else 0

        if quality < 3:
            # 遗忘：重来，明天再见
            reps = 0
            lapses += 1
            interval = 1
            ease = max(_EASE_MIN, ease - 0.2)
        else:
            reps += 1
            if reps == 1:
                interval = 1
            elif reps == 2:
                interval = 6
            else:
                interval = round(interval_before * ease)
            if quality == 3:
                ease = max(_EASE_MIN, ease - 0.15)
            elif quality == 5:
                ease = min(_EASE_MAX, ease + 0.1)

        due_at = (date.today() + timedelta(days=interval)).isoformat()
        now = _now()
        conn.execute(
            """
            INSERT INTO review_card
                (card_id, ease, interval_days, reps, lapses, due_at, last_review, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(card_id) DO UPDATE SET
                ease = excluded.ease,
                interval_days = excluded.interval_days,
                reps = excluded.reps,
                lapses = excluded.lapses,
                due_at = excluded.due_at,
                last_review = excluded.last_review
            """,
            (card_id, ease, interval, reps, lapses, due_at, now, now),
        )
        conn.execute(
            "INSERT INTO review_log (card_id, reviewed_at, quality, interval_before, interval_after)"
            " VALUES (?, ?, ?, ?, ?)",
            (card_id, now, quality, interval_before, interval),
        )
        conn.commit()
        return {"next_interval_days": interval, "due_at": due_at}
    finally:
        conn.close()


def enqueue_today(card_ids: list[str]) -> dict:
    """把指定卡片提前到今日复习队列（仅对已有复习记录的卡片生效；未学的新卡本就在队列里）。"""
    today = _today()
    conn = _connect()
    try:
        updated = []
        for cid in card_ids or []:
            cur = conn.execute(
                "UPDATE review_card SET due_at = ? WHERE card_id = ?", (today, cid)
            )
            if cur.rowcount:
                updated.append(cid)
        conn.commit()
        return {"updated": updated,
                "skipped_new": [c for c in (card_ids or []) if c not in updated]}
    finally:
        conn.close()


def log_review(card_id: str, quality: int) -> dict:
    """只记流水、不推进 SM-2 间隔。

    用于面试模拟中「同一张卡片在本场已被评过分」的后续回答：成绩要留档供报告
    使用，但不应让一次面试把间隔层层放大（否则卡片被推到几年后，等于退出复习队列）。
    """
    if quality not in _VALID_QUALITY:
        raise ValueError(f"quality 必须是 {sorted(_VALID_QUALITY)} 之一，收到 {quality!r}")
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT interval_days FROM review_card WHERE card_id = ?", (card_id,)
        ).fetchone()
        interval = row["interval_days"] if row else 0.0
        conn.execute(
            "INSERT INTO review_log (card_id, reviewed_at, quality, interval_before, interval_after)"
            " VALUES (?, ?, ?, ?, ?)",
            (card_id, _now(), quality, interval, interval),
        )
        conn.commit()
        return {"skipped": True, "next_interval_days": interval}
    finally:
        conn.close()


def queue(cards: list[dict], limit: int = 20, module: str | None = None) -> list[dict]:
    """今日复习队列。

    到期卡片（due_at <= today）按 due_at 升序在前；不足 limit 时用
    「从未刷过」的卡片随机补齐，保证新卡持续进入。孤儿卡片自动排除。
    """
    by_id = {c["id"]: c for c in cards}
    if module is not None:
        by_id = {cid: c for cid, c in by_id.items() if c["module"] == module}
    today = _today()

    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT card_id, due_at FROM review_card ORDER BY due_at ASC"
        ).fetchall()
    finally:
        conn.close()

    out: list[dict] = []
    studied: set[str] = set()
    for r in rows:
        cid = r["card_id"]
        if cid not in by_id:
            continue
        studied.add(cid)
        if r["due_at"] <= today and len(out) < limit:
            c = by_id[cid]
            out.append({"card_id": cid, "title": c["title"], "module": c["module"],
                        "is_new": False, "due_at": r["due_at"]})
    if len(out) < limit:
        fresh = [cid for cid in by_id if cid not in studied]
        random.shuffle(fresh)
        for cid in fresh[: limit - len(out)]:
            c = by_id[cid]
            out.append({"card_id": cid, "title": c["title"], "module": c["module"],
                        "is_new": True, "due_at": ""})
    return out


def quiz_view(card_id: str, cards: list[dict]) -> dict | None:
    """刷题视图：题目信息 + 章节标题，不返回正文（避免用户瞄到答案）。"""
    for c in cards:
        if c["id"] == card_id:
            return {
                "card_id": card_id,
                "title": c["title"],
                "module": c["module"],
                "tags": c["tags"],
                "summary": c["summary"],
                "headings": [s["heading"] for s in c["sections"] if s["heading"]],
            }
    return None


def _streak_days(review_dates: list[str]) -> int:
    """连续学习天数：从今天（或昨天）往回数的连续有复习记录的天数。"""
    day_set = {d for d in review_dates if d}
    if not day_set:
        return 0
    d = date.today()
    if d.isoformat() not in day_set:
        d -= timedelta(days=1)  # 今天还没学不打断连击，从昨天起算
    streak = 0
    while d.isoformat() in day_set:
        streak += 1
        d -= timedelta(days=1)
    return streak


def stats(cards: list[dict]) -> dict:
    """学习仪表盘：总量/已开始/已掌握/今日待复习/连续天数/分模块通过率。"""
    by_id = {c["id"]: c for c in cards}
    conn = _connect()
    try:
        rc = conn.execute(
            "SELECT card_id, interval_days, due_at FROM review_card"
        ).fetchall()
        dates = [r["reviewed_at"][:10] for r in
                 conn.execute("SELECT reviewed_at FROM review_log").fetchall()]
        mod_rows = conn.execute(
            """
            SELECT card_id, COUNT(*) AS total,
                   SUM(CASE WHEN quality >= 3 THEN 1 ELSE 0 END) AS passed
            FROM review_log GROUP BY card_id
            """
        ).fetchall()
    finally:
        conn.close()

    valid = [r for r in rc if r["card_id"] in by_id]
    # 模块归属在知识库卡片上而非 DB，按 by_id 归组
    agg: dict[str, dict] = {}
    for r in mod_rows:
        card = by_id.get(r["card_id"])
        if card is None:
            continue
        m = agg.setdefault(card["module"], {"answers": 0, "passed": 0})
        m["answers"] += r["total"]
        m["passed"] += r["passed"] or 0
    by_module = [
        {"module": m, "answers": v["answers"],
         "pass_rate": round(v["passed"] / v["answers"], 3) if v["answers"] else 0.0}
        for m, v in agg.items()
    ]
    by_module.sort(key=lambda x: x["module"])
    return {
        "total_cards": len(cards),
        "started": len(valid),
        "mastered": sum(1 for r in valid if r["interval_days"] >= _MASTERED_INTERVAL),
        "due_today": sum(1 for r in valid if r["due_at"] <= _today()),
        "streak_days": _streak_days(dates),
        "by_module": by_module,
    }


def wrong_book(cards: list[dict], module: str | None = None) -> list[dict]:
    """错题本：lapses > 0 或近 3 次自评均值 <= 2.5 的卡片。"""
    by_id = {c["id"]: c for c in cards}
    if module is not None:
        by_id = {cid: c for cid, c in by_id.items() if c["module"] == module}

    conn = _connect()
    try:
        lapse_rows = conn.execute(
            "SELECT card_id, lapses FROM review_card WHERE lapses > 0"
        ).fetchall()
        avg_rows = conn.execute(
            """
            SELECT card_id, AVG(quality) AS avg_q FROM (
                SELECT card_id, quality,
                       ROW_NUMBER() OVER (PARTITION BY card_id ORDER BY reviewed_at DESC) AS rn
                FROM review_log
            ) WHERE rn <= 3 GROUP BY card_id
            """
        ).fetchall()
    finally:
        conn.close()

    picked: dict[str, dict] = {}
    for r in lapse_rows:
        if r["card_id"] in by_id:
            picked[r["card_id"]] = {"lapses": r["lapses"], "avg_quality3": None}
    for r in avg_rows:
        if r["card_id"] in by_id and r["avg_q"] is not None and r["avg_q"] <= 2.5:
            item = picked.setdefault(r["card_id"], {"lapses": 0, "avg_quality3": None})
            item["avg_quality3"] = round(r["avg_q"], 2)

    out = [
        {"card_id": cid, "title": by_id[cid]["title"], "module": by_id[cid]["module"], **info}
        for cid, info in picked.items()
    ]
    out.sort(key=lambda x: (-x["lapses"], x["card_id"]))
    return out
