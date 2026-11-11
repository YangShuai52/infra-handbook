"""知识库加载与解析。

读取 backend/kb 下的 Markdown 卡片（带 YAML frontmatter），解析为结构化对象，
并切分正文章节，供检索引擎与前端展示使用。

卡片 frontmatter 字段约定：
    id, title, module, tags, difficulty, freq, updated, summary, source, questions
模块说明文件 _module.md 字段：
    id, title, module, order, is_module, desc
"""
from __future__ import annotations

import os
import re
import yaml

KB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")

_FREQ_WEIGHT = {"高": 1.25, "中": 1.0, "低": 0.85, "": 1.0}
_DIFF_LABEL = {1: "入门", 2: "基础", 3: "进阶", 4: "深入", 5: "专家"}


def _parse_frontmatter(text: str):
    """提取并解析 YAML frontmatter，返回 (meta: dict, body: str)。"""
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, m.group(2)


def _split_sections(body: str):
    """将正文按 H2 (## ) 切分为章节列表。

    返回 [{"heading": str, "content": str}, ...]，第一个无标题块（如一句话结论前的
    引言）会落入 heading 为 "" 的章节。
    """
    sections = []
    cur_heading = ""
    cur_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            if cur_heading != "" or cur_lines:
                sections.append({"heading": cur_heading, "content": "\n".join(cur_lines).strip()})
            cur_heading = line[3:].strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_heading != "" or cur_lines:
        sections.append({"heading": cur_heading, "content": "\n".join(cur_lines).strip()})
    return sections


def _first_paragraph(body: str, max_len: int = 220) -> str:
    """取正文中第一个非空段落，用于搜索摘要。"""
    for sec in _split_sections(body):
        for para in sec["content"].split("\n\n"):
            para = para.strip()
            if para and not para.startswith("#") and not para.startswith("```"):
                return para[:max_len] + ("…" if len(para) > max_len else "")
    return ""


def load_cards() -> list[dict]:
    """加载全部知识卡片（不含 _module.md）。"""
    cards = []
    for dirpath, _dirs, files in os.walk(KB_ROOT):
        for fn in files:
            if not fn.endswith(".md") or fn == "_module.md":
                continue
            path = os.path.join(dirpath, fn)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            meta, body = _parse_frontmatter(text)
            if not meta.get("id"):
                continue
            sections = _split_sections(body)
            cards.append({
                "id": meta["id"],
                "title": meta.get("title", fn),
                "module": meta.get("module", ""),
                "tags": meta.get("tags", []) or [],
                "difficulty": int(meta.get("difficulty", 3) or 3),
                "diff_label": _DIFF_LABEL.get(int(meta.get("difficulty", 3) or 3), "基础"),
                "freq": meta.get("freq", ""),
                "freq_weight": _FREQ_WEIGHT.get(str(meta.get("freq", "")), 1.0),
                "updated": str(meta.get("updated", "")),
                "summary": meta.get("summary", ""),
                "source": ([meta["source"]] if isinstance(meta.get("source"), str)
                           else (meta.get("source", []) or [])),
                "questions": meta.get("questions", []) or [],
                "body": body,
                "sections": sections,
                "abstract": _first_paragraph(body),
                "path": path,
                "file": fn,
            })
    return cards


def load_modules() -> list[dict]:
    """加载模块说明（_module.md），按 order 排序。"""
    modules = []
    for dirpath, _dirs, files in os.walk(KB_ROOT):
        if "_module.md" not in files:
            continue
        path = os.path.join(dirpath, "_module.md")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        meta, body = _parse_frontmatter(text)
        modules.append({
            "id": meta.get("id", ""),
            "title": meta.get("title", ""),
            "module": meta.get("module", ""),
            "order": int(meta.get("order", 99) or 99),
            "desc": meta.get("desc", ""),
            "body": body,
        })
    modules.sort(key=lambda m: m["order"])
    return modules


def get_card_by_id(card_id: str, cards: list[dict] | None = None) -> dict | None:
    if cards is None:
        cards = load_cards()
    for c in cards:
        if c["id"] == card_id:
            return c
    return None
