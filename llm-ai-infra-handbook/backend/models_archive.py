"""模型档案模块：整理开源模型技术细节。

数据来源（均免 Key 的公开接口）：
1. 本地种子卡片 models/seeds.json —— 人工整理的经典/里程碑模型技术要点，稳定可复现。
2. HuggingFace Hub API —— /api/models?sort=trendingScore 拉取当前 trending 模型，
   再抓取每个模型的 README（model card）提取技术细节。
3. arXiv API —— 检索 LLM 相关最新论文作为前沿补充。

所有联网结果写入 data/ 缓存（默认 6 小时 TTL），避免重复请求与限流。
"""
from __future__ import annotations

import os
import json
import time
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEEDS_PATH = os.path.join(_HERE, "models", "seeds.json")
_DATA_DIR = os.path.join(_HERE, "data")
_CACHE_TTL = 6 * 3600  # 秒

os.makedirs(_DATA_DIR, exist_ok=True)

HF_API = "https://huggingface.co/api/models"
HF_README = "https://huggingface.co/{}/raw/main/README.md"
ARXIV_API = "http://export.arxiv.org/api/query"


def load_local_seeds() -> list[dict]:
    if not os.path.exists(_SEEDS_PATH):
        return []
    try:
        with open(_SEEDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _cache_get(name: str):
    path = os.path.join(_DATA_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if time.time() - obj.get("_ts", 0) > _CACHE_TTL:
            return None
        return obj.get("data")
    except (json.JSONDecodeError, OSError):
        return None


def _cache_set(name: str, data):
    path = os.path.join(_DATA_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"_ts": time.time(), "data": data}, f, ensure_ascii=False)


async def fetch_trending(limit: int = 12, filter_tag: str = "text-generation") -> dict:
    """拉取 HuggingFace trending 模型。返回 {ok, models, error}。"""
    cache_key = f"hf_trending_{limit}_{filter_tag}.json"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {"ok": True, "models": cached, "error": None, "cached": True}

    url = f"{HF_API}?sort=trendingScore&direction=-1&limit={limit}&filter={quote(filter_tag)}"
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            items = r.json()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "models": [], "error": str(e)}

    models = []
    for it in items:
        models.append({
            "id": it.get("id", ""),
            "likes": it.get("likes", 0),
            "downloads": it.get("downloads", 0),
            "pipeline_tag": it.get("pipeline_tag", ""),
            "last_modified": it.get("lastModified", ""),
            "tags": it.get("tags", []) or [],
            "author": (it.get("id", "").split("/")[0] if it.get("id") else ""),
        })
    _cache_set(cache_key, models)
    return {"ok": True, "models": models, "error": None, "cached": False}


# README 中常见的「技术细节」小标题，用于抽取
_DETAIL_SECTIONS = [
    "Model Details", "Model details", "Architecture", " architecture",
    "Training", "training", "Intended Use", "intended use",
    "Hardware", "hardware", "Tokenizer", "tokenizer", "License", "license",
]


async def fetch_model_card(model_id: str) -> dict:
    """抓取单个模型的 README，做轻量结构化抽取。"""
    cache_key = f"hf_card_{model_id.replace('/', '__')}.json"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    url = HF_README.format(quote(model_id, safe=""))
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            readme = r.text
    except Exception as e:  # noqa: BLE001
        result = {"id": model_id, "readme_available": False, "error": str(e),
                  "sections": {}, "highlights": [], "raw_len": 0}
        _cache_set(cache_key, result)
        return result

    # 按 H2 切分，抽取与架构/训练/细节相关的段落
    sections: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in readme.splitlines():
        if line.startswith("## "):
            if cur is not None:
                sections[cur] = "\n".join(buf).strip()
            cur = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if cur is not None:
        sections[cur] = "\n".join(buf).strip()

    picked = {k: v for k, v in sections.items()
              if any(sig.lower() in k.lower() for sig in
                     ["detail", "architect", "train", "use", "hardware", "token", "license"])}
    # 抽取亮点：含数字 + 参数的句子
    highlights = []
    for line in readme.splitlines():
        s = line.strip("-* ").strip()
        if re.search(r"\b(params|parameters|B|billion|trillion|context|tokens|layers|hidden)\b", s, re.I) \
                and 10 < len(s) < 200:
            highlights.append(s)
        if len(highlights) >= 12:
            break

    result = {
        "id": model_id,
        "readme_available": True,
        "error": None,
        "sections": picked,
        "highlights": highlights,
        "raw_len": len(readme),
    }
    _cache_set(cache_key, result)
    return result


async def fetch_arxiv(query: str = "cat:cs.CL AND (LLM OR large language model)",
                      max_results: int = 10) -> dict:
    """检索 arXiv 最新相关论文。"""
    cache_key = f"arxiv_{abs(hash(query)) % 100000}_{max_results}.json"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {"ok": True, "papers": cached, "error": None, "cached": True}

    url = (f"{ARXIV_API}?search_query={quote(query)}"
           f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}")
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            xml_text = r.text
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "papers": [], "error": str(e)}

    papers = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = (entry.find("a:title", ns).text or "").strip().replace("\n", " ")
            summary = (entry.find("a:summary", ns).text or "").strip().replace("\n", " ")
            published = (entry.find("a:published", ns).text or "")[:10]
            link = (entry.find("a:id", ns).text or "")
            authors = [a.find("a:name", ns).text for a in entry.findall("a:author", ns)][:5]
            papers.append({
                "title": title, "summary": summary[:400],
                "published": published, "link": link, "authors": authors,
            })
    except ET.ParseError as e:
        return {"ok": False, "papers": [], "error": f"parse: {e}"}

    _cache_set(cache_key, papers)
    return {"ok": True, "papers": papers, "error": None, "cached": False}
