"""中英混合 BM25 检索引擎（零额外依赖）。

中文无分词器，采用「字二元（bigram）」切分 + 单字回退，对短查询与包含二字词的查询
效果较好；英文按词切分并小写化。再叠加元数据加权：标签命中、标题命中、频率字段。

对外提供：build_index(cards) -> index；search(index, query, top_k) -> [hit...]
"""
from __future__ import annotations

import math
import re

_ENG_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """对一段文本做混合分词，返回 term 列表。"""
    text = text.lower()
    tokens: list[str] = []
    # 英文 / 数字词
    tokens.extend(_ENG_RE.findall(text))
    # 中文：先抽取连续汉字串，再整体做 bigram
    for han_seg in re.findall(r"[\u4e00-\u9fff]+", text):
        for i in range(len(han_seg) - 1):
            tokens.append(han_seg[i:i + 2])
        if han_seg:
            tokens.append(han_seg[-1])  # 末字单字，保证单字查询可命中
    return tokens


def _tokenize_query(text: str) -> list[str]:
    """查询分词：英文 + 中文二元；同时把中文查询串也整体作为单字序列加入，提升召回。"""
    terms = tokenize(text)
    # 原始查询里的中文片段额外补单字（bigram 已部分覆盖，这里确保单字在）
    for han_seg in re.findall(r"[\u4e00-\u9fff]+", text.lower()):
        terms.extend(list(han_seg))
    # 去重但保留顺序稳定性
    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


class Index:
    def __init__(self, cards: list[dict]):
        self.cards = cards
        self.doc_count = len(cards)
        # 每篇文档的 term -> tf
        self.doc_tf: list[dict[str, int]] = []
        # term -> df
        self.df: dict[str, int] = {}
        # 每篇文档的标签集合与标题（用于元数据加权）
        self.doc_tags: list[set[str]] = []
        self.doc_title: list[str] = []
        # 平均文档长度（按 token 数）
        self.avg_dl = 0.0

        all_lens = []
        for c in cards:
            # 索引语料 = 标题 + 标签 + summary + 常见问法 + 正文前 4000 字。
            # questions 让用户自然语言问法即使不复用正文措辞也能命中卡片。
            corpus = " ".join([
                c["title"], " ".join(c["tags"]),
                c.get("summary", ""), " ".join(c.get("questions", [])),
                c["body"][:4000],
            ])
            tf: dict[str, int] = {}
            toks = tokenize(corpus)
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tf.append(tf)
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
            self.doc_tags.append(set(c["tags"]))
            self.doc_title.append(c["title"].lower())
            all_lens.append(len(toks))
        self.avg_dl = (sum(all_lens) / len(all_lens)) if all_lens else 1.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        # BM25 标准 idf，加 1 平滑
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))


def build_index(cards: list[dict]) -> Index:
    return Index(cards)


_K1 = 1.5
_B = 0.75


def search(index: Index, query: str, top_k: int = 8, min_score: float = 0.0) -> list[dict]:
    """返回命中列表，按分数降序，含调试信息。"""
    q_terms = _tokenize_query(query)
    if not q_terms:
        return []

    # 查询 term 频率（同一词重复出现视为强调）
    q_tf: dict[str, int] = {}
    for t in q_terms:
        q_tf[t] = q_tf.get(t, 0) + 1

    results = []
    tag_terms = set(query.lower().split())

    for i, card in enumerate(index.cards):
        score = 0.0
        tf_map = index.doc_tf[i]
        dl = sum(tf_map.values())
        for term, qf in q_tf.items():
            f = tf_map.get(term, 0)
            if f == 0:
                continue
            idf = index.idf(term)
            if idf == 0:
                continue
            # 标准 BM25 饱和度项
            denom = f * (_K1 + 1) / (f + _K1 * (1 - _B + _B * dl / index.avg_dl))
            bm25 = idf * denom
            # 查询词频轻微加权
            bm25 *= (1 + 0.1 * (qf - 1))
            score += bm25

        if score <= 0:
            continue

        # ---- 元数据加权 ----
        # 1) 标签命中（精确标签词），强加权
        tag_hit = 0
        for tag in index.doc_tags[i]:
            if tag.lower() in tag_terms:
                tag_hit += 1
        if tag_hit:
            score *= (1 + 0.35 * tag_hit)
        # 2) 标题命中（任一查询词出现在标题）
        title = index.doc_title[i]
        if any(t in title for t in q_terms):
            score *= 1.2
        # 3) 频率字段加权（高频考点略加权）
        score *= card.get("freq_weight", 1.0)

        results.append({
            "card": card,
            "score": round(score, 4),
            "matched_terms": [t for t in q_terms if tf_map.get(t, 0) > 0],
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    if min_score > 0:
        results = [r for r in results if r["score"] >= min_score]
    return results[:top_k]


def snippet(card: dict, query: str, max_chars: int = 240) -> str:
    """从正文中找出最匹配查询的段落作为摘要。"""
    q_terms = set(_tokenize_query(query))
    best_sec, best_score = card.get("abstract", ""), -1
    for sec in card["sections"]:
        text = sec["content"]
        # 用段落检索最相关片段
        paras = [p for p in text.split("\n\n") if len(p.strip()) > 20]
        for p in paras:
            score = sum(p.lower().count(t) for t in q_terms)
            if score > best_score:
                best_score, best_sec = score, p.strip()
    if best_score <= 0:
        return card.get("abstract", "")[:max_chars]
    # 截断并去 markdown 噪音
    s = re.sub(r"[#*`>|]", "", best_sec)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars] + ("…" if len(s) > max_chars else "")
