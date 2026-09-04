"""答案生成器：检索增强生成（RAG）或降级为纯检索卡片。

- 检测到 LLM API 配置（环境变量）时：用检索到的卡片作为上下文，调用 OpenAI 兼容接口
  进行流式生成，并要求模型标注引用卡片。
- 未配置时：不调用大模型，直接把排序后的卡片作为答案返回（功能不降级，只是没有整合讲解）。

环境变量：
    LLM_API_KEY      必填（有值才启用 RAG）
    LLM_BASE_URL     兼容接口地址，默认 https://api.openai.com/v1
    LLM_MODEL        模型名，默认 gpt-4o-mini
    LLM_TEMPERATURE  默认 0.3
"""
from __future__ import annotations

import os
import json

import httpx

_DEFAULT_BASE = "https://api.openai.com/v1"
_SYSTEM_PROMPT = """你是一名资深 LLM 算法工程师，正在辅导候选人准备大模型面试（"八股"）。
你的回答必须遵循以下风格：
1. 结论先行：用一两句话直接给出核心答案。
2. 原理推导：关键结论要给出公式或推导，尤其是面试官常考的推导（如参数量、RoPE、DPO、显存）。
3. 具体例子：用真实配置（如 LLaMA-7B、A100 80G）给可复现的数字。
4. 末尾用「参考卡片」列出你引用的卡片标题（来自下面提供的上下文标题）。
5. 如果提供的上下文不足以回答问题，明确说明"知识库未覆盖"，并给出最相关的卡片供用户进一步阅读，不要编造。

请用中文作答，保持专业、简洁、可记忆。"""


def is_llm_enabled() -> bool:
    return bool(os.environ.get("LLM_API_KEY", "").strip())


def llm_config() -> dict:
    return {
        "base_url": os.environ.get("LLM_BASE_URL", _DEFAULT_BASE).rstrip("/"),
        "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.3")),
        "api_key": os.environ.get("LLM_API_KEY", ""),
    }


async def chat_complete(messages: list[dict], temperature: float | None = None) -> str:
    """非流式对话补全，供面试官等不需要 SSE 的场景使用。

    未配置 LLM_API_KEY 或接口异常时抛异常，由调用方决定降级行为。
    """
    cfg = llm_config()
    if not cfg["api_key"]:
        raise RuntimeError("LLM_API_KEY 未配置")
    payload = {
        "model": cfg["model"],
        "temperature": cfg["temperature"] if temperature is None else temperature,
        "messages": messages,
    }
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{cfg['base_url']}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def chat_stream(messages: list[dict], temperature: float | None = None):
    """流式对话补全，逐段 yield 内容增量。

    与 stream_answer（检索问答专用）不同，本函数接受任意 messages，
    接口异常/未配置 Key 时抛异常，由调用方决定降级行为。
    """
    cfg = llm_config()
    if not cfg["api_key"]:
        raise RuntimeError("LLM_API_KEY 未配置")
    payload = {
        "model": cfg["model"],
        "temperature": cfg["temperature"] if temperature is None else temperature,
        "stream": True,
        "messages": messages,
    }
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", f"{cfg['base_url']}/chat/completions", json=payload, headers=headers
        ) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                raise RuntimeError(f"接口返回 {resp.status_code}: {err[:200]!r}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if text:
                    yield text


def _build_user_prompt(query: str, hits: list[dict], ctx_chars: int = 6000) -> str:
    parts = [f"用户问题：{query}\n\n以下是知识库中检索到的相关卡片（按相关度排序）：\n"]
    for i, h in enumerate(hits, 1):
        card = h["card"]
        body = card["body"][:ctx_chars]
        parts.append(
            f"--- 卡片{i}（标题：{card['title']}；id：{card['id']}；频率：{card.get('freq','')}）---\n"
            f"{card.get('summary','')}\n{body}\n"
        )
    parts.append(
        "请基于以上卡片回答用户问题。若上下文足够，请整合讲解并标注参考卡片标题；"
        "若不足，请说明并列出相关卡片标题。"
    )
    return "\n".join(parts)


async def stream_answer(query: str, hits: list[dict]):
    """异步生成器，逐步 yield SSE 事件字符串（已含末尾的 \\n\\n）。

    yield 内容形如：
        data: {"type":"token","text":"..."}\n\n
        data: {"type":"sources","data":[...]}\n\n
        data: {"type":"done"}\n\n
    """
    base_url = os.environ.get("LLM_BASE_URL", _DEFAULT_BASE).rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
    api_key = os.environ["LLM_API_KEY"]

    sources = [
        {
            "id": h["card"]["id"],
            "title": h["card"]["title"],
            "module": h["card"]["module"],
            "score": h["score"],
            "snippet": h.get("snippet", ""),
        }
        for h in hits
    ]
    yield "data: " + json.dumps({"type": "sources", "data": sources}, ensure_ascii=False) + "\n\n"

    user_prompt = _build_user_prompt(query, hits)
    payload = {
        "model": model,
        "temperature": temperature,
        "stream": True,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", f"{base_url}/chat/completions", json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    yield "data: " + json.dumps(
                        {"type": "token", "text": f"\n[生成失败：接口返回 {resp.status_code}] {err[:300]}"},
                        ensure_ascii=False,
                    ) + "\n\n"
                    yield "data: " + json.dumps({"type": "done"}) + "\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        yield "data: " + json.dumps(
                            {"type": "token", "text": text}, ensure_ascii=False
                        ) + "\n\n"
    except Exception as e:  # noqa: BLE001
        yield "data: " + json.dumps(
            {"type": "token", "text": f"\n[生成异常：{e}]"}, ensure_ascii=False
        ) + "\n\n"

    yield "data: " + json.dumps({"type": "done"}) + "\n\n"
