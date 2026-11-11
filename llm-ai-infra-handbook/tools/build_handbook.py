#!/usr/bin/env python3
"""Build the Docsify handbook artifact from the canonical Markdown cards."""
from __future__ import annotations

import argparse
import html
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "backend" / "kb"
SITE = ROOT / "site"
TRACKS = {
    "01-llm-foundations": ("Part I", "LLM 基础"),
    "02-model-lifecycle": ("Part II", "模型生命周期"),
    "03-ai-infrastructure": ("Part III", "AI 基础设施"),
    "04-llm-systems": ("Part IV", "LLM 系统"),
}


def parse_markdown(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not match:
        raise ValueError(f"missing frontmatter: {path}")
    return yaml.safe_load(match.group(1)) or {}, match.group(2).strip()


def safe_label(text: str) -> str:
    return str(text).replace("[", "\\[").replace("]", "\\]")


def route_for(path: Path) -> str:
    return "/" + path.relative_to(KB).with_suffix(".md").as_posix()


def discover() -> tuple[list[dict], list[dict]]:
    modules: list[dict] = []
    cards: list[dict] = []
    for path in sorted(KB.rglob("_module.md")):
        meta, body = parse_markdown(path)
        modules.append({**meta, "body": body, "dir": path.parent, "route": route_for(path.parent / "README.md")})
    modules.sort(key=lambda item: int(item["order"]))

    for path in sorted(KB.rglob("*.md")):
        if path.name in {"_module.md", "README.md"}:
            continue
        meta, body = parse_markdown(path)
        cards.append({**meta, "body": body, "path": path, "route": route_for(path)})
    cards.sort(key=lambda item: (int(item.get("difficulty", 3)), str(item["title"])))
    return modules, cards


def rewrite_internal_links(body: str, id_to_route: dict[str, str]) -> str:
    pattern = re.compile(r"(\[[^\]]+\]\()([^)/#:][^)]*)(\))")

    def replace(match: re.Match[str]) -> str:
        target = match.group(2).strip()
        route = id_to_route.get(target)
        return f"{match.group(1)}{route}{match.group(3)}" if route else match.group(0)

    return pattern.sub(replace, body)


def render_card(card: dict, id_to_route: dict[str, str]) -> str:
    tags = "".join(f'<span class="meta-pill">{html.escape(str(tag))}</span>' for tag in card.get("tags", []))
    meta = (
        '<div class="card-meta">'
        f'<span class="meta-pill">难度 {html.escape(str(card.get("difficulty", 3)))}</span>'
        f'<span class="meta-pill">频率 {html.escape(str(card.get("freq", "—")))}</span>'
        f"{tags}</div>"
    )
    sources = card.get("source", [])
    source_lines = "\n".join(f"{i}. [{url}]({url})" for i, url in enumerate(sources, 1))
    source_block = f"\n\n<div class=\"source-list\">\n\n## 延伸来源\n\n{source_lines}\n\n</div>" if sources else ""
    body = rewrite_internal_links(card["body"], id_to_route)
    return (
        f"# {card['title']}\n\n"
        f"> {card.get('summary', '')}\n\n"
        f"{meta}\n\n"
        f"{body}{source_block}\n"
    )


def render_module(module: dict, cards: list[dict]) -> str:
    lines = [
        f"# {module['order']:02d}. {module['title']}", "",
        f"> {module.get('desc', '')}", "", module["body"], "", "## 本章卡片", "",
        '<ul class="module-list">',
    ]
    for card in cards:
        lines.append(
            f'<li><a href="#{card["route"]}">{html.escape(str(card["title"]))}</a>'
            f'<small>{html.escape(str(card.get("summary", "")))}</small></li>'
        )
    lines.extend(["</ul>", ""])
    return "\n".join(lines)


def render_home(modules: list[dict], cards_by_module: dict[str, list[dict]]) -> str:
    lines = [
        "# LLM & AI Infra Handbook", "",
        '<p class="handbook-lead">从模型第一性原理走到训练集群、推理服务和 Agent 生产系统。'
        f'当前收录 <strong>{sum(len(v) for v in cards_by_module.values())}</strong> 张知识卡片、'
        f'<strong>{len(modules)}</strong> 个模块。</p>', "",
        '<div class="path-grid">',
        '<div class="path-card"><strong>LLM 算法</strong><span>模型原理 → 数据与训练 → 对齐与评估</span></div>',
        '<div class="path-card"><strong>训练 Infra</strong><span>GPU → 多维并行 → 平台与可靠性</span></div>',
        '<div class="path-card"><strong>推理 Infra</strong><span>KV 与量化 → Runtime → 调度与成本</span></div>',
        '<div class="path-card"><strong>RAG / Agent</strong><span>检索 → 工具与上下文 → 安全治理</span></div>',
        '</div>', "", "## 从哪里开始", "",
        "第一次接触 LLM，建议从 Part I 顺序阅读；已有模型基础，可直接选择训练 Infra、推理 Infra 或 RAG / Agent 路线。每个模块首页都列出本章卡片和学习验收方式。", "",
        "[查看四条学习路径 →](/learning-paths.md)", "",
    ]
    current_track = None
    for module in modules:
        track = module["dir"].relative_to(KB).parts[0]
        if track != current_track:
            part, title = TRACKS[track]
            lines.extend([f"## {part} · {title}", "", '<ul class="module-list">'])
            current_track = track
        count = len(cards_by_module[module["module"]])
        lines.append(
            f'<li><a href="#{module["route"]}">{module["order"]:02d}. {html.escape(str(module["title"]))}</a>'
            f'<small>{html.escape(str(module.get("desc", "")))} · {count} 张卡片</small></li>'
        )
        next_index = modules.index(module) + 1
        next_track = None if next_index == len(modules) else modules[next_index]["dir"].relative_to(KB).parts[0]
        if next_track != track:
            lines.extend(["</ul>", ""])
    lines.extend([
        "## 阅读原则", "",
        "本手册优先使用论文、规范、官方文档和源码作为证据。性能数字需要结合模型、精度、硬件、拓扑和负载理解，不应脱离条件机械复用。", "",
    ])
    return "\n".join(lines)


def render_sidebar(modules: list[dict], cards_by_module: dict[str, list[dict]]) -> str:
    lines = ["- [手册首页](/README.md)", "- [学习路径](/learning-paths.md)", ""]
    for track, (part, title) in TRACKS.items():
        lines.append(f"- **{part} · {title}**")
        for module in [m for m in modules if m["dir"].relative_to(KB).parts[0] == track]:
            lines.append(f"  - [{module['order']:02d}. {safe_label(module['title'])}]({module['route']})")
            for card in cards_by_module[module["module"]]:
                lines.append(f"    - [{safe_label(card['title'])}]({card['route']})")
        lines.append("")
    return "\n".join(lines)


def build(output: Path) -> None:
    output = output.resolve()
    if ROOT not in output.parents or output in {ROOT, SITE} or SITE in output.parents:
        raise ValueError("output must be a dedicated directory inside the repository")
    modules, cards = discover()
    cards_by_module: dict[str, list[dict]] = defaultdict(list)
    for card in cards:
        cards_by_module[card["module"]].append(card)
    id_to_route = {card["id"]: card["route"] for card in cards}

    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(SITE, output)
    (output / ".nojekyll").touch()
    (output / "README.md").write_text(render_home(modules, cards_by_module), encoding="utf-8")
    (output / "_sidebar.md").write_text(render_sidebar(modules, cards_by_module), encoding="utf-8")

    for module in modules:
        rel_dir = module["dir"].relative_to(KB)
        target_dir = output / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_dir.joinpath("README.md").write_text(
            render_module(module, cards_by_module[module["module"]]), encoding="utf-8"
        )
    for card in cards:
        target = output / card["path"].relative_to(KB)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_card(card, id_to_route), encoding="utf-8")

    print(f"Built {len(cards)} cards / {len(modules)} modules into {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
