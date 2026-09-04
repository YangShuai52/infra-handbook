#!/usr/bin/env python3
"""Validate knowledge-card metadata and taxonomy integrity."""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "backend" / "kb"
REQUIRED = {
    "id", "title", "module", "tags", "difficulty", "freq",
    "updated", "summary", "source", "questions",
}


def parse(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not match:
        raise ValueError("missing or malformed YAML frontmatter")
    return yaml.safe_load(match.group(1)) or {}, match.group(2)


def main() -> int:
    errors: list[str] = []
    module_paths: dict[str, Path] = {}
    module_orders: list[int] = []

    for path in sorted(KB.rglob("_module.md")):
        try:
            meta, _ = parse(path)
            module = str(meta.get("module", ""))
            if not module or module in module_paths:
                errors.append(f"{path}: missing or duplicate module id {module!r}")
            module_paths[module] = path.parent
            module_orders.append(int(meta.get("order", -1)))
        except (ValueError, TypeError, yaml.YAMLError) as exc:
            errors.append(f"{path}: {exc}")

    if len(module_paths) != 20:
        errors.append(f"expected 20 modules, found {len(module_paths)}")
    if len(module_orders) != len(set(module_orders)):
        errors.append("module order values must be unique")

    seen: dict[str, Path] = {}
    counts: Counter[str] = Counter()
    card_paths = [
        p for p in sorted(KB.rglob("*.md"))
        if p.name not in {"_module.md", "README.md"}
    ]
    for path in card_paths:
        try:
            meta, body = parse(path)
        except (ValueError, yaml.YAMLError) as exc:
            errors.append(f"{path}: {exc}")
            continue

        missing = sorted(REQUIRED - set(meta))
        if missing:
            errors.append(f"{path}: missing fields {', '.join(missing)}")
        card_id = str(meta.get("id", ""))
        if card_id in seen:
            errors.append(f"{path}: duplicate id {card_id!r}; first seen at {seen[card_id]}")
        elif card_id:
            seen[card_id] = path

        module = str(meta.get("module", ""))
        counts[module] += 1
        if module not in module_paths:
            errors.append(f"{path}: unknown module {module!r}")
        elif path.parent != module_paths[module]:
            errors.append(f"{path}: file is outside its declared module directory")

        sources = meta.get("source", [])
        if not isinstance(sources, list) or not sources or not all(
            isinstance(url, str) and url.startswith(("https://", "http://")) for url in sources
        ):
            errors.append(f"{path}: source must be a non-empty URL list")
        questions = meta.get("questions", [])
        if not isinstance(questions, list) or not 3 <= len(questions) <= 5:
            errors.append(f"{path}: questions must contain 3 to 5 items")
        if "## 面试追问" not in body:
            errors.append(f"{path}: missing 面试追问 section")

    if len(card_paths) < 150:
        errors.append(f"expected at least 150 cards, found {len(card_paths)}")
    empty = sorted(set(module_paths) - set(counts))
    if empty:
        errors.append(f"empty modules: {', '.join(empty)}")

    if errors:
        print("Knowledge-base validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"OK: {len(card_paths)} cards / {len(module_paths)} modules / {len(seen)} unique ids")
    for module in sorted(counts, key=lambda item: int(item.split('-', 1)[0])):
        print(f"  {module}: {counts[module]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
