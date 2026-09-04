#!/usr/bin/env python3
"""Validate the generated GitHub Pages handbook artifact."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "backend" / "kb"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site", nargs="?", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    site = args.site.resolve()
    errors: list[str] = []

    for required in ("index.html", "README.md", "_sidebar.md", "_navbar.md", "styles.css", "theme.js", "favicon.svg", ".nojekyll"):
        if not (site / required).exists():
            errors.append(f"missing {required}")

    source_cards = [p for p in KB.rglob("*.md") if p.name not in {"_module.md", "README.md"}]
    generated_cards = [
        p for p in site.rglob("*.md")
        if p.name not in {"README.md", "_sidebar.md", "_navbar.md", "learning-paths.md"}
    ]
    if len(generated_cards) != len(source_cards):
        errors.append(f"expected {len(source_cards)} generated cards, found {len(generated_cards)}")

    for source in source_cards:
        target = site / source.relative_to(KB)
        if not target.exists():
            errors.append(f"missing generated card {target.relative_to(site)}")
        elif target.read_text(encoding="utf-8").startswith("---"):
            errors.append(f"frontmatter leaked into {target.relative_to(site)}")

    link_pattern = re.compile(r"\[[^\]]*\]\((/[^)#?]+)(?:#[^)]*)?\)")
    for path in site.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for route in link_pattern.findall(text):
            target = site / route.lstrip("/")
            if not target.exists():
                errors.append(f"{path.relative_to(site)} links to missing {route}")
        for route in re.findall(r'href="#(/[^"#?]+)"', text):
            target = site / route.lstrip("/")
            if not target.exists():
                errors.append(f"{path.relative_to(site)} links to missing #{route}")

    index = (site / "index.html").read_text(encoding="utf-8") if (site / "index.html").exists() else ""
    if "@latest" in index:
        errors.append("index.html must pin CDN dependencies")
    if "docsify@4.13.1" not in index:
        errors.append("Docsify runtime is missing or unpinned")
    sidebar = (site / "_sidebar.md").read_text(encoding="utf-8") if (site / "_sidebar.md").exists() else ""
    sidebar_card_links = sum(1 for p in generated_cards if f"](/{p.relative_to(site).as_posix()})" in sidebar)
    if sidebar_card_links != len(source_cards):
        errors.append(f"sidebar exposes {sidebar_card_links}/{len(source_cards)} cards")

    if errors:
        print("Handbook validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"OK: static handbook contains {len(generated_cards)} cards and no broken internal routes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
