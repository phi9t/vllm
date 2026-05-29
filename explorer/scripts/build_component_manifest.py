#!/usr/bin/env python3
"""Build the V1-engine component manifest for the Explorer's Component Deep Dive.

Source of truth is ``HACKERS_GUIDE.md`` (no network needed):
  * §2 "30-second architecture" table -> the core component nodes (box/file/symbol)
  * §2 mermaid flow                    -> the graph edges (hard-coded below, mirrors it)
  * §16 "Hands-on hacks" table         -> the runnable hacks, each paired with a section
  * ``## N. Title`` headers            -> one "section" node per major subsystem

Line numbers in the guide drift (the guide says so). We therefore resolve each
node's current line by grepping its symbol in the referenced file, falling back
to the guide's hint only if the symbol is not found.

Run via the repo venv (AGENTS.md):

    .venv/bin/python explorer/scripts/build_component_manifest.py \
        --repo-root . --out explorer/public/data/components.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# Regex helpers -------------------------------------------------------------
CODE_REF = re.compile(r"`([^`]+?\.py):(\d+)`")  # `path/to/file.py:123`
SYMBOL = re.compile(r"`((?:class|def)\s+[A-Za-z_][\w.]*)`")  # `class Foo` / `def bar`
TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
HEADER = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")
HACK_REF = re.compile(r"`(hacks/\d+_[A-Za-z0-9_]+\.py)`")


def slugify(text: str) -> str:
    text = text.strip().strip("`")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text


def resolve_line(repo_root: Path, file: str, symbol: str | None, hint: int) -> int:
    """Grep ``symbol`` in ``file`` to find its real current line (drift-proof)."""
    path = repo_root / file
    if not path.is_file() or not symbol:
        return hint
    needle = symbol.strip().strip("`")
    # Search for the full symbol, then progressively shorter prefixes.
    candidates = [needle]
    parts = needle.split()
    if len(parts) >= 2:
        candidates.append(f"{parts[0]} {parts[1]}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return hint
    for cand in candidates:
        for i, line in enumerate(lines, start=1):
            idx = line.find(cand)
            if idx == -1:
                continue
            # Word-boundary: the char after the symbol must not continue the
            # identifier (so "class OutputProcessor" does not match
            # "class OutputProcessorOutput").
            after = line[idx + len(cand) : idx + len(cand) + 1]
            if after and (after.isalnum() or after == "_"):
                continue
            return i
    return hint


def section_blocks(md: str) -> list[tuple[str, str, str]]:
    """Split the guide into (number, title, body) per ``## N. Title`` header."""
    blocks: list[tuple[str, str, str]] = []
    cur: list[str] | None = None
    num = title = ""
    for line in md.splitlines():
        m = HEADER.match(line)
        if m:
            if cur is not None:
                blocks.append((num, title, "\n".join(cur)))
            num, title = m.group(1), m.group(2)
            cur = []
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        blocks.append((num, title, "\n".join(cur)))
    return blocks


def parse_core_nodes(md: str, repo_root: Path) -> list[dict]:
    """Parse the §2 box->file->symbol table into core graph nodes."""
    nodes: list[dict] = []
    in_table = False
    for line in md.splitlines():
        if line.strip().startswith("| Box ") or line.strip().startswith("| `LLM`"):
            in_table = True
        if not in_table:
            continue
        m = TABLE_ROW.match(line)
        if not m:
            if nodes:  # table ended
                break
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if len(cells) < 3 or cells[0] in ("Box", "---") or set(cells[0]) <= {"-", " "}:
            continue
        box, file_cell, symbol_cell = cells[0], cells[1], cells[2]
        ref = CODE_REF.search(file_cell)
        if not ref:
            continue
        file, hint = ref.group(1), int(ref.group(2))
        sym = SYMBOL.search(symbol_cell)
        symbol = sym.group(1) if sym else None
        label = box.replace("`", "").strip()
        nodes.append(
            {
                "id": slugify(box),
                "label": label,
                "group": "core",
                "file": file,
                "line": resolve_line(repo_root, file, symbol, hint),
                "symbol": symbol,
            }
        )
    return nodes


def parse_hacks(md: str) -> list[dict]:
    """Parse the §16 hacks table: # | script | pairs-with | one-liner."""
    hacks: list[dict] = []
    for line in md.splitlines():
        m = TABLE_ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if len(cells) < 4:
            continue
        ref = HACK_REF.search(cells[1])
        if not ref or not re.match(r"^\d+$", cells[0]):
            continue
        section = cells[2].replace("§", "").strip()
        hacks.append(
            {"n": cells[0], "script": ref.group(1), "section": section, "desc": cells[3]}
        )
    return hacks


def parse_sections(md: str, repo_root: Path, hacks: list[dict]) -> list[dict]:
    """One node per major subsystem section, with its first code ref + paired hack."""
    by_section: dict[str, list[str]] = {}
    for h in hacks:
        by_section.setdefault(h["section"], []).append(h["script"])

    sections: list[dict] = []
    for num, title, body in section_blocks(md):
        ref = CODE_REF.search(body)
        sym = SYMBOL.search(body)
        file = ref.group(1) if ref else None
        line = (
            resolve_line(repo_root, file, sym.group(1) if sym else None, int(ref.group(2)))
            if ref
            else None
        )
        sections.append(
            {
                "id": f"section-{num}",
                "section": num,
                "title": title,
                "group": "section",
                "file": file,
                "line": line,
                "symbol": sym.group(1) if sym else None,
                "hacks": by_section.get(num, []),
            }
        )
    return sections


# The §2 mermaid flow, expressed over slugified §2 node ids (stable). -------
CORE_EDGES = [
    ("llm-asyncllm", "enginecoreproc"),
    ("enginecoreproc", "enginecore-step"),
    ("enginecore-step", "scheduler"),
    ("scheduler", "kvcachemanager"),
    ("enginecore-step", "executor"),
    ("executor", "gpumodelrunner"),
    ("gpumodelrunner", "sampler"),
    ("sampler", "enginecore-step"),
    ("enginecore-step", "outputprocessor"),
    ("outputprocessor", "llm-asyncllm"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--guide", default="HACKERS_GUIDE.md")
    ap.add_argument("--out", default="explorer/public/data/components.json")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    md = (repo_root / args.guide).read_text(encoding="utf-8")

    nodes = parse_core_nodes(md, repo_root)
    hacks = parse_hacks(md)
    sections = parse_sections(md, repo_root, hacks)

    node_ids = {n["id"] for n in nodes}
    edges = [{"from": a, "to": b} for a, b in CORE_EDGES if a in node_ids and b in node_ids]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "guide": args.guide,
        "nodes": nodes,
        "edges": edges,
        "sections": sections,
        "hacks": hacks,
    }

    out = (repo_root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # graphs/index.json — the switchable-subject index for Component mode.
    # `manifest` is relative to public/data/ (where the file above lives).
    graphs_dir = out.parent / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    (graphs_dir / "index.json").write_text(
        json.dumps([{"slug": "v1-engine", "label": "V1 engine", "manifest": out.name}], indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(
        f"wrote {out} : {len(nodes)} core nodes, {len(edges)} edges, "
        f"{len(sections)} sections, {len(hacks)} hacks; graphs/index.json"
    )


if __name__ == "__main__":
    main()
