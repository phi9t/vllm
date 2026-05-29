"""Shared symbol-resolution helpers for Explorer manifest scripts.

``resolve_line`` greps a symbol in a source file to find its real current line,
falling back to a hint only if the symbol is not found.  Import via::

    from _refs import resolve_line
"""

from __future__ import annotations

from pathlib import Path


def resolve_line(repo_root: Path, file: str, symbol: str | None, hint: int) -> int:
    """Grep ``symbol`` in ``file`` to find its real current line (drift-proof).

    Tries the full symbol string first, then progressively shorter prefixes
    (first two whitespace-separated tokens).  Falls back to ``hint`` when the
    file does not exist, ``symbol`` is None/empty, or no match is found.
    """
    path = repo_root / file
    if not path.is_file() or not symbol:
        return hint
    needle = symbol.strip().strip("`")
    # Build a list of candidates: full needle, then "kind name" prefix.
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
            # Word-boundary check: the char after the match must not continue
            # the identifier (prevents "class Foo" matching "class FooBar").
            after = line[idx + len(cand) : idx + len(cand) + 1]
            if after and (after.isalnum() or after == "_"):
                continue
            return i
    return hint
