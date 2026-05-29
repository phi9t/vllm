#!/usr/bin/env python3
"""Build the Data Exploration manifests from a small fineweb-edu subset.

Emits three files into ``public/data/``:
  * ``fineweb_schema.json``  — column docs for the format view (always written)
  * ``fineweb_sample.json``  — N sample rows (text + metadata columns)
  * ``tokenization.json``    — Qwen3 token pieces + counts for a few samples,
                               compared against the dataset's own ``token_count``
                               (which fineweb-edu computes with the GPT-2 tokenizer).

Real path uses ``datasets`` (streaming) + the Qwen3 HF tokenizer. If either is
unavailable (no network / libs), it falls back to a small embedded sample and a
whitespace byte-ish tokenization so the UI still works offline.

Run via the repo venv (AGENTS.md):

    .venv/bin/python explorer/scripts/build_fineweb_sample.py \
        --rows 200 --out-dir explorer/public/data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-10BT"
TOKENIZER_MODEL = "Qwen/Qwen3-0.6B"
TOK_PREVIEW_CHARS = 280  # cap text fed to the tokenization view
TEXT_PREVIEW_CHARS = 1200  # cap text stored per sample row
N_TOKENIZE = 8  # rows to fully tokenize for the tokenization view

SCHEMA = {
    "dataset": DATASET,
    "columns": [
        {"name": "text", "type": "string", "desc": "The extracted, cleaned web-page text."},
        {"name": "id", "type": "string", "desc": "Stable per-document identifier."},
        {"name": "dump", "type": "string", "desc": "CommonCrawl dump the doc came from (e.g. CC-MAIN-2024-10)."},
        {"name": "url", "type": "string", "desc": "Source URL of the page."},
        {"name": "date", "type": "string", "desc": "Crawl date."},
        {"name": "file_path", "type": "string", "desc": "Path of the source WARC/parquet shard."},
        {"name": "language", "type": "string", "desc": "Detected language code (this subset is 'en')."},
        {"name": "language_score", "type": "float", "desc": "fastText language-ID confidence (0-1)."},
        {"name": "token_count", "type": "int", "desc": "Token count per the GPT-2 tokenizer (dataset-provided)."},
        {"name": "score", "type": "float", "desc": "Edu-quality classifier score (raw, ~0-5)."},
        {"name": "int_score", "type": "int", "desc": "Rounded integer edu score used for filtering."},
    ],
}

EMBEDDED_ROWS = [
    {
        "text": "Photosynthesis is the process by which green plants and some other organisms "
        "use sunlight to synthesize foods from carbon dioxide and water. In plants, "
        "photosynthesis generally takes place in the leaves, inside organelles called "
        "chloroplasts. The overall reaction converts light energy into chemical energy "
        "stored in glucose.",
        "id": "<urn:uuid:0001>",
        "dump": "CC-MAIN-2024-10",
        "url": "https://example.edu/biology/photosynthesis",
        "date": "2024-03-01T00:00:00Z",
        "file_path": "sample/000_00000.parquet",
        "language": "en",
        "language_score": 0.98,
        "token_count": 64,
        "score": 4.1,
        "int_score": 4,
    },
    {
        "text": "The French Revolution was a period of radical political and societal change "
        "in France that began with the Estates General of 1789 and ended with the "
        "formation of the French Consulate in November 1799. Many of its ideas are "
        "considered fundamental principles of liberal democracy.",
        "id": "<urn:uuid:0002>",
        "dump": "CC-MAIN-2024-10",
        "url": "https://example.edu/history/french-revolution",
        "date": "2024-03-02T00:00:00Z",
        "file_path": "sample/000_00000.parquet",
        "language": "en",
        "language_score": 0.97,
        "token_count": 52,
        "score": 3.7,
        "int_score": 4,
    },
    {
        "text": "A prime number is a natural number greater than 1 that is not a product of two "
        "smaller natural numbers. The first few primes are 2, 3, 5, 7, 11 and 13. The "
        "fundamental theorem of arithmetic establishes the central role of primes in "
        "number theory: every integer greater than 1 is either prime or can be "
        "factorized uniquely into primes.",
        "id": "<urn:uuid:0003>",
        "dump": "CC-MAIN-2024-18",
        "url": "https://example.edu/math/prime-numbers",
        "date": "2024-04-12T00:00:00Z",
        "file_path": "sample/000_00001.parquet",
        "language": "en",
        "language_score": 0.99,
        "token_count": 71,
        "score": 4.5,
        "int_score": 5,
    },
    {
        "text": "Newton's laws of motion are three basic laws of classical mechanics that "
        "describe the relationship between the motion of an object and the forces "
        "acting on it. The first law states that an object remains at rest or in "
        "uniform motion unless acted upon by a net external force.",
        "id": "<urn:uuid:0004>",
        "dump": "CC-MAIN-2024-18",
        "url": "https://example.edu/physics/newtons-laws",
        "date": "2024-04-18T00:00:00Z",
        "file_path": "sample/000_00001.parquet",
        "language": "en",
        "language_score": 0.96,
        "token_count": 58,
        "score": 3.9,
        "int_score": 4,
    },
    {
        "text": "Cellular respiration is a set of metabolic reactions and processes that take "
        "place in the cells of organisms to convert chemical energy from nutrients "
        "into adenosine triphosphate (ATP), and then release waste products.",
        "id": "<urn:uuid:0005>",
        "dump": "CC-MAIN-2024-22",
        "url": "https://example.edu/biology/cellular-respiration",
        "date": "2024-05-02T00:00:00Z",
        "file_path": "sample/000_00002.parquet",
        "language": "en",
        "language_score": 0.98,
        "token_count": 45,
        "score": 4.0,
        "int_score": 4,
    },
    {
        "text": "In computer science, a hash table is a data structure that implements an "
        "associative array, a structure that can map keys to values. A hash table "
        "uses a hash function to compute an index into an array of buckets from which "
        "the desired value can be found. Ideally lookups run in amortized O(1) time.",
        "id": "<urn:uuid:0006>",
        "dump": "CC-MAIN-2024-22",
        "url": "https://example.edu/cs/hash-tables",
        "date": "2024-05-09T00:00:00Z",
        "file_path": "sample/000_00002.parquet",
        "language": "en",
        "language_score": 0.97,
        "token_count": 69,
        "score": 4.3,
        "int_score": 4,
    },
]

ROW_KEYS = [
    "id", "dump", "url", "date", "file_path",
    "language", "language_score", "token_count", "score", "int_score",
]


def load_rows(n: int, config: str) -> tuple[list[dict], str]:
    """Return (rows, source). Falls back to embedded rows on any failure."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] datasets unavailable ({exc}); using embedded sample")
        return list(EMBEDDED_ROWS), "embedded"
    try:
        ds = load_dataset(DATASET, name=config, split="train", streaming=True)
        rows: list[dict] = []
        for ex in ds:
            row = {"text": (ex.get("text") or "")[:TEXT_PREVIEW_CHARS]}
            for k in ROW_KEYS:
                if k in ex:
                    row[k] = ex[k]
            rows.append(row)
            if len(rows) >= n:
                break
        if not rows:
            raise RuntimeError("no rows streamed")
        return rows, f"{DATASET}:{config}"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not stream {DATASET} ({exc}); using embedded sample")
        return list(EMBEDDED_ROWS), "embedded"


def get_tokenizer():
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(TOKENIZER_MODEL), "qwen3"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Qwen3 tokenizer unavailable ({exc}); using whitespace fallback")
        return None, "fallback"


_WS_SPLIT = re.compile(r"\s+|\S+")


def tokenize(text: str, tok) -> list[dict]:
    """Return a list of {id, text, special} token pieces."""
    if tok is not None:
        ids = tok.encode(text, add_special_tokens=False)
        specials = set(getattr(tok, "all_special_ids", []) or [])
        out = []
        for tid in ids:
            piece = tok.decode([tid])
            out.append({"id": int(tid), "text": piece, "special": tid in specials})
        return out
    # Fallback: whitespace byte-ish split with stable synthetic ids.
    out = []
    for piece in _WS_SPLIT.findall(text):
        tid = (hash(piece) & 0x7FFFFFFF) % 151936
        out.append({"id": tid, "text": piece, "special": False})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", default="explorer/public/data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, source = load_rows(args.rows, args.config)
    tok, tok_source = get_tokenizer()

    # fineweb_sample.json
    (out_dir / "fineweb_sample.json").write_text(
        json.dumps({"source": source, "count": len(rows), "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    # fineweb_schema.json
    (out_dir / "fineweb_schema.json").write_text(
        json.dumps(SCHEMA, indent=2) + "\n", encoding="utf-8"
    )

    # tokenization.json — a few rows fully tokenized for the token view.
    samples = []
    for row in rows[:N_TOKENIZE]:
        text = (row.get("text") or "")[:TOK_PREVIEW_CHARS]
        toks = tokenize(text, tok)
        samples.append(
            {
                "id": row.get("id"),
                "text": text,
                "tokens": toks,
                "qwen3_token_count": len(toks),
                "dataset_token_count": row.get("token_count"),
            }
        )
    (out_dir / "tokenization.json").write_text(
        json.dumps(
            {
                "tokenizer": TOKENIZER_MODEL,
                "tokenizer_source": tok_source,
                "note": "dataset_token_count is computed by fineweb-edu with the GPT-2 "
                "tokenizer; qwen3_token_count is recomputed here with the Qwen3 tokenizer.",
                "samples": samples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # datasets/index.json — the switchable-subject index for Data mode. File
    # names are relative to public/data/ (where the three files above live).
    datasets_dir = out_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / "index.json").write_text(
        json.dumps(
            [
                {
                    "slug": "fineweb-edu",
                    "label": "FineWeb-Edu",
                    "schema": "fineweb_schema.json",
                    "sample": "fineweb_sample.json",
                    "tokenization": "tokenization.json",
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"wrote fineweb_sample.json ({len(rows)} rows, {source}), fineweb_schema.json, "
        f"tokenization.json ({len(samples)} samples, tokenizer={tok_source}), datasets/index.json"
    )


if __name__ == "__main__":
    main()
