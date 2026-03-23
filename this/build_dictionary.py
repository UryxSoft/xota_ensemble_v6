"""
build_dictionary.py — N-gram Dictionary Builder for PerplexityProfiler
======================================================================
Two modes:
  - Small corpus (<500k texts): loads into memory, fast
  - Large corpus (500k–100M+ texts): streaming with periodic pruning

Performance on 37M texts (4-core CPU, 8 GB RAM):
  Build time:   ~2-4 hours
  Peak RAM:     ~500 MB (with pruning every 500k texts)
  Output file:  2-10 MB JSON
  Lookup speed: 0.2ms per text (unchanged regardless of corpus size)

Usage
-----
  # Small corpus from CSV
  python build_dictionary.py --input corpus.csv --output ngram_dict.json

  # LARGE corpus from JSONL (37M texts) — streaming auto-detected
  python build_dictionary.py --input huge.jsonl --output ngram_dict.json

  # HuggingFace dataset
  python build_dictionary.py --hf Hello-SimpleAI/HC3 --column answer --output ngram_dict.json

  # Custom parameters for very large corpus
  python build_dictionary.py --input data.jsonl --output ngram_dict.json \\
      --max-rows 37000000 --top-k 200000 --prune-every 500000
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_STREAMING_THRESHOLD = 500_000


def _gen_from_csv(path: str, column: str, max_rows: int) -> Generator[str, None, None]:
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if count >= max_rows:
                break
            text = row.get(column, "").strip()
            if len(text.split()) >= 20:
                yield text
                count += 1


def _gen_from_jsonl(path: str, column: str, max_rows: int) -> Generator[str, None, None]:
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if count >= max_rows:
                break
            try:
                obj = json.loads(line)
                text = obj.get(column, "").strip()
                if len(text.split()) >= 20:
                    yield text
                    count += 1
            except json.JSONDecodeError:
                continue


def _gen_from_folder(path: str, max_files: int) -> Generator[str, None, None]:
    count = 0
    for f in sorted(Path(path).glob("*.txt")):
        if count >= max_files:
            break
        text = f.read_text(encoding="utf-8", errors="replace").strip()
        if len(text.split()) >= 20:
            yield text
            count += 1


def _gen_from_huggingface(dataset_name: str, column: str,
                           split: str, max_rows: int) -> Generator[str, None, None]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    logger.info("Loading %s (split=%s, streaming=True)...", dataset_name, split)
    ds = load_dataset(dataset_name, split=split, streaming=True)
    count = 0
    for row in ds:
        if count >= max_rows:
            break
        val = row.get(column, "")
        if isinstance(val, list):
            for item in val:
                if count >= max_rows:
                    break
                text = str(item).strip()
                if len(text.split()) >= 20:
                    yield text
                    count += 1
        else:
            text = str(val).strip()
            if len(text.split()) >= 20:
                yield text
                count += 1


def _estimate_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    count = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Build n-gram dictionary for PerplexityProfiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_dictionary.py --input corpus.csv --output ngram_dict.json
  python build_dictionary.py --input huge.jsonl --output ngram_dict.json --max-rows 37000000
  python build_dictionary.py --hf Hello-SimpleAI/HC3 --column answer --output ngram_dict.json
        """,
    )
    parser.add_argument("--input", type=str, help="CSV, JSONL, or folder of .txt files")
    parser.add_argument("--hf", type=str, help="HuggingFace dataset name")
    parser.add_argument("--column", type=str, default="text", help="Text column (default: text)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--output", type=str, default="ngram_dict.json", help="Output path")
    parser.add_argument("--max-rows", type=int, default=50_000_000, help="Max texts (default: 50M)")
    parser.add_argument("--orders", type=str, default="2,3,4", help="N-gram orders (default: 2,3,4)")
    parser.add_argument("--top-k", type=int, default=200_000, help="Final top K per order (default: 200k)")
    parser.add_argument("--prune-every", type=int, default=500_000, help="Prune every N texts (default: 500k)")
    parser.add_argument("--prune-keep", type=int, default=500_000, help="Keep top N during prune (default: 500k)")
    parser.add_argument("--force-streaming", action="store_true", help="Force streaming mode")
    args = parser.parse_args()

    if not args.input and not args.hf:
        parser.error("Provide --input or --hf")

    orders = tuple(int(x) for x in args.orders.split(","))
    use_streaming = args.force_streaming
    generator = None

    if args.hf:
        use_streaming = True
        generator = _gen_from_huggingface(args.hf, args.column, args.split, args.max_rows)
        logger.info("Source: HuggingFace %s (streaming)", args.hf)
    elif args.input:
        p = args.input
        if os.path.isdir(p):
            fc = len(list(Path(p).glob("*.txt")))
            use_streaming = use_streaming or fc > _STREAMING_THRESHOLD
            generator = _gen_from_folder(p, args.max_rows)
            logger.info("Source: folder %s (%d files)", p, fc)
        elif p.endswith((".jsonl", ".ndjson")):
            est = _estimate_lines(p)
            use_streaming = use_streaming or est > _STREAMING_THRESHOLD
            generator = _gen_from_jsonl(p, args.column, args.max_rows)
            logger.info("Source: JSONL %s (~%d lines)", p, est)
        elif p.endswith(".csv"):
            est = _estimate_lines(p)
            use_streaming = use_streaming or est > _STREAMING_THRESHOLD
            generator = _gen_from_csv(p, args.column, args.max_rows)
            logger.info("Source: CSV %s (~%d lines)", p, est)
        else:
            print(f"ERROR: Unsupported format: {p} (use .csv, .jsonl, or folder)")
            sys.exit(1)

    if generator is None:
        print("ERROR: No data source.")
        sys.exit(1)

    from perplexity_profiler import NgramDictionary
    builder = NgramDictionary()

    if use_streaming:
        print(f"\n{'='*60}")
        print(f"  STREAMING BUILD (memory-efficient)")
        print(f"  top_k={args.top_k:,} | prune_every={args.prune_every:,}")
        print(f"  orders={orders} | max_rows={args.max_rows:,}")
        print(f"{'='*60}\n")

        builder.build_and_save_streaming(
            text_iterator=generator,
            output_path=args.output,
            orders=orders,
            top_k=args.top_k,
            prune_every=args.prune_every,
            prune_keep=args.prune_keep,
            log_every=100_000,
        )
    else:
        print("Loading texts into memory...")
        texts = list(generator)
        print(f"Loaded {len(texts):,} texts")
        builder.build_and_save(texts, args.output, orders=orders, top_k=args.top_k)
        size = os.path.getsize(args.output)
        print(f"\nDictionary saved: {args.output} ({size:,} bytes)")

    print(f'\nUse: PerplexityProfiler(ngram_dict_path="{args.output}")')


if __name__ == "__main__":
    main()
