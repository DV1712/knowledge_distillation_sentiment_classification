#!/usr/bin/env python3
"""Analyze dataset token lengths and recommend a max_length value.

Usage examples:
python analyze_max_length.py
python analyze_max_length.py --dataset dataset/multiclass_sentiments.jsonl --target-coverage 0.98
python analyze_max_length.py --models bert-base-uncased roberta-base distilbert-base-uncased
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer


DEFAULT_MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased",
]

DEFAULT_TEXT_KEYS = ["text", "sentence", "content", "review", "tweet"]
DEFAULT_CANDIDATES = [64, 96, 128, 160, 192, 224, 256, 320, 384, 512]
DEFAULT_PERCENTILES = [90, 95, 97, 98, 99, 99.5]


@dataclass
class LengthStats:
    model_id: str
    lengths: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find best max_length from dataset statistics.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/multiclass_sentiments.jsonl"),
        help="Path to JSONL dataset.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Tokenizer model IDs to evaluate.",
    )
    parser.add_argument(
        "--text-keys",
        nargs="+",
        default=DEFAULT_TEXT_KEYS,
        help="Preferred JSON keys for text fields.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=int,
        default=DEFAULT_CANDIDATES,
        help="Candidate max_length values to evaluate.",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.95,
        help="Target non-truncation coverage in [0, 1], for recommendation.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc


def pick_text(record: dict, text_keys: list[str]) -> str | None:
    for key in text_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value

    for value in record.values():
        if isinstance(value, str) and value.strip():
            return value

    return None


def load_texts(dataset_path: Path, text_keys: list[str]) -> list[str]:
    texts: list[str] = []
    for record in iter_jsonl(dataset_path):
        text = pick_text(record, text_keys)
        if text is not None:
            texts.append(text)
    return texts


def token_lengths(texts: list[str], model_id: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    lengths = [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in texts
    ]
    return np.asarray(lengths, dtype=np.int32)


def recommend_candidate(lengths: np.ndarray, candidates: list[int], target_coverage: float) -> int:
    valid_candidates = sorted(set(candidates))
    needed_quantile = int(np.ceil(np.quantile(lengths, target_coverage)))
    for c in valid_candidates:
        if c >= needed_quantile:
            return c
    return valid_candidates[-1]


def print_stats(stats: LengthStats, candidates: list[int], target_coverage: float) -> None:
    lengths = stats.lengths
    print(f"\nModel: {stats.model_id}")
    print(
        "Length summary: "
        f"min={int(lengths.min())}, mean={float(lengths.mean()):.2f}, "
        f"median={int(np.median(lengths))}, max={int(lengths.max())}"
    )

    print("Percentiles:")
    for p in DEFAULT_PERCENTILES:
        print(f"  p{p}: {int(np.percentile(lengths, p))}")

    print("Truncation rates at candidates:")
    for c in sorted(set(candidates)):
        trunc_rate = float((lengths > c).mean()) * 100.0
        print(f"  max_length={c:>3}: {trunc_rate:>6.2f}% truncated")

    rec = recommend_candidate(lengths, candidates, target_coverage)
    covered = float((lengths <= rec).mean()) * 100.0
    print(
        f"Recommendation for coverage {target_coverage:.2%}: "
        f"max_length={rec} (covers {covered:.2f}% of samples)"
    )


def main() -> None:
    args = parse_args()

    if not (0.0 < args.target_coverage <= 1.0):
        raise ValueError("--target-coverage must be in (0, 1].")

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    texts = load_texts(args.dataset, args.text_keys)
    if not texts:
        raise ValueError("No text samples found. Check --text-keys and dataset format.")

    print(f"Loaded {len(texts)} text samples from {args.dataset}")

    all_recommendations: list[int] = []
    for model_id in args.models:
        lengths = token_lengths(texts, model_id)
        stats = LengthStats(model_id=model_id, lengths=lengths)
        print_stats(stats, args.candidates, args.target_coverage)
        all_recommendations.append(recommend_candidate(lengths, args.candidates, args.target_coverage))

    combined_recommendation = max(all_recommendations)
    print("\nCombined recommendation across all models:")
    print(
        f"  max_length={combined_recommendation} "
        f"(safe choice for all evaluated tokenizers at target coverage {args.target_coverage:.2%})"
    )


if __name__ == "__main__":
    main()
