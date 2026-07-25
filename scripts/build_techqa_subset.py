from __future__ import annotations

import argparse
import sys
from pathlib import Path

from resolve_ai.techqa.subset import (
    TechQASubsetError,
    build_benchmark,
    write_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build a reproducible closed-corpus TechQA benchmark.")
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-questions",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation-questions",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/processed/techqa_small"),
    )

    parser.add_argument(
        "--train-answerable",
        type=int,
        default=75,
    )

    parser.add_argument(
        "--train-unanswerable",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--validation-answerable",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--validation-unanswerable",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--extra-distractors",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        artifacts = build_benchmark(
            corpus_path=args.corpus,
            train_questions_path=args.train_questions,
            validation_questions_path=args.validation_questions,
            train_answerable_count=args.train_answerable,
            train_unanswerable_count=args.train_unanswerable,
            validation_answerable_count=(args.validation_answerable),
            validation_unanswerable_count=(args.validation_unanswerable),
            extra_distractor_count=args.extra_distractors,
            seed=args.seed,
        )

        write_benchmark(
            artifacts,
            output_directory=args.output_directory,
        )
    except TechQASubsetError as exc:
        print(
            f"TechQA benchmark creation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    counts = artifacts.manifest["counts"]

    print("TechQA benchmark created successfully.")
    print(f"Output directory: {args.output_directory}")
    print(f"Train queries: {counts['train_queries']}")
    print(f"Validation queries: {counts['validation_queries']}")
    print(f"Documents: {counts['documents']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
