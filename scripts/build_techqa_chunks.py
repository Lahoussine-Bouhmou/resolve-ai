from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from resolve_ai.techqa.chunking import (
    ChunkingConfig,
    TechQAChunkingError,
    chunk_corpus,
    evaluate_answer_span_coverage,
    summarize_chunks,
    write_chunks_jsonl,
    write_json,
)


def load_json(path: Path) -> object:
    if not path.exists():
        raise TechQAChunkingError(f"File does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise TechQAChunkingError(
            f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}."
        ) from exc
    except OSError as exc:
        raise TechQAChunkingError(f"Unable to read {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create offset-preserving chunks for the reduced TechQA benchmark."
        )
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/processed/techqa_small/documents.json"),
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/processed/techqa_small/ground_truth.json"),
    )

    parser.add_argument(
        "--chunks-output",
        type=Path,
        default=Path("data/processed/techqa_small/chunks.jsonl"),
    )

    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("data/processed/techqa_small/chunking_report.json"),
    )

    parser.add_argument(
        "--target-chars",
        type=int,
        default=1800,
    )

    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--boundary-search-chars",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--minimum-chunk-chars",
        type=int,
        default=300,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = ChunkingConfig(
            target_chars=args.target_chars,
            overlap_chars=args.overlap_chars,
            boundary_search_chars=(args.boundary_search_chars),
            minimum_chunk_chars=(args.minimum_chunk_chars),
        )

        documents = load_json(args.documents)
        ground_truth = load_json(args.ground_truth)

        chunks = chunk_corpus(
            documents,
            config=config,
        )

        coverage = evaluate_answer_span_coverage(
            chunks=chunks,
            ground_truth=ground_truth,
        )

        report = {
            "configuration": {
                "target_chars": config.target_chars,
                "overlap_chars": config.overlap_chars,
                "boundary_search_chars": (config.boundary_search_chars),
                "minimum_chunk_chars": (config.minimum_chunk_chars),
            },
            "chunk_statistics": summarize_chunks(chunks),
            "answer_span_evaluation": coverage,
        }

        write_chunks_jsonl(
            chunks,
            output_path=args.chunks_output,
        )

        write_json(
            report,
            output_path=args.report_output,
        )

    except (
        TechQAChunkingError,
        ValueError,
    ) as exc:
        print(
            f"TechQA chunking failed: {exc}",
            file=sys.stderr,
        )
        return 1

    statistics = report["chunk_statistics"]
    evaluation = report["answer_span_evaluation"]

    print("TechQA chunking completed.")
    print(f"Documents: {statistics['document_count']}")
    print(f"Chunks: {statistics['chunk_count']}")
    print(f"Mean chunk length: {statistics['mean_length']} characters")
    print(f"Full answer-span coverage: {evaluation['full_span_coverage_rate']:.2%}")
    print(f"Chunks file: {args.chunks_output}")
    print(f"Report file: {args.report_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
