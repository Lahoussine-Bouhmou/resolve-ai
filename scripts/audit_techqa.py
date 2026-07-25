from __future__ import annotations

import argparse
import sys
from pathlib import Path

from resolve_ai.techqa.audit import (
    TechQAAuditError,
    audit_techqa,
    write_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit TechQA question and corpus JSON files."
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to the TechQA corpus JSON file.",
    )

    parser.add_argument(
        "--questions",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to one or more TechQA question JSON files.",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/reports/techqa/audit.json"),
        help="Path of the generated JSON report.",
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("data/reports/techqa/audit.md"),
        help="Path of the generated Markdown report.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        report = audit_techqa(
            corpus_path=args.corpus,
            question_paths=args.questions,
        )

        write_reports(
            report,
            json_output_path=args.json_output,
            markdown_output_path=args.markdown_output,
        )
    except TechQAAuditError as exc:
        print(f"TechQA audit failed: {exc}", file=sys.stderr)
        return 1

    print("TechQA audit completed.")
    print(f"JSON report: {args.json_output}")
    print(f"Markdown report: {args.markdown_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
