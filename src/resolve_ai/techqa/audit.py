from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class TechQAAuditError(RuntimeError):
    """Raised when a TechQA file cannot be audited."""


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file with a useful domain error."""

    if not path.exists():
        raise TechQAAuditError(f"File does not exist: {path}")

    if not path.is_file():
        raise TechQAAuditError(f"Path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise TechQAAuditError(
            f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise TechQAAuditError(f"Unable to read {path}: {exc}") from exc


def _percentile(values: list[int], percentile: float) -> int | None:
    """Return a nearest-rank percentile without external dependencies."""

    if not values:
        return None

    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _summarize_lengths(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "mean": round(sum(values) / len(values), 2),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _as_string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _parse_offset(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def audit_corpus(corpus: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Audit the TechQA document corpus."""

    if not isinstance(corpus, dict):
        raise TechQAAuditError(
            "The corpus root must be a JSON object mapping document IDs "
            "to document objects."
        )

    key_frequency: Counter[str] = Counter()
    text_lengths: list[int] = []

    invalid_document_objects = 0
    missing_title = 0
    missing_text = 0
    empty_text = 0
    id_mismatches = 0

    normalized_documents: dict[str, dict[str, Any]] = {}

    for document_id, raw_document in corpus.items():
        if not isinstance(document_id, str):
            document_id = str(document_id)

        if not isinstance(raw_document, dict):
            invalid_document_objects += 1
            continue

        normalized_documents[document_id] = raw_document
        key_frequency.update(raw_document.keys())

        title = raw_document.get("title")
        text = raw_document.get("text")

        if not isinstance(title, str) or not title.strip():
            missing_title += 1

        if not isinstance(text, str):
            missing_text += 1
            continue

        text_lengths.append(len(text))

        if not text.strip():
            empty_text += 1

        embedded_id = raw_document.get("_id")
        if embedded_id is not None and str(embedded_id) != document_id:
            id_mismatches += 1

    report = {
        "document_count": len(corpus),
        "valid_document_objects": len(normalized_documents),
        "invalid_document_objects": invalid_document_objects,
        "missing_or_empty_titles": missing_title,
        "missing_text": missing_text,
        "empty_text": empty_text,
        "embedded_id_mismatches": id_mismatches,
        "field_frequency": dict(key_frequency.most_common()),
        "text_length_characters": _summarize_lengths(text_lengths),
    }

    return report, normalized_documents


def audit_questions(
    questions: Any,
    *,
    source_name: str,
    documents: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Audit one TechQA question split."""

    if not isinstance(questions, list):
        raise TechQAAuditError(
            f"The question file {source_name} must contain a JSON list."
        )

    field_frequency: Counter[str] = Counter()
    answerable_values: Counter[str] = Counter()

    question_ids: list[str] = []
    title_lengths: list[int] = []
    text_lengths: list[int] = []
    candidate_counts: list[int] = []

    invalid_question_objects = 0
    missing_question_ids = 0
    duplicate_question_ids = 0
    missing_candidate_documents: set[str] = set()
    missing_answer_documents: set[str] = set()
    answer_document_not_in_candidates = 0
    invalid_offsets = 0
    empty_answer_spans = 0
    valid_answer_spans = 0

    seen_question_ids: set[str] = set()

    for raw_question in questions:
        if not isinstance(raw_question, dict):
            invalid_question_objects += 1
            continue

        field_frequency.update(raw_question.keys())

        question_id = _as_string(raw_question.get("QUESTION_ID")).strip()
        if not question_id:
            missing_question_ids += 1
        elif question_id in seen_question_ids:
            duplicate_question_ids += 1
        else:
            seen_question_ids.add(question_id)
            question_ids.append(question_id)

        title = _as_string(raw_question.get("QUESTION_TITLE"))
        question_text = _as_string(raw_question.get("QUESTION_TEXT"))

        title_lengths.append(len(title))
        text_lengths.append(len(question_text))

        answerable = _as_string(raw_question.get("ANSWERABLE")).strip()
        answerable_values[answerable or "<missing>"] += 1

        raw_doc_ids = raw_question.get("DOC_IDS", [])
        if isinstance(raw_doc_ids, list):
            candidate_ids = [
                str(document_id).strip()
                for document_id in raw_doc_ids
                if str(document_id).strip()
            ]
        else:
            candidate_ids = []

        candidate_counts.append(len(candidate_ids))

        for candidate_id in candidate_ids:
            if candidate_id not in documents:
                missing_candidate_documents.add(candidate_id)

        if answerable != "Y":
            continue

        answer_document_id = _as_string(raw_question.get("DOCUMENT")).strip()

        if not answer_document_id or answer_document_id not in documents:
            if answer_document_id:
                missing_answer_documents.add(answer_document_id)
            continue

        if answer_document_id not in candidate_ids:
            answer_document_not_in_candidates += 1

        start_offset = _parse_offset(raw_question.get("START_OFFSET"))
        end_offset = _parse_offset(raw_question.get("END_OFFSET"))

        document_text = _as_string(documents[answer_document_id].get("text"))

        offsets_are_valid = (
            start_offset is not None
            and end_offset is not None
            and 0 <= start_offset < end_offset <= len(document_text)
        )

        if not offsets_are_valid:
            invalid_offsets += 1
            continue

        answer_span = document_text[start_offset:end_offset]

        if not answer_span.strip():
            empty_answer_spans += 1
            continue

        valid_answer_spans += 1

    return {
        "source": source_name,
        "question_count": len(questions),
        "valid_question_objects": len(questions) - invalid_question_objects,
        "invalid_question_objects": invalid_question_objects,
        "missing_question_ids": missing_question_ids,
        "duplicate_question_ids": duplicate_question_ids,
        "answerable_values": dict(answerable_values),
        "field_frequency": dict(field_frequency.most_common()),
        "question_title_length_characters": _summarize_lengths(title_lengths),
        "question_text_length_characters": _summarize_lengths(text_lengths),
        "candidate_document_count": _summarize_lengths(candidate_counts),
        "missing_candidate_document_count": len(missing_candidate_documents),
        "missing_candidate_document_examples": sorted(missing_candidate_documents)[:20],
        "missing_answer_document_count": len(missing_answer_documents),
        "missing_answer_document_examples": sorted(missing_answer_documents)[:20],
        "answer_document_not_in_candidates": answer_document_not_in_candidates,
        "valid_answer_spans": valid_answer_spans,
        "invalid_offsets": invalid_offsets,
        "empty_answer_spans": empty_answer_spans,
    }


def audit_techqa(
    *,
    corpus_path: Path,
    question_paths: list[Path],
) -> dict[str, Any]:
    """Audit a TechQA corpus and one or more question splits."""

    corpus_raw = load_json(corpus_path)
    corpus_report, documents = audit_corpus(corpus_raw)

    question_reports: list[dict[str, Any]] = []

    for question_path in question_paths:
        questions_raw = load_json(question_path)
        question_reports.append(
            audit_questions(
                questions_raw,
                source_name=question_path.name,
                documents=documents,
            )
        )

    return {
        "corpus_file": str(corpus_path),
        "question_files": [str(path) for path in question_paths],
        "corpus": corpus_report,
        "question_splits": question_reports,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown summary."""

    corpus = report["corpus"]

    lines = [
        "# TechQA audit report",
        "",
        "## Corpus",
        "",
        f"- Documents: {corpus['document_count']}",
        f"- Valid document objects: {corpus['valid_document_objects']}",
        f"- Documents without usable title: {corpus['missing_or_empty_titles']}",
        f"- Documents without text: {corpus['missing_text']}",
        f"- Documents with empty text: {corpus['empty_text']}",
        f"- Embedded ID mismatches: {corpus['embedded_id_mismatches']}",
        "",
        "### Document text lengths",
        "",
    ]

    for key, value in corpus["text_length_characters"].items():
        lines.append(f"- {key}: {value}")

    for split in report["question_splits"]:
        lines.extend(
            [
                "",
                f"## Questions: {split['source']}",
                "",
                f"- Questions: {split['question_count']}",
                f"- Missing question IDs: {split['missing_question_ids']}",
                f"- Duplicate question IDs: {split['duplicate_question_ids']}",
                f"- Answerable values: {split['answerable_values']}",
                (
                    "- Missing candidate documents: "
                    f"{split['missing_candidate_document_count']}"
                ),
                (
                    "- Missing answer documents: "
                    f"{split['missing_answer_document_count']}"
                ),
                (
                    "- Answer document absent from candidates: "
                    f"{split['answer_document_not_in_candidates']}"
                ),
                f"- Valid answer spans: {split['valid_answer_spans']}",
                f"- Invalid offsets: {split['invalid_offsets']}",
                f"- Empty answer spans: {split['empty_answer_spans']}",
                "",
                "### Question text lengths",
                "",
            ]
        )

        for key, value in split["question_text_length_characters"].items():
            lines.append(f"- {key}: {value}")

        lines.extend(
            [
                "",
                "### Candidate document counts",
                "",
            ]
        )

        for key, value in split["candidate_document_count"].items():
            lines.append(f"- {key}: {value}")

    lines.append("")
    return "\n".join(lines)


def write_reports(
    report: dict[str, Any],
    *,
    json_output_path: Path,
    markdown_output_path: Path,
) -> None:
    """Write machine-readable and human-readable reports."""

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)

    with json_output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    markdown_output_path.write_text(
        render_markdown(report),
        encoding="utf-8",
    )
