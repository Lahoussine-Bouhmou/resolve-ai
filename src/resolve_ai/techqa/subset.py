from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resolve_ai.techqa.audit import TechQAAuditError, load_json


class TechQASubsetError(RuntimeError):
    """Raised when a reproducible TechQA subset cannot be created."""


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Files produced for the reduced TechQA benchmark."""

    documents: dict[str, dict[str, Any]]
    train_queries: list[dict[str, Any]]
    validation_queries: list[dict[str, Any]]
    ground_truth: list[dict[str, Any]]
    manifest: dict[str, Any]


def _question_id(question: dict[str, Any]) -> str:
    value = question.get("QUESTION_ID")

    if not isinstance(value, str) or not value.strip():
        raise TechQASubsetError(
            "Every selected question must have a non-empty QUESTION_ID."
        )

    return value.strip()


def _is_answerable(question: dict[str, Any]) -> bool:
    value = question.get("ANSWERABLE")

    if not isinstance(value, str):
        raise TechQASubsetError(
            f"Question {_question_id(question)} has no valid ANSWERABLE value."
        )

    normalized = value.strip().upper()

    if normalized == "Y":
        return True

    if normalized == "N":
        return False

    raise TechQASubsetError(
        f"Question {_question_id(question)} has unsupported ANSWERABLE value: {value!r}"
    )


def _validate_question_collection(
    questions: Any,
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(questions, list):
        raise TechQASubsetError(f"{source_name} must contain a JSON list of questions.")

    validated: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()

    for raw_question in questions:
        if not isinstance(raw_question, dict):
            raise TechQASubsetError(
                f"{source_name} contains a question that is not a JSON object."
            )

        question_id = _question_id(raw_question)

        if question_id in seen_question_ids:
            raise TechQASubsetError(
                f"Duplicate question ID in {source_name}: {question_id}"
            )

        seen_question_ids.add(question_id)
        validated.append(raw_question)

    return validated


def _select_stratified_questions(
    questions: list[dict[str, Any]],
    *,
    answerable_count: int,
    unanswerable_count: int,
    rng: random.Random,
    split_name: str,
) -> list[dict[str, Any]]:
    if answerable_count < 0 or unanswerable_count < 0:
        raise TechQASubsetError("Requested question counts cannot be negative.")

    answerable_questions = sorted(
        (question for question in questions if _is_answerable(question)),
        key=_question_id,
    )

    unanswerable_questions = sorted(
        (question for question in questions if not _is_answerable(question)),
        key=_question_id,
    )

    if answerable_count > len(answerable_questions):
        raise TechQASubsetError(
            f"Not enough answerable questions in {split_name}: "
            f"requested {answerable_count}, available "
            f"{len(answerable_questions)}."
        )

    if unanswerable_count > len(unanswerable_questions):
        raise TechQASubsetError(
            f"Not enough unanswerable questions in {split_name}: "
            f"requested {unanswerable_count}, available "
            f"{len(unanswerable_questions)}."
        )

    selected = [
        *rng.sample(answerable_questions, answerable_count),
        *rng.sample(unanswerable_questions, unanswerable_count),
    ]

    return sorted(selected, key=_question_id)


def _extract_candidate_document_ids(
    question: dict[str, Any],
) -> set[str]:
    raw_document_ids = question.get("DOC_IDS")

    if not isinstance(raw_document_ids, list):
        raise TechQASubsetError(
            f"Question {_question_id(question)} has no valid DOC_IDS list."
        )

    document_ids: set[str] = set()

    for raw_document_id in raw_document_ids:
        document_id = str(raw_document_id).strip()

        if document_id:
            document_ids.add(document_id)

    return document_ids


def _extract_answer_document_id(
    question: dict[str, Any],
) -> str | None:
    if not _is_answerable(question):
        return None

    raw_document_id = question.get("DOCUMENT")

    if not isinstance(raw_document_id, str) or not raw_document_id.strip():
        raise TechQASubsetError(
            f"Answerable question {_question_id(question)} has no answer DOCUMENT."
        )

    return raw_document_id.strip()


def _parse_required_offset(
    question: dict[str, Any],
    field_name: str,
) -> int:
    raw_value = question.get(field_name)

    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise TechQASubsetError(
            f"Question {_question_id(question)} has an invalid "
            f"{field_name}: {raw_value!r}"
        ) from exc

    if value < 0:
        raise TechQASubsetError(
            f"Question {_question_id(question)} has a negative {field_name}: {value}"
        )

    return value


def _to_safe_query(
    question: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    """
    Convert a TechQA question into a runtime-safe query.

    DOC_IDS, answer document IDs and answer offsets are deliberately omitted.
    """

    title = question.get("QUESTION_TITLE")
    text = question.get("QUESTION_TEXT")

    return {
        "question_id": _question_id(question),
        "split": split,
        "title": title if isinstance(title, str) else "",
        "text": text if isinstance(text, str) else "",
    }


def _to_ground_truth(
    question: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    answerable = _is_answerable(question)

    if not answerable:
        return {
            "question_id": _question_id(question),
            "split": split,
            "answerable": False,
            "document_id": None,
            "start_offset": None,
            "end_offset": None,
        }

    return {
        "question_id": _question_id(question),
        "split": split,
        "answerable": True,
        "document_id": _extract_answer_document_id(question),
        "start_offset": _parse_required_offset(
            question,
            "START_OFFSET",
        ),
        "end_offset": _parse_required_offset(
            question,
            "END_OFFSET",
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _sha256_identifiers(identifiers: list[str]) -> str:
    normalized = "\n".join(sorted(identifiers)).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _validate_corpus(corpus: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(corpus, dict):
        raise TechQASubsetError(
            "The corpus root must be a JSON object indexed by document ID."
        )

    validated: dict[str, dict[str, Any]] = {}

    for raw_document_id, raw_document in corpus.items():
        document_id = str(raw_document_id).strip()

        if not document_id:
            raise TechQASubsetError("The corpus contains an empty document ID.")

        if not isinstance(raw_document, dict):
            raise TechQASubsetError(f"Document {document_id} is not a JSON object.")

        if not isinstance(raw_document.get("text"), str):
            raise TechQASubsetError(f"Document {document_id} has no textual content.")

        validated[document_id] = raw_document

    return validated


def build_benchmark(
    *,
    corpus_path: Path,
    train_questions_path: Path,
    validation_questions_path: Path,
    train_answerable_count: int,
    train_unanswerable_count: int,
    validation_answerable_count: int,
    validation_unanswerable_count: int,
    extra_distractor_count: int,
    seed: int,
) -> BenchmarkArtifacts:
    """
    Build a closed-corpus TechQA benchmark.

    The corpus is the global union of:
    - candidate documents from all selected questions;
    - all answer documents;
    - optional random distractor documents.

    Candidate lists are never exposed in runtime query files.
    """

    try:
        raw_corpus = load_json(corpus_path)
        raw_train_questions = load_json(train_questions_path)
        raw_validation_questions = load_json(validation_questions_path)
    except TechQAAuditError as exc:
        raise TechQASubsetError(str(exc)) from exc

    corpus = _validate_corpus(raw_corpus)

    train_questions = _validate_question_collection(
        raw_train_questions,
        source_name=train_questions_path.name,
    )

    validation_questions = _validate_question_collection(
        raw_validation_questions,
        source_name=validation_questions_path.name,
    )

    train_ids = {_question_id(question) for question in train_questions}
    validation_ids = {_question_id(question) for question in validation_questions}

    overlapping_question_ids = train_ids & validation_ids

    if overlapping_question_ids:
        examples = sorted(overlapping_question_ids)[:10]
        raise TechQASubsetError(
            "Training and validation files contain overlapping question IDs: "
            f"{examples}"
        )

    rng = random.Random(seed)

    selected_train = _select_stratified_questions(
        train_questions,
        answerable_count=train_answerable_count,
        unanswerable_count=train_unanswerable_count,
        rng=rng,
        split_name="train",
    )

    selected_validation = _select_stratified_questions(
        validation_questions,
        answerable_count=validation_answerable_count,
        unanswerable_count=validation_unanswerable_count,
        rng=rng,
        split_name="validation",
    )

    selected_questions = [
        *selected_train,
        *selected_validation,
    ]

    candidate_document_ids: set[str] = set()
    answer_document_ids: set[str] = set()
    gold_documents_absent_from_candidates: set[str] = set()

    for question in selected_questions:
        question_candidate_ids = _extract_candidate_document_ids(question)
        candidate_document_ids.update(question_candidate_ids)

        missing_candidate_ids = question_candidate_ids - corpus.keys()

        if missing_candidate_ids:
            examples = sorted(missing_candidate_ids)[:10]
            raise TechQASubsetError(
                f"Question {_question_id(question)} references candidate "
                f"documents absent from the corpus: {examples}"
            )

        answer_document_id = _extract_answer_document_id(question)

        if answer_document_id is None:
            continue

        if answer_document_id not in corpus:
            raise TechQASubsetError(
                f"Question {_question_id(question)} references answer "
                f"document {answer_document_id}, which is absent "
                "from the corpus."
            )

        answer_document_ids.add(answer_document_id)

        if answer_document_id not in question_candidate_ids:
            gold_documents_absent_from_candidates.add(answer_document_id)

    base_document_ids = candidate_document_ids | answer_document_ids
    remaining_document_ids = sorted(corpus.keys() - base_document_ids)

    if extra_distractor_count < 0:
        raise TechQASubsetError("extra_distractor_count cannot be negative.")

    if extra_distractor_count > len(remaining_document_ids):
        raise TechQASubsetError(
            "Not enough remaining documents for the requested distractors: "
            f"requested {extra_distractor_count}, available "
            f"{len(remaining_document_ids)}."
        )

    random_distractor_ids = set(
        rng.sample(
            remaining_document_ids,
            extra_distractor_count,
        )
    )

    selected_document_ids = sorted(base_document_ids | random_distractor_ids)

    selected_documents = {
        document_id: corpus[document_id] for document_id in selected_document_ids
    }

    train_queries = [
        _to_safe_query(question, split="train") for question in selected_train
    ]

    validation_queries = [
        _to_safe_query(question, split="validation") for question in selected_validation
    ]

    ground_truth = [
        *(_to_ground_truth(question, split="train") for question in selected_train),
        *(
            _to_ground_truth(question, split="validation")
            for question in selected_validation
        ),
    ]

    train_question_ids = [query["question_id"] for query in train_queries]

    validation_question_ids = [query["question_id"] for query in validation_queries]

    manifest = {
        "benchmark_name": "techqa-small-closed-corpus",
        "benchmark_version": 1,
        "seed": seed,
        "construction": {
            "description": (
                "Global union of DOC_IDS from all selected questions, "
                "all answer documents, and random distractor documents."
            ),
            "runtime_rule": (
                "DOC_IDS and ground-truth annotations must never be passed "
                "to the retrieval engine."
            ),
        },
        "selection": {
            "train_answerable": train_answerable_count,
            "train_unanswerable": train_unanswerable_count,
            "validation_answerable": validation_answerable_count,
            "validation_unanswerable": validation_unanswerable_count,
            "extra_random_distractors": extra_distractor_count,
        },
        "counts": {
            "train_queries": len(train_queries),
            "validation_queries": len(validation_queries),
            "documents": len(selected_documents),
            "candidate_union_documents": len(candidate_document_ids),
            "answer_documents": len(answer_document_ids),
            "gold_documents_absent_from_candidates": len(
                gold_documents_absent_from_candidates
            ),
            "random_distractor_documents": len(random_distractor_ids),
        },
        "source_files": {
            "corpus": {
                "path": str(corpus_path),
                "sha256": _sha256_file(corpus_path),
            },
            "train_questions": {
                "path": str(train_questions_path),
                "sha256": _sha256_file(train_questions_path),
            },
            "validation_questions": {
                "path": str(validation_questions_path),
                "sha256": _sha256_file(validation_questions_path),
            },
        },
        "selected_ids": {
            "train_question_ids": train_question_ids,
            "validation_question_ids": validation_question_ids,
        },
        "integrity": {
            "train_question_ids_sha256": _sha256_identifiers(train_question_ids),
            "validation_question_ids_sha256": _sha256_identifiers(
                validation_question_ids
            ),
            "document_ids_sha256": _sha256_identifiers(selected_document_ids),
        },
    }

    return BenchmarkArtifacts(
        documents=selected_documents,
        train_queries=train_queries,
        validation_queries=validation_queries,
        ground_truth=ground_truth,
        manifest=manifest,
    )


def _write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )
        file.write("\n")


def write_benchmark(
    artifacts: BenchmarkArtifacts,
    *,
    output_directory: Path,
) -> None:
    """Write all benchmark files into a dedicated directory."""

    output_directory.mkdir(parents=True, exist_ok=True)

    _write_json(
        output_directory / "documents.json",
        artifacts.documents,
    )

    _write_json(
        output_directory / "train_queries.json",
        artifacts.train_queries,
    )

    _write_json(
        output_directory / "validation_queries.json",
        artifacts.validation_queries,
    )

    _write_json(
        output_directory / "ground_truth.json",
        artifacts.ground_truth,
    )

    _write_json(
        output_directory / "manifest.json",
        artifacts.manifest,
    )
