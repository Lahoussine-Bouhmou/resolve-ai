from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from resolve_ai.techqa.subset import (
    TechQASubsetError,
    build_benchmark,
)


def write_json(path: Path, content: Any) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False),
        encoding="utf-8",
    )


def make_question(
    *,
    question_id: str,
    answerable: bool,
    candidate_document_ids: list[str],
    answer_document_id: str | None = None,
) -> dict[str, Any]:
    question: dict[str, Any] = {
        "QUESTION_ID": question_id,
        "QUESTION_TITLE": f"Title for {question_id}",
        "QUESTION_TEXT": f"Text for {question_id}",
        "ANSWERABLE": "Y" if answerable else "N",
        "DOC_IDS": candidate_document_ids,
    }

    if answerable:
        if answer_document_id is None:
            raise ValueError("An answerable fixture requires an answer document.")

        question.update(
            {
                "DOCUMENT": answer_document_id,
                "START_OFFSET": 0,
                "END_OFFSET": 6,
            }
        )

    return question


def create_fixture_files(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    corpus_path = tmp_path / "documents.json"
    train_path = tmp_path / "training.json"
    validation_path = tmp_path / "validation.json"

    corpus = {
        f"doc-{index}": {
            "_id": f"doc-{index}",
            "title": f"Document {index}",
            "text": f"answer text for document {index}",
        }
        for index in range(1, 13)
    }

    train_questions = [
        make_question(
            question_id="train-a-1",
            answerable=True,
            candidate_document_ids=["doc-1", "doc-2"],
            answer_document_id="doc-1",
        ),
        make_question(
            question_id="train-a-2",
            answerable=True,
            candidate_document_ids=["doc-3", "doc-4"],
            answer_document_id="doc-3",
        ),
        make_question(
            question_id="train-u-1",
            answerable=False,
            candidate_document_ids=["doc-5", "doc-6"],
        ),
        make_question(
            question_id="train-u-2",
            answerable=False,
            candidate_document_ids=["doc-7", "doc-8"],
        ),
    ]

    validation_questions = [
        make_question(
            question_id="validation-a-1",
            answerable=True,
            candidate_document_ids=["doc-9", "doc-10"],
            answer_document_id="doc-9",
        ),
        make_question(
            question_id="validation-u-1",
            answerable=False,
            candidate_document_ids=["doc-10", "doc-11"],
        ),
    ]

    write_json(corpus_path, corpus)
    write_json(train_path, train_questions)
    write_json(validation_path, validation_questions)

    return corpus_path, train_path, validation_path


def test_build_benchmark_is_reproducible(
    tmp_path: Path,
) -> None:
    corpus_path, train_path, validation_path = create_fixture_files(tmp_path)

    first = build_benchmark(
        corpus_path=corpus_path,
        train_questions_path=train_path,
        validation_questions_path=validation_path,
        train_answerable_count=1,
        train_unanswerable_count=1,
        validation_answerable_count=1,
        validation_unanswerable_count=1,
        extra_distractor_count=1,
        seed=42,
    )

    second = build_benchmark(
        corpus_path=corpus_path,
        train_questions_path=train_path,
        validation_questions_path=validation_path,
        train_answerable_count=1,
        train_unanswerable_count=1,
        validation_answerable_count=1,
        validation_unanswerable_count=1,
        extra_distractor_count=1,
        seed=42,
    )

    assert first.train_queries == second.train_queries
    assert first.validation_queries == second.validation_queries
    assert first.ground_truth == second.ground_truth
    assert first.manifest["integrity"] == second.manifest["integrity"]


def test_runtime_queries_do_not_expose_ground_truth(
    tmp_path: Path,
) -> None:
    corpus_path, train_path, validation_path = create_fixture_files(tmp_path)

    artifacts = build_benchmark(
        corpus_path=corpus_path,
        train_questions_path=train_path,
        validation_questions_path=validation_path,
        train_answerable_count=1,
        train_unanswerable_count=1,
        validation_answerable_count=1,
        validation_unanswerable_count=1,
        extra_distractor_count=0,
        seed=10,
    )

    all_queries = [
        *artifacts.train_queries,
        *artifacts.validation_queries,
    ]

    for query in all_queries:
        assert "DOC_IDS" not in query
        assert "DOCUMENT" not in query
        assert "START_OFFSET" not in query
        assert "END_OFFSET" not in query
        assert "answerable" not in query


def test_answer_documents_are_in_global_corpus(
    tmp_path: Path,
) -> None:
    corpus_path, train_path, validation_path = create_fixture_files(tmp_path)

    artifacts = build_benchmark(
        corpus_path=corpus_path,
        train_questions_path=train_path,
        validation_questions_path=validation_path,
        train_answerable_count=1,
        train_unanswerable_count=1,
        validation_answerable_count=1,
        validation_unanswerable_count=1,
        extra_distractor_count=0,
        seed=4,
    )

    for truth in artifacts.ground_truth:
        if truth["answerable"]:
            assert truth["document_id"] in artifacts.documents


def test_build_fails_when_requested_sample_is_too_large(
    tmp_path: Path,
) -> None:
    corpus_path, train_path, validation_path = create_fixture_files(tmp_path)

    with pytest.raises(
        TechQASubsetError,
        match="Not enough answerable questions",
    ):
        build_benchmark(
            corpus_path=corpus_path,
            train_questions_path=train_path,
            validation_questions_path=validation_path,
            train_answerable_count=100,
            train_unanswerable_count=1,
            validation_answerable_count=1,
            validation_unanswerable_count=1,
            extra_distractor_count=0,
            seed=42,
        )
