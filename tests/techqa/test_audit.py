from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resolve_ai.techqa.audit import audit_techqa


def write_json(path: Path, content: Any) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False),
        encoding="utf-8",
    )


def test_audit_detects_valid_answer_span(tmp_path: Path) -> None:
    corpus_path = tmp_path / "technotes.json"
    questions_path = tmp_path / "questions.json"

    document_text = "prefix answer suffix"

    write_json(
        corpus_path,
        {
            "doc-1": {
                "_id": "doc-1",
                "title": "Example Technote",
                "text": document_text,
            }
        },
    )

    write_json(
        questions_path,
        [
            {
                "QUESTION_ID": "question-1",
                "QUESTION_TITLE": "Example problem",
                "QUESTION_TEXT": "How do I solve this?",
                "ANSWERABLE": "Y",
                "DOCUMENT": "doc-1",
                "START_OFFSET": 7,
                "END_OFFSET": 13,
                "DOC_IDS": ["doc-1"],
            }
        ],
    )

    report = audit_techqa(
        corpus_path=corpus_path,
        question_paths=[questions_path],
    )

    split = report["question_splits"][0]

    assert report["corpus"]["document_count"] == 1
    assert split["question_count"] == 1
    assert split["valid_answer_spans"] == 1
    assert split["invalid_offsets"] == 0
    assert split["missing_answer_document_count"] == 0


def test_audit_detects_invalid_offset(tmp_path: Path) -> None:
    corpus_path = tmp_path / "technotes.json"
    questions_path = tmp_path / "questions.json"

    write_json(
        corpus_path,
        {
            "doc-1": {
                "_id": "doc-1",
                "title": "Example Technote",
                "text": "short text",
            }
        },
    )

    write_json(
        questions_path,
        [
            {
                "QUESTION_ID": "question-1",
                "QUESTION_TITLE": "Example problem",
                "QUESTION_TEXT": "How do I solve this?",
                "ANSWERABLE": "Y",
                "DOCUMENT": "doc-1",
                "START_OFFSET": 0,
                "END_OFFSET": 500,
                "DOC_IDS": ["doc-1"],
            }
        ],
    )

    report = audit_techqa(
        corpus_path=corpus_path,
        question_paths=[questions_path],
    )

    split = report["question_splits"][0]

    assert split["valid_answer_spans"] == 0
    assert split["invalid_offsets"] == 1


def test_audit_detects_missing_documents(tmp_path: Path) -> None:
    corpus_path = tmp_path / "technotes.json"
    questions_path = tmp_path / "questions.json"

    write_json(corpus_path, {})

    write_json(
        questions_path,
        [
            {
                "QUESTION_ID": "question-1",
                "QUESTION_TITLE": "Example problem",
                "QUESTION_TEXT": "How do I solve this?",
                "ANSWERABLE": "Y",
                "DOCUMENT": "missing-answer-document",
                "START_OFFSET": 0,
                "END_OFFSET": 10,
                "DOC_IDS": [
                    "missing-answer-document",
                    "missing-candidate-document",
                ],
            }
        ],
    )

    report = audit_techqa(
        corpus_path=corpus_path,
        question_paths=[questions_path],
    )

    split = report["question_splits"][0]

    assert split["missing_answer_document_count"] == 1
    assert split["missing_candidate_document_count"] == 2
