from __future__ import annotations

from resolve_ai.techqa.chunking import (
    ChunkingConfig,
    chunk_document,
    evaluate_answer_span_coverage,
    normalize_for_search,
)


def test_normalization_is_separate_from_raw_content() -> None:
    raw_text = "SQL30082N\u00a0  authentication\n\nfailed"

    normalized = normalize_for_search(raw_text)

    assert normalized == ("SQL30082N authentication failed")

    # The original value is not modified.
    assert raw_text == ("SQL30082N\u00a0  authentication\n\nfailed")


def test_chunk_content_matches_original_offsets() -> None:
    text = (
        "First paragraph with useful content.\n\n"
        "Second paragraph with additional content.\n\n"
        "Third paragraph containing the resolution."
    )

    chunks = chunk_document(
        document_id="doc-1",
        document_title="Example document",
        text=text,
        config=ChunkingConfig(
            target_chars=60,
            overlap_chars=15,
            boundary_search_chars=15,
            minimum_chunk_chars=20,
        ),
    )

    assert len(chunks) >= 2

    for chunk in chunks:
        assert chunk.content == text[chunk.start_offset : chunk.end_offset]

        assert chunk.end_offset > chunk.start_offset


def test_chunking_is_deterministic() -> None:
    text = "Technical documentation. " * 100

    config = ChunkingConfig(
        target_chars=200,
        overlap_chars=40,
        boundary_search_chars=30,
        minimum_chunk_chars=80,
    )

    first = chunk_document(
        document_id="doc-1",
        document_title="Document",
        text=text,
        config=config,
    )

    second = chunk_document(
        document_id="doc-1",
        document_title="Document",
        text=text,
        config=config,
    )

    assert first == second


def test_answer_span_is_fully_covered() -> None:
    text = "A" * 100 + "ANSWER" + "B" * 100

    chunks = chunk_document(
        document_id="doc-1",
        document_title="Document",
        text=text,
        config=ChunkingConfig(
            target_chars=100,
            overlap_chars=30,
            boundary_search_chars=0,
            minimum_chunk_chars=20,
        ),
    )

    report = evaluate_answer_span_coverage(
        chunks=chunks,
        ground_truth=[
            {
                "question_id": "question-1",
                "answerable": True,
                "document_id": "doc-1",
                "start_offset": 100,
                "end_offset": 106,
            }
        ],
    )

    assert report["answerable_questions"] == 1
    assert report["fully_covered_questions"] == 1
    assert report["full_span_coverage_rate"] == 1.0


def test_evaluation_ignores_unanswerable_questions() -> None:
    report = evaluate_answer_span_coverage(
        chunks=[],
        ground_truth=[
            {
                "question_id": "question-1",
                "answerable": False,
                "document_id": None,
                "start_offset": None,
                "end_offset": None,
            }
        ],
    )

    assert report["answerable_questions"] == 0
    assert report["missing_gold_documents"] == 0
