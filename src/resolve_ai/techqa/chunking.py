from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class TechQAChunkingError(RuntimeError):
    """Raised when TechQA documents cannot be chunked safely."""


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Character-based baseline chunking configuration.

    Character offsets are used because TechQA answer annotations are
    expressed as character offsets in the original document text.
    """

    target_chars: int = 1800
    overlap_chars: int = 300
    boundary_search_chars: int = 250
    minimum_chunk_chars: int = 300

    def __post_init__(self) -> None:
        if self.target_chars <= 0:
            raise ValueError("target_chars must be positive.")

        if self.overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative.")

        if self.overlap_chars >= self.target_chars:
            raise ValueError("overlap_chars must be smaller than target_chars.")

        if self.boundary_search_chars < 0:
            raise ValueError("boundary_search_chars cannot be negative.")

        if self.minimum_chunk_chars <= 0:
            raise ValueError("minimum_chunk_chars must be positive.")

        if self.minimum_chunk_chars >= self.target_chars:
            raise ValueError("minimum_chunk_chars must be smaller than target_chars.")


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_title: str
    position: int
    start_offset: int
    end_offset: int
    content: str
    search_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_WHITESPACE_PATTERN = re.compile(r"\s+")

_ZERO_WIDTH_TRANSLATION = str.maketrans(
    {
        "\u200b": None,
        "\u200c": None,
        "\u200d": None,
        "\ufeff": None,
    }
)

# Boundaries are ordered from strongest to weakest.
_BOUNDARY_TOKENS = (
    "\n\n",
    "\r\n\r\n",
    "\n",
    "\r\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ": ",
    " ",
)


def normalize_for_search(text: str) -> str:
    """
    Create a normalized representation for retrieval.

    This function must never be used to calculate TechQA offsets.
    The original text remains the source of truth.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    normalized = html.unescape(text)
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.translate(_ZERO_WIDTH_TRANSLATION)
    normalized = normalized.replace("\u00a0", " ")
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized)

    return normalized.strip()


def _deterministic_chunk_id(
    *,
    document_id: str,
    start_offset: int,
    end_offset: int,
) -> str:
    payload = (f"{document_id}:{start_offset}:{end_offset}").encode("utf-8")

    digest = hashlib.sha256(payload).hexdigest()[:16]

    return f"chunk_{digest}"


def _find_previous_boundary(
    text: str,
    *,
    lower_bound: int,
    tentative_end: int,
) -> int | None:
    for token in _BOUNDARY_TOKENS:
        position = text.rfind(
            token,
            lower_bound,
            tentative_end,
        )

        if position != -1:
            return position + len(token)

    return None


def _find_next_boundary(
    text: str,
    *,
    tentative_end: int,
    upper_bound: int,
) -> int | None:
    for token in _BOUNDARY_TOKENS:
        position = text.find(
            token,
            tentative_end,
            upper_bound,
        )

        if position != -1:
            return position + len(token)

    return None


def _choose_chunk_end(
    text: str,
    *,
    start_offset: int,
    config: ChunkingConfig,
) -> int:
    document_length = len(text)

    tentative_end = min(
        start_offset + config.target_chars,
        document_length,
    )

    if tentative_end >= document_length:
        return document_length

    minimum_end = min(
        start_offset + config.minimum_chunk_chars,
        document_length,
    )

    lower_bound = max(
        minimum_end,
        tentative_end - config.boundary_search_chars,
    )

    previous_boundary = _find_previous_boundary(
        text,
        lower_bound=lower_bound,
        tentative_end=tentative_end,
    )

    if previous_boundary is not None:
        return previous_boundary

    upper_bound = min(
        document_length,
        tentative_end + config.boundary_search_chars,
    )

    next_boundary = _find_next_boundary(
        text,
        tentative_end=tentative_end,
        upper_bound=upper_bound,
    )

    if next_boundary is not None:
        return next_boundary

    return tentative_end


def chunk_document(
    *,
    document_id: str,
    document_title: str,
    text: str,
    config: ChunkingConfig,
) -> list[DocumentChunk]:
    """
    Split a document while preserving raw character offsets.

    Chunk content is always exactly equal to:
    original_text[start_offset:end_offset]
    """

    if not document_id.strip():
        raise TechQAChunkingError("document_id cannot be empty.")

    if not isinstance(text, str):
        raise TechQAChunkingError(f"Document {document_id} has no valid text.")

    if not text:
        return []

    chunks: list[DocumentChunk] = []
    start_offset = 0
    position = 0

    while start_offset < len(text):
        end_offset = _choose_chunk_end(
            text,
            start_offset=start_offset,
            config=config,
        )

        if end_offset <= start_offset:
            raise TechQAChunkingError(
                f"Chunking made no progress for document "
                f"{document_id} at offset {start_offset}."
            )

        content = text[start_offset:end_offset]

        # Pure whitespace chunks are not useful for retrieval.
        if content.strip():
            chunks.append(
                DocumentChunk(
                    chunk_id=_deterministic_chunk_id(
                        document_id=document_id,
                        start_offset=start_offset,
                        end_offset=end_offset,
                    ),
                    document_id=document_id,
                    document_title=document_title,
                    position=position,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    content=content,
                    search_text=normalize_for_search(content),
                )
            )

            position += 1

        if end_offset >= len(text):
            break

        next_start = end_offset - config.overlap_chars

        # Defensive protection against infinite loops.
        if next_start <= start_offset:
            next_start = end_offset

        start_offset = next_start

    return chunks


def chunk_corpus(
    documents: Any,
    *,
    config: ChunkingConfig,
) -> list[DocumentChunk]:
    """Chunk every document in deterministic document-ID order."""

    if not isinstance(documents, dict):
        raise TechQAChunkingError(
            "documents.json must contain an object indexed by document ID."
        )

    chunks: list[DocumentChunk] = []

    for raw_document_id in sorted(documents):
        document_id = str(raw_document_id)
        document = documents[raw_document_id]

        if not isinstance(document, dict):
            raise TechQAChunkingError(f"Document {document_id} is not a JSON object.")

        text = document.get("text")

        if not isinstance(text, str):
            raise TechQAChunkingError(f"Document {document_id} has no textual content.")

        raw_title = document.get("title")
        title = raw_title if isinstance(raw_title, str) else ""

        chunks.extend(
            chunk_document(
                document_id=document_id,
                document_title=title,
                text=text,
                config=config,
            )
        )

    return chunks


def evaluate_answer_span_coverage(
    *,
    chunks: Iterable[DocumentChunk],
    ground_truth: Any,
) -> dict[str, Any]:
    """
    Measure whether each answer span is fully contained in at least one chunk.

    Ground truth is used only for evaluation, never during chunk construction.
    """

    if not isinstance(ground_truth, list):
        raise TechQAChunkingError("ground_truth.json must contain a JSON list.")

    chunks_by_document: dict[str, list[DocumentChunk]] = defaultdict(list)

    for chunk in chunks:
        chunks_by_document[chunk.document_id].append(chunk)

    answerable_count = 0
    fully_covered_count = 0
    partially_covered_count = 0
    missing_document_count = 0
    invalid_annotation_count = 0
    uncovered_examples: list[dict[str, Any]] = []

    for annotation in ground_truth:
        if not isinstance(annotation, dict):
            invalid_annotation_count += 1
            continue

        if annotation.get("answerable") is not True:
            continue

        answerable_count += 1

        question_id = str(annotation.get("question_id", ""))

        document_id = annotation.get("document_id")
        start_offset = annotation.get("start_offset")
        end_offset = annotation.get("end_offset")

        if (
            not isinstance(document_id, str)
            or not isinstance(start_offset, int)
            or not isinstance(end_offset, int)
            or start_offset < 0
            or end_offset <= start_offset
        ):
            invalid_annotation_count += 1
            continue

        document_chunks = chunks_by_document.get(
            document_id,
            [],
        )

        if not document_chunks:
            missing_document_count += 1

            uncovered_examples.append(
                {
                    "question_id": question_id,
                    "document_id": document_id,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "reason": "gold_document_has_no_chunks",
                }
            )
            continue

        fully_containing_chunks = [
            chunk
            for chunk in document_chunks
            if (chunk.start_offset <= start_offset and chunk.end_offset >= end_offset)
        ]

        if fully_containing_chunks:
            fully_covered_count += 1
            continue

        overlapping_chunks = [
            chunk
            for chunk in document_chunks
            if (chunk.start_offset < end_offset and chunk.end_offset > start_offset)
        ]

        if overlapping_chunks:
            partially_covered_count += 1
            reason = "span_crosses_chunk_boundaries"
        else:
            reason = "span_not_found_in_any_chunk"

        if len(uncovered_examples) < 25:
            uncovered_examples.append(
                {
                    "question_id": question_id,
                    "document_id": document_id,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "answer_length": end_offset - start_offset,
                    "reason": reason,
                    "overlapping_chunk_ids": [
                        chunk.chunk_id for chunk in overlapping_chunks
                    ],
                }
            )

    coverage_rate = fully_covered_count / answerable_count if answerable_count else 0.0

    return {
        "answerable_questions": answerable_count,
        "fully_covered_questions": fully_covered_count,
        "partially_covered_questions": partially_covered_count,
        "missing_gold_documents": missing_document_count,
        "invalid_annotations": invalid_annotation_count,
        "full_span_coverage_rate": round(
            coverage_rate,
            6,
        ),
        "uncovered_examples": uncovered_examples,
    }


def summarize_chunks(
    chunks: list[DocumentChunk],
) -> dict[str, Any]:
    lengths = [chunk.end_offset - chunk.start_offset for chunk in chunks]

    document_ids = {chunk.document_id for chunk in chunks}

    if not lengths:
        return {
            "chunk_count": 0,
            "document_count": 0,
            "minimum_length": None,
            "mean_length": None,
            "maximum_length": None,
        }

    return {
        "chunk_count": len(chunks),
        "document_count": len(document_ids),
        "minimum_length": min(lengths),
        "mean_length": round(
            sum(lengths) / len(lengths),
            2,
        ),
        "maximum_length": max(lengths),
    }


def write_chunks_jsonl(
    chunks: Iterable[DocumentChunk],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            json.dump(
                chunk.to_dict(),
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def write_json(
    content: Any,
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")
