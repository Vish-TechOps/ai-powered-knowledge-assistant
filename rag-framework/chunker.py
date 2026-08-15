"""Semantic paragraph-aware text chunker.

Splits clean Confluence text into overlapping chunks that respect heading
and paragraph boundaries. Uses tiktoken to stay within token budget.

Prerequisites:
  pip install tiktoken
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import tiktoken

import config

logger = logging.getLogger(__name__)

# Tiktoken encoding for token counting (cl100k_base covers GPT-3.5/4/embedding models)
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Regex to detect the heading markers injected by confluence_loader
_HEADING_RE = re.compile(r"###\s*HEADING:h\d\s+(.*?)\s*###")


def _count_tokens(text: str) -> int:
    """Return the number of tokens in a text string."""
    return len(_ENCODING.encode(text))


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines or heading markers.

    Args:
        text: Clean plain text (may contain HEADING markers).

    Returns:
        List of non-empty paragraph strings.
    """
    # Split on blank lines or immediately before a HEADING marker
    parts = re.split(r"\n{2,}|(?=###\s*HEADING:)", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Chunk:
    """A single text chunk with its metadata."""

    text: str
    chunk_index: int
    heading_context: str = ""
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = _count_tokens(self.text)


def chunk_text(
    clean_text: str,
    page_id: str,
    page_title: str,
    url: str,
    tags: list[str],
    target_tokens: int = config.CHUNK_TARGET_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """Chunk clean text into token-bounded, semantically coherent pieces.

    Strategy:
    - Split on blank lines / heading markers (paragraph boundaries).
    - Accumulate paragraphs until the chunk would exceed `target_tokens`.
    - When flushing a chunk, keep the last `overlap_tokens` worth of text
      as the start of the next chunk (overlap window).
    - Each chunk carries the most recent heading as context prefix.

    Args:
        clean_text: Pre-cleaned plain text from confluence_loader.
        page_id: Confluence page ID for metadata.
        page_title: Human-readable page title.
        url: Canonical page URL.
        tags: List of topic tags from inventory.json.
        target_tokens: Maximum tokens per chunk (default from config).
        overlap_tokens: Tokens of overlap between adjacent chunks (default from config).

    Returns:
        List of payload dicts ready for Qdrant (without vector).
    """
    paragraphs = _split_into_paragraphs(clean_text)
    ingested_at = datetime.now(timezone.utc).isoformat()

    chunks: list[dict] = []
    current_paragraphs: list[str] = []
    current_tokens: int = 0
    current_heading: str = ""
    overlap_buffer: str = ""

    def _flush_chunk(chunk_index: int) -> dict:
        """Assemble and return the current chunk payload."""
        body = "\n\n".join(current_paragraphs)
        # Prepend heading context if available
        prefix = f"[{current_heading}]\n" if current_heading else ""
        full_text = (prefix + body).strip()

        return {
            "text": full_text,
            "page_id": page_id,
            "page_title": page_title,
            "url": url,
            "tags": tags,
            "chunk_index": chunk_index,
            "total_chunks": -1,        # back-filled after all chunks known
            "source": "confluence",
            "ingested_at": ingested_at,
            "heading_context": current_heading,
        }

    chunk_index = 0

    # Seed with overlap from previous chunk if exists
    if overlap_buffer:
        current_paragraphs = [overlap_buffer]
        current_tokens = _count_tokens(overlap_buffer)

    for para in paragraphs:
        # Check if this paragraph is a heading marker
        heading_match = _HEADING_RE.match(para)
        if heading_match:
            current_heading = heading_match.group(1).strip()
            # Don't add raw marker to chunk text — heading is captured as context
            continue

        para_tokens = _count_tokens(para)

        # If adding this paragraph exceeds target, flush current chunk first
        if current_tokens + para_tokens > target_tokens and current_paragraphs:
            chunk_payload = _flush_chunk(chunk_index)
            chunks.append(chunk_payload)
            chunk_index += 1

            # Build overlap: take the last paragraph(s) up to overlap_tokens
            overlap_text = _build_overlap(current_paragraphs, overlap_tokens)
            current_paragraphs = [overlap_text] if overlap_text else []
            current_tokens = _count_tokens(overlap_text) if overlap_text else 0

        current_paragraphs.append(para)
        current_tokens += para_tokens

    # Flush remaining paragraphs
    if current_paragraphs:
        chunk_payload = _flush_chunk(chunk_index)
        chunks.append(chunk_payload)

    # Back-fill total_chunks now that we know the final count
    total = len(chunks)
    for ch in chunks:
        ch["total_chunks"] = total

    logger.info(
        "Chunked page '%s' into %d chunk(s) (target=%d tokens)",
        page_title, total, target_tokens,
    )
    return chunks


def _build_overlap(paragraphs: list[str], overlap_tokens: int) -> str:
    """Return the tail of paragraphs that fits within overlap_tokens.

    Args:
        paragraphs: Current list of paragraphs in the chunk.
        overlap_tokens: Maximum tokens allowed in the overlap buffer.

    Returns:
        A single string containing the overlap text.
    """
    overlap_parts: list[str] = []
    token_count = 0

    for para in reversed(paragraphs):
        para_tokens = _count_tokens(para)
        if token_count + para_tokens <= overlap_tokens:
            overlap_parts.insert(0, para)
            token_count += para_tokens
        else:
            break

    return "\n\n".join(overlap_parts)
