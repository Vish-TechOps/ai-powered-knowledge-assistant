"""Qdrant vector store operations.

Handles collection lifecycle, idempotent upserts (deterministic IDs),
and semantic search with score-threshold filtering.

Prerequisites:
  pip install qdrant-client
"""

import hashlib
import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

import config

logger = logging.getLogger(__name__)


def _get_client(qdrant_url: str = config.QDRANT_URL) -> QdrantClient:
    """Instantiate and return a Qdrant client.

    Args:
        qdrant_url: Qdrant instance URL (default from config).

    Returns:
        Configured QdrantClient instance.
    """
    return QdrantClient(url=qdrant_url)


def make_chunk_id(page_id: str, chunk_index: int) -> str:
    """Generate a deterministic UUID-compatible ID for a chunk.

    Uses SHA-256 of `{page_id}_{chunk_index}` and formats it as a UUID string
    so Qdrant accepts it natively.

    Args:
        page_id: Confluence page ID.
        chunk_index: Zero-based chunk sequence number.

    Returns:
        UUID-formatted string derived from the hash.
    """
    raw = f"{page_id}_{chunk_index}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    # Format first 32 hex chars as UUID: 8-4-4-4-12
    return f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def ensure_collection(
    vector_size: int,
    collection_name: str = config.QDRANT_COLLECTION,
    qdrant_url: str = config.QDRANT_URL,
) -> None:
    """Create a Qdrant collection if it does not already exist.

    Uses cosine similarity as per project standards.

    Args:
        vector_size: Dimensionality of the embedding vectors.
        collection_name: Target collection name (default from config).
        qdrant_url: Qdrant instance URL (default from config).
    """
    client = _get_client(qdrant_url)
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (dim=%d, cosine)", collection_name, vector_size)
    else:
        logger.info("Qdrant collection '%s' already exists — skipping creation", collection_name)


def upsert_chunks(
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    collection_name: str = config.QDRANT_COLLECTION,
    qdrant_url: str = config.QDRANT_URL,
) -> int:
    """Upsert a list of chunk payloads + their vectors into Qdrant.

    Chunk IDs are deterministic (sha256-based), so re-running produces
    identical results — no duplicates.

    Args:
        chunks: List of payload dicts from chunker (must include page_id, chunk_index).
        vectors: Parallel list of embedding vectors, same length as chunks.
        collection_name: Target collection name (default from config).
        qdrant_url: Qdrant instance URL (default from config).

    Returns:
        Number of points upserted.

    Raises:
        ValueError: If chunks and vectors lengths do not match.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(vectors)} vectors provided."
        )

    client = _get_client(qdrant_url)
    points: list[PointStruct] = []

    for chunk, vector in zip(chunks, vectors):
        point_id = make_chunk_id(chunk["page_id"], chunk["chunk_index"])
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk,
            )
        )

    try:
        client.upsert(collection_name=collection_name, points=points)
    except Exception as exc:
        logger.error("Qdrant upsert failed for collection '%s': %s", collection_name, exc)
        raise

    logger.info(
        "Upserted %d chunk(s) into collection '%s'",
        len(points), collection_name,
    )
    return len(points)


def search(
    query_vector: list[float],
    top_k: int = config.TOP_K,
    score_threshold: float = config.SCORE_THRESHOLD,
    collection_name: str = config.QDRANT_COLLECTION,
    qdrant_url: str = config.QDRANT_URL,
) -> list[ScoredPoint]:
    """Retrieve the top-K most similar chunks above the score threshold.

    Args:
        query_vector: Embedding of the user's query.
        top_k: Maximum number of results to return (default from config).
        score_threshold: Minimum cosine similarity to include (default from config).
        collection_name: Collection to search (default from config).
        qdrant_url: Qdrant instance URL (default from config).

    Returns:
        List of ScoredPoint objects (may be empty if no results pass threshold).

    Raises:
        Exception: On Qdrant connectivity or query failure.
    """
    client = _get_client(qdrant_url)

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        logger.error("Qdrant search failed on collection '%s': %s", collection_name, exc)
        raise

    hits = results.points
    logger.info(
        "Search returned %d result(s) above threshold=%.2f (top_k=%d)",
        len(hits), score_threshold, top_k,
    )
    return hits
