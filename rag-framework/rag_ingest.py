"""RAG Ingest Pipeline — Orchestrator.

Reads Confluence page references from inventory.json, fetches each page,
chunks the content semantically, generates embeddings, and stores everything
in Qdrant with full metadata.

Run:
  cd rag-framework
  python rag_ingest.py

Prerequisites:
  pip install requests beautifulsoup4 tiktoken openai qdrant-client

Environment variables required:
  LITELLM_KEY        — LiteLLM API key
  CONFLUENCE_TOKEN   — Confluence Bearer token

Optional overrides:
  LITELLM_URL, LITELLM_EMBED_MODEL, CONFLUENCE_URL, QDRANT_URL, QDRANT_COLLECTION
"""

import json
import logging
import sys
from pathlib import Path

import config
import confluence_loader
import chunker
import embedder
import qdrant_store

# ---------------------------------------------------------------------------
# Logging — INFO to stdout so the terminal shows progress milestones
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_inventory(inventory_path: Path) -> list[dict]:
    """Load and validate the page inventory from JSON.

    Args:
        inventory_path: Path to inventory.json.

    Returns:
        List of enabled page descriptor dicts.

    Raises:
        FileNotFoundError: If inventory.json does not exist.
        KeyError: If the JSON schema is missing required fields.
    """
    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")

    with inventory_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    all_pages: list[dict] = data.get("pages", [])
    enabled = [p for p in all_pages if p.get("enabled", True)]

    logger.info(
        "Inventory: %d total page(s), %d enabled for ingestion",
        len(all_pages), len(enabled),
    )
    return enabled


def ingest_page(page_descriptor: dict, vector_size: int) -> int:
    """Run the full ingest pipeline for a single Confluence page.

    Pipeline: fetch → clean → chunk → embed → upsert

    Args:
        page_descriptor: Dict from inventory.json (page_id, title, url, tags, …).
        vector_size: Embedding dimension (used to ensure collection exists).

    Returns:
        Number of chunks upserted for this page.
    """
    page_id: str = page_descriptor["page_id"]
    page_title: str = page_descriptor["title"]
    url: str = page_descriptor["url"]
    tags: list[str] = page_descriptor.get("tags", [])

    logger.info("─── Ingesting: %s (id=%s) ───", page_title, page_id)

    # Step 1: Fetch + clean from Confluence
    page_data = confluence_loader.load_page(page_id)
    clean_text: str = page_data["clean_text"]

    if not clean_text:
        logger.warning("Skipping page %s — empty content after cleaning", page_id)
        return 0

    # Step 2: Semantic chunking
    chunks = chunker.chunk_text(
        clean_text=clean_text,
        page_id=page_id,
        page_title=page_title,
        url=url,
        tags=tags,
    )

    if not chunks:
        logger.warning("Skipping page %s — no chunks produced", page_id)
        return 0

    # Step 3: Generate embeddings for all chunks
    chunk_texts = [ch["text"] for ch in chunks]
    vectors = embedder.embed_texts(chunk_texts)

    # Step 4: Ensure collection exists, then upsert
    qdrant_store.ensure_collection(vector_size=vector_size)
    count = qdrant_store.upsert_chunks(chunks=chunks, vectors=vectors)

    return count


def main() -> None:
    """Entry point: orchestrate full ingestion of all enabled Confluence pages."""
    config.validate_required_env_vars()

    pages = load_inventory(config.INVENTORY_PATH)

    if not pages:
        print("No enabled pages found in inventory. Nothing to ingest.")
        sys.exit(0)

    # Probe embedding dimension once (avoids redundant API calls per page)
    logger.info("Probing embedding dimension …")
    vector_size = embedder.get_embedding_dimension()

    total_chunks = 0
    failed_pages: list[str] = []

    for page_descriptor in pages:
        try:
            count = ingest_page(page_descriptor, vector_size)
            total_chunks += count
        except Exception as exc:
            page_id = page_descriptor.get("page_id", "unknown")
            logger.error("Failed to ingest page %s: %s", page_id, exc)
            failed_pages.append(page_id)
            # Continue with remaining pages
            continue

    # Human-readable summary
    print("\n" + "=" * 60)
    print("✅  INGEST COMPLETE")
    print(f"    Pages processed : {len(pages) - len(failed_pages)} / {len(pages)}")
    print(f"    Chunks stored   : {total_chunks}")
    print(f"    Collection      : {config.QDRANT_COLLECTION}")
    print(f"    Qdrant          : {config.QDRANT_URL}")
    if failed_pages:
        print(f"    ❌ Failed page IDs: {', '.join(failed_pages)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
