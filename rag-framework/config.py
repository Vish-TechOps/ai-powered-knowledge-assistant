"""Central configuration for the RAG pipeline.

All secrets and tuneable constants live here.
Values are read from environment variables — never hardcode secrets.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
INVENTORY_PATH = ROOT_DIR / "data-sources" / "inventory.json"

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "admin123")
NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "ai-knowledge-assistant")

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "ai-knowledge-assistant")

# ---------------------------------------------------------------------------
# LiteLLM / OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
LITELLM_URL: str = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY: str = os.getenv("LITELLM_KEY", "")
LITELLM_EMBED_MODEL: str = os.getenv("LITELLM_EMBED_MODEL", "text-embedding-004")
LITELLM_LLM_MODEL: str = os.getenv("LITELLM_LLM_MODEL", "gpt-5.4-mini")

# ---------------------------------------------------------------------------
# Confluence REST API
# ---------------------------------------------------------------------------
CONFLUENCE_URL: str = os.getenv("CONFLUENCE_URL", "")
CONFLUENCE_TOKEN: str = os.getenv("CONFLUENCE_TOKEN", "")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_TARGET_TOKENS: int = 400   # target chunk size in tokens
CHUNK_OVERLAP_TOKENS: int = 50   # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = 5
SCORE_THRESHOLD: float = 0.50   # cosine similarity minimum

# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT: str = (
    "You are a helpful enterprise knowledge assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the context does not contain enough information, say so clearly. "
    "Be concise, accurate, and cite the page title when referencing a source."
)

RAG_USER_PROMPT_TEMPLATE: str = (
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:"
)


def validate_required_env_vars() -> None:
    """Raise RuntimeError if any mandatory secret is missing."""
    missing = []
    if not LITELLM_KEY:
        missing.append("LITELLM_KEY")
    if not CONFLUENCE_TOKEN:
        missing.append("CONFLUENCE_TOKEN")
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Export them before running."
        )
