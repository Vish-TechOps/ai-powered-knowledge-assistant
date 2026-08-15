"""Embedding generation via OpenAI-compatible LiteLLM endpoint.

Wraps the OpenAI client to generate text embeddings and expose a clean
interface that the ingest and query pipelines can both use.

Prerequisites:
  pip install openai
"""

import logging

from openai import OpenAI, OpenAIError

import config

logger = logging.getLogger(__name__)


def _get_client(litellm_url: str, litellm_key: str) -> OpenAI:
    """Instantiate and return an OpenAI-compatible client.

    Args:
        litellm_url: Base URL of the LiteLLM LLM endpoint.
        litellm_key: API key / Bearer token for LiteLLM.

    Returns:
        Configured OpenAI client instance.
    """
    return OpenAI(api_key=litellm_key, base_url=litellm_url)


def embed_texts(
    texts: list[str],
    model: str = config.LITELLM_EMBED_MODEL,
    litellm_url: str = config.LITELLM_URL,
    litellm_key: str = config.LITELLM_KEY,
) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Calls the LiteLLM embedding endpoint once per text (sequential) to
    avoid batching issues with custom endpoints. For large-scale ingestion,
    this can be extended to true batch calls.

    Args:
        texts: List of strings to embed.
        model: Embedding model name (default from config).
        litellm_url: LiteLLM base URL (default from config).
        litellm_key: LiteLLM API key (default from config).

    Returns:
        List of embedding vectors (list[float]) in the same order as input.

    Raises:
        OpenAIError: On API-level error (rate limit, auth failure, etc.).
        RuntimeError: If the response shape is unexpected.
    """
    client = _get_client(litellm_url, litellm_key)
    vectors: list[list[float]] = []

    for i, text in enumerate(texts):
        logger.debug("Embedding text %d/%d (len=%d chars)", i + 1, len(texts), len(text))
        try:
            response = client.embeddings.create(model=model, input=text)
            vector = response.data[0].embedding
            vectors.append(vector)
        except OpenAIError as exc:
            logger.error("Embedding failed for text index %d: %s", i, exc)
            raise

    logger.info("Generated %d embedding(s) using model '%s'", len(vectors), model)
    return vectors


def embed_query(
    query: str,
    model: str = config.LITELLM_EMBED_MODEL,
    litellm_url: str = config.LITELLM_URL,
    litellm_key: str = config.LITELLM_KEY,
) -> list[float]:
    """Generate a single embedding for a query string.

    Args:
        query: The user's search query.
        model: Embedding model name (default from config).
        litellm_url: LiteLLM base URL (default from config).
        litellm_key: LiteLLM API key (default from config).

    Returns:
        A single embedding vector (list[float]).

    Raises:
        OpenAIError: On API-level error.
    """
    vectors = embed_texts([query], model=model, litellm_url=litellm_url, litellm_key=litellm_key)
    return vectors[0]


def get_embedding_dimension(
    model: str = config.LITELLM_EMBED_MODEL,
    litellm_url: str = config.LITELLM_URL,
    litellm_key: str = config.LITELLM_KEY,
) -> int:
    """Probe the embedding model to determine the vector dimension.

    Args:
        model: Embedding model name (default from config).
        litellm_url: LiteLLM base URL (default from config).
        litellm_key: LiteLLM API key (default from config).

    Returns:
        Integer dimension of the embedding vectors.
    """
    vector = embed_query("dimension probe", model=model, litellm_url=litellm_url, litellm_key=litellm_key)
    dim = len(vector)
    logger.info("Embedding dimension for model '%s': %d", model, dim)
    return dim
