"""RAG Query Pipeline — Orchestrator.

Accepts a user query (via command-line argument or interactive prompt),
retrieves the most relevant Confluence chunks from Qdrant, and generates
a context-aware answer via the LiteLLM LLM endpoint.

This is the entry point Cline uses for user queries.

Usage (CLI):
  cd rag-framework
  python rag_query.py "What is the AI-Driven Test Evaluation initiative?"

Usage (interactive, no argument):
  python rag_query.py

Environment variables required:
  LITELLM_KEY   — LiteLLM API key

Optional overrides:
  LITELLM_URL, LITELLM_LLM_MODEL, LITELLM_EMBED_MODEL,
  QDRANT_URL, QDRANT_COLLECTION
"""

import logging
import sys

from openai import OpenAI, OpenAIError

import config
import embedder
import qdrant_store

# ---------------------------------------------------------------------------
# Logging — WARNING level for query path (keep output clean for the user)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_context(hits: list) -> tuple[str, list[dict]]:
    """Format retrieved chunks into a numbered context block.

    Args:
        hits: List of ScoredPoint objects from Qdrant search.

    Returns:
        Tuple of:
          - context_text: Formatted string for the LLM prompt.
          - sources: List of source dicts (title, url, score) for citation.
    """
    context_parts: list[str] = []
    sources: list[dict] = []

    for i, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        title = payload.get("page_title", "Unknown")
        url = payload.get("url", "")
        text = payload.get("text", "").strip()
        score = hit.score if hit.score is not None else 0.0

        context_parts.append(f"[{i}] Source: {title}\n{text}")
        sources.append({"index": i, "title": title, "url": url, "score": round(score, 4)})

    return "\n\n---\n\n".join(context_parts), sources


def generate_answer(query: str, context: str) -> str:
    """Call the LiteLLM LLM to generate a RAG answer.

    Args:
        query: The user's original question.
        context: Numbered context block built from retrieved chunks.

    Returns:
        The LLM-generated answer string.

    Raises:
        OpenAIError: On API-level failure.
    """
    client = OpenAI(api_key=config.LITELLM_KEY, base_url=config.LITELLM_URL)

    user_prompt = config.RAG_USER_PROMPT_TEMPLATE.format(
        context=context,
        question=query,
    )

    try:
        response = client.chat.completions.create(
            model=config.LITELLM_LLM_MODEL,
            messages=[
                {"role": "system", "content": config.RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,        # low temperature for factual, grounded answers
            max_tokens=1024,
        )
    except OpenAIError as exc:
        logger.error("LLM generation failed: %s", exc)
        raise

    content = response.choices[0].message.content
    return content.strip() if content else ""


def query_pipeline(user_query: str) -> dict:
    """Execute the full RAG query pipeline for a given user query.

    Pipeline: embed query → retrieve chunks → build context → generate answer

    Args:
        user_query: The question submitted by the user.

    Returns:
        Dict with keys: query, answer, sources, chunk_count.
    """
    # Step 1: Embed the query
    query_vector = embedder.embed_query(user_query)

    # Step 2: Retrieve top-K chunks above score threshold
    hits = qdrant_store.search(query_vector)

    if not hits:
        return {
            "query": user_query,
            "answer": (
                "I could not find relevant information in the knowledge base for your query. "
                "Please try rephrasing, or ensure the relevant Confluence pages have been ingested."
            ),
            "sources": [],
            "chunk_count": 0,
        }

    # Step 3: Build context from retrieved chunks
    context, sources = build_context(hits)

    # Step 4: Generate answer via LLM
    answer = generate_answer(query=user_query, context=context)

    return {
        "query": user_query,
        "answer": answer,
        "sources": sources,
        "chunk_count": len(hits),
    }


def print_result(result: dict) -> None:
    """Print the RAG query result in a clean, readable format.

    Args:
        result: Output dict from query_pipeline().
    """
    print("\n" + "=" * 70)
    print(f"🔍  QUERY   : {result['query']}")
    print("=" * 70)
    print(f"\n💡  ANSWER\n{result['answer']}")

    if result["sources"]:
        print(f"\n📚  SOURCES  ({result['chunk_count']} chunk(s) retrieved)")
        for src in result["sources"]:
            print(f"  [{src['index']}] {src['title']}  (score={src['score']})")
            if src["url"]:
                print(f"       {src['url']}")
    else:
        print("\n📚  No sources retrieved.")

    print("=" * 70 + "\n")


def main() -> None:
    """Entry point: run the RAG query pipeline for a user question."""
    # Only LITELLM_KEY is required for queries (no Confluence access needed)
    if not config.LITELLM_KEY:
        raise RuntimeError(
            "Missing required environment variable: LITELLM_KEY. "
            "Export it before running."
        )

    # Accept query from CLI arg or interactive prompt
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:]).strip()
    else:
        user_query = input("Enter your query: ").strip()

    if not user_query:
        print("❌  No query provided. Exiting.")
        sys.exit(1)

    result = query_pipeline(user_query)
    print_result(result)


if __name__ == "__main__":
    main()
