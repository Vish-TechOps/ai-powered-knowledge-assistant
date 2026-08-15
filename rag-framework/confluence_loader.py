"""Confluence REST API loader.

Fetches page content by page_id using a Bearer token, then strips HTML
to return clean plain text ready for chunking.

Prerequisites:
  pip install requests beautifulsoup4
"""

import logging
import re

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)


def _auth_headers(token: str) -> dict[str, str]:
    """Return HTTP headers with Bearer token auth."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def fetch_page_storage(page_id: str, confluence_url: str, token: str) -> dict:
    """Fetch raw Confluence page data (storage format) via REST API.

    Args:
        page_id: Numeric Confluence page ID (e.g. '1911031070').
        confluence_url: Base URL of the Confluence instance.
        token: Bearer token for authentication.

    Returns:
        Parsed JSON response dict from the Confluence API.

    Raises:
        requests.HTTPError: On non-2xx response.
        requests.ConnectionError: On network failure.
    """
    url = f"{confluence_url.rstrip('/')}/rest/api/content/{page_id}"
    params = {"expand": "body.storage,title,space,version"}

    logger.info("Fetching Confluence page id=%s", page_id)
    try:
        response = requests.get(
            url,
            headers=_auth_headers(token),
            params=params,
            timeout=30,
            verify=False,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("HTTP error fetching page %s: %s", page_id, exc)
        raise
    except requests.ConnectionError as exc:
        logger.error("Connection error reaching Confluence: %s", exc)
        raise

    return response.json()


def html_to_clean_text(html: str) -> str:
    """Convert Confluence storage-format HTML to clean plain text.

    Preserves heading structure by prepending heading text with a marker
    so the chunker can detect semantic boundaries.

    Args:
        html: Raw HTML string from Confluence storage format.

    Returns:
        Clean, normalised plain text string.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Replace heading tags with marker lines so the chunker can split on them
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        level = tag.name  # e.g. 'h2'
        tag.replace_with(f"\n### HEADING:{level} {tag.get_text()} ###\n")

    # Replace list items with bullet markers
    for li in soup.find_all("li"):
        li.replace_with(f"\n• {li.get_text(strip=True)}")

    # Replace table cells with tab-separated text
    for td in soup.find_all(["td", "th"]):
        td.replace_with(f" {td.get_text(strip=True)} |")

    text = soup.get_text(separator="\n")

    # Normalise whitespace: collapse multiple blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def load_page(page_id: str) -> dict:
    """High-level loader: fetch a Confluence page and return structured data.

    Args:
        page_id: Numeric Confluence page ID.

    Returns:
        Dict with keys: page_id, title, space_key, clean_text.

    Raises:
        RuntimeError: If the page body cannot be extracted.
    """
    raw = fetch_page_storage(
        page_id=page_id,
        confluence_url=config.CONFLUENCE_URL,
        token=config.CONFLUENCE_TOKEN,
    )

    title = raw.get("title", "Untitled")
    space_key = raw.get("space", {}).get("key", "")

    body_html = raw.get("body", {}).get("storage", {}).get("value", "")
    if not body_html:
        raise RuntimeError(f"Empty body returned for page_id={page_id} (title='{title}')")

    clean_text = html_to_clean_text(body_html)
    logger.info("Loaded page '%s' (%s chars clean text)", title, len(clean_text))

    return {
        "page_id": page_id,
        "title": title,
        "space_key": space_key,
        "clean_text": clean_text,
    }
