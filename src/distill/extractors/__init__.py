"""Content extraction from URLs and HTML."""

from __future__ import annotations

import httpx
from readability import Document


def extract_from_url(url: str, timeout: float = 15.0) -> dict:
    """Fetch a URL and extract readable content.

    Returns:
        dict with keys: title, text, url, word_count
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; Distill/0.1; +https://github.com/YOUR_USERNAME/distill)"
        ),
    }

    response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    response.raise_for_status()

    return extract_from_html(response.text, url=url)


def extract_from_html(html: str, url: str = "") -> dict:
    """Extract readable text from HTML using readability."""
    doc = Document(html)
    title = doc.title()

    # Get the readable HTML and strip tags for plain text
    summary_html = doc.summary()
    text = _strip_html(summary_html)

    return {
        "title": title,
        "text": text,
        "url": url,
        "word_count": len(text.split()),
    }


def _strip_html(html: str) -> str:
    """Simple HTML tag stripping. Preserves paragraph structure."""
    import re

    # Replace block elements with newlines
    text = re.sub(r"<(?:p|br|div|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Clean up whitespace
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()
