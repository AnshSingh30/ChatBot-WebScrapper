#!/usr/bin/env python3
"""
=============================================================================
ASSIGNMENT: Website Chatbot using Google Gemini API
Target website: https://botpenguin.com/
Submission format: Single .py file (this file)
=============================================================================

STEP-BY-STEP PROCESS FOLLOWED
-----------------------------

Step 1 — Environment Setup
    • Created a Python virtual environment: python3 -m venv venv
    • Installed dependencies: requests, beautifulsoup4, python-dotenv
      (see requirements.txt)
    • Obtained a Google Gemini API key from Google AI Studio
      (https://aistudio.google.com/apikey)
    • Stored the key in a local .env file as GOOGLE_API_KEY=<your-key>

Step 2 — Extracting Website Data (Web Scraping)
    • Used the requests library to fetch HTML from the target URL
    • Parsed HTML with BeautifulSoup (bs4)
    • Removed non-content tags (script, style, nav, footer)
    • Extracted readable text from headings, paragraphs, and list items
    • Crawled internal pages on the same domain (BFS) and optionally
      seeded URLs from sitemap.xml for broader coverage

Step 3 — Processing & Structuring Data
    • Cleaned whitespace and normalized extracted text
    • Split content into overlapping word-based chunks (500 words, 50 overlap)
    • Labeled each chunk with its source page URL and title
    • Generated Gemini embeddings (gemini-embedding-001) for every chunk
    • Cached the index locally (.cache/) to avoid re-crawling on restart

Step 4 — Implementing the Chatbot (Gemini API)
    • Used Google Gemini REST API instead of OpenAI ChatGPT:
        - Embeddings: models/gemini-embedding-001:batchEmbedContents
        - Chat:       models/gemini-2.5-flash-lite:generateContent
    • For each user question:
        a) Embed the question (RETRIEVAL_QUERY task type)
        b) Retrieve top-k most similar chunks via cosine similarity
        c) Send retrieved context + question to Gemini with a system prompt
           that restricts answers to the provided context only

Step 5 — Console Demonstration
    • Run: python chatbot.py
    • Enter any website URL when prompted (e.g. https://botpenguin.com/)
    • The script crawls all discoverable pages on that site, then enters a REPL
    • Ask questions about the website; the bot answers from scraped content
    • Type "exit" to quit

HOW TO RUN
----------
    1. pip install -r requirements.txt
    2. Create .env with: GOOGLE_API_KEY=your_gemini_api_key
    3. python chatbot.py

RATE LIMITS (HTTP 429)
----------------------
    If you hit Gemini quota errors, try:
    • USE_KEYWORD_MODE=true   — skip embeddings, use keyword search + Gemini chat
    • MAX_CRAWL_PAGES=10      — crawl fewer pages
    • EMBED_BATCH_SIZE=1      — one embedding request at a time
    • EMBED_DELAY_SECONDS=10  — wait longer between embedding batches
    • REFRESH_CACHE=false     — reuse a previously saved index in .cache/

=============================================================================
Console-based Retrieval-Augmented Generation (RAG) chatbot.

Crawls a website starting from a configurable URL, chunks the cleaned text,
embeds chunks with Google Gemini, retrieves relevant passages for each
question via cosine similarity, and answers using Gemini strictly from
that context.
=============================================================================
"""

from __future__ import annotations

import math
import os
import re
import sys
import textwrap
import time
import warnings
import json
import hashlib
from collections import deque
from typing import Any, Deque, Dict, List, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Gemini REST API base URL (Google AI Studio).
GEMINI_API_BASE: str = "https://generativelanguage.googleapis.com/v1beta"

# Word-based chunking parameters (per assessment requirements).
CHUNK_SIZE_WORDS: int = 500
CHUNK_OVERLAP_WORDS: int = 50

# Number of context chunks to inject into the prompt.
TOP_K_CHUNKS: int = 6
TOP_K_BROAD: int = 10

# Keywords that indicate a broad/list-style question needing more context.
BROAD_QUERY_KEYWORDS: frozenset = frozenset({
    "pricing", "price", "prices", "plan", "plans", "cost", "costs",
    "fee", "fees", "subscription", "package", "packages", "tier", "tiers",
    "feature", "features", "service", "services", "product", "products",
})

# Crawl these paths early so pricing/product pages are indexed first.
HIGH_PRIORITY_PATH_KEYWORDS: Tuple[str, ...] = (
    "/pricing", "/plans", "/price", "/packages", "/subscription",
)

# Multi-page crawl limits (override via .env). Crawl continues until the
# queue is empty or max_pages is reached — whichever comes first.
DEFAULT_MAX_CRAWL_PAGES: int = 30
DEFAULT_MAX_CRAWL_DEPTH: int = 5
DEFAULT_CRAWL_DELAY_SECONDS: float = 0.3

# File extensions and URL patterns to skip during crawling.
SKIP_EXTENSIONS: Tuple[str, ...] = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".rar", ".mp4", ".mp3", ".css", ".js", ".xml", ".json",
)
SKIP_URL_KEYWORDS: Tuple[str, ...] = (
    "/login", "/signin", "/signup", "/register", "/cart", "/checkout",
    "/account", "/wp-admin", "/feed", "/rss",
)
# Paths crawled only after higher-value product/marketing pages are indexed.
LOW_PRIORITY_PATH_KEYWORDS: Tuple[str, ...] = (
    "/blogs/", "/blog/", "/tag/", "/author/", "/category/",
)

# Gemini model identifiers.
EMBEDDING_MODEL: str = "gemini-embedding-001"
CHAT_MODEL: str = "gemini-2.5-flash-lite"

# Network timeouts (seconds) to avoid hanging indefinitely.
HTTP_TIMEOUT_SECONDS: int = 30
GEMINI_TIMEOUT_SECONDS: int = 60

# Maximum retries for transient Gemini / network failures.
MAX_RETRIES: int = 6
RETRY_BACKOFF_SECONDS: float = 2.0

# Longer exponential backoff for HTTP 429 (quota / rate-limit).
RATE_LIMIT_BACKOFF_SECONDS: float = 15.0
MAX_RATE_LIMIT_BACKOFF_SECONDS: float = 120.0

# Embedding batch size and delay — conservative defaults for free-tier keys.
EMBED_BATCH_SIZE: int = 2
EMBED_DELAY_SECONDS: float = 5.0

# Local cache directory for crawled + embedded index (speeds up restarts).
CACHE_DIR: str = ".cache"
CACHE_FILENAME: str = "knowledge_index.json"

# HTTP status codes that are safe to retry (rate limits / server errors).
RETRYABLE_STATUS_CODES: Tuple[int, ...] = (429, 500, 502, 503, 504)

# System prompt enforces grounded answers only from supplied context.
SYSTEM_PROMPT: str = textwrap.dedent(
    """
    You are a helpful assistant that answers questions using ONLY the
    context provided below. Follow these rules strictly:

    1. Base every answer solely on the supplied context passages.
    2. Answer directly with whatever relevant facts appear in the context.
       If the context partially answers the question, share those facts
       clearly (use bullet points or short paragraphs as appropriate).
       Do NOT say "I don't have enough information" or mention missing,
       incomplete, or comprehensive data when some relevant facts are present.
       Only say you cannot answer when the context contains zero relevant
       facts for the question.
    3. Do not invent facts, speculate, or use outside knowledge.
    4. Keep answers concise, accurate, and focused on what the user asked.
       Do not add disclaimers, meta-commentary, or apologies.
    """
).strip()


# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------


def load_configuration() -> str:
    """
    Load environment variables from a local .env file.

    Returns:
        The Google Gemini API key.

    Raises:
        SystemExit: If the API key is missing or empty.
    """
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key:
        print(
            "Error: GOOGLE_API_KEY is not set. "
            "Create a .env file with GOOGLE_API_KEY=... and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    return api_key


def normalize_user_url(raw: str) -> str:
    """
    Normalize a user-entered URL into a canonical absolute form.

    Adds https:// when the scheme is omitted and strips trailing slashes
    via normalize_url for consistent deduplication during crawling.
    """
    url = raw.strip()
    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    parsed = urlparse(url)
    if not parsed.netloc:
        return ""

    return normalize_url(url, url)


def prompt_for_website_url() -> str:
    """
    Ask the user for a website URL and validate basic reachability.

    Returns:
        A normalized absolute URL to crawl.

    Raises:
        SystemExit: If the user cancels or chooses not to retry after failure.
    """
    print("=" * 60)
    print("  Website Chatbot — powered by Google Gemini")
    print("=" * 60)
    print("\nEnter the URL of the website you want to learn about.")
    print("The chatbot will crawl all pages on that site before answering.")
    print("Example: https://botpenguin.com/\n")

    while True:
        try:
            raw = input("Website URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not raw:
            print("Please enter a URL.\n")
            continue

        url = normalize_user_url(raw)
        if not url:
            print("Invalid URL. Include a domain such as example.com\n")
            continue

        print(f"\nVerifying {url} ...")
        try:
            fetch_html(url)
        except requests.RequestException as exc:
            print(f"Could not reach that website: {exc}")
            try:
                retry = input("Try a different URL? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                sys.exit(0)
            if retry != "y":
                sys.exit(1)
            print()
            continue

        print("Website reachable. Crawling all discoverable pages...\n")
        return url


def get_embed_settings() -> Tuple[int, float]:
    """Read embedding batch size and inter-batch delay from environment."""
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", str(EMBED_BATCH_SIZE)))
    delay = float(os.getenv("EMBED_DELAY_SECONDS", str(EMBED_DELAY_SECONDS)))
    return max(batch_size, 1), max(delay, 0.0)


def get_crawl_settings() -> Tuple[int, int, float]:
    """
    Read crawl tuning parameters from environment variables.

    Returns:
        Tuple of (max_pages, max_depth, delay_seconds).
    """
    max_pages = int(os.getenv("MAX_CRAWL_PAGES", str(DEFAULT_MAX_CRAWL_PAGES)))
    max_depth = int(os.getenv("MAX_CRAWL_DEPTH", str(DEFAULT_MAX_CRAWL_DEPTH)))
    delay = float(os.getenv("CRAWL_DELAY_SECONDS", str(DEFAULT_CRAWL_DELAY_SECONDS)))
    return max(max_pages, 1), max(max_depth, 0), max(delay, 0.0)


# ---------------------------------------------------------------------------
# Web scraping & text extraction
# ---------------------------------------------------------------------------


def fetch_html(url: str) -> str:
    """
    Download raw HTML from the given URL using the requests library.

    Args:
        url: Fully qualified HTTP/HTTPS address of the page to scrape.

    Returns:
        The response body as a string.

    Raises:
        requests.RequestException: On network, timeout, or HTTP errors.
    """
    headers = {
        # Some sites block generic bots; a browser-like UA improves success rate.
        "User-Agent": (
            "Mozilla/5.0 (compatible; RAG-Chatbot/1.0; "
            "+https://example.com/bot)"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    # Raise for 4xx/5xx so callers can handle failures explicitly.
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def get_root_domain(url: str) -> str:
    """Return the registrable domain (netloc) for same-site crawl checks."""
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def normalize_url(url: str, base_url: str) -> str:
    """
    Resolve relative URLs and strip fragments/query tracking noise.

    Args:
        url: Raw href or absolute URL discovered on a page.
        base_url: Page URL used to resolve relative paths.

    Returns:
        Canonical absolute URL string.
    """
    absolute = urljoin(base_url, url.strip())
    parsed = urlparse(absolute)

    # Drop fragments (#section) and common tracking params for deduplication.
    clean_path = parsed.path.rstrip("/") or "/"
    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        clean_path,
        "",  # params
        "",  # query — omit to avoid duplicate content URLs
        "",  # fragment
    ))
    return normalized


def is_same_domain(url: str, root_domain: str) -> bool:
    """True when url belongs to the same site as root_domain."""
    return get_root_domain(url) == root_domain


def should_skip_url(url: str) -> bool:
    """Filter out non-HTML assets and low-value navigation targets."""
    parsed = urlparse(url)
    path_lower = parsed.path.lower()

    if parsed.scheme not in ("http", "https"):
        return True

    if any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True

    if any(keyword in path_lower for keyword in SKIP_URL_KEYWORDS):
        return True

    return False


def is_high_priority_url(url: str) -> bool:
    """Pricing and product pages are crawled before generic content."""
    path_lower = urlparse(url).path.lower()
    return any(keyword in path_lower for keyword in HIGH_PRIORITY_PATH_KEYWORDS)


def is_low_priority_url(url: str) -> bool:
    """Blog and archive pages are indexed after core product/marketing pages."""
    path_lower = urlparse(url).path.lower()
    return any(keyword in path_lower for keyword in LOW_PRIORITY_PATH_KEYWORDS)


def enqueue_crawl_url(
    queue: Deque[Tuple[str, int]],
    url: str,
    depth: int,
    visited: Set[str],
) -> None:
    """Add a URL to the crawl queue; pricing pages and marketing pages first."""
    if url in visited:
        return
    entry = (url, depth)
    if is_low_priority_url(url):
        queue.append(entry)
    else:
        queue.appendleft(entry)


def discover_links(html: str, base_url: str) -> List[str]:
    """
    Extract crawlable same-page links from anchor tags.

    Args:
        html: Raw HTML of the current page.
        base_url: URL of the page being parsed (for relative link resolution).

    Returns:
        List of normalized absolute URLs found in href attributes.
    """
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        links.append(normalize_url(href, base_url))

    return links


def fetch_sitemap_urls(start_url: str, max_urls: int = 200) -> List[str]:
    """
    Attempt to discover page URLs from sitemap.xml or sitemap_index.xml.

    Many sites expose their full page list here, which is faster and more
    complete than relying on link-following alone.

    Args:
        start_url: Seed URL used to infer the site origin.
        max_urls: Cap on sitemap URLs to avoid unbounded memory use.

    Returns:
        List of absolute page URLs found in sitemaps (may be empty).
    """
    parsed = urlparse(start_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    sitemap_candidates = [
        urljoin(origin, "/sitemap.xml"),
        urljoin(origin, "/sitemap_index.xml"),
    ]

    discovered: List[str] = []
    seen_sitemaps: Set[str] = set()
    pending_sitemaps: Deque[str] = deque(sitemap_candidates)

    while pending_sitemaps and len(discovered) < max_urls:
        sitemap_url = pending_sitemaps.popleft()
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)

        try:
            response = requests.get(
                sitemap_url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-Chatbot/1.0)"},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            if not response.ok:
                continue

            # Sitemaps are XML; html.parser still finds <loc> tags reliably.
            soup = BeautifulSoup(response.text, "html.parser")
            for loc_tag in soup.find_all("loc"):
                loc_text = loc_tag.get_text(strip=True)
                if not loc_text:
                    continue

                if loc_text.endswith(".xml") and "sitemap" in loc_text.lower():
                    pending_sitemaps.append(loc_text)
                else:
                    discovered.append(normalize_url(loc_text, origin))

                if len(discovered) >= max_urls:
                    break

        except requests.RequestException:
            continue

    return discovered


def crawl_website(
    start_url: str,
    max_pages: int,
    max_depth: int,
    crawl_delay: float,
) -> Dict[str, str]:
    """
    Breadth-first crawl of all internal pages on a website.

    Seeds the queue from the start URL, sitemap.xml (when available), and
    links discovered on each fetched page. Stays within the same domain and
    respects max_pages / max_depth limits.

    Args:
        start_url: Entry-point URL (any page on the target site).
        max_pages: Maximum number of pages to fetch.
        max_depth: Maximum link depth from the start URL.
        crawl_delay: Seconds to wait between requests (politeness).

    Returns:
        Mapping of {page_url: raw_html} for every successfully fetched page.
    """
    root_domain = get_root_domain(start_url)
    start_normalized = normalize_url(start_url, start_url)

    visited: Set[str] = set()
    queue: Deque[Tuple[str, int]] = deque([(start_normalized, 0)])
    pages: Dict[str, str] = {}

    # Seed queue from sitemap; product pages are enqueued ahead of blog posts.
    for sitemap_url in fetch_sitemap_urls(start_url, max_urls=max_pages * 4):
        if (
            is_same_domain(sitemap_url, root_domain)
            and not should_skip_url(sitemap_url)
        ):
            enqueue_crawl_url(queue, sitemap_url, 1, visited)

    # Seed common pricing paths so plan details are indexed early.
    parsed_start = urlparse(start_normalized)
    origin = f"{parsed_start.scheme}://{parsed_start.netloc}"
    for path in HIGH_PRIORITY_PATH_KEYWORDS:
        pricing_url = normalize_url(urljoin(origin, path), origin)
        if is_same_domain(pricing_url, root_domain) and not should_skip_url(pricing_url):
            enqueue_crawl_url(queue, pricing_url, 1, visited)

    print(
        f"Crawling site (max {max_pages} pages, depth {max_depth}) "
        f"starting at: {start_normalized}"
    )

    while queue and len(pages) < max_pages:
        url, depth = queue.popleft()

        if url in visited:
            continue
        visited.add(url)

        if not is_same_domain(url, root_domain) or should_skip_url(url):
            continue

        try:
            html = fetch_html(url)
            pages[url] = html
            print(f"  [{len(pages)}/{max_pages}] {url}")
        except requests.RequestException as exc:
            print(f"  Skipped {url}: {exc}", file=sys.stderr)
            continue

        if depth < max_depth:
            for link in discover_links(html, url):
                enqueue_crawl_url(queue, link, depth + 1, visited)

        if crawl_delay > 0:
            time.sleep(crawl_delay)

    print(f"Crawl complete: {len(pages)} page(s) indexed.")
    return pages


def extract_page_content(html: str) -> Tuple[str, str]:
    """
    Parse HTML and extract human-readable semantic text with structure.

    Removes non-content elements (<script>, <style>, <nav>, <footer>),
    preserves headings for context, and returns the page title alongside
    cleaned prose suitable for chunking and embedding.

    Args:
        html: Raw HTML string.

    Returns:
        Tuple of (page_title, cleaned_text).
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else ""

    h1_tag = soup.find("h1")
    if h1_tag and not page_title:
        page_title = h1_tag.get_text(strip=True)

    # Decompose removes tags entirely from the parse tree (not just hide them).
    for tag_name in ("script", "style", "nav", "footer"):
        for element in soup.find_all(tag_name):
            element.decompose()

    # Prefer main/article when present; otherwise fall back to full body text.
    content_root = (
        soup.find("main")
        or soup.find("article")
        or soup.find("body")
        or soup
    )

    text_parts: List[str] = []

    # Walk semantic elements so headings and list/table content stay structured.
    for element in content_root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th"]
    ):
        text = element.get_text(separator=" ", strip=True)
        if not text:
            continue
        if element.name.startswith("h"):
            text_parts.append(f"[{element.name.upper()}] {text}")
        else:
            text_parts.append(text)

    if text_parts:
        raw_text = " ".join(text_parts)
    else:
        raw_text = content_root.get_text(separator=" ", strip=True)

    cleaned_text = " ".join(raw_text.split())
    return page_title, cleaned_text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Example: 500-word chunks with 50-word overlap means each chunk (after
    the first) shares 50 words with its predecessor, improving retrieval
    recall across chunk boundaries.

    Args:
        text: Full document plain text.
        chunk_size: Target number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        A list of non-empty text chunks.

    Raises:
        ValueError: If chunk_size or overlap are invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    stride = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += stride

    return chunks


# ---------------------------------------------------------------------------
# Gemini API helpers (via requests)
# ---------------------------------------------------------------------------


def _retry_sleep_seconds(status_code: int | None, attempt: int) -> float:
    """Compute wait time before retry; 429 uses longer exponential backoff."""
    if status_code == 429:
        delay = RATE_LIMIT_BACKOFF_SECONDS * (2 ** (attempt - 1))
        return min(delay, MAX_RATE_LIMIT_BACKOFF_SECONDS)
    return RETRY_BACKOFF_SECONDS * attempt


def _gemini_post(
    api_key: str,
    endpoint: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    POST to a Gemini REST endpoint with retry logic for transient failures.

    Args:
        api_key: Google AI Studio / Gemini API key.
        endpoint: Path segment after the API base (e.g. models/...:embedContent).
        payload: JSON request body.

    Returns:
        Parsed JSON response body.

    Raises:
        requests.RequestException: After retries are exhausted.
        RuntimeError: If the API returns a non-retryable error payload.
    """
    url = f"{GEMINI_API_BASE}/{endpoint}"
    params = {"key": api_key}
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                params=params,
                json=payload,
                timeout=GEMINI_TIMEOUT_SECONDS,
            )

            if response.status_code in RETRYABLE_STATUS_CODES:
                last_error = requests.HTTPError(
                    f"{response.status_code} {response.reason}: {response.text}",
                    response=response,
                )
                if attempt == MAX_RETRIES:
                    break
                sleep_seconds = _retry_sleep_seconds(response.status_code, attempt)
                print(
                    f"Gemini request failed (HTTP {response.status_code}). "
                    f"Retrying in {sleep_seconds:.1f}s "
                    f"({attempt}/{MAX_RETRIES})...",
                    file=sys.stderr,
                )
                time.sleep(sleep_seconds)
                continue

            if not response.ok:
                raise RuntimeError(
                    f"Gemini API error ({response.status_code}): {response.text}"
                )

            return response.json()

        except requests.Timeout as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            sleep_seconds = _retry_sleep_seconds(None, attempt)
            print(
                f"Gemini request timed out. Retrying in {sleep_seconds:.1f}s "
                f"({attempt}/{MAX_RETRIES})...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

        except requests.ConnectionError as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            sleep_seconds = _retry_sleep_seconds(None, attempt)
            print(
                f"Gemini connection failed. Retrying in {sleep_seconds:.1f}s "
                f"({attempt}/{MAX_RETRIES})...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Embeddings & retrieval
# ---------------------------------------------------------------------------


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """
    Compute cosine similarity between two vectors using pure Python math.

    Cosine similarity = (A · B) / (||A|| * ||B||), ranging from -1 to 1.
    Higher values indicate greater semantic alignment.

    Args:
        vector_a: First embedding vector.
        vector_b: Second embedding vector (must share length with vector_a).

    Returns:
        Similarity score in [-1.0, 1.0], or 0.0 for zero-magnitude vectors.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensionality.")

    dot_product = 0.0
    magnitude_a = 0.0
    magnitude_b = 0.0

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b
        magnitude_a += a * a
        magnitude_b += b * b

    magnitude_a = math.sqrt(magnitude_a)
    magnitude_b = math.sqrt(magnitude_b)

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def embed_texts(
    api_key: str,
    texts: Sequence[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> List[List[float]]:
    """
    Convert strings into embedding vectors via the Gemini REST API.

    Uses batchEmbedContents where possible to reduce API calls and adds a
    short delay between batches to avoid rate-limit errors on large sites.

    Args:
        api_key: Google Gemini API key.
        texts: Iterable of strings to embed.
        task_type: Gemini embedding task hint (document vs. query).

    Returns:
        List of embedding vectors aligned with the input order.
    """
    if not texts:
        return []

    all_embeddings: List[List[float]] = []
    text_list = list(texts)
    total = len(text_list)
    batch_size, inter_batch_delay = get_embed_settings()

    for batch_start in range(0, total, batch_size):
        batch = text_list[batch_start: batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        batch_total = math.ceil(total / batch_size)
        print(
            f"  Embedding batch {batch_num}/{batch_total} "
            f"({len(all_embeddings)}/{total} chunks)...",
            file=sys.stderr,
        )

        payload = {
            "requests": [
                {
                    "model": f"models/{EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                }
                for text in batch
            ]
        }
        endpoint = f"models/{EMBEDDING_MODEL}:batchEmbedContents"
        response = _gemini_post(api_key, endpoint, payload)

        embeddings_block = response.get("embeddings", [])
        if len(embeddings_block) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} embeddings, got {len(embeddings_block)}"
            )

        for item in embeddings_block:
            values = item.get("values")
            if not values:
                raise RuntimeError(f"Gemini returned empty embedding: {item}")
            all_embeddings.append(values)

        if batch_start + batch_size < total and inter_batch_delay > 0:
            time.sleep(inter_batch_delay)

    return all_embeddings


def _cache_path(source_url: str, max_pages: int, max_depth: int) -> str:
    """Build a deterministic cache file path for a crawl configuration."""
    key = f"{source_url}|{max_pages}|{max_depth}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{CACHE_FILENAME}.{digest}")


def load_cached_index(
    source_url: str,
    max_pages: int,
    max_depth: int,
) -> Tuple[List[str], List[List[float]], str] | None:
    """
    Load a previously saved index from disk, if one exists.

    Returns:
        Tuple of (chunks, embeddings, mode) where mode is "embedding" or
        "keyword", or None if no valid cache exists.
    """
    path = _cache_path(source_url, max_pages, max_depth)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        chunks = data["chunks"]
        embeddings = data.get("embeddings", [])
        mode = data.get("mode", "embedding" if embeddings else "keyword")
        if mode == "embedding" and len(chunks) != len(embeddings):
            return None
        print(f"Loaded cached index ({len(chunks)} chunks, {mode} mode) from {path}")
        return chunks, embeddings, mode
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def save_cached_index(
    source_url: str,
    max_pages: int,
    max_depth: int,
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    mode: str = "embedding",
) -> None:
    """Persist the index to disk so subsequent runs skip crawl + embed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(source_url, max_pages, max_depth)
    payload = {
        "chunks": list(chunks),
        "embeddings": [list(e) for e in embeddings],
        "mode": mode,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"Saved index cache to {path}")


def retrieve_top_k_chunks(
    query_embedding: Sequence[float],
    chunk_embeddings: Sequence[Sequence[float]],
    chunks: Sequence[str],
    question: str = "",
    k: int = TOP_K_CHUNKS,
) -> List[str]:
    """
    Rank chunks by cosine similarity to the query and return the top k.

    Args:
        query_embedding: Vector representation of the user's question.
        chunk_embeddings: Precomputed vectors for each text chunk.
        chunks: Original chunk strings (same order as chunk_embeddings).
        k: Number of highest-scoring chunks to return.

    Returns:
        Up to k chunk strings, ordered from most to least similar.
    """
    if not chunks or not chunk_embeddings:
        return []

    scored: List[Tuple[float, str]] = []

    for chunk, embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, embedding)
        scored.append((score, chunk))

    # Pull extra candidates, then re-rank with topic bonuses for broad queries.
    pool_size = min(len(scored), max(k * 3, k))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return _finalize_ranked_chunks(question, scored[:pool_size], k)


def _tokenize(text: str) -> Set[str]:
    """Lowercase alphanumeric tokens for keyword retrieval."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def get_retrieval_k(question: str) -> int:
    """Return more chunks for broad questions (pricing, plans, features)."""
    query_terms = _tokenize(question)
    if query_terms & BROAD_QUERY_KEYWORDS:
        return int(os.getenv("TOP_K_BROAD", str(TOP_K_BROAD)))
    return int(os.getenv("TOP_K_CHUNKS", str(TOP_K_CHUNKS)))


def _chunk_relevance_bonus(question: str, chunk: str) -> float:
    """Boost chunks that likely contain facts the user asked for."""
    bonus = 0.0
    chunk_lower = chunk.lower()
    query_terms = _tokenize(question)

    if query_terms & BROAD_QUERY_KEYWORDS:
        if re.search(r"\$\d", chunk):
            bonus += 3.0
        if re.search(r"\bfree\b", chunk_lower):
            bonus += 1.5
        for signal in ("pricing", "plan", "per month", "/month", "/year"):
            if signal in chunk_lower:
                bonus += 1.0
        if "pricing" in chunk_lower or "/pricing" in chunk_lower:
            bonus += 2.0

    return bonus


def _finalize_ranked_chunks(
    question: str,
    scored: List[Tuple[float, str]],
    k: int,
) -> List[str]:
    """Apply topic bonuses and return the top-k chunks."""
    adjusted = [
        (score + _chunk_relevance_bonus(question, chunk), chunk)
        for score, chunk in scored
    ]
    adjusted.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in adjusted[:k]]


def keyword_retrieve_top_k(
    question: str,
    chunks: Sequence[str],
    k: int | None = None,
) -> List[str]:
    """
    Rank chunks by keyword overlap when embeddings are unavailable.

    Used as a free-tier fallback when Gemini embedding quota is exhausted.
    """
    if not chunks:
        return []

    if k is None:
        k = get_retrieval_k(question)

    query_terms = _tokenize(question)
    if not query_terms:
        return list(chunks[:k])

    question_lower = question.lower()
    scored: List[Tuple[float, str]] = []

    for chunk in chunks:
        chunk_terms = _tokenize(chunk)
        overlap = len(query_terms & chunk_terms)
        phrase_bonus = 2.0 if question_lower in chunk.lower() else 0.0
        scored.append((overlap + phrase_bonus, chunk))

    return _finalize_ranked_chunks(question, scored, k)


def retrieve_context_chunks(
    api_key: str,
    question: str,
    chunks: Sequence[str],
    chunk_embeddings: Sequence[Sequence[float]],
    retrieval_mode: str,
    k: int | None = None,
) -> List[str]:
    """
    Retrieve relevant chunks using embeddings or keyword fallback.

    Falls back to keyword search if embedding the query fails at runtime.
    """
    if k is None:
        k = get_retrieval_k(question)

    if retrieval_mode == "keyword" or not chunk_embeddings:
        return keyword_retrieve_top_k(question, chunks, k=k)

    try:
        query_embeddings = embed_texts(
            api_key,
            [question],
            task_type="RETRIEVAL_QUERY",
        )
        return retrieve_top_k_chunks(
            query_embeddings[0],
            chunk_embeddings,
            chunks,
            question=question,
            k=k,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"  Embedding query failed ({exc}). Using keyword search.",
            file=sys.stderr,
        )
        return keyword_retrieve_top_k(question, chunks, k=k)


# ---------------------------------------------------------------------------
# Generative chat
# ---------------------------------------------------------------------------


def build_user_prompt(question: str, context_chunks: Sequence[str]) -> str:
    """
    Assemble the user message with labeled context passages.

    Args:
        question: Natural-language query from the console user.
        context_chunks: Retrieved text segments to ground the answer.

    Returns:
        A formatted prompt string for the Gemini generateContent API.
    """
    if context_chunks:
        context_block = "\n\n".join(
            f"[Context {index + 1}]\n{chunk}"
            for index, chunk in enumerate(context_chunks)
        )
    else:
        context_block = "(No relevant context was retrieved.)"

    return textwrap.dedent(
        f"""
        Context:
        {context_block}

        Question:
        {question}

        Answer the question directly using only the context above.
        Give only the relevant facts the user asked for — no disclaimers
        about missing or incomplete information when partial facts are present.
        """
    ).strip()


def generate_answer(
    api_key: str,
    question: str,
    context_chunks: Sequence[str],
) -> str:
    """
    Call Gemini generateContent to produce a grounded answer.

    Args:
        api_key: Google Gemini API key.
        question: User's question from the console loop.
        context_chunks: Top-k retrieved passages injected as context.

    Returns:
        Assistant reply text, or a user-friendly error message on failure.
    """
    user_prompt = build_user_prompt(question, context_chunks)
    endpoint = f"models/{CHAT_MODEL}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    try:
        response = _gemini_post(api_key, endpoint, payload)
        candidates = response.get("candidates", [])

        if not candidates:
            return (
                "I received an empty response from the language model. "
                "Please try rephrasing your question."
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        message = "".join(part.get("text", "") for part in parts).strip()

        if message:
            return message

        return (
            "I received an empty response from the language model. "
            "Please try rephrasing your question."
        )

    except Exception as exc:  # noqa: BLE001 — keep console app alive
        return (
            "Sorry, an unexpected error occurred while generating a "
            f"response: {exc}"
        )


# ---------------------------------------------------------------------------
# Index building (one-time startup)
# ---------------------------------------------------------------------------


def build_knowledge_index(
    api_key: str,
    source_url: str,
) -> Tuple[List[str], List[List[float]], str]:
    """
    Crawl the website, clean each page, chunk, and embed all content.

    Args:
        api_key: Gemini API key used for batch embedding.
        source_url: Starting URL; all same-domain pages are discovered from here.

    Returns:
        Tuple of (text_chunks, chunk_embeddings, retrieval_mode).
        retrieval_mode is "embedding" or "keyword" when embeddings fail.

    Raises:
        SystemExit: If crawling or indexing fails irrecoverably.
    """
    max_pages, max_depth, crawl_delay = get_crawl_settings()

    refresh_cache = os.getenv("REFRESH_CACHE", "").strip().lower() in ("1", "true", "yes")
    if not refresh_cache:
        cached = load_cached_index(source_url, max_pages, max_depth)
        if cached is not None:
            return cached

    try:
        pages = crawl_website(
            source_url,
            max_pages=max_pages,
            max_depth=max_depth,
            crawl_delay=crawl_delay,
        )
    except requests.RequestException as exc:
        print(f"Error: Website crawl failed — {exc}", file=sys.stderr)
        sys.exit(1)

    if not pages:
        print("Error: No pages could be fetched from the website.", file=sys.stderr)
        sys.exit(1)

    all_chunks: List[str] = []
    total_words = 0

    for page_url, html in pages.items():
        page_title, cleaned_text = extract_page_content(html)
        if not cleaned_text.strip():
            continue

        # Label every chunk with its source so retrieval can surface pricing etc.
        page_header = f"[Page: {page_url}]"
        if page_title:
            page_header += f"\n[Title: {page_title}]"

        labeled_text = f"{page_header}\n{cleaned_text}"
        page_chunks = chunk_text(labeled_text)
        all_chunks.extend(page_chunks)
        total_words += len(cleaned_text.split())

    if not all_chunks:
        print(
            "Error: No readable text could be extracted from any crawled page.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Indexed {len(pages)} page(s), {total_words} words, "
        f"{len(all_chunks)} chunk(s)."
    )

    force_keyword = os.getenv("USE_KEYWORD_MODE", "").strip().lower() in (
        "1", "true", "yes"
    )
    if force_keyword:
        print("USE_KEYWORD_MODE enabled — skipping embeddings.\n")
        save_cached_index(
            source_url, max_pages, max_depth, all_chunks, [], mode="keyword"
        )
        return all_chunks, [], "keyword"

    print("Generating embeddings (this may take a moment)...")

    retrieval_mode = "embedding"
    embeddings: List[List[float]] = []
    try:
        embeddings = embed_texts(
            api_key,
            all_chunks,
            task_type="RETRIEVAL_DOCUMENT",
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"\nWarning: Embedding generation failed — {exc}",
            file=sys.stderr,
        )
        print(
            "Switching to keyword search mode (no embeddings). "
            "Answers will still use Gemini chat with scraped text.\n",
            file=sys.stderr,
        )
        retrieval_mode = "keyword"

    save_cached_index(
        source_url,
        max_pages,
        max_depth,
        all_chunks,
        embeddings,
        mode=retrieval_mode,
    )
    return all_chunks, embeddings, retrieval_mode


# ---------------------------------------------------------------------------
# Console interaction loop
# ---------------------------------------------------------------------------


def run_chat_loop(
    api_key: str,
    source_url: str,
    chunks: List[str],
    chunk_embeddings: List[List[float]],
    retrieval_mode: str,
) -> None:
    """
    Interactive REPL: accept questions, retrieve context, print answers.

    Type 'exit' (case-insensitive) to quit gracefully.
    """
    mode_label = "semantic (embeddings)" if retrieval_mode == "embedding" else "keyword"
    print(f"\nChatbot is ready for: {source_url}")
    print(f"Search mode: {mode_label}")
    print("Ask any question about the website content.")
    print('Type "exit" to quit.\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # --- Retrieval phase ------------------------------------------------
        top_chunks = retrieve_context_chunks(
            api_key,
            user_input,
            chunks,
            chunk_embeddings,
            retrieval_mode,
        )

        # --- Generation phase -----------------------------------------------
        answer = generate_answer(api_key, user_input, top_chunks)
        print(f"Assistant: {answer}\n")


def main() -> None:
    """Entry point: load API key, ask for URL, crawl site, start chat loop."""
    api_key = load_configuration()
    source_url = prompt_for_website_url()

    chunks, chunk_embeddings, retrieval_mode = build_knowledge_index(
        api_key, source_url
    )
    run_chat_loop(api_key, source_url, chunks, chunk_embeddings, retrieval_mode)


if __name__ == "__main__":
    main()
