import sys
import argparse
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
import PyPDF2
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from .database import upsert_chunks
from .embeddings import get_embedding_function

# ---------------------------------------------------------------------------
# Helpers for extracting raw text
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a PDF file using PyPDF2.
    Returns a single concatenated string.
    """
    reader = PyPDF2.PdfReader(str(pdf_path))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)


def extract_pages_from_pdf(pdf_path: Path) -> List[str]:
    """Extract PDF text page by page so retrieved chunks can cite their page."""
    reader = PyPDF2.PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def fetch_html(url: str) -> str:
    """Download a web page and return its HTML content.
    Simple GET with a short timeout; raises for non‑200.
    """
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text


def extract_text_from_web(url: str) -> str:
    """Strip HTML tags and return readable text.
    Uses BeautifulSoup's get_text with whitespace collapsing.
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts / styles
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _youtube_video_id(video_url: str) -> str:
    """Extract a YouTube video ID from a URL or an 11-character video ID."""
    parsed_url = urlparse(video_url)
    host = parsed_url.hostname or ""
    if host == "youtu.be" or host.endswith(".youtu.be"):
        video_id = parsed_url.path.strip("/").split("/", 1)[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed_url.path == "/watch":
            video_id = parse_qs(parsed_url.query).get("v", [""])[0]
        else:
            path_parts = parsed_url.path.strip("/").split("/")
            video_id = (
                path_parts[1]
                if len(path_parts) == 2
                and path_parts[0] in {"embed", "shorts", "live"}
                else ""
            )
    elif not host:
        video_id = video_url.strip()
    else:
        video_id = ""

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(
            "Enter a valid YouTube video URL or 11-character video ID."
        )
    return video_id


def extract_transcript_from_youtube(video_url: str) -> str:
    """Fetch English or Hindi YouTube captions and return plain transcript text."""
    video_id = _youtube_video_id(video_url)
    try:
        transcript = YouTubeTranscriptApi().fetch(
            video_id, languages=["en", "hi"]
        )
    except TranscriptsDisabled as exc:
        raise ValueError(
            f"YouTube captions are disabled for video {video_id}."
        ) from exc
    except NoTranscriptFound as exc:
        raise ValueError(
            f"No English or Hindi transcript is available for video {video_id}."
        ) from exc
    except VideoUnavailable as exc:
        raise ValueError(
            f"YouTube video {video_id} is unavailable."
        ) from exc

    text = "\n".join(snippet.text for snippet in transcript.snippets).strip()
    if not text:
        raise ValueError(f"The transcript for video {video_id} is empty.")
    return text

# ---------------------------------------------------------------------------
# Chunking & storage
# ---------------------------------------------------------------------------

def chunk_and_store(
    text: str,
    source: str,
    splitter: RecursiveCharacterTextSplitter,
    page_texts: Optional[List[str]] = None,
) -> List[int]:
    """Split ``text`` into chunks, embed them, and upsert into PostgreSQL.

    When ``page_texts`` is provided, keep each PDF page separate and attach its
    one-based page number to every chunk from that page.
    Returns a list of document IDs inserted.
    """
    if page_texts is None:
        docs = splitter.create_documents([text], metadatas=[{"source": source}])
    else:
        page_texts_with_content = [
            (page_number, page_text)
            for page_number, page_text in enumerate(page_texts, start=1)
            if page_text.strip()
        ]
        if not page_texts_with_content:
            raise ValueError(f"No extractable text found in {source}.")
        docs = splitter.create_documents(
            [page_text for _, page_text in page_texts_with_content],
            metadatas=[
                {"source": source, "page": page_number}
                for page_number, _ in page_texts_with_content
            ],
        )
    if not docs:
        raise ValueError(f"No extractable text found in {source}.")

    embed_fn = get_embedding_function()
    vectors = [embed_fn(doc.page_content) for doc in docs]
    return upsert_chunks(docs, source, vectors)

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ingestion pipeline for RAG chatbot")
    parser.add_argument("--pdf", type=Path, help="Path to a PDF file to ingest")
    parser.add_argument("--url", type=str, help="Web page URL to ingest")
    parser.add_argument("--youtube", type=str, help="YouTube video URL to ingest (captions required)")
    args = parser.parse_args(argv)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    if args.pdf:
        if not args.pdf.is_file():
            sys.exit(f"PDF not found: {args.pdf}")
        print(f" Extracting PDF: {args.pdf}")
        page_texts = extract_pages_from_pdf(args.pdf)
        ids = chunk_and_store(
            "\n".join(page_texts),
            source=str(args.pdf),
            splitter=splitter,
            page_texts=page_texts,
        )
        print(f"Stored {len(ids)} chunks from PDF")

    if args.url:
        print(f"🌐 Fetching web page: {args.url}")
        raw = extract_text_from_web(args.url)
        ids = chunk_and_store(raw, source=args.url, splitter=splitter)
        print(f" Stored {len(ids)} chunks from URL")

    if args.youtube:
        print(f"🎥 Downloading YouTube captions: {args.youtube}")
        raw = extract_transcript_from_youtube(args.youtube)
        ids = chunk_and_store(raw, source=args.youtube, splitter=splitter)
        print(f" Stored {len(ids)} chunks from YouTube video")

    if not any([args.pdf, args.url, args.youtube]):
        parser.print_help()

if __name__ == "__main__":
    main()
