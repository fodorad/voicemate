"""Text extraction from HTML pages and PDF files."""

from __future__ import annotations

import io


def html_to_text(html: str, url: str | None = None) -> str:
    """Main-content text of an HTML page (boilerplate, menus and ads removed).

    Returns:
        Extracted text, or an empty string when nothing readable was found.
    """
    import trafilatura

    return trafilatura.extract(html, url=url, include_comments=False, include_tables=True) or ""


def pdf_to_text(data: bytes) -> tuple[str, int]:
    """Text of a PDF.

    Returns:
        ``(text, page_count)``; pages are separated by blank lines.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip()), len(pages)
