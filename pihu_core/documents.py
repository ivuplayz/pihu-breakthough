"""Document analysis and lightweight In-Context RAG parsing for Pihu-BreakThough.

Stage 9 implements serverless-friendly document text extraction (PDF, TXT, MD, CSV, JSON)
using pypdf/PyPDF2 with a 50,000 character safety threshold for prompt context injection.
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Safe import for PDF extraction libraries
try:
    import pypdf as pdf_lib
    PDF_LIBRARY = "pypdf"
except ImportError:
    try:
        import PyPDF2 as pdf_lib
        PDF_LIBRARY = "PyPDF2"
    except ImportError:
        pdf_lib = None
        PDF_LIBRARY = None

MAX_DOCUMENT_CHARS = 50_000
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024  # 10 MB per document
MAX_DOCUMENTS_PER_REQUEST = 2

SUPPORTED_DOCUMENT_MIMES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "text/csv",
    "text/html",
    "application/json",
    "text/json",
}

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".htm",
    ".css",
    ".xml",
    ".yaml",
    ".yml",
    ".log",
    ".sql",
    ".env",
}


class DocumentParseError(RuntimeError):
    """Raised when document extraction fails due to unsupported or corrupt format."""
    pass


def normalize_text(text: str) -> str:
    """Normalize whitespace and newlines for clean context injection."""
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def parse_pdf(file_data: bytes, filename: str) -> str:
    """Extract text from PDF binary data using pypdf or PyPDF2.

    Args:
        file_data: Raw binary content of the PDF.
        filename: Name of the file being processed.

    Returns:
        Extracted text content.

    Raises:
        DocumentParseError: If PDF library is unavailable or file is corrupted/encrypted.
    """
    if pdf_lib is None:
        raise DocumentParseError(
            f"PDF extraction library (pypdf / PyPDF2) is not installed. Cannot parse '{filename}'."
        )

    try:
        reader = pdf_lib.PdfReader(io.BytesIO(file_data))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                raise DocumentParseError(
                    f"PDF document '{filename}' is password-protected or encrypted."
                )

        num_pages = len(reader.pages)
        if num_pages == 0:
            return f"[Empty PDF document: '{filename}' contains 0 pages.]"

        extracted_pages = []
        for idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                page_clean = page_text.strip()
                if page_clean:
                    extracted_pages.append(f"--- Page {idx} ---\n{page_clean}")
            except Exception as page_err:
                logger.warning("Failed to extract text from page %d of '%s': %s", idx, filename, page_err)

        if not extracted_pages:
            return f"[PDF '{filename}' contains {num_pages} page(s), but no selectable text was found (document may consist of scanned images).]"

        return "\n\n".join(extracted_pages)
    except DocumentParseError:
        raise
    except Exception as exc:
        logger.error("Failed to parse PDF '%s': %s", filename, exc)
        raise DocumentParseError(f"Failed to parse PDF document '{filename}': {exc}") from exc


def parse_text_file(file_data: bytes, filename: str) -> str:
    """Extract and decode text from plain text formats with encoding fallback.

    Args:
        file_data: Raw binary content of the text document.
        filename: Name of the file being processed.

    Returns:
        Decoded text string.

    Raises:
        DocumentParseError: If decoding fails across all attempted encodings.
    """
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return file_data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    raise DocumentParseError(f"Unable to decode text document '{filename}'. Unsupported encoding.")


def parse_document(file_data: bytes, filename: str = "document", mime_type: Optional[str] = None) -> str:
    """Extract and normalize text from a document payload with a 50,000 character limit.

    Args:
        file_data: Raw binary bytes of the document.
        filename: Original name of the document file.
        mime_type: Declared or detected MIME type of the document.

    Returns:
        Cleaned, normalized text content, truncated to 50,000 characters if necessary.

    Raises:
        DocumentParseError: If file is empty, corrupted, unsupported, or exceeds size limits.
    """
    if not file_data or len(file_data) == 0:
        raise DocumentParseError(f"Document '{filename}' is empty (0 bytes).")

    if len(file_data) > MAX_DOCUMENT_BYTES:
        raise DocumentParseError(
            f"Document '{filename}' exceeds maximum allowed size of {MAX_DOCUMENT_BYTES // (1024 * 1024)}MB."
        )

    clean_filename = (filename or "document").strip()
    _, ext = os.path.splitext(clean_filename.lower())
    clean_mime = (mime_type or "").strip().lower()

    # Determine parser based on MIME type and extension
    is_pdf = (clean_mime == "application/pdf") or (ext == ".pdf")
    is_plain_text = (
        clean_mime in SUPPORTED_DOCUMENT_MIMES
        or clean_mime.startswith("text/")
        or ext in TEXT_EXTENSIONS
    )

    if is_pdf:
        raw_text = parse_pdf(file_data, clean_filename)
    elif is_plain_text or not clean_mime or clean_mime == "application/octet-stream":
        try:
            raw_text = parse_text_file(file_data, clean_filename)
        except DocumentParseError:
            if not is_plain_text:
                raise DocumentParseError(
                    f"Unsupported document format '{clean_mime or ext}' for file '{clean_filename}'. "
                    f"Supported formats: PDF and plain text formats (.txt, .md, .csv, .json, .html)."
                )
            raise
    else:
        raise DocumentParseError(
            f"Unsupported document MIME type '{clean_mime}'. "
            f"Supported formats: application/pdf, text/plain, text/markdown, text/csv, application/json."
        )

    normalized = normalize_text(raw_text)
    if not normalized:
        normalized = f"[Document '{clean_filename}' contains no readable text.]"

    # Enforce 50,000 character limit
    if len(normalized) > MAX_DOCUMENT_CHARS:
        logger.info(
            "Document '%s' length (%d chars) exceeds limit of %d chars. Truncating.",
            clean_filename,
            len(normalized),
            MAX_DOCUMENT_CHARS,
        )
        normalized = (
            normalized[:MAX_DOCUMENT_CHARS].rstrip()
            + f"\n\n[Note: Document '{clean_filename}' truncated to first {MAX_DOCUMENT_CHARS} characters for context limit]"
        )

    return normalized
