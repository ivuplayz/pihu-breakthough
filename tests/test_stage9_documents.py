"""Unit and integration test suite for Stage 9: Document Analysis and In-Context RAG.

Tests:
- pihu_core/documents.py:
  - Text, Markdown, CSV, JSON extraction.
  - Mocked and in-memory PDF extraction.
  - Character limit enforcement (50,000 max).
  - Encrypted, corrupted, empty, and oversized document error handling.
- api/chat.py:
  - Acceptance and validation of the 'documents' payload list (max 2 docs).
  - Context injection into system_prompt before routing.
  - Database metadata persistence (storing filenames, not base64 content).
  - Document-first requests without message.
  - Streaming with document attachments.
"""

from __future__ import annotations

import base64
import io
import json
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask

from api.chat import chat_bp
from pihu_core.documents import (
    DocumentParseError,
    MAX_DOCUMENT_CHARS,
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENTS_PER_REQUEST,
    normalize_text,
    parse_document,
)


class TestDocumentParser(unittest.TestCase):
    """Unit test suite for pihu_core/documents.py."""

    def test_parse_plain_text(self) -> None:
        """Verify plain text files decode and normalize properly."""
        content = b"Line 1\r\nLine 2\r\n\r\n\r\n\r\nLine 3"
        result = parse_document(content, filename="sample.txt", mime_type="text/plain")
        self.assertIn("Line 1\nLine 2\n\nLine 3", result)

    def test_parse_markdown(self) -> None:
        """Verify Markdown files decode with preserved markdown formatting."""
        md_bytes = b"# Project Overview\n\n- Feature A\n- Feature B\n\n```python\nprint('hello')\n```"
        result = parse_document(md_bytes, filename="README.md", mime_type="text/markdown")
        self.assertIn("# Project Overview", result)
        self.assertIn("- Feature A", result)
        self.assertIn("```python", result)

    def test_parse_csv_and_json(self) -> None:
        """Verify tabular CSV and structured JSON files parse cleanly."""
        csv_bytes = b"id,name,role\n1,Alice,Engineer\n2,Bob,Designer"
        res_csv = parse_document(csv_bytes, filename="users.csv", mime_type="text/csv")
        self.assertIn("Alice,Engineer", res_csv)

        json_bytes = b'{"version": "1.0", "status": "active"}'
        res_json = parse_document(json_bytes, filename="config.json", mime_type="application/json")
        self.assertIn('"version": "1.0"', res_json)

    def test_parse_in_memory_pdf(self) -> None:
        """Verify real PDF extraction using in-memory generated PDF."""
        try:
            import pypdf
            writer = pypdf.PdfWriter()
            writer.add_blank_page(width=200, height=200)
            buf = io.BytesIO()
            writer.write(buf)
            pdf_bytes = buf.getvalue()

            res = parse_document(pdf_bytes, filename="blank.pdf", mime_type="application/pdf")
            self.assertIn("blank.pdf", res)
        except ImportError:
            self.skipTest("pypdf not available for in-memory PDF test")

    def test_parse_mocked_pdf_success(self) -> None:
        """Verify multi-page PDF extraction with mocked PdfReader."""
        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1 Content: Executive Summary"
        page2 = MagicMock()
        page2.extract_text.return_value = "Page 2 Content: Detailed Financials"

        mock_reader = MagicMock()
        mock_reader.pages = [page1, page2]
        mock_reader.is_encrypted = False

        with patch("pypdf.PdfReader", return_value=mock_reader):
            res = parse_document(b"%PDF-1.5 simulated", filename="annual_report.pdf", mime_type="application/pdf")
            self.assertIn("--- Page 1 ---", res)
            self.assertIn("Executive Summary", res)
            self.assertIn("--- Page 2 ---", res)
            self.assertIn("Detailed Financials", res)

    def test_parse_pdf_encrypted_failure(self) -> None:
        """Verify encrypted password-protected PDFs raise DocumentParseError."""
        mock_reader = MagicMock()
        mock_reader.is_encrypted = True
        mock_reader.decrypt.side_effect = RuntimeError("Invalid password")

        with patch("pypdf.PdfReader", return_value=mock_reader):
            with self.assertRaises(DocumentParseError) as ctx:
                parse_document(b"%PDF-1.4 encrypted", filename="secret.pdf", mime_type="application/pdf")
            self.assertIn("password-protected", str(ctx.exception).lower())

    def test_parse_empty_document(self) -> None:
        """Verify empty byte payload raises DocumentParseError."""
        with self.assertRaises(DocumentParseError) as ctx:
            parse_document(b"", filename="empty.txt")
        self.assertIn("is empty", str(ctx.exception))

    def test_parse_oversized_document(self) -> None:
        """Verify documents exceeding 10MB limit raise DocumentParseError."""
        oversized = b"X" * (MAX_DOCUMENT_BYTES + 10)
        with self.assertRaises(DocumentParseError) as ctx:
            parse_document(oversized, filename="giant.txt")
        self.assertIn("exceeds maximum allowed size", str(ctx.exception))

    def test_parse_unsupported_mime(self) -> None:
        """Verify unsupported binaries raise DocumentParseError."""
        with self.assertRaises(DocumentParseError) as ctx:
            parse_document(b"\x7fELF...", filename="binary.bin", mime_type="application/x-executable")
        self.assertIn("Unsupported document", str(ctx.exception))

    def test_50k_character_limit_enforcement(self) -> None:
        """Verify extracted text exceeding 50,000 characters is capped with a notice."""
        large_text = ("Sentence number %d with some filler words.\n" % i for i in range(3000))
        large_bytes = "".join(large_text).encode("utf-8")
        self.assertTrue(len(large_bytes) > MAX_DOCUMENT_CHARS)

        res = parse_document(large_bytes, filename="large_doc.txt", mime_type="text/plain")
        self.assertIn("[Note: Document 'large_doc.txt' truncated to first 50000 characters", res)
        # Verify length is bounded around 50k + truncated notice length
        self.assertTrue(len(res) <= MAX_DOCUMENT_CHARS + 200)

    def test_normalize_text(self) -> None:
        """Verify whitespace normalization collapses multiple blank lines."""
        raw = "Line A\r\n\r\n\r\n\r\n\r\nLine B"
        normalized = normalize_text(raw)
        self.assertEqual(normalized, "Line A\n\nLine B")
        self.assertEqual(normalize_text(""), "")


class TestChatAPIDocuments(unittest.TestCase):
    """Integration test suite for /api/chat document handling."""

    def setUp(self) -> None:
        """Initialize Flask test client."""
        app = Flask(__name__)
        app.register_blueprint(chat_bp, url_prefix="/api")
        self.client = app.test_client()

    @patch("api.chat.router.route_chat")
    def test_chat_with_valid_text_document(self, mock_route: MagicMock) -> None:
        """Verify /api/chat extracts text and injects document context into system_prompt."""
        mock_route.return_value = {
            "response": "The document specifies that the revenue grew by 25%.",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "metadata": {"stage": "Stage 9"},
        }

        doc_content = "Q3 Revenue Report: Total revenue was $10M, representing 25% YoY growth."
        b64_doc = base64.b64encode(doc_content.encode("utf-8")).decode("utf-8")

        payload = {
            "message": "What was the revenue growth?",
            "documents": [
                {
                    "data": b64_doc,
                    "filename": "revenue.txt",
                    "mime_type": "text/plain",
                }
            ],
        }

        resp = self.client.post("/api/chat", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["metadata"]["documents_attached"], ["revenue.txt"])
        self.assertEqual(data["metadata"]["document_count"], 1)

        # Verify system_prompt passed to route_chat includes the document context
        self.assertTrue(mock_route.called)
        kwargs = mock_route.call_args[1]
        system_prompt = kwargs.get("system_prompt", "")
        self.assertIn("[Attached Document: revenue.txt]", system_prompt)
        self.assertIn("Q3 Revenue Report", system_prompt)
        self.assertIn("Total revenue was $10M", system_prompt)

    @patch("api.chat.router.route_chat")
    def test_chat_document_only_request(self, mock_route: MagicMock) -> None:
        """Verify /api/chat allows document-first requests without explicit message."""
        mock_route.return_value = {
            "response": "Summary: This document outlines system architecture.",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "metadata": {"stage": "Stage 9"},
        }

        doc_content = "System Architecture: Microservices deployed on Kubernetes with Postgres."
        b64_doc = base64.b64encode(doc_content.encode("utf-8")).decode("utf-8")

        payload = {
            "documents": [
                {
                    "data": b64_doc,
                    "filename": "architecture.md",
                    "mime_type": "text/markdown",
                }
            ]
        }

        resp = self.client.post("/api/chat", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["ok"])

        # Check default message was assigned
        kwargs = mock_route.call_args[1]
        self.assertIn("analyze", kwargs["message"].lower())

    def test_chat_too_many_documents(self) -> None:
        """Verify /api/chat rejects requests with more than 2 documents."""
        b64_dummy = base64.b64encode(b"Dummy").decode("utf-8")
        payload = {
            "message": "Analyze all 3 files",
            "documents": [
                {"data": b64_dummy, "filename": "1.txt"},
                {"data": b64_dummy, "filename": "2.txt"},
                {"data": b64_dummy, "filename": "3.txt"},
            ],
        }

        resp = self.client.post("/api/chat", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["error"]["code"], "TOO_MANY_DOCUMENTS")

    def test_chat_invalid_documents_type(self) -> None:
        """Verify non-list documents field returns 400."""
        resp = self.client.post("/api/chat", json={"message": "Hi", "documents": "not-a-list"})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["error"]["code"], "INVALID_DOCUMENTS_TYPE")

    def test_chat_missing_document_data(self) -> None:
        """Verify document item missing data field returns 400."""
        payload = {
            "message": "Check this",
            "documents": [{"filename": "nofile.txt"}],
        }
        resp = self.client.post("/api/chat", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["error"]["code"], "MISSING_DOCUMENT_DATA")

    def test_chat_invalid_base64_document(self) -> None:
        """Verify malformed base64 in document data returns 400."""
        payload = {
            "message": "Check this",
            "documents": [{"data": "!!!malformed_b64!!!", "filename": "bad.txt"}],
        }
        resp = self.client.post("/api/chat", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["error"]["code"], "INVALID_BASE64_DOCUMENT")

    def test_chat_empty_document_data(self) -> None:
        """Verify empty document data returns 400."""
        payload = {
            "message": "Check this",
            "documents": [{"data": "   ", "filename": "empty.txt"}],
        }
        resp = self.client.post("/api/chat", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["error"]["code"], "EMPTY_DOCUMENT_DATA")

    @patch("api.chat.conversation_repo")
    @patch("api.chat.router.route_chat")
    def test_chat_database_metadata_persistence(
        self,
        mock_route: MagicMock,
        mock_repo: MagicMock,
    ) -> None:
        """Verify database message persistence records filename tag and not base64 content."""
        mock_repo.is_available.return_value = True
        mock_repo.create_conversation.return_value = {"id": "conv-123", "user_id": "u-1"}
        mock_repo.get_conversation.return_value = {"id": "conv-123", "user_id": "u-1"}
        mock_repo.get_messages.return_value = []
        mock_repo.get_memories.return_value = []

        mock_route.return_value = {
            "response": "Analysis complete.",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "metadata": {},
        }

        b64_content = base64.b64encode(b"Important research findings").decode("utf-8")
        payload = {
            "message": "Analyze research findings",
            "documents": [
                {
                    "data": b64_content,
                    "filename": "findings.txt",
                    "mime_type": "text/plain",
                }
            ],
        }

        resp = self.client.post("/api/chat", json=payload)
        self.assertEqual(resp.status_code, 200)

        # Check call to save_message for the user message
        user_save_call = mock_repo.save_message.call_args_list[0]
        saved_content = user_save_call[1].get("content") or user_save_call[0][2]
        self.assertIn("Analyze research findings", saved_content)
        self.assertIn("[Attached Document(s): findings.txt]", saved_content)
        # Ensure base64 payload is NOT in saved_content
        self.assertNotIn(b64_content, saved_content)

    @patch("api.chat.router.route_stream")
    def test_chat_streaming_with_documents(self, mock_stream: MagicMock) -> None:
        """Verify SSE streaming endpoint handles document context injection."""
        from pihu_core.providers import StreamChunk

        def mock_generator(*args, **kwargs):
            yield {"text": "Document ", "provider": "gemini"}
            yield {"text": "summary.", "provider": "gemini"}

        mock_stream.side_effect = mock_generator

        b64_content = base64.b64encode(b"Doc content").decode("utf-8")
        payload = {
            "message": "Summarize",
            "stream": True,
            "documents": [
                {
                    "data": b64_content,
                    "filename": "doc.txt",
                    "mime_type": "text/plain",
                }
            ],
        }

        resp = self.client.post("/api/chat", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "text/event-stream")

        # Verify system_prompt in stream call contains document
        self.assertTrue(mock_stream.called)
        kwargs = mock_stream.call_args[1]
        self.assertIn("[Attached Document: doc.txt]", kwargs["system_prompt"])


if __name__ == "__main__":
    unittest.main()
