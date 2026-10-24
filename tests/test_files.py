"""File loader tests. Tiny PDF/DOCX fixtures are generated at test time."""

from __future__ import annotations

from pathlib import Path

import pymupdf
from docx import Document as DocxDocument

from kbforge.documents import compute_content_hash
from kbforge.ingest.files import load_file, load_files


def make_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def make_docx(path: Path, title: str, paragraphs: list[str]) -> None:
    doc = DocxDocument()
    if title:
        doc.core_properties.title = title
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(path)


def test_load_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    make_pdf(path, "Hello PDF world")
    doc = load_file(path)
    assert doc.kind == "file"
    assert doc.content_type == "pdf"
    assert "Hello PDF world" in doc.raw_text
    assert doc.content_hash == compute_content_hash(doc.raw_text)


def test_load_docx(tmp_path):
    path = tmp_path / "report.docx"
    make_docx(path, title="Quarterly Report", paragraphs=["First point.", "Second point."])
    doc = load_file(path)
    assert doc.title == "Quarterly Report"
    assert "First point." in doc.raw_text
    assert "Second point." in doc.raw_text


def test_load_html_file(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>My Page</title><style>.x{}</style></head>"
        "<body><p>Visible text</p><script>var x=1;</script></body></html>",
        encoding="utf-8",
    )
    doc = load_file(path)
    assert doc.title == "My Page"
    assert "Visible text" in doc.raw_text
    assert "var x=1" not in doc.raw_text


def test_load_txt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("plain text body", encoding="utf-8")
    doc = load_file(path)
    assert doc.raw_text == "plain text body"
    assert doc.content_type == "txt"
    assert doc.title == "notes"


def test_load_files_recurses_and_skips_unsupported(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "notes.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub" / "more.txt").write_text("b", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    docs = load_files(tmp_path)
    names = sorted(d.metadata["filename"] for d in docs)
    assert names == ["more.txt", "notes.txt"]


def test_load_files_missing_dir_returns_empty(tmp_path):
    assert load_files(tmp_path / "nope") == []


def test_file_schema_fields(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("body", encoding="utf-8")
    doc = load_file(path)
    assert doc.doc_id.startswith("file-")
    assert doc.source.endswith("notes.txt")
    assert doc.fetched_at  # ISO timestamp from file mtime
    assert len(doc.content_hash) == 64


def test_load_files_reports_missing_directory(tmp_path):
    errors: list[dict] = []
    assert load_files(tmp_path / "missing", errors=errors) == []
    assert len(errors) == 1
    assert errors[0]["path"].endswith("missing")
    assert "does not exist" in errors[0]["error"]
