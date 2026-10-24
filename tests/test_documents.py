from kbforge.documents import (
    Document,
    compute_content_hash,
    read_documents_jsonl,
    stable_doc_id,
    write_documents_jsonl,
)


def make_doc(source="http://kb.test/a", text="<html>hi</html>"):
    return Document.from_parts(
        source=source,
        title="T",
        raw_text=text,
        fetched_at="2026-01-01T00:00:00+00:00",
        kind="web",
        content_type="html",
        metadata={"depth": 0},
    )


def test_content_hash_is_sha256_of_text():
    doc = make_doc()
    assert doc.content_hash == compute_content_hash("<html>hi</html>")
    assert len(doc.content_hash) == 64


def test_doc_id_is_stable_and_kind_prefixed():
    assert stable_doc_id("web", "http://kb.test/a") == stable_doc_id("web", "http://kb.test/a")
    assert stable_doc_id("web", "http://kb.test/a").startswith("web-")
    assert stable_doc_id("file", "/tmp/a.pdf") != stable_doc_id("web", "/tmp/a.pdf")


def test_jsonl_roundtrip(tmp_path):
    docs = [make_doc(), make_doc(source="http://kb.test/b", text="other")]
    path = tmp_path / "documents.jsonl"
    assert write_documents_jsonl(docs, path) == 2
    loaded = read_documents_jsonl(path)
    assert [d.to_dict() for d in loaded] == [d.to_dict() for d in docs]


def test_different_sources_get_different_ids():
    assert make_doc(source="http://kb.test/a").doc_id != make_doc(source="http://kb.test/b").doc_id
