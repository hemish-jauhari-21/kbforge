"""Phase 5 FastAPI endpoint tests using the in-process test client."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from kbforge.api import create_app
from kbforge.package import build_package


@pytest.fixture
def api_client(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    return TestClient(create_app(package_path=output_dir))


def test_health_endpoint(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["knowledge_base_loaded"] is True
    assert body["ranking_modes"] == ["bm25", "tfidf"]
    assert body["chunks"] > 0
    assert body["format_version"]


@pytest.mark.parametrize("mode", ["bm25", "tfidf"])
def test_search_endpoint_returns_metadata(api_client, mode):
    response = api_client.post(
        "/search", json={"query": "python", "mode": mode, "limit": 2}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "python"
    assert body["mode"] == mode
    assert body["limit"] == 2
    assert body["total"] == len(body["results"]) <= 2
    assert body["results"]
    top = body["results"][0]
    assert top["rank"] == 1
    assert top["source"] == "https://kb.test/python"
    assert top["title"] == "Python guide"
    assert top["chunk_id"]
    assert top["doc_id"].startswith("web-")
    assert top["score"] > 0
    assert isinstance(top["metadata"], dict)


def test_search_no_match_returns_empty_list(api_client):
    response = api_client.post("/search", json={"query": "nonexistentterm"})

    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "python", "mode": "dense"},
        {"query": "python", "limit": 0},
        {"query": "python", "limit": 101},
        {"query": "python", "limit": "many"},
        {},
    ],
)
def test_search_request_validation(api_client, payload):
    response = api_client.post("/search", json=payload)
    assert response.status_code == 422


def test_search_defaults(api_client):
    response = api_client.post("/search", json={"query": "python"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "bm25"
    assert body["limit"] == 5


def test_health_without_package_is_unavailable(tmp_path):
    client = TestClient(create_app(package_path=tmp_path / "missing"))

    assert client.get("/health").status_code == 503
    assert client.post("/search", json={"query": "python"}).status_code == 503


def test_search_with_corrupt_package_is_unavailable(sample_kb):
    config, output_dir = sample_kb
    build_package(config)
    (output_dir / "index" / "df.json").write_text("broken", encoding="utf-8")

    client = TestClient(create_app(package_path=output_dir))

    response = client.get("/health")
    assert response.status_code == 503
    assert "traceback" not in response.text.lower()
    assert str(output_dir) not in response.text


def test_api_does_not_rebuild(sample_kb, monkeypatch):
    config, output_dir = sample_kb
    build_package(config)

    def fail(*args, **kwargs):
        raise AssertionError("API must not crawl, chunk, index, or package")

    monkeypatch.setattr("kbforge.ingest.crawler.run_ingest", fail)
    monkeypatch.setattr("kbforge.process.run_process", fail)
    monkeypatch.setattr("kbforge.process.run_chunk", fail)
    monkeypatch.setattr("kbforge.process.run_index", fail)
    monkeypatch.setattr("kbforge.package.build_package", fail)

    client = TestClient(create_app(package_path=output_dir))
    assert client.get("/health").status_code == 200
    assert client.post("/search", json={"query": "python"}).status_code == 200


def test_api_config_path_selects_output_dir(sample_kb, tmp_path, monkeypatch):
    config, output_dir = sample_kb
    build_package(config)
    config_path = tmp_path / "client.yaml"
    config_path.write_text(
        json.dumps(
            {
                "client_name": "api-test",
                "sources": {"urls": ["https://kb.test/"]},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(config_path=config_path))
    assert client.get("/health").status_code == 200
    assert client.post("/search", json={"query": "search"}).json()["results"]
