from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from kbforge import __version__
from kbforge.config import ConfigError, load_config
from kbforge.package import KnowledgeBase, PackageError, SUPPORTED_RANKING_MODES

logger = logging.getLogger("kbforge.api")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Text to search for")
    mode: Literal["bm25", "tfidf"] = Field(
        default="bm25", description="Lexical ranking mode"
    )
    limit: int = Field(default=5, ge=1, le=100, description="Maximum results (1-100)")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class SearchResultModel(BaseModel):
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    mode: str
    limit: int
    total: int
    results: list[SearchResultModel]


class HealthResponse(BaseModel):
    status: str
    knowledge_base_loaded: bool
    kbforge_version: str
    ranking_modes: list[str] = []
    chunks: int | None = None
    format_version: str | None = None


def _resolve_source(
    package_path: str | Path | None,
    config_path: str | Path | None,
    *,
    use_default_config: bool,
) -> Path | None:
    if package_path is not None:
        return Path(package_path)
    env_package = os.environ.get("KBFORGE_PACKAGE")
    if env_package:
        return Path(env_package)
    env_config = os.environ.get("KBFORGE_CONFIG")
    if config_path is not None:
        candidate = Path(config_path)
    elif env_config:
        candidate = Path(env_config)
    elif use_default_config:
        candidate = Path("config.yaml")
    else:
        return None
    if not candidate.exists():
        return None
    config = load_config(candidate)
    return Path(config.output_dir)


def create_app(
    package_path: str | Path | None = None,
    config_path: str | Path | None = None,
    *,
    use_default_config: bool = True,
) -> FastAPI:
    """Create the API app and load the knowledge base at startup.

    Loading is read-only and occurs once per app instance.  If artifacts are
    unavailable, the app still starts so ``/health`` can report the problem;
    ``/search`` returns HTTP 503 until a valid package is configured.
    """
    app = FastAPI(
        title="kbforge search API",
        version=__version__,
        description="Local lexical search over a packaged kbforge knowledge base.",
    )

    kb: KnowledgeBase | None = None
    load_error: str | None = None
    try:
        source = _resolve_source(
            package_path, config_path, use_default_config=use_default_config
        )
        if source is None:
            load_error = "no knowledge base configured"
        else:
            kb = KnowledgeBase.load(source)
    except (PackageError, ConfigError, OSError, ValueError) as exc:
        load_error = str(exc)
        logger.warning("knowledge base unavailable: %s", exc)

    app.state.kb = kb
    app.state.load_error = load_error

    def _get_kb() -> KnowledgeBase:
        if app.state.kb is None:
            # Do not leak filesystem paths or exception traces to clients.
            raise HTTPException(
                status_code=503,
                detail="knowledge base is unavailable; configure a valid package",
            )
        return app.state.kb

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current: KnowledgeBase | None = app.state.kb
        if current is None:
            raise HTTPException(
                status_code=503,
                detail="knowledge base is unavailable; configure a valid package",
            )
        manifest = current.manifest or {}
        statistics = manifest.get("statistics", {}) if isinstance(manifest, dict) else {}
        return HealthResponse(
            status="ok",
            knowledge_base_loaded=True,
            kbforge_version=__version__,
            ranking_modes=list(current.supported_modes),
            chunks=statistics.get("chunks") or current.index.N,
            format_version=manifest.get("format_version"),
        )

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        current = _get_kb()
        try:
            results = current.search(
                request.query, mode=request.mode, limit=request.limit
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            logger.exception("search failed")
            raise HTTPException(status_code=500, detail="search failed") from None

        payload = [
            SearchResultModel(
                rank=result.rank + 1,
                score=result.score,
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                source=result.source,
                title=result.title,
                text=result.text,
                metadata=result.metadata,
            )
            for result in results
        ]
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            limit=request.limit,
            total=len(payload),
            results=payload,
        )

    return app


# Convenience instance for ``uvicorn kbforge.api:app``.  Configure it with
# KBFORGE_PACKAGE (preferred) or KBFORGE_CONFIG; without either it starts in
# an unavailable state and only /health reports that.
app = create_app(
    package_path=os.environ.get("KBFORGE_PACKAGE"),
    config_path=os.environ.get("KBFORGE_CONFIG"),
    use_default_config=False,
)

__all__ = [
    "app",
    "create_app",
    "HealthResponse",
    "SearchRequest",
    "SearchResponse",
    "SearchResultModel",
    "SUPPORTED_RANKING_MODES",
]
