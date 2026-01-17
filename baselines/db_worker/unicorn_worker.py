from __future__ import annotations

import logging
import os
import asyncio
from typing import Any, Dict, List, Mapping, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from halo_dev.db import MissingQueryInputs, PostgresDatabaseExecutor
from halo_dev.executor import DBNodeExecutor
from halo_dev.models import DBQuery, Node

LOGGER = logging.getLogger(__name__)
DB_POOL_SIZE_DEFAULT = 32
DB_POOL_SIZE = int(os.getenv("HALO_PG_POOL_SIZE", str(DB_POOL_SIZE_DEFAULT)) or DB_POOL_SIZE_DEFAULT)


class DBQueryPayload(BaseModel):
    name: str
    sql: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    post_llm: bool = False
    result_mappings: Dict[str, str] = Field(default_factory=dict)
    required_inputs: Sequence[str] = Field(default_factory=tuple)
    param_types: Dict[str, str] = Field(default_factory=dict)


class ExecuteBatchRequest(BaseModel):
    node_id: str = "db_worker"
    queries: List[DBQueryPayload]
    contexts: List[Dict[str, Any]] | Dict[str, Any] = Field(default_factory=list)
    db_concurrency: int = DB_POOL_SIZE

    @validator("contexts", pre=True)
    def normalize_contexts(
        cls,
        value: List[Dict[str, Any]] | Dict[str, Any] | None,
    ) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            return [dict(value)]
        if isinstance(value, list):
            return value
        raise ValueError("contexts must be a mapping or list of mappings")


class ExecuteResponse(BaseModel):
    node_id: str
    outputs: List[Dict[str, Any]]
    stats: Dict[str, Any]


def _default_connect_kwargs() -> Dict[str, Any]:
    """Fallback to halo defaults if env not set."""
    pool_size_val = DB_POOL_SIZE
    return {
        "host": os.getenv("HALO_PG_HOST", "/var/run/postgresql"),
        "port": int(os.getenv("HALO_PG_PORT", "4032")),
        "dbname": os.getenv("HALO_PG_DBNAME", "imdb"),
        "user": os.getenv("HALO_PG_USER", os.getenv("USER", "postgres")),
        # password intentionally optional
        **({"password": os.getenv("HALO_PG_PASSWORD")} if os.getenv("HALO_PG_PASSWORD") else {}),
        "pool_size": pool_size_val,
    }


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="Halo Postgres DB Worker", version="0.1.0")
db_executor = PostgresDatabaseExecutor(
    dsn=os.getenv("HALO_PG_DSN"),
    connect_kwargs={k: v for k, v in _default_connect_kwargs().items() if k != "pool_size"},
    pool_size=max(0, _default_connect_kwargs().get("pool_size", 0)),
)


@app.middleware("http")
async def log_exceptions(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception:
        LOGGER.exception("Unhandled error during request %s %s", request.method, request.url.path)
        raise


def _build_node(node_id: str, queries: Sequence[DBQueryPayload]) -> Node:
    db_queries = [DBQuery(**query.dict()) for query in queries]
    return Node(
        id=node_id,
        type="db",
        engine="db",
        inputs=(),
        outputs=(),
        db_queries=tuple(db_queries),
        raw={},
    )


async def _execute_queries(
    payload: ExecuteBatchRequest,
) -> ExecuteResponse:
    contexts = payload.contexts or [{}]
    if not payload.queries:
        raise HTTPException(status_code=400, detail="At least one query is required")
    if not contexts:
        raise HTTPException(status_code=400, detail="At least one context is required")

    node = _build_node(payload.node_id, payload.queries)
    db_concurrency = min(payload.db_concurrency, DB_POOL_SIZE if DB_POOL_SIZE > 0 else payload.db_concurrency)
    executor = DBNodeExecutor(db_executor=db_executor, db_concurrency=db_concurrency)
    try:
        outputs = await asyncio.to_thread(executor.execute_batch, node, contexts)
    except MissingQueryInputs as exc:
        LOGGER.exception("Missing inputs while executing node %s", node.id)
        executor.consume_stats()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        LOGGER.exception("Failed to execute node %s", node.id)
        executor.consume_stats()
        raise HTTPException(status_code=500, detail=str(exc))
    stats = executor.consume_stats()
    return ExecuteResponse(node_id=node.id, outputs=outputs, stats=stats)


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteBatchRequest) -> ExecuteResponse:
    if len(request.queries) != 1:
        raise HTTPException(status_code=400, detail="POST /execute expects exactly one query")
    return await _execute_queries(request)


@app.post("/execute_batch", response_model=ExecuteResponse)
async def execute_batch(request: ExecuteBatchRequest) -> ExecuteResponse:
    return await _execute_queries(request)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
