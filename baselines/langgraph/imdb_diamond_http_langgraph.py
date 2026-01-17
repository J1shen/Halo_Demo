from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict

import requests
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from baselines.metrics import BaselineMetrics, get_global_metrics, set_global_metrics
from halo_dev.executor import HTTPNodeExecutor
from halo_dev.models import Node
from halo_dev.utils import render_template

# Defaults aligned with local vLLM workers (ports 9101–9104)
API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
MODEL_A = os.getenv("IMDB_MODEL_A", "openai/gpt-oss-20b")
MODEL_B = os.getenv("IMDB_MODEL_B", "Qwen/Qwen3-14B")
MODEL_C = os.getenv("IMDB_MODEL_C", "Qwen/Qwen3-32B")
BASE_A = os.getenv("MODEL_A_BASE_URL", "http://localhost:9101/v1")
BASE_B = os.getenv("MODEL_B_BASE_URL", "http://localhost:9102/v1")
BASE_C = os.getenv("MODEL_C_BASE_URL", "http://localhost:9103/v1")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))
HTTP_DEFAULT_SLEEP_S = float(os.getenv("HALO_HTTP_DEFAULT_SLEEP_S", "0.0"))
HTTP_CONCURRENCY = int(os.getenv("HALO_HTTP_CONCURRENCY", "1"))
MAX_FIELD_CHARS = 2000
MAX_ROWS = 5

HTTP_EXECUTOR = HTTPNodeExecutor(
    http_concurrency=HTTP_CONCURRENCY,
    default_sleep_s=HTTP_DEFAULT_SLEEP_S,
)


class State(TypedDict, total=False):
    user_query: str
    search_keyword: str
    entry_movie_notes: str
    entry_person_notes: str
    entry_akas_notes: str
    hub_movie_person_notes: str
    hub_full_context: str
    tail_focus_notes: str
    final_answer: str


def make_model(model_name: str, base_url: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        api_key=API_KEY,
        openai_api_base=base_url,
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
    )


def _sanitize_value(value: Any, max_chars: int = MAX_FIELD_CHARS) -> Any:
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars] + "...[truncated]"
        return value
    if isinstance(value, Mapping):
        return {k: _sanitize_value(v, max_chars) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_value(v, max_chars) for v in value]
    return value


def _prepare_result_for_prompt(result: Mapping[str, Any]) -> Mapping[str, Any]:
    prepared: Dict[str, Any] = dict(result)
    rows = prepared.get("rows")
    if isinstance(rows, list):
        trimmed: List[Any] = []
        for row in rows[:MAX_ROWS]:
            if isinstance(row, Mapping):
                trimmed.append({k: _sanitize_value(v) for k, v in row.items()})
            else:
                trimmed.append(row)
        prepared["rows"] = trimmed
        if len(rows) > MAX_ROWS:
            prepared["rows_truncated"] = f"{len(rows) - MAX_ROWS} rows omitted"
    return prepared


def _render_input_value(value: Any) -> str:
    if isinstance(value, Mapping) and "rows" in value:
        value = _prepare_result_for_prompt(value)
    try:
        if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
            return json.dumps(value, ensure_ascii=False)
    except Exception:
        pass
    return str(value)


def _parse_db_payload(payload: str) -> List[Mapping[str, Any]]:
    try:
        data = json.loads(payload)
        if isinstance(data, Mapping):
            if "query" in data:
                return [data]
            results: List[Mapping[str, Any]] = []
            for val in data.values():
                if isinstance(val, Mapping):
                    results.append(val)
            return results
    except Exception:
        return []
    return []


def build_halo_prompt(system_prompt: str | None, inputs: Mapping[str, Any], db_payloads: List[str]) -> str:
    parts: List[str] = []
    if system_prompt:
        parts.append(render_template(system_prompt.strip(), inputs))
    for name, value in inputs.items():
        rendered = _render_input_value(value)
        parts.append(f"[Input::{name}]\n{rendered}")
    for payload in db_payloads:
        for result in _parse_db_payload(payload):
            prepared = _prepare_result_for_prompt(result)
            parts.append(
                f"[DB::{prepared.get('query', 'unknown')}]\n"
                f"{json.dumps(prepared, ensure_ascii=False)}"
            )
    return "\n\n".join(parts)


def maybe_parse_structured_response(response: str) -> Any:
    candidates: List[str] = [response.strip()]
    block_pattern = re.compile(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", re.IGNORECASE)
    candidates.extend(block_pattern.findall(response))
    first_brace = response.find("{")
    last_brace = response.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(response[first_brace:last_brace + 1])
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    text = response.strip()
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        brace = text.find("{", idx)
        if brace == -1:
            break
        try:
            parsed, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(parsed, dict):
            return parsed
        idx = end
    return response


def extract_outputs(expected_keys: List[str], response: str) -> Dict[str, Any]:
    parsed = maybe_parse_structured_response(response)
    outputs: Dict[str, Any] = {}
    if isinstance(parsed, dict):
        for key in expected_keys:
            if key in parsed:
                outputs[key] = parsed[key]
        if not outputs and expected_keys:
            outputs[expected_keys[0]] = parsed
    else:
        if expected_keys:
            outputs[expected_keys[0]] = parsed
    if "search_keyword" in outputs and isinstance(outputs["search_keyword"], str):
        first = outputs["search_keyword"].split(",")[0].strip()
        if first:
            outputs["search_keyword"] = first
    return outputs


def _record_db_metrics(stats: Mapping[str, Any], *, elapsed: float, fallback_count: int) -> None:
    metrics = get_global_metrics()
    if not metrics:
        return
    db_calls = int(stats.get("db_calls") or 0)
    db_time = float(stats.get("db_time") or 0.0)
    if db_calls <= 0:
        db_calls = max(1, fallback_count)
    if db_time <= 0:
        db_time = elapsed
    metrics.record_db(db_time, count=db_calls)


def call_db_worker_sync(node_id: str, queries: List[Dict[str, Any]], context: Mapping[str, Any]) -> str:
    start = time.perf_counter()
    payload = {"node_id": node_id, "queries": queries, "contexts": [dict(context)]}
    resp = requests.post(f"{DB_WORKER_URL}/execute_batch", json=payload, timeout=DB_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    _record_db_metrics(data.get("stats") or {}, elapsed=time.perf_counter() - start, fallback_count=len(queries))
    outputs = data.get("outputs") or [{}]
    return json.dumps(outputs[0], ensure_ascii=False)


def call_http_node_sync(
    node_id: str,
    output_name: str,
    sleep_s: float,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    node = Node(
        id=node_id,
        type="http",
        engine="http",
        outputs=(output_name,),
        raw={"sleep_s": sleep_s},
    )
    outputs = HTTP_EXECUTOR.execute(node, context)
    return outputs.get(output_name, {})


def invoke_llm(model: ChatOpenAI, messages: List[Mapping[str, str]]):
    start = time.perf_counter()
    resp = model.invoke(messages)
    metrics = get_global_metrics()
    if metrics:
        metrics.record_llm(time.perf_counter() - start, call_count=1, prompt_count=1)
    return resp


# DB query definitions (from templates/imdb_diamond_http.yaml)
# HTTP stand-ins (must match template node ids/outputs).
ENTRY_MOVIE_HTTP_NODES: List[Dict[str, Any]] = [
    {
        "id": "http_entry_movie_by_title",
        "output": "http_entry_movie_by_title",
        "sleep_s": 2.0,
    },
    {
        "id": "http_entry_movie_recent",
        "output": "http_entry_movie_recent",
        "sleep_s": 2.0,
    },
]


def _run_entry_movie_http_nodes(search_keyword: str) -> Dict[str, Any]:
    context = {"search_keyword": search_keyword}
    outputs: Dict[str, Any] = {}
    for spec in ENTRY_MOVIE_HTTP_NODES:
        outputs[spec["output"]] = call_http_node_sync(
            spec["id"],
            spec["output"],
            spec["sleep_s"],
            context,
        )
    return outputs

ENTRY_PERSON_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "imdb_entry_person_by_name",
        "sql": """
            SELECT
              n.nconst,
              n.primary_name,
              n.primary_profession,
              n.known_for_titles
            FROM name_basics AS n
            WHERE
              n.primary_name ILIKE '%%' || :keyword || '%%'
            ORDER BY n.primary_name
            LIMIT 25;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    },
    {
        "name": "imdb_entry_person_by_profession",
        "sql": """
            SELECT
              n.nconst,
              n.primary_name,
              n.primary_profession,
              n.known_for_titles
            FROM name_basics AS n
            WHERE
              n.primary_profession ILIKE '%%' || :keyword || '%%'
            ORDER BY n.nconst
            LIMIT 20;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    },
]

ENTRY_AKAS_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "imdb_entry_akas_by_title",
        "sql": """
            SELECT
              a.title_id,
              a.title,
              a.region,
              a.language,
              a.is_original_title
            FROM title_akas AS a
            WHERE
              a.title ILIKE '%%' || :keyword || '%%'
            ORDER BY a.is_original_title DESC, a.ordering
            LIMIT 25;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    },
    {
        "name": "imdb_entry_akas_by_region",
        "sql": """
            SELECT
              a.title_id,
              a.title,
              a.region,
              a.language
            FROM title_akas AS a
            WHERE
              a.region ILIKE '%%' || :keyword || '%%'
            ORDER BY a.ordering
            LIMIT 20;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    },
]

HUB_MOVIE_PERSON_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "imdb_hub_movie_person_stats",
        "sql": """
            SELECT
              COUNT(*) AS movie_count,
              AVG(r.average_rating) AS avg_rating
            FROM title_basics AS b
            LEFT JOIN title_ratings AS r
              ON r.tconst = b.tconst
            WHERE
              b.title_type IN ('movie','tvMovie','tvSeries')
              AND (
                b.primary_title ILIKE '%%' || :keyword || '%%'
                OR b.original_title ILIKE '%%' || :keyword || '%%'
              );
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    }
]

HUB_GENRE_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "imdb_hub_genre_stats_filtered",
        "sql": """
            SELECT
              g AS genre,
              COUNT(*) AS cnt
            FROM (
              SELECT
                regexp_split_to_table(b.genres, ',') AS g
              FROM title_basics AS b
              WHERE
                b.genres IS NOT NULL
                AND b.genres ILIKE '%%' || :keyword || '%%'
            ) AS x
            GROUP BY g
            ORDER BY cnt DESC
            LIMIT 10;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    }
]

FINAL_SAMPLE_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "imdb_final_sample_titles",
        "sql": """
            SELECT
              b.tconst,
              b.primary_title,
              b.start_year,
              r.average_rating,
              r.num_votes
            FROM title_basics AS b
            LEFT JOIN title_ratings AS r
              ON r.tconst = b.tconst
            WHERE
              b.title_type IN ('movie','tvMovie','tvSeries')
              AND (
                b.primary_title ILIKE '%%' || :keyword || '%%'
                OR b.original_title ILIKE '%%' || :keyword || '%%'
              )
            ORDER BY COALESCE(r.num_votes, 0) DESC
            LIMIT 20;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    }
]


# === LangGraph node functions ===
def kw_planner(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You are the keyword planner.\n"
        "From user_query, generate a concise search_keyword suitable for\n"
        "ILIKE filters on movie titles, person names, and AKA titles.\n"
        "Respond ONLY with:\n"
        '{\n  "search_keyword": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"]}, [])
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["search_keyword"], resp.content)
    return {"search_keyword": parsed.get("search_keyword", "")}


def entry_movie(state: State, model_b: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Entry A (movie-centric).\n"
        "Use http_entry_movie_by_title and http_entry_movie_recent payloads\n"
        "to summarize representative candidate titles related to the\n"
        "search_keyword.\n"
        "Respond ONLY with:\n"
        '{\n  "entry_movie_notes": "string"\n}'
    )
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "search_keyword": state["search_keyword"],
            **_run_entry_movie_http_nodes(state["search_keyword"]),
        },
        [],
    )
    resp = invoke_llm(model_b, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["entry_movie_notes"], resp.content)
    return {"entry_movie_notes": parsed.get("entry_movie_notes", "")}


def entry_person(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Entry B (people-centric).\n"
        "Use imdb_entry_person_by_name and imdb_entry_person_by_profession\n"
        "to identify people relevant to the search_keyword and summarize\n"
        "why they matter.\n"
        "Respond ONLY with:\n"
        '{\n  "entry_person_notes": "string"\n}'
    )
    entry_person_db = call_db_worker_sync(
        "entry_person",
        ENTRY_PERSON_QUERIES,
        {"search_keyword": state["search_keyword"]},
    )
    prepared_entry_person_db = json.dumps(_prepare_result_for_prompt(json.loads(entry_person_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "search_keyword": state["search_keyword"]},
        [prepared_entry_person_db],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["entry_person_notes"], resp.content)
    return {"entry_person_notes": parsed.get("entry_person_notes", "")}


def entry_akas(state: State, model_c: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Entry C (AKA / international titles).\n"
        "Use imdb_entry_akas_by_title and imdb_entry_akas_by_region to\n"
        "summarize alternative and international titles related to the\n"
        "search_keyword.\n"
        "Respond ONLY with:\n"
        '{\n  "entry_akas_notes": "string"\n}'
    )
    entry_akas_db = call_db_worker_sync(
        "entry_akas",
        ENTRY_AKAS_QUERIES,
        {"search_keyword": state["search_keyword"]},
    )
    prepared_entry_akas_db = json.dumps(_prepare_result_for_prompt(json.loads(entry_akas_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "search_keyword": state["search_keyword"]},
        [prepared_entry_akas_db],
    )
    resp = invoke_llm(model_c, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["entry_akas_notes"], resp.content)
    return {"entry_akas_notes": parsed.get("entry_akas_notes", "")}


def hub_movie_person(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Hub1: fuse entry_movie_notes and entry_person_notes into a joint\n"
        "movie-person perspective on the user_query and search_keyword.\n"
        "You may reference simple aggregates from imdb_hub_movie_person_stats.\n"
        "Respond ONLY with:\n"
        '{\n  "hub_movie_person_notes": "string"\n}'
    )
    hub_movie_person_db = call_db_worker_sync(
        "hub_movie_person",
        HUB_MOVIE_PERSON_QUERIES,
        {"search_keyword": state["search_keyword"]},
    )
    prepared_hub_movie_person_db = json.dumps(_prepare_result_for_prompt(json.loads(hub_movie_person_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "search_keyword": state["search_keyword"],
            "entry_movie_notes": state.get("entry_movie_notes", ""),
            "entry_person_notes": state.get("entry_person_notes", ""),
        },
        [prepared_hub_movie_person_db],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["hub_movie_person_notes"], resp.content)
    return {"hub_movie_person_notes": parsed.get("hub_movie_person_notes", "")}


def hub_full_context(state: State, model_b: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Hub2: combine hub_movie_person_notes with entry_akas_notes into\n"
        "a single full-context view of the candidate space around\n"
        "search_keyword.\n"
        "You may optionally reference imdb_hub_genre_stats_filtered for\n"
        "genre distribution cues.\n"
        "Respond ONLY with:\n"
        '{\n  "hub_full_context": "string"\n}'
    )
    hub_genre_db = call_db_worker_sync(
        "hub_full_context",
        HUB_GENRE_QUERIES,
        {"search_keyword": state["search_keyword"]},
    )
    prepared_hub_genre_db = json.dumps(_prepare_result_for_prompt(json.loads(hub_genre_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "search_keyword": state["search_keyword"],
            "entry_akas_notes": state.get("entry_akas_notes", ""),
            "hub_movie_person_notes": state.get("hub_movie_person_notes", ""),
        },
        [prepared_hub_genre_db],
    )
    resp = invoke_llm(model_b, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["hub_full_context"], resp.content)
    return {"hub_full_context": parsed.get("hub_full_context", "")}


def tail_focus(state: State, model_c: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Tail: from hub_full_context, produce focused viewing guidance\n"
        "(e.g., what to watch first, how to explore related titles or\n"
        "people) tailored to the user_query.\n"
        "Respond ONLY with:\n"
        '{\n  "tail_focus_notes": "string"\n}'
    )
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "hub_full_context": state.get("hub_full_context", "")},
        [],
    )
    resp = invoke_llm(model_c, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["tail_focus_notes"], resp.content)
    return {"tail_focus_notes": parsed.get("tail_focus_notes", "")}


def final_answer(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Final answer assistant.\n"
        "Using user_query, hub_full_context, and tail_focus_notes,\n"
        "produce a clear and concise final_answer that suggests how the\n"
        "user should explore movies and people related to the\n"
        "search_keyword.\n"
        "You may reference imdb_final_sample_titles as generic background\n"
        "but keep the answer grounded in the earlier context.\n"
        "Respond ONLY with:\n"
        '{\n  "final_answer": "string"\n}'
    )
    final_titles_db = call_db_worker_sync(
        "final_answer",
        FINAL_SAMPLE_QUERIES,
        {"search_keyword": state["search_keyword"]},
    )
    prepared_final_titles_db = json.dumps(_prepare_result_for_prompt(json.loads(final_titles_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "search_keyword": state["search_keyword"],
            "hub_full_context": state.get("hub_full_context", ""),
            "tail_focus_notes": state.get("tail_focus_notes", ""),
        },
        [prepared_final_titles_db],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["final_answer"], resp.content)
    return {"final_answer": parsed.get("final_answer", "")}


def build_graph(model_a: ChatOpenAI, model_b: ChatOpenAI, model_c: ChatOpenAI) -> StateGraph[State]:
    g = StateGraph(State)
    g.add_node("keyword_planner", lambda s: kw_planner(s, model_a))
    g.add_node("entry_movie", lambda s: entry_movie(s, model_b))
    g.add_node("entry_person", lambda s: entry_person(s, model_a))
    g.add_node("entry_akas", lambda s: entry_akas(s, model_c))
    g.add_node("hub_movie_person", lambda s: hub_movie_person(s, model_a))
    g.add_node("hub_full_context", lambda s: hub_full_context(s, model_b))
    g.add_node("tail_focus", lambda s: tail_focus(s, model_c))
    g.add_node("final_answer", lambda s: final_answer(s, model_a))

    g.add_edge(START, "keyword_planner")
    g.add_edge("keyword_planner", "entry_movie")
    g.add_edge("keyword_planner", "entry_person")
    g.add_edge("keyword_planner", "entry_akas")
    g.add_edge("entry_movie", "hub_movie_person")
    g.add_edge("entry_person", "hub_movie_person")
    g.add_edge("hub_movie_person", "hub_full_context")
    g.add_edge("entry_akas", "hub_full_context")
    g.add_edge("hub_full_context", "tail_focus")
    g.add_edge("hub_full_context", "final_answer")
    g.add_edge("tail_focus", "final_answer")
    g.add_edge("final_answer", END)
    return g


def load_queries(path: Path, limit: int | None = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    queries: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"):
                continue
            queries.append(q)
            if limit is not None and len(queries) >= limit:
                break
    return queries


async def process_one_query(query: str, models: Dict[str, ChatOpenAI], workflow) -> tuple[str, State]:
    full_state: State = {"user_query": query}
    final_state: State = await asyncio.get_event_loop().run_in_executor(None, workflow.invoke, full_state)
    return query, final_state


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="IMDb diamond HTTP LangGraph workflow (Halo-aligned prompts)."
    )
    parser.add_argument("--query", type=str, default=None, help="单条用户查询；若提供则忽略文件。")
    parser.add_argument("--file", type=str, default="data/imdb_input.txt", help="批量查询文件，每行一条。")
    parser.add_argument("--limit", type=int, default=None, help="最多读取前 N 条（可选）。")
    parser.add_argument("--count", type=int, default=None, help="仅处理前 count 条（早于 limit 截断）。")
    args = parser.parse_args(argv)

    metrics = BaselineMetrics()
    models = {
        "a": make_model(MODEL_A, BASE_A),
        "b": make_model(MODEL_B, BASE_B),
        "c": make_model(MODEL_C, BASE_C),
    }

    graph = build_graph(models["a"], models["b"], models["c"]).compile()

    if args.query:
        queries = [args.query.strip()]
    else:
        path = Path(args.file)
        queries = load_queries(path, args.limit)
        if args.count is not None:
            queries = queries[: args.count]
        if not queries:
            print("No queries found.")
            return 1
    metrics.start(run_name="imdb_diamond_http_langgraph", total_queries=len(queries))
    metrics.add_metadata("runner", "langgraph")
    metrics.add_metadata("input_source", "inline" if args.query else args.file)
    if args.limit is not None:
        metrics.add_metadata("limit", args.limit)
    if args.count is not None:
        metrics.add_metadata("count", args.count)

    async def runner():
        total = len(queries)

        async def timed_task(idx: int, query: str):
            start = time.perf_counter()
            result = await process_one_query(query, models, graph)
            elapsed = time.perf_counter() - start
            return idx, *result, elapsed

        tasks = [asyncio.create_task(timed_task(idx, q)) for idx, q in enumerate(queries, 1)]
        for coro in asyncio.as_completed(tasks):
            idx, q, final_state, elapsed = await coro
            metrics.record_query(elapsed)
            print(f"\n===== Query #{idx}/{total} =====\n{q}")
            print("\n=== Final Answer ===")
            print(final_state.get("final_answer", ""))
            print("\n--- Tail Focus ---")
            print(final_state.get("tail_focus_notes", ""))
            print("\n" + "=" * 60)

    set_global_metrics(metrics)
    asyncio.run(runner())
    set_global_metrics(None)
    metrics.finish()
    print("\n" + metrics.format_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
