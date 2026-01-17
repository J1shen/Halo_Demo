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
from halo_dev.utils import render_template

API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
MODEL_A = os.getenv("IMDB_MODEL_A", "openai/gpt-oss-20b")
MODEL_B = os.getenv("IMDB_MODEL_B", "Qwen/Qwen3-14B")
MODEL_C = os.getenv("IMDB_MODEL_C", "Qwen/Qwen3-32B")
BASE_A = os.getenv("MODEL_A_BASE_URL", "http://localhost:9101/v1")
BASE_B = os.getenv("MODEL_B_BASE_URL", "http://localhost:9102/v1")
BASE_C = os.getenv("MODEL_C_BASE_URL", "http://localhost:9103/v1")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))
MAX_FIELD_CHARS = 2000
MAX_ROWS = 5

class State(TypedDict, total=False):
    user_query: str
    search_keyword: str
    movie_overview: str
    movie_final_notes: str
    people_overview: str
    people_final_notes: str
    crew_overview: str
    crew_insights: str
    crew_final_notes: str
    canonical_query: str
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


def invoke_llm(model: ChatOpenAI, messages: List[Mapping[str, str]]):
    start = time.perf_counter()
    resp = model.invoke(messages)
    metrics = get_global_metrics()
    if metrics:
        metrics.record_llm(time.perf_counter() - start, call_count=1, prompt_count=1)
    return resp

# DB queries (from templates/imdb_triple_chain.yaml)
MOVIE_QUERIES = [
    {
        "name": "imdb_movie_overview_main",
        "sql": """
            SELECT
              b.tconst,
              b.primary_title,
              b.original_title,
              b.start_year,
              b.genres,
              r.average_rating,
              r.num_votes
            FROM title_basics AS b
            LEFT JOIN title_ratings AS r
              ON r.tconst = b.tconst
            WHERE
              b.title_type IN ('movie','tvMovie','tvSeries')
              AND (
                b.primary_title     ILIKE '%%' || :keyword || '%%'
                OR b.original_title ILIKE '%%' || :keyword || '%%'
              )
            ORDER BY COALESCE(r.num_votes,0)     DESC,
                     COALESCE(r.average_rating,0) DESC,
                     b.start_year DESC NULLS LAST
            LIMIT 40;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    },
]

PEOPLE_QUERIES = [
    {
        "name": "imdb_people_overview_main",
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
            LIMIT 40;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    },
]

CREW_QUERIES = [
    {
        "name": "imdb_crew_by_title_keyword",
        "sql": """
            SELECT
              b.tconst,
              b.primary_title,
              b.start_year,
              c.directors,
              c.writers
            FROM title_basics AS b
            JOIN title_crew AS c
              ON c.tconst = b.tconst
            WHERE
              b.title_type IN ('movie','tvMovie','tvSeries')
              AND (
                b.primary_title     ILIKE '%%' || :keyword || '%%'
                OR b.original_title ILIKE '%%' || :keyword || '%%'
              )
            ORDER BY b.start_year DESC NULLS LAST
            LIMIT 40;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    },
]

FINAL_QUERIES = []

# LangGraph node functions
def kw_planner(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You are a keyword planning assistant for an IMDb-like database.\n"
        "Given user_query, produce a concise search_keyword suitable for ILIKE\n"
        "on titles, names, and related text fields.\n"
        "Respond ONLY with:\n"
        '{\n  "search_keyword": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"]}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["search_keyword"], resp.content)
    return {"search_keyword": parsed.get("search_keyword", "")}

def node_movie_overview(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Movie chain: overview.\n"
        "Use rows from imdb_movie_overview_main and imdb_movie_overview_recent\n"
        "to identify candidate titles and describe the overall landscape.\n"
        "Respond ONLY with:\n"
        '{\n  "movie_overview": "string"\n}'
    )
    movie_db = call_db_worker_sync(
        "movie_overview_qwen32b",
        MOVIE_QUERIES,
        {"search_keyword": state["search_keyword"]},
    )
    prepared_movie_db = json.dumps(_prepare_result_for_prompt(json.loads(movie_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "search_keyword": state["search_keyword"]},
        [prepared_movie_db],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["movie_overview"], resp.content)
    return {"movie_overview": parsed.get("movie_overview", "")}

def node_movie_final(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Movie chain: final notes.\n"
        "Based on movie_overview, highlight 3–5 representative titles and\n"
        "explain why they matter for the user_query.\n"
        "Respond ONLY with:\n"
        '{\n  "movie_final_notes": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"], "movie_overview": state.get("movie_overview", "")}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["movie_final_notes"], resp.content)
    return {"movie_final_notes": parsed.get("movie_final_notes", "")}

def node_people_overview(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "People chain: overview.\n"
        "Use imdb_people_overview_main to summarize which actors/directors/\n"
        "writers are most relevant to the search_keyword.\n"
        "Respond ONLY with:\n"
        '{\n  "people_overview": "string"\n}'
    )
    people_db = call_db_worker_sync(
        "people_overview_openai20b",
        PEOPLE_QUERIES,
        {"search_keyword": state["search_keyword"]},
    )
    prepared_people_db = json.dumps(_prepare_result_for_prompt(json.loads(people_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "search_keyword": state["search_keyword"]},
        [prepared_people_db],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["people_overview"], resp.content)
    return {"people_overview": parsed.get("people_overview", "")}

def node_people_final(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "People chain: final notes.\n"
        "From people_overview, explain which people are most central to\n"
        "answering the user_query, and why.\n"
        "Respond ONLY with:\n"
        '{\n  "people_final_notes": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"], "people_overview": state.get("people_overview", "")}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["people_final_notes"], resp.content)
    return {"people_final_notes": parsed.get("people_final_notes", "")}

def node_crew_overview(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Crew chain: overview.\n"
        "Use imdb_crew_by_title_keyword to summarize typical directors/\n"
        "writers and crew patterns relevant to the search_keyword.\n"
        "Respond ONLY with:\n"
        '{\n  "crew_overview": "string"\n}'
    )
    crew_db = call_db_worker_sync(
        "crew_overview_qwen14b",
        CREW_QUERIES,
        {"search_keyword": state["search_keyword"]},
    )
    prepared_crew_db = json.dumps(_prepare_result_for_prompt(json.loads(crew_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "search_keyword": state["search_keyword"]},
        [prepared_crew_db],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["crew_overview"], resp.content)
    return {"crew_overview": parsed.get("crew_overview", "")}

def node_crew_insights(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Crew chain: mid-level insights.\n"
        "Turn crew_overview into a few concrete, human-readable insights\n"
        "about directing/writing styles that matter for the user_query.\n"
        "Respond ONLY with:\n"
        '{\n  "crew_insights": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"], "crew_overview": state.get("crew_overview", "")}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["crew_insights"], resp.content)
    return {"crew_insights": parsed.get("crew_insights", "")}

def node_crew_final(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Crew chain: final notes.\n"
        "Produce crew-centric final notes that downstream nodes can\n"
        "consume, focusing on how directors/writers shape interpretation\n"
        "of the titles.\n"
        "Respond ONLY with:\n"
        '{\n  "crew_final_notes": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"], "crew_insights": state.get("crew_insights", "")}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["crew_final_notes"], resp.content)
    return {"crew_final_notes": parsed.get("crew_final_notes", "")}

def node_sanity(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You are a lightweight sanity checker.\n"
        "Rewrite user_query into a shorter, unambiguous canonical_query.\n"
        "Respond ONLY with:\n"
        '{\n  "canonical_query": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"]}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["canonical_query"], resp.content)
    return {"canonical_query": parsed.get("canonical_query", "")}

def node_final(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Final answer assistant.\n"
        "Combine movie_final_notes, people_final_notes, crew_final_notes,\n"
        "and canonical_query into a single clear, grounded answer.\n"
        "Respond ONLY with:\n"
        '{\n  "final_answer": "string"\n}'
    )
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "search_keyword": state["search_keyword"],
            "canonical_query": state.get("canonical_query", ""),
            "movie_final_notes": state.get("movie_final_notes", ""),
            "people_final_notes": state.get("people_final_notes", ""),
            "crew_final_notes": state.get("crew_final_notes", ""),
        },
        [],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["final_answer"], resp.content)
    return {"final_answer": parsed.get("final_answer", "")}

def build_graph(models: Dict[str, ChatOpenAI]) -> StateGraph[State]:
    g = StateGraph(State)
    g.add_node("keyword_planner", lambda s: kw_planner(s, models["a"]))
    g.add_node("movie_overview_qwen32b", lambda s: node_movie_overview(s, models["c"]))
    g.add_node("movie_final_notes_qwen32b", lambda s: node_movie_final(s, models["c"]))
    g.add_node("people_overview_openai20b", lambda s: node_people_overview(s, models["a"]))
    g.add_node("people_final_notes_openai20b", lambda s: node_people_final(s, models["a"]))
    g.add_node("crew_overview_qwen14b", lambda s: node_crew_overview(s, models["b"]))
    g.add_node("crew_insights_qwen14b", lambda s: node_crew_insights(s, models["b"]))
    g.add_node("crew_final_notes_qwen14b", lambda s: node_crew_final(s, models["b"]))
    g.add_node("small_sanity_check_qwen32b", lambda s: node_sanity(s, models["c"]))
    g.add_node("final_answer_qwen14b", lambda s: node_final(s, models["b"]))

    g.add_edge(START, "keyword_planner")
    g.add_edge("keyword_planner", "movie_overview_qwen32b")
    g.add_edge("movie_overview_qwen32b", "movie_final_notes_qwen32b")
    g.add_edge("keyword_planner", "people_overview_openai20b")
    g.add_edge("people_overview_openai20b", "people_final_notes_openai20b")
    g.add_edge("keyword_planner", "crew_overview_qwen14b")
    g.add_edge("crew_overview_qwen14b", "crew_insights_qwen14b")
    g.add_edge("crew_insights_qwen14b", "crew_final_notes_qwen14b")
    g.add_edge("keyword_planner", "final_answer_qwen14b")
    g.add_edge("movie_final_notes_qwen32b", "final_answer_qwen14b")
    g.add_edge("people_final_notes_openai20b", "final_answer_qwen14b")
    g.add_edge("crew_final_notes_qwen14b", "final_answer_qwen14b")
    g.add_edge(START, "small_sanity_check_qwen32b")
    g.add_edge("small_sanity_check_qwen32b", "final_answer_qwen14b")
    g.add_edge("final_answer_qwen14b", END)
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

async def process_one_query(query: str, models: Dict[str, ChatOpenAI], graph) -> tuple[str, State]:
    full_state: State = {"user_query": query}
    loop = asyncio.get_event_loop()
    final_state: State = await loop.run_in_executor(None, graph.invoke, full_state)
    return query, final_state

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IMDb triple-chain LangGraph baseline (Halo-aligned).")
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
    graph = build_graph(models).compile()

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
    metrics.start(run_name="imdb_triple_chain_langgraph", total_queries=len(queries))
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
            idx, q, state, elapsed = await coro
            metrics.record_query(elapsed)
            print(f"\n===== Query #{idx}/{total} =====\n{q}")
            print("\n=== Final Answer ===")
            print(state.get("final_answer", ""))
            print("\n" + "=" * 60)

    set_global_metrics(metrics)
    asyncio.run(runner())
    set_global_metrics(None)
    metrics.finish()
    print("\n" + metrics.format_summary())
    return 0

if __name__ == "__main__":
    sys.exit(main())
