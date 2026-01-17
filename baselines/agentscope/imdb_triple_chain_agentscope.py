from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import requests
from agentscope.model import OpenAIChatModel
from baselines.metrics import BaselineMetrics
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
LLM_TIMEOUT = float(os.getenv("HALO_LLM_TIMEOUT", "1200"))
MAX_FIELD_CHARS = 2000
MAX_ROWS = 5

def make_model(model_name: str, base_url: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=model_name,
        api_key=API_KEY,
        client_kwargs={"base_url": base_url, "timeout": LLM_TIMEOUT},
        stream=False,
        generate_kwargs={"temperature": 0.2, "top_p": 0.9, "max_tokens": 1024},
    )

async def chat(model: OpenAIChatModel, messages: List[Dict[str, Any]]) -> str:
    response = await model(messages)
    text_parts: List[str] = []
    for block in response.content:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "thinking":
            text_parts.append(block.get("thinking", ""))
        else:
            text_parts.append(str(block))
    return "\n".join(text_parts).strip()

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

def call_db_worker_sync(node_id: str, queries: List[Dict[str, Any]], context: Mapping[str, Any]) -> str:
    payload = {"node_id": node_id, "queries": queries, "contexts": [dict(context)]}
    resp = requests.post(f"{DB_WORKER_URL}/execute_batch", json=payload, timeout=DB_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    outputs = data.get("outputs") or [{}]
    return json.dumps(outputs[0], ensure_ascii=False)

async def call_db_worker(node_id: str, queries: List[Dict[str, Any]], context: Mapping[str, Any]) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, call_db_worker_sync, node_id, queries, context)

# DB queries mirroring templates/imdb_triple_chain.yaml
MOVIE_OVERVIEW_QUERIES = [
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

FINAL_SAMPLE_QUERIES = []

async def run_one_query(user_query: str, models: Dict[str, OpenAIChatModel]) -> str:
    # keyword planner
    kw_system = (
        "You are a keyword planning assistant for an IMDb-like database.\n"
        "Given user_query, produce a concise search_keyword suitable for ILIKE\n"
        "on titles, names, and related text fields.\n"
        "Respond ONLY with:\n"
        '{\n  "search_keyword": "string"\n}'
    )
    kw_prompt = build_halo_prompt(None, {"user_query": user_query}, [])
    kw_resp = await chat(models["a"], [{"role": "system", "content": kw_system}, {"role": "user", "content": kw_prompt}])
    search_keyword = extract_outputs(["search_keyword"], kw_resp).get("search_keyword", "")

    # small_sanity_check_qwen32b (independent of search_keyword)
    sanity_system = (
        "You are a lightweight sanity checker.\n"
        "Rewrite user_query into a shorter, unambiguous canonical_query.\n"
        "Respond ONLY with:\n"
        '{\n  "canonical_query": "string"\n}'
    )
    sanity_prompt = build_halo_prompt(None, {"user_query": user_query}, [])
    sanity_resp = await chat(
        models["c"], [{"role": "system", "content": sanity_system}, {"role": "user", "content": sanity_prompt}]
    )
    canonical_query = extract_outputs(["canonical_query"], sanity_resp).get("canonical_query", "")

    # movie_overview_qwen32b
    movie_overview_system = (
        "Movie chain: overview.\n"
        "Use rows from imdb_movie_overview_main\n"
        "to identify candidate titles and describe the overall landscape.\n"
        "Respond ONLY with:\n"
        '{\n  "movie_overview": "string"\n}'
    )
    movie_db = await call_db_worker("movie_overview_qwen32b", MOVIE_OVERVIEW_QUERIES, {"search_keyword": search_keyword})
    movie_db = json.dumps(_prepare_result_for_prompt(json.loads(movie_db)), ensure_ascii=False)
    movie_overview_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "search_keyword": search_keyword},
        [movie_db],
    )
    movie_overview_resp = await chat(
        models["c"],
        [{"role": "system", "content": movie_overview_system}, {"role": "user", "content": movie_overview_prompt}],
    )
    movie_overview = extract_outputs(["movie_overview"], movie_overview_resp).get("movie_overview", "")

    # movie_final_notes_qwen32b
    movie_notes_system = (
        "Movie chain: final notes.\n"
        "Based on movie_overview, highlight 3–5 representative titles and\n"
        "explain why they matter for the user_query.\n"
        "Respond ONLY with:\n"
        '{\n  "movie_final_notes": "string"\n}'
    )
    movie_notes_prompt = build_halo_prompt(None, {"user_query": user_query, "movie_overview": movie_overview}, [])
    movie_notes_resp = await chat(
        models["c"], [{"role": "system", "content": movie_notes_system}, {"role": "user", "content": movie_notes_prompt}]
    )
    movie_final_notes = extract_outputs(["movie_final_notes"], movie_notes_resp).get("movie_final_notes", "")

    # people_overview_openai20b
    people_overview_system = (
        "People chain: overview.\n"
        "Use imdb_people_overview_main\n"
        "to summarize which actors/directors/writers are most relevant to\n"
        "the search_keyword.\n"
        "Respond ONLY with:\n"
        '{\n  "people_overview": "string"\n}'
    )
    people_db = await call_db_worker("people_overview_openai20b", PEOPLE_QUERIES, {"search_keyword": search_keyword})
    people_db = json.dumps(_prepare_result_for_prompt(json.loads(people_db)), ensure_ascii=False)
    people_overview_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "search_keyword": search_keyword},
        [people_db],
    )
    people_overview_resp = await chat(
        models["a"],
        [{"role": "system", "content": people_overview_system}, {"role": "user", "content": people_overview_prompt}],
    )
    people_overview = extract_outputs(["people_overview"], people_overview_resp).get("people_overview", "")

    # people_final_notes_openai20b
    people_notes_system = (
        "People chain: final notes.\n"
        "From people_overview, explain which people are most central to\n"
        "answering the user_query, and why.\n"
        "Respond ONLY with:\n"
        '{\n  "people_final_notes": "string"\n}'
    )
    people_notes_prompt = build_halo_prompt(None, {"user_query": user_query, "people_overview": people_overview}, [])
    people_notes_resp = await chat(
        models["a"],
        [{"role": "system", "content": people_notes_system}, {"role": "user", "content": people_notes_prompt}],
    )
    people_final_notes = extract_outputs(["people_final_notes"], people_notes_resp).get("people_final_notes", "")

    # crew_overview_qwen14b
    crew_overview_system = (
        "Crew chain: overview.\n"
        "Use imdb_crew_by_title_keyword\n"
        "to summarize typical directors/writers and crew patterns relevant\n"
        "to the search_keyword.\n"
        "Respond ONLY with:\n"
        '{\n  "crew_overview": "string"\n}'
    )
    crew_db = await call_db_worker("crew_overview_qwen14b", CREW_QUERIES, {"search_keyword": search_keyword})
    crew_db = json.dumps(_prepare_result_for_prompt(json.loads(crew_db)), ensure_ascii=False)
    crew_overview_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "search_keyword": search_keyword},
        [crew_db],
    )
    crew_overview_resp = await chat(
        models["b"],
        [{"role": "system", "content": crew_overview_system}, {"role": "user", "content": crew_overview_prompt}],
    )
    crew_overview = extract_outputs(["crew_overview"], crew_overview_resp).get("crew_overview", "")

    # crew_insights_qwen14b
    crew_insights_system = (
        "Crew chain: mid-level insights.\n"
        "Turn crew_overview into a few concrete, human-readable insights\n"
        "about directing/writing styles that matter for the user_query.\n"
        "Respond ONLY with:\n"
        '{\n  "crew_insights": "string"\n}'
    )
    crew_insights_prompt = build_halo_prompt(None, {"user_query": user_query, "crew_overview": crew_overview}, [])
    crew_insights_resp = await chat(
        models["b"],
        [{"role": "system", "content": crew_insights_system}, {"role": "user", "content": crew_insights_prompt}],
    )
    crew_insights = extract_outputs(["crew_insights"], crew_insights_resp).get("crew_insights", "")

    # crew_final_notes_qwen14b
    crew_final_system = (
        "Crew chain: final notes.\n"
        "Produce crew-centric final notes that downstream nodes can\n"
        "consume, focusing on how directors/writers shape interpretation\n"
        "of the titles.\n"
        "Respond ONLY with:\n"
        '{\n  "crew_final_notes": "string"\n}'
    )
    crew_final_prompt = build_halo_prompt(None, {"user_query": user_query, "crew_insights": crew_insights}, [])
    crew_final_resp = await chat(
        models["b"],
        [{"role": "system", "content": crew_final_system}, {"role": "user", "content": crew_final_prompt}],
    )
    crew_final_notes = extract_outputs(["crew_final_notes"], crew_final_resp).get("crew_final_notes", "")

    # final_answer_qwen14b
    final_system = (
        "Final answer assistant.\n"
        "Combine movie_final_notes, people_final_notes, crew_final_notes,\n"
        "and canonical_query into a single clear, grounded answer.\n"
        "Do not contradict the earlier notes.\n"
        "Respond ONLY with:\n"
        '{\n  "final_answer": "string"\n}'
    )
    final_prompt = build_halo_prompt(
        None,
        {
            "user_query": user_query,
            "search_keyword": search_keyword,
            "canonical_query": canonical_query,
            "movie_final_notes": movie_final_notes,
            "people_final_notes": people_final_notes,
            "crew_final_notes": crew_final_notes,
        },
        [],
    )
    final_resp = await chat(models["b"], [{"role": "system", "content": final_system}, {"role": "user", "content": final_prompt}])
    final_answer = extract_outputs(["final_answer"], final_resp).get("final_answer", "")
    return str(final_answer)

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

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IMDb triple-chain workflow (Agentscope, Halo-aligned prompts).")
    parser.add_argument("--query", type=str, default=None, help="单条用户查询；若提供则忽略文件。")
    parser.add_argument("--file", type=str, default="data/imdb_input.txt", help="批量查询文件，每行一条。")
    parser.add_argument("--limit", type=int, default=None, help="最多读取前 N 条（可选）。")
    parser.add_argument("--count", type=int, default=None, help="仅处理前 count 条（早于 limit 截断）。")
    args = parser.parse_args(argv)

    models = {
        "a": make_model(MODEL_A, BASE_A),
        "b": make_model(MODEL_B, BASE_B),
        "c": make_model(MODEL_C, BASE_C),
    }
    metrics = BaselineMetrics()

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

    metrics.start()

    async def worker(q: str, idx: int) -> tuple[int, str, str, float]:
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        ans = await run_one_query(q, models)
        dur = loop.time() - t0
        metrics.record_query(dur)
        return idx, q, ans, dur

    async def run_all() -> int:
        total = len(queries)
        tasks = [asyncio.create_task(worker(q, idx)) for idx, q in enumerate(queries, 1)]
        for coro in asyncio.as_completed(tasks):
            idx, q, ans, dur = await coro
            print(f"\n===== Query #{idx}/{total} =====")
            print(q)
            print("\n=== Final Answer ===")
            print(ans)
            print(f"\n(latency: {dur:.3f}s)")
            print("\n" + "=" * 60)
        metrics.finish()
        print("\n" + metrics.format_summary())
        return 0

    return asyncio.run(run_all())

if __name__ == "__main__":
    sys.exit(main())
