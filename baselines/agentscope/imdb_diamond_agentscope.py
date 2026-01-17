from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import requests
from agentscope.model import OpenAIChatModel
from baselines.metrics import BaselineMetrics
from halo_dev.utils import render_template
from openai import APITimeoutError

# Defaults aligned with Halo
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
MAX_FIELD_CHARS = 2048
MAX_ROWS = 5
METRICS: BaselineMetrics | None = None


def make_model(model_name: str, base_url: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=model_name,
        api_key=API_KEY,
        client_kwargs={"base_url": base_url, "timeout": LLM_TIMEOUT},
        stream=False,
        generate_kwargs={"temperature": 0.2, "top_p": 0.9, "max_tokens": 1024},
    )


async def chat(model: OpenAIChatModel, messages: List[Dict[str, Any]]) -> str:
    metrics = METRICS
    last_err: Exception | None = None
    for attempt in range(3):
        start = time.perf_counter()
        try:
            response = await model(messages)
            duration = time.perf_counter() - start
            if metrics:
                metrics.record_llm(duration, call_count=1, prompt_count=1)
            text_parts: List[str] = []
            for block in response.content:
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    text_parts.append(block.get("thinking", ""))
                else:
                    text_parts.append(str(block))
            return "\n".join(text_parts).strip()
        except APITimeoutError as err:
            last_err = err
            await asyncio.sleep(1 + attempt)
            continue
    raise last_err if last_err else RuntimeError("LLM call failed without exception")


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


def parse_json_field(payload: str, key: str) -> str:
    parsed = extract_outputs([key], payload)
    return str(parsed.get(key, payload)).strip()


def call_db_worker_sync(node_id: str, queries: List[Dict[str, Any]], context: Mapping[str, Any]) -> str:
    payload = {"node_id": node_id, "queries": queries, "contexts": [dict(context)]}
    start = time.perf_counter()
    resp = requests.post(f"{DB_WORKER_URL}/execute_batch", json=payload, timeout=DB_TIMEOUT)
    resp.raise_for_status()
    duration = time.perf_counter() - start
    metrics = METRICS
    if metrics:
        metrics.record_db(duration, count=len(queries))
    data = resp.json()
    outputs = data.get("outputs") or [{}]
    return json.dumps(outputs[0], ensure_ascii=False)


async def call_db_worker(node_id: str, queries: List[Dict[str, Any]], context: Mapping[str, Any]) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, call_db_worker_sync, node_id, queries, context)


# DB query definitions (from templates/imdb_diamond.yaml)
ENTRY_MOVIE_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "imdb_entry_movie_by_title",
        "sql": """
            SELECT
              b.tconst,
              b.primary_title,
              b.original_title,
              b.start_year,
              b.title_type,
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
            LIMIT 25;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    },
    {
        "name": "imdb_entry_movie_recent",
        "sql": """
            SELECT
              b.tconst,
              b.primary_title,
              b.start_year,
              b.genres,
              r.average_rating,
              r.num_votes
            FROM title_basics AS b
            LEFT JOIN title_ratings AS r
              ON r.tconst = b.tconst
            WHERE
              b.title_type IN ('movie','tvMovie')
              AND b.start_year >= 2000
              AND (
                b.primary_title ILIKE '%%' || :keyword || '%%'
                OR b.original_title ILIKE '%%' || :keyword || '%%'
              )
            ORDER BY COALESCE(r.num_votes, 0) DESC
            LIMIT 15;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    },
]

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


async def run_one_query(user_query: str, models: Dict[str, OpenAIChatModel]) -> str:
    # keyword planner
    kw_system = (
        "You are the keyword planner.\n"
        "From user_query, generate a concise search_keyword suitable for\n"
        "ILIKE filters on movie titles, person names, and AKA titles.\n"
        "Respond ONLY with:\n"
        '{\n  "search_keyword": "string"\n}'
    )
    kw_prompt = build_halo_prompt(None, {"user_query": user_query}, [])
    kw_resp = await chat(models["a"], [{"role": "system", "content": kw_system}, {"role": "user", "content": kw_prompt}])
    kw_parsed = extract_outputs(["search_keyword"], kw_resp)
    search_keyword = kw_parsed.get("search_keyword", "")

    # entry nodes
    entry_movie_system = (
        "Entry A (movie-centric).\n"
        "Use imdb_entry_movie_by_title and imdb_entry_movie_recent to\n"
        "summarize representative candidate titles related to the\n"
        "search_keyword.\n"
        "Respond ONLY with:\n"
        '{\n  "entry_movie_notes": "string"\n}'
    )
    entry_movie_db = await call_db_worker("entry_movie", ENTRY_MOVIE_QUERIES, {"search_keyword": search_keyword})
    entry_movie_db = json.dumps(_prepare_result_for_prompt(json.loads(entry_movie_db)), ensure_ascii=False)
    entry_movie_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "search_keyword": search_keyword},
        [entry_movie_db],
    )
    entry_movie_resp = await chat(models["b"], [{"role": "system", "content": entry_movie_system}, {"role": "user", "content": entry_movie_prompt}])
    entry_movie_notes = extract_outputs(["entry_movie_notes"], entry_movie_resp).get("entry_movie_notes", "")

    entry_person_system = (
        "Entry B (people-centric).\n"
        "Use imdb_entry_person_by_name and imdb_entry_person_by_profession\n"
        "to identify people relevant to the search_keyword and summarize\n"
        "why they matter.\n"
        "Respond ONLY with:\n"
        '{\n  "entry_person_notes": "string"\n}'
    )
    entry_person_db = await call_db_worker("entry_person", ENTRY_PERSON_QUERIES, {"search_keyword": search_keyword})
    entry_person_db = json.dumps(_prepare_result_for_prompt(json.loads(entry_person_db)), ensure_ascii=False)
    entry_person_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "search_keyword": search_keyword},
        [entry_person_db],
    )
    entry_person_resp = await chat(models["a"], [{"role": "system", "content": entry_person_system}, {"role": "user", "content": entry_person_prompt}])
    entry_person_notes = extract_outputs(["entry_person_notes"], entry_person_resp).get("entry_person_notes", "")

    entry_akas_system = (
        "Entry C (AKA / international titles).\n"
        "Use imdb_entry_akas_by_title and imdb_entry_akas_by_region to\n"
        "summarize alternative and international titles related to the\n"
        "search_keyword.\n"
        "Respond ONLY with:\n"
        '{\n  "entry_akas_notes": "string"\n}'
    )
    entry_akas_db = await call_db_worker("entry_akas", ENTRY_AKAS_QUERIES, {"search_keyword": search_keyword})
    entry_akas_db = json.dumps(_prepare_result_for_prompt(json.loads(entry_akas_db)), ensure_ascii=False)
    entry_akas_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "search_keyword": search_keyword},
        [entry_akas_db],
    )
    entry_akas_resp = await chat(models["c"], [{"role": "system", "content": entry_akas_system}, {"role": "user", "content": entry_akas_prompt}])
    entry_akas_notes = extract_outputs(["entry_akas_notes"], entry_akas_resp).get("entry_akas_notes", "")

    # hub nodes
    hub_movie_person_db = await call_db_worker("hub_movie_person", HUB_MOVIE_PERSON_QUERIES, {"search_keyword": search_keyword})
    hub_movie_person_db = json.dumps(_prepare_result_for_prompt(json.loads(hub_movie_person_db)), ensure_ascii=False)
    hub_movie_person_system = (
        "Hub1: fuse entry_movie_notes and entry_person_notes into a joint\n"
        "movie-person perspective on the user_query and search_keyword.\n"
        "You may reference simple aggregates from imdb_hub_movie_person_stats.\n"
        "Respond ONLY with:\n"
        '{\n  "hub_movie_person_notes": "string"\n}'
    )
    hub_movie_person_prompt = build_halo_prompt(
        None,
        {
            "user_query": user_query,
            "search_keyword": search_keyword,
            "entry_movie_notes": entry_movie_notes,
            "entry_person_notes": entry_person_notes,
        },
        [hub_movie_person_db],
    )
    hub_movie_person_resp = await chat(models["a"], [{"role": "system", "content": hub_movie_person_system}, {"role": "user", "content": hub_movie_person_prompt}])
    hub_movie_person_notes = extract_outputs(["hub_movie_person_notes"], hub_movie_person_resp).get("hub_movie_person_notes", "")

    hub_genre_db = await call_db_worker("hub_full_context", HUB_GENRE_QUERIES, {"search_keyword": search_keyword})
    hub_genre_db = json.dumps(_prepare_result_for_prompt(json.loads(hub_genre_db)), ensure_ascii=False)
    hub_full_context_system = (
        "Hub2: combine hub_movie_person_notes with entry_akas_notes into\n"
        "a single full-context view of the candidate space around\n"
        "search_keyword.\n"
        "You may optionally reference imdb_hub_genre_stats_filtered for\n"
        "genre distribution cues.\n"
        "Respond ONLY with:\n"
        '{\n  "hub_full_context": "string"\n}'
    )
    hub_full_context_prompt = build_halo_prompt(
        None,
        {
            "user_query": user_query,
            "search_keyword": search_keyword,
            "entry_akas_notes": entry_akas_notes,
            "hub_movie_person_notes": hub_movie_person_notes,
        },
        [hub_genre_db],
    )
    hub_full_context_resp = await chat(models["b"], [{"role": "system", "content": hub_full_context_system}, {"role": "user", "content": hub_full_context_prompt}])
    hub_full_context = extract_outputs(["hub_full_context"], hub_full_context_resp).get("hub_full_context", "")

    # tail and final
    tail_focus_system = (
        "Tail: from hub_full_context, produce focused viewing guidance\n"
        "(e.g., what to watch first, how to explore related titles or\n"
        "people) tailored to the user_query.\n"
        "Respond ONLY with:\n"
        '{\n  "tail_focus_notes": "string"\n}'
    )
    tail_focus_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "hub_full_context": hub_full_context},
        [],
    )
    tail_focus_resp = await chat(models["c"], [{"role": "system", "content": tail_focus_system}, {"role": "user", "content": tail_focus_prompt}])
    tail_focus_notes = extract_outputs(["tail_focus_notes"], tail_focus_resp).get("tail_focus_notes", "")

    final_titles_db = await call_db_worker("final_answer", FINAL_SAMPLE_QUERIES, {"search_keyword": search_keyword})
    final_titles_db = json.dumps(_prepare_result_for_prompt(json.loads(final_titles_db)), ensure_ascii=False)
    final_system = (
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
    final_prompt = build_halo_prompt(
        None,
        {
            "user_query": user_query,
            "search_keyword": search_keyword,
            "hub_full_context": hub_full_context,
            "tail_focus_notes": tail_focus_notes,
        },
        [final_titles_db],
    )
    final_resp = await chat(models["a"], [{"role": "system", "content": final_system}, {"role": "user", "content": final_prompt}])
    final_answer = extract_outputs(["final_answer"], final_resp).get("final_answer", "")
    return final_answer


def load_queries(path: str, limit: int | None = None) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    queries: List[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            queries.append(text)
            if limit is not None and len(queries) >= limit:
                break
    return queries


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IMDb diamond workflow with OpenAIChatModel + DB worker (Halo-aligned prompts).")
    parser.add_argument("--query", type=str, default=None, help="单条用户查询；若提供，将忽略文件。")
    parser.add_argument("--file", type=str, default="data/imdb_input.txt", help="批量查询文件（每行一条，空行跳过）。")
    parser.add_argument("--limit", type=int, default=None, help="最多处理前 N 条（可选）。")
    parser.add_argument("--count", type=int, default=None, help="仅处理前 count 条（早于 limit 执行，主要用于小批量调试）。")
    args = parser.parse_args(argv)

    models = {
        "a": make_model(MODEL_A, BASE_A),
        "b": make_model(MODEL_B, BASE_B),
        "c": make_model(MODEL_C, BASE_C),
    }
    metrics = BaselineMetrics()
    global METRICS
    METRICS = metrics

    if args.query:
        queries = [args.query.strip()]
    else:
        queries = load_queries(args.file, args.limit)
        if not queries:
            print("No queries found. Exit.")
            return 1
        if args.count is not None:
            queries = queries[: args.count]

    metrics.start(run_name="imdb", total_queries=len(queries))
    metrics.add_metadata("runner", "agentscope_imdb_diamond")
    metrics.add_metadata("input_file", args.file)
    metrics.add_metadata("sample_count", len(queries))
    if args.limit is not None:
        metrics.add_metadata("limit", args.limit)
    if args.count is not None:
        metrics.add_metadata("count", args.count)

    async def worker(q: str, idx: int) -> tuple[int, str, str, float]:
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        final_answer = await run_one_query(q, models)
        duration = loop.time() - t0
        metrics.record_query(duration)
        return idx, q, final_answer, duration

    async def run_all() -> int:
        total = len(queries)
        tasks = [asyncio.create_task(worker(q, idx)) for idx, q in enumerate(queries, 1)]
        for coro in asyncio.as_completed(tasks):
            idx, q, ans, duration = await coro
            print(f"\n===== Query #{idx}/{total} =====")
            print(q)
            print("\n=== Final Answer ===")
            print(ans)
            print(f"\n(latency: {duration:.3f}s)")
            print("\n" + "=" * 60)
        metrics.finish()
        print("\n" + metrics.format_summary())
        return 0

    return asyncio.run(run_all())


if __name__ == "__main__":
    sys.exit(main())
