"""Baseline Parrot workflow for the IMDb diamond template (template-aligned).

- Mirrors the nodes from templates/imdb_diamond.yaml (imdb_qa_agent_diamond_small_v1).
- Keeps baseline behavior: sequential orchestration in Python (per query).
- Uses the shared DB worker (gunicorn FastAPI) for all SQL.
- Semantic functions use ONLY (input, output) variables (diamond-style).
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests
from parrot import P
from parrot.frontend.pfunc.semantic_variable import SemanticVariable
from baselines.metrics import BaselineMetrics, get_global_metrics, set_global_metrics
from baselines.parrot.parrot_utils import (
    build_halo_prompt,
    extract_outputs,
    guard_input as _guard_input,
    prepare_result_for_prompt as _prepare_result_for_prompt,
    safe_aget as _safe_aget,
)

# ---- Runtime wiring ----
CORE_HTTP_ADDR = os.getenv("PARROT_CORE_HTTP", "http://localhost:9000")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))

MAX_FIELD_CHARS = 2048
MAX_ROWS = 5

# Model routing: must match the template model assignment.
MODEL_A = os.getenv("IMDB_MODEL_A", "openai/gpt-oss-20b")  # keyword_planner, entry_person, hub1, final
MODEL_B = os.getenv("IMDB_MODEL_B", "Qwen/Qwen3-14B")      # entry_movie, hub2
MODEL_C = os.getenv("IMDB_MODEL_C", "Qwen/Qwen3-32B")      # entry_akas, tail

SAMPLING_CONFIG = P.SamplingConfig(
    temperature=0.2,
    top_p=0.9,
    max_gen_length=1024,
)

vm = P.VirtualMachine(core_http_addr=CORE_HTTP_ADDR, mode="debug")

# Pre-bind VM to avoid registration warnings when decorators run at import time.
from parrot.frontend.pfunc.function import BasicFunction as _BF  # noqa: E402
from parrot.frontend.pfunc.semantic_variable import SemanticVariable as _SV  # noqa: E402
_BF._virtual_machine_env = vm
_SV._virtual_machine_env = vm


def load_queries(path: Path, limit: int | None = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    queries: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            queries.append(line)
            if limit is not None and len(queries) >= limit:
                break
    return queries


def _unwrap(val: Any) -> Any:
    if isinstance(val, SemanticVariable):
        if not val.is_registered:
            val.assign_id(vm.register_semantic_variable_handler(val.name))
        if val.content is not None:
            return val.content
        return val.get()
    return val


def _normalize_keyword(val: Any) -> str:
    text = str(_unwrap(val)) if val is not None else ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "search_keyword" in parsed:
            text = str(parsed["search_keyword"])
    except Exception:
        pass
    first = text.split(",")[0].strip()
    return first


# ---- DB query payloads (must match template names/sql/params) ----
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


# ---- Semantic functions (ONLY input/output; system_prompt text aligned to template) ----
@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def keyword_planner(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are the keyword planner.
    From user_query, generate a concise search_keyword suitable for
    ILIKE filters on movie titles, person names, and AKA titles.
    Respond ONLY with:
    {
      "search_keyword": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def entry_movie_qwen14b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Entry A (movie-centric).
    Use imdb_entry_movie_by_title and imdb_entry_movie_recent to
    summarize representative candidate titles related to the
    search_keyword.
    Respond ONLY with:
    {
      "entry_movie_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def entry_person_openai20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Entry B (people-centric).
    Use imdb_entry_person_by_name and imdb_entry_person_by_profession
    to identify people relevant to the search_keyword and summarize
    why they matter.
    Respond ONLY with:
    {
      "entry_person_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def entry_akas_qwen32b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Entry C (AKA / international titles).
    Use imdb_entry_akas_by_title and imdb_entry_akas_by_region to
    summarize alternative and international titles related to the
    search_keyword.
    Respond ONLY with:
    {
      "entry_akas_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def hub_movie_person_openai20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Hub1: fuse entry_movie_notes and entry_person_notes into a joint
    movie-person perspective on the user_query and search_keyword.
    You may reference simple aggregates from imdb_hub_movie_person_stats.
    Respond ONLY with:
    {
      "hub_movie_person_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def hub_full_context_qwen14b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Hub2: combine hub_movie_person_notes with entry_akas_notes into
    a single full-context view of the candidate space around
    search_keyword.
    You may optionally reference imdb_hub_genre_stats_filtered for
    genre distribution cues.
    Respond ONLY with:
    {
      "hub_full_context": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def tail_focus_qwen32b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Tail: from hub_full_context, produce focused viewing guidance
    (e.g., what to watch first, how to explore related titles or
    people) tailored to the user_query.
    Respond ONLY with:
    {
      "tail_focus_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def imdb_final_answer(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Final answer assistant.
    Using user_query, hub_full_context, and tail_focus_notes,
    produce a clear and concise final_answer that suggests how the
    user should explore movies and people related to the
    search_keyword.
    You may reference imdb_final_sample_titles as generic background
    but keep the answer grounded in the earlier context.
    Respond ONLY with:
    {
      "final_answer": "string"
    }

    {{input}}
    {{output}}
    """


async def _call_db(node_id: str, queries: Sequence[Dict[str, Any]], search_keyword: str) -> str:
    """Async DB worker call (template-aligned node_id)."""

    def _normalize_db_payload(payload: Any) -> str:
        if payload is None:
            return "{}"
        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
        except Exception:
            return "{}"

        if isinstance(data, dict):
            data = _prepare_result_for_prompt(data)
            try:
                return json.dumps(data, ensure_ascii=False)
            except Exception:
                return "{}"

        try:
            text = json.dumps(data, ensure_ascii=False)
        except Exception:
            return "{}"
        lines = text.splitlines() if text else []
        if not lines:
            return "{}"
        trimmed = [line[:MAX_FIELD_CHARS] for line in lines[:MAX_ROWS]]
        return "\n".join(trimmed) if trimmed else "{}"

    def _do_call() -> str:
        start = time.perf_counter()
        payload = {
            "node_id": node_id,
            "queries": list(queries),
            "contexts": [{"search_keyword": search_keyword}],
        }
        resp = requests.post(f"{DB_WORKER_URL}/execute_batch", json=payload, timeout=DB_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        stats = data.get("stats") or {}
        metrics = get_global_metrics()
        if metrics:
            db_calls = int(stats.get("db_calls") or 0)
            db_time = float(stats.get("db_time") or 0.0)
            if db_calls <= 0:
                db_calls = len(queries)
            if db_time <= 0:
                db_time = time.perf_counter() - start
            metrics.record_db(db_time, count=db_calls)
        outputs = data.get("outputs") or [{}]
        first = outputs[0] if isinstance(outputs, list) and outputs else outputs
        return _normalize_db_payload(first)

    return await asyncio.to_thread(_do_call)


async def run_one_query_async(user_query: str, idx: int | None = None, total: int | None = None) -> None:
    uq = _guard_input(user_query)

    # ===== 1) keyword_planner =====
    kp_prompt = build_halo_prompt(None, {"user_query": uq}, [])
    kp_in = P.variable("user_query", content=kp_prompt)
    kp_out_var = keyword_planner(input=kp_in)
    kp_raw = await _safe_aget(kp_out_var, "search_keyword")
    sk_parsed = extract_outputs(["search_keyword"], kp_raw)
    search_keyword = _guard_input(sk_parsed.get("search_keyword", _normalize_keyword(kp_raw)))

    # ===== 2–4) entry fan-out =====
    # entry_movie_qwen14b
    entry_movie_db = await _call_db("entry_movie_qwen14b", ENTRY_MOVIE_QUERIES, search_keyword)
    entry_movie_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "search_keyword": search_keyword},
        [_guard_input(entry_movie_db or "{}")],
    )
    em_in = P.variable("user_query", content=entry_movie_prompt)
    em_out_var = entry_movie_qwen14b(input=em_in)

    # entry_person_openai20b
    entry_person_db = await _call_db("entry_person_openai20b", ENTRY_PERSON_QUERIES, search_keyword)
    entry_person_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "search_keyword": search_keyword},
        [_guard_input(entry_person_db or "{}")],
    )
    ep_in = P.variable("user_query", content=entry_person_prompt)
    ep_out_var = entry_person_openai20b(input=ep_in)

    # entry_akas_qwen32b
    entry_akas_db = await _call_db("entry_akas_qwen32b", ENTRY_AKAS_QUERIES, search_keyword)
    entry_akas_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "search_keyword": search_keyword},
        [_guard_input(entry_akas_db or "{}")],
    )
    ea_in = P.variable("user_query", content=entry_akas_prompt)
    ea_out_var = entry_akas_qwen32b(input=ea_in)

    # collect entry outputs
    em_raw = await _safe_aget(em_out_var, "entry_movie_notes")
    ep_raw = await _safe_aget(ep_out_var, "entry_person_notes")
    ea_raw = await _safe_aget(ea_out_var, "entry_akas_notes")

    entry_movie_notes = _guard_input(extract_outputs(["entry_movie_notes"], em_raw or "{}").get("entry_movie_notes", ""))
    entry_person_notes = _guard_input(extract_outputs(["entry_person_notes"], ep_raw or "{}").get("entry_person_notes", ""))
    entry_akas_notes = _guard_input(extract_outputs(["entry_akas_notes"], ea_raw or "{}").get("entry_akas_notes", ""))

    # ===== 5) hub_movie_person_openai20b =====
    hub_mp_db = await _call_db("hub_movie_person_openai20b", HUB_MOVIE_PERSON_QUERIES, search_keyword)
    hub_mp_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "search_keyword": search_keyword,
            "entry_movie_notes": entry_movie_notes,
            "entry_person_notes": entry_person_notes,
        },
        [_guard_input(hub_mp_db or "{}")],
    )
    hmp_in = P.variable("user_query", content=hub_mp_prompt)
    hmp_out_var = hub_movie_person_openai20b(input=hmp_in)
    hmp_raw = await _safe_aget(hmp_out_var, "hub_movie_person_notes")
    hub_movie_person_notes = _guard_input(
        extract_outputs(["hub_movie_person_notes"], hmp_raw or "{}").get("hub_movie_person_notes", "")
    )

    # ===== 6) hub_full_context_qwen14b =====
    hub_genre_db = await _call_db("hub_full_context_qwen14b", HUB_GENRE_QUERIES, search_keyword)
    hub_fc_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "search_keyword": search_keyword,
            "hub_movie_person_notes": hub_movie_person_notes,
            "entry_akas_notes": entry_akas_notes,
        },
        [_guard_input(hub_genre_db or "{}")],
    )
    hfc_in = P.variable("user_query", content=hub_fc_prompt)
    hfc_out_var = hub_full_context_qwen14b(input=hfc_in)
    hfc_raw = await _safe_aget(hfc_out_var, "hub_full_context")
    hub_full_context = _guard_input(extract_outputs(["hub_full_context"], hfc_raw or "{}").get("hub_full_context", ""))

    # ===== 7) tail_focus_qwen32b =====
    tail_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "hub_full_context": hub_full_context},
        [],
    )
    tf_in = P.variable("user_query", content=tail_prompt)
    tf_out_var = tail_focus_qwen32b(input=tf_in)
    tf_raw = await _safe_aget(tf_out_var, "tail_focus_notes")
    tail_focus_notes = _guard_input(extract_outputs(["tail_focus_notes"], tf_raw or "{}").get("tail_focus_notes", ""))

    # ===== 8) imdb_final_answer =====
    final_titles_db = await _call_db("imdb_final_answer", FINAL_SAMPLE_QUERIES, search_keyword)
    final_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "search_keyword": search_keyword,
            "hub_full_context": hub_full_context,
            "tail_focus_notes": tail_focus_notes,
        },
        [_guard_input(final_titles_db or "{}")],
    )
    fa_in = P.variable("user_query", content=final_prompt)
    fa_out_var = imdb_final_answer(input=fa_in)
    fa_raw = await _safe_aget(fa_out_var, "final_answer")
    final_answer = _guard_input(extract_outputs(["final_answer"], fa_raw or "{}").get("final_answer", ""))

    if idx is not None and total is not None:
        print(f"\n===== Query #{idx}/{total} =====\n{user_query}")
    else:
        print(f"\n===== Query =====\n{user_query}")
    print("\n=== Final Answer ===")
    print(final_answer)
    print("\n--- Tail Focus ---")
    print(tail_focus_notes)


async def _run_query_async_wrapper(q: str, idx: int, total: int, start: float, metrics: BaselineMetrics):
    await run_one_query_async(q, idx=idx, total=total)
    metrics.record_query(time.perf_counter() - start)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IMDb diamond Parrot workflow (template-aligned; input/output only).")
    parser.add_argument("--query", type=str, default=None, help="单条用户查询；若提供则忽略文件。")
    parser.add_argument("--file", type=str, default="data/imdb_input.txt", help="批量查询文件，每行一条。")
    parser.add_argument("--limit", type=int, default=None, help="最多读取前 N 条（可选）。")
    parser.add_argument("--count", type=int, default=None, help="仅处理前 count 条（早于 limit 截断）。")
    args = parser.parse_args(argv)

    if args.query:
        queries = [args.query.strip()]
    else:
        queries = load_queries(Path(args.file), args.limit)
        if args.count is not None:
            queries = queries[: args.count]

    if not queries:
        print("No queries found.")
        return 1

    metrics = BaselineMetrics()
    metrics.start(run_name="imdb_diamond_parrot", total_queries=len(queries))
    metrics.add_metadata("runner", "parrot")
    metrics.add_metadata("input_source", "inline" if args.query else args.file)
    if args.limit is not None:
        metrics.add_metadata("limit", args.limit)
    if args.count is not None:
        metrics.add_metadata("count", args.count)

    async def _runner():
        tasks = []
        total = len(queries)
        for idx, q in enumerate(queries, 1):
            start = time.perf_counter()
            tasks.append(asyncio.create_task(_run_query_async_wrapper(q, idx, total, start, metrics)))
        await asyncio.gather(*tasks)

    set_global_metrics(metrics)
    vm.set_global_env()
    try:
        asyncio.run(_runner())
    finally:
        vm.unset_global_env()
        set_global_metrics(None)

    metrics.finish()
    print("\n" + metrics.format_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
