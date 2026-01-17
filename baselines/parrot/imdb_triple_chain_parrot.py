"""
Parrot workflow for the IMDb triple-chain topology (template-aligned, implicit-output style).

Template: imdb_qa_agent_triple_chain_10nodes_3models_32b
- 10 LLM nodes (excluding user_input)
- 3 DB queries total (movie/people/crew); final head has NO DB queries
- Models:
  - openai/gpt-oss-20b: keyword planner + people chain
  - Qwen3-32B: movie chain + small sanity
  - Qwen3-14B: crew chain + final head
- All DB queries depend only on search_keyword (derived from user_query)

Implicit-output alignment:
- Semantic functions still defined as (input, output) in signature (template style).
- Call sites DO NOT explicitly allocate output vars; we rely on Parrot's implicit output binding
  by capturing the function call's return value.
- Dataset reading / CLI flags: matches imdb_diamond_parrot.py (file/limit/count/query)
"""

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests
from parrot import P
from baselines.metrics import BaselineMetrics, get_global_metrics, set_global_metrics
from baselines.parrot.parrot_utils import (
    build_halo_prompt,
    extract_outputs,
    guard_input as _guard_input,
    prepare_db_payload_for_prompt as _prepare_db_payload_for_prompt,
    safe_aget as _safe_aget,
)

# ---- Runtime wiring ----
CORE_HTTP_ADDR = os.getenv("PARROT_CORE_HTTP", "http://localhost:9000")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))

# ---- Model routing (exactly three models) ----
MODEL_OPENAI20B = os.getenv("IMDB_MODEL_OPENAI20B", "openai/gpt-oss-20b")
MODEL_QWEN14B = os.getenv("IMDB_MODEL_QWEN14B", "Qwen/Qwen3-14B")
MODEL_QWEN32B = os.getenv("IMDB_MODEL_QWEN32B", "Qwen/Qwen3-32B")

# Backward-compatible envs (optional)
MODEL_A = os.getenv("IMDB_MODEL_A", MODEL_OPENAI20B)  # openai/gpt-oss-20b
MODEL_B = os.getenv("IMDB_MODEL_B", MODEL_QWEN14B)    # Qwen/Qwen3-14B
MODEL_C = os.getenv("IMDB_MODEL_C", MODEL_QWEN32B)    # Qwen/Qwen3-32B

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


# -------------------------
# Dataset reading (diamond-aligned)
# -------------------------
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


async def _call_db(node_id: str, queries: Sequence[Dict[str, Any]], context: Dict[str, Any]) -> Any:
    """Call shared DB worker and return raw output object (dict/list/string)."""

    def _do_call():
        start = time.perf_counter()
        payload = {"node_id": node_id, "queries": list(queries), "contexts": [context]}
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
        return outputs[0] if isinstance(outputs, list) and outputs else outputs

    return await asyncio.to_thread(_do_call)


# =========================
# DB queries (exactly 1 per chain; final has none)
# =========================
MOVIE_QUERIES: List[Dict[str, Any]] = [
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
            ORDER BY COALESCE(r.num_votes,0)       DESC,
                     COALESCE(r.average_rating,0) DESC,
                     b.start_year DESC NULLS LAST
            LIMIT 40;
        """,
        "parameters": {"keyword": "{{ search_keyword }}"},
        "param_types": {"keyword": "text"},
    }
]

PEOPLE_QUERIES: List[Dict[str, Any]] = [
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
    }
]

CREW_QUERIES: List[Dict[str, Any]] = [
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
    }
]


# =========================
# Semantic nodes (input/output ONLY; prompt text aligned to template)
# =========================

@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def keyword_planner(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are a keyword planning assistant for an IMDb-like database.
    Given user_query, produce a concise search_keyword suitable for ILIKE
    on titles, names, and related text fields.
    Respond ONLY with:
    {
      "search_keyword": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def movie_overview_qwen32b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Movie chain: overview.
    Use rows from imdb_movie_overview_main to identify candidate titles
    and describe the overall landscape.
    Respond ONLY with:
    {
      "movie_overview": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def movie_final_notes_qwen32b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Movie chain: final notes.
    Based on movie_overview, highlight 3–5 representative titles and
    explain why they matter for the user_query.
    Respond ONLY with:
    {
      "movie_final_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def people_overview_openai20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """People chain: overview.
    Use imdb_people_overview_main to summarize which actors/directors/writers
    are most relevant to the search_keyword.
    Respond ONLY with:
    {
      "people_overview": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def people_final_notes_openai20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """People chain: final notes.
    From people_overview, explain which people are most central to
    answering the user_query, and why.
    Respond ONLY with:
    {
      "people_final_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def crew_overview_qwen14b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Crew chain: overview.
    Use imdb_crew_by_title_keyword to summarize typical directors/writers
    and crew patterns relevant to the search_keyword.
    Respond ONLY with:
    {
      "crew_overview": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def crew_insights_qwen14b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Crew chain: mid-level insights.
    Turn crew_overview into a few concrete, human-readable insights
    about directing/writing styles that matter for the user_query.
    Respond ONLY with:
    {
      "crew_insights": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def crew_final_notes_qwen14b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Crew chain: final notes.
    Produce crew-centric final notes that downstream nodes can
    consume, focusing on how directors/writers shape interpretation
    of the titles.
    Respond ONLY with:
    {
      "crew_final_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def small_sanity_check_qwen32b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are a lightweight sanity checker.
    Rewrite user_query into a shorter, unambiguous canonical_query.
    Respond ONLY with:
    {
      "canonical_query": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def final_answer_qwen14b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Final answer assistant.
    Combine movie_final_notes, people_final_notes, crew_final_notes,
    and canonical_query into a single clear, grounded answer.
    Respond ONLY with:
    {
      "final_answer": "string"
    }

    {{input}}
    {{output}}
    """


# =========================
# Orchestration (implicit output vars)
# =========================
async def run_one_query_async(user_query: str, idx: int | None = None, total: int | None = None) -> None:
    uq = _guard_input(user_query)

    # --- keyword planner ---
    kw_input_text = build_halo_prompt(None, {"user_query": uq}, [])
    kw_in = P.variable("user_query", content=kw_input_text)

    kw_out_var = keyword_planner(input=kw_in)  # implicit output var
    kw_raw = await _safe_aget(kw_out_var, "keyword_planner")
    kw_parsed = extract_outputs(["search_keyword"], kw_raw)
    search_keyword = _guard_input(kw_parsed.get("search_keyword", ""))

    # --- sanity ---
    sanity_input_text = build_halo_prompt(None, {"user_query": uq}, [])
    sanity_in = P.variable("user_query", content=sanity_input_text)

    sanity_out_var = small_sanity_check_qwen32b(input=sanity_in)  # implicit output var
    sanity_raw = await _safe_aget(sanity_out_var, "small_sanity_check_qwen32b")
    sanity_parsed = extract_outputs(["canonical_query"], sanity_raw)
    canonical_query = _guard_input(sanity_parsed.get("canonical_query", ""))

    # --- movie overview (DB) ---
    movie_db_raw = await _call_db("movie_overview_qwen32b", MOVIE_QUERIES, {"search_keyword": search_keyword})
    movie_overview_input_text = build_halo_prompt(
        None,
        {"user_query": uq, "search_keyword": search_keyword},
        [_prepare_db_payload_for_prompt(movie_db_raw)],
    )
    movie_overview_in = P.variable("user_query", content=movie_overview_input_text)

    movie_overview_out_var = movie_overview_qwen32b(input=movie_overview_in)  # implicit output var
    movie_overview_raw = await _safe_aget(movie_overview_out_var, "movie_overview_qwen32b")
    movie_overview_parsed = extract_outputs(["movie_overview"], movie_overview_raw)
    movie_overview = _guard_input(movie_overview_parsed.get("movie_overview", ""))

    # --- movie final notes ---
    movie_final_input_text = build_halo_prompt(None, {"user_query": uq, "movie_overview": movie_overview}, [])
    movie_final_in = P.variable("user_query", content=movie_final_input_text)

    movie_final_out_var = movie_final_notes_qwen32b(input=movie_final_in)  # implicit output var
    movie_final_raw = await _safe_aget(movie_final_out_var, "movie_final_notes_qwen32b")
    movie_final_parsed = extract_outputs(["movie_final_notes"], movie_final_raw)
    movie_final_notes = _guard_input(movie_final_parsed.get("movie_final_notes", ""))

    # --- people overview (DB) ---
    people_db_raw = await _call_db("people_overview_openai20b", PEOPLE_QUERIES, {"search_keyword": search_keyword})
    people_overview_input_text = build_halo_prompt(
        None,
        {"user_query": uq, "search_keyword": search_keyword},
        [_prepare_db_payload_for_prompt(people_db_raw)],
    )
    people_overview_in = P.variable("user_query", content=people_overview_input_text)

    people_overview_out_var = people_overview_openai20b(input=people_overview_in)  # implicit output var
    people_overview_raw = await _safe_aget(people_overview_out_var, "people_overview_openai20b")
    people_overview_parsed = extract_outputs(["people_overview"], people_overview_raw)
    people_overview = _guard_input(people_overview_parsed.get("people_overview", ""))

    # --- people final notes ---
    people_final_input_text = build_halo_prompt(None, {"user_query": uq, "people_overview": people_overview}, [])
    people_final_in = P.variable("user_query", content=people_final_input_text)

    people_final_out_var = people_final_notes_openai20b(input=people_final_in)  # implicit output var
    people_final_raw = await _safe_aget(people_final_out_var, "people_final_notes_openai20b")
    people_final_parsed = extract_outputs(["people_final_notes"], people_final_raw)
    people_final_notes = _guard_input(people_final_parsed.get("people_final_notes", ""))

    # --- crew overview (DB) ---
    crew_db_raw = await _call_db("crew_overview_qwen14b", CREW_QUERIES, {"search_keyword": search_keyword})
    crew_overview_input_text = build_halo_prompt(
        None,
        {"user_query": uq, "search_keyword": search_keyword},
        [_prepare_db_payload_for_prompt(crew_db_raw)],
    )
    crew_overview_in = P.variable("user_query", content=crew_overview_input_text)

    crew_overview_out_var = crew_overview_qwen14b(input=crew_overview_in)  # implicit output var
    crew_overview_raw = await _safe_aget(crew_overview_out_var, "crew_overview_qwen14b")
    crew_overview_parsed = extract_outputs(["crew_overview"], crew_overview_raw)
    crew_overview = _guard_input(crew_overview_parsed.get("crew_overview", ""))

    # --- crew insights ---
    crew_insights_input_text = build_halo_prompt(None, {"user_query": uq, "crew_overview": crew_overview}, [])
    crew_insights_in = P.variable("user_query", content=crew_insights_input_text)

    crew_insights_out_var = crew_insights_qwen14b(input=crew_insights_in)  # implicit output var
    crew_insights_raw = await _safe_aget(crew_insights_out_var, "crew_insights_qwen14b")
    crew_insights_parsed = extract_outputs(["crew_insights"], crew_insights_raw)
    crew_insights = _guard_input(crew_insights_parsed.get("crew_insights", ""))

    # --- crew final notes ---
    crew_final_input_text = build_halo_prompt(None, {"user_query": uq, "crew_insights": crew_insights}, [])
    crew_final_in = P.variable("user_query", content=crew_final_input_text)

    crew_final_out_var = crew_final_notes_qwen14b(input=crew_final_in)  # implicit output var
    crew_final_raw = await _safe_aget(crew_final_out_var, "crew_final_notes_qwen14b")
    crew_final_parsed = extract_outputs(["crew_final_notes"], crew_final_raw)
    crew_final_notes = _guard_input(crew_final_parsed.get("crew_final_notes", ""))

    # --- final answer (NO DB) ---
    final_input_text = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "search_keyword": search_keyword,
            "canonical_query": canonical_query,
            "movie_final_notes": movie_final_notes,
            "people_final_notes": people_final_notes,
            "crew_final_notes": crew_final_notes,
        },
        [],
    )
    final_in = P.variable("user_query", content=final_input_text)

    final_out_var = final_answer_qwen14b(input=final_in)  # implicit output var
    final_raw = await _safe_aget(final_out_var, "final_answer_qwen14b")
    final_parsed = extract_outputs(["final_answer"], final_raw)
    final_answer = _guard_input(final_parsed.get("final_answer", ""))

    if idx is not None and total is not None:
        print(f"\n===== Query #{idx}/{total} =====")
    else:
        print("\n===== Query =====")
    print(user_query)
    print("\n=== Final Answer ===")
    print(final_answer)


async def _run_query_async_wrapper(q: str, idx: int, total: int, start: float, metrics: BaselineMetrics):
    await run_one_query_async(q, idx=idx, total=total)
    metrics.record_query(time.perf_counter() - start)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IMDb triple-chain Parrot (template-aligned; implicit output vars).")
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
    metrics.start(run_name="imdb_triple_chain_parrot", total_queries=len(queries))
    metrics.add_metadata("runner", "parrot")
    metrics.add_metadata("graph", "imdb_qa_agent_triple_chain_10nodes_3models_32b")
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
