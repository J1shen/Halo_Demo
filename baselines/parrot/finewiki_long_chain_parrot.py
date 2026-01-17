"""
Parrot workflow for FineWiki long-chain + branches topology
(templates/finewiki_qa_agent_longchain_branches_9nodes_3models).

Baseline behavior:
- Mirrors the YAML graph's node semantics and system_prompt wording.
- No optimizer tricks: orchestration is explicit in Python.
- Uses the shared DB worker (gunicorn FastAPI) for all SQL.
- LLM functions are IMDb-aligned: ONLY two variables (input/output).
  All structured inputs are folded into a single prompt string.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
import sys
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
from run.query_sampler import (
    DEFAULT_SAMPLE_POOL_SIZE,
    SampleResult,
    default_extract_query,
    sample_queries,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- Runtime wiring ----
CORE_HTTP_ADDR = os.getenv("PARROT_CORE_HTTP", "http://localhost:9000")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))
MAX_FIELD_CHARS = 2048
MAX_ROWS = 5

# Model routing (matches YAML)
MODEL_A = os.getenv("FINEWIKI_MODEL_A", "openai/gpt-oss-20b")  # M_A: global/timeline/narrow
MODEL_B = os.getenv("FINEWIKI_MODEL_B", "Qwen/Qwen3-32B")      # M_B: category
MODEL_C = os.getenv("FINEWIKI_MODEL_C", "Qwen/Qwen3-14B")      # M_C: links + final

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
# DB queries (match YAML)
# -------------------------
GLOBAL_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "fw_global_wikitext_fulltext",
        "sql": """
            SELECT page_id, title, url, wikitext
            FROM pages
            WHERE in_language = 'en'
              AND to_tsvector('english', coalesce(wikitext, ''))
                  @@ plainto_tsquery('english', :keyword)
            ORDER BY date_modified DESC
            LIMIT 40;
        """,
        "parameters": {"keyword": "{{ user_query }}"},
        "param_types": {"keyword": "text"},
    },
    {
        "name": "fw_global_title_keyword",
        "sql": """
            SELECT page_id, title, url, wikitext
            FROM pages
            WHERE in_language = 'en'
              AND title ILIKE '%%' || :keyword || '%%'
            ORDER BY date_modified DESC
            LIMIT 25;
        """,
        "parameters": {"keyword": "{{ user_query }}"},
        "param_types": {"keyword": "text"},
    },
]

TIMELINE_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "fw_timeline_from_keyword",
        "sql": """
            SELECT page_id, title, url, wikitext, date_modified
            FROM pages
            WHERE in_language = 'en'
              AND to_tsvector('english', coalesce(wikitext, ''))
                  @@ plainto_tsquery('english', :keyword)
            ORDER BY date_modified ASC
            LIMIT 40;
        """,
        "parameters": {"keyword": "{{ user_query }}"},
        "param_types": {"keyword": "text"},
    }
]

CATEGORY_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "fw_category_pages",
        "sql": """
            SELECT page_id, title, url, wikitext
            FROM pages
            WHERE in_language = 'en'
              AND LOWER(title) LIKE LOWER('Category:%%' || :keyword || '%%')
            LIMIT 30;
        """,
        "parameters": {"keyword": "{{ user_query }}"},
        "param_types": {"keyword": "text"},
    }
]

OUTBOUND_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "fw_outbound_links",
        "sql": """
            SELECT page_id, title, url, wikitext
            FROM pages
            WHERE in_language = 'en'
              AND title ILIKE '%%' || :keyword || '%%'
            ORDER BY page_id ASC
            LIMIT 40;
        """,
        "parameters": {"keyword": "{{ user_query }}"},
        "param_types": {"keyword": "text"},
    }
]

FINAL_SNIPPET_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "fw_supporting_snippets_global",
        "sql": """
            SELECT page_id, title, url, wikitext
            FROM pages
            WHERE in_language = 'en'
              AND to_tsvector('english', coalesce(wikitext, ''))
                  @@ plainto_tsquery('english', :keyword)
            ORDER BY date_modified DESC
            LIMIT 20;
        """,
        "parameters": {"keyword": "{{ user_query }}"},
        "param_types": {"keyword": "text"},
    }
]


# -------------------------
# DB call (IMDb-aligned)
# -------------------------
async def _call_db(node_id: str, queries: Sequence[Dict[str, Any]], context: Dict[str, Any]) -> str:
    """Async DB worker call; normalize payload for prompt."""
    def _normalize_db_payload(payload: Any) -> str:
        if payload is None:
            return "{}"

        data = payload
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except Exception:
                data = payload

        # Use FineWiki-specific sanitizer if possible
        try:
            data = _prepare_db_payload_for_prompt(data)
        except Exception:
            pass

        if isinstance(data, str):
            text = data
        else:
            try:
                text = json.dumps(data, ensure_ascii=False)
            except Exception:
                text = "{}"

        lines = text.splitlines() if text else []
        if not lines:
            return "{}"
        trimmed = [line[:MAX_FIELD_CHARS] for line in lines[:MAX_ROWS]]
        return "\n".join(trimmed) if trimmed else "{}"

    def _do_call() -> str:
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
        first = outputs[0] if isinstance(outputs, list) and outputs else outputs
        return _guard_input(_normalize_db_payload(first))

    return await asyncio.to_thread(_do_call)


# -------------------------
# Semantic nodes (YAML system_prompt aligned; only input/output)
# -------------------------
@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def global_topic_expander(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are a global topic expander over a Wikipedia-like corpus.
    Use fw_global_wikitext_fulltext and fw_global_title_keyword to build
    a broad but coherent summary of the topic and propose a small list
    of central candidate page_ids (as plain integers in text).
    Respond ONLY with:
    {
      "global_summary": "string",
      "central_page_ids": [integer, ...]
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def timeline_refiner_1(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are a timeline-focused refiner.
    Given global_summary and (textual) central_page_ids, and using
    fw_timeline_from_keyword as context, construct a coarse
    chronological narrative of the topic.
    Respond ONLY with:
    {
      "coarse_timeline": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def timeline_refiner_2(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are a second-pass timeline refiner.
    Take the coarse_timeline and improve clarity and ordering, and
    highlight 3–5 key milestones.
    Respond ONLY with:
    {
      "refined_timeline": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def category_overview(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are a category-focused summarizer.
    Use fw_category_pages to provide a high-level taxonomy of the topic.
    Respond ONLY with:
    {
      "category_overview": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def category_drilldown(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are a drill-down assistant for category structures.
    Take category_overview and extract 3–5 important subtopics that
    should be mentioned in a final answer.
    Respond ONLY with:
    {
      "category_details": ["string", ...]
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def outbound_link_view(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are an outbound link summarizer.
    Given pages whose titles match user_query, summarize which topics
    they tend to link out to (based on their wikitext).
    Respond ONLY with:
    {
      "outbound_links_overview": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def inbound_link_view(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You summarize who links into this topic.
    Given outbound_links_overview and global_summary, infer which
    other pages or domains frequently point to this topic and why
    that matters for understanding it.
    Respond ONLY with:
    {
      "inbound_links_overview": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def narrow_focus_rewriter(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You rewrite the user_query into a narrower focus question
    suitable for a final answer over this topic.
    Respond ONLY with:
    {
      "narrowed_query": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def final_answer_summarizer(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are the final answer assistant.
    Combine global_summary, refined_timeline, category_details,
    outbound_links_overview, inbound_links_overview, and
    narrowed_query (plus fw_supporting_snippets_global as loose
    background) into a single clear, grounded answer.
    Respond ONLY with:
    {
      "final_answer": "string"
    }

    {{input}}
    {{output}}
    """


# -------------------------
# Orchestration (baseline)
# -------------------------
async def run_one_query_async(user_query: str, idx: int | None = None, total: int | None = None) -> None:
    uq = _guard_input(user_query)

    # (1) global_topic_expander (DB + LLM)
    global_db_str = await _call_db("global_topic_expander", GLOBAL_QUERIES, {"user_query": uq})
    global_prompt = build_halo_prompt(None, {"user_query": uq}, [global_db_str])
    global_val = P.variable("user_query", content=global_prompt)
    global_out = global_topic_expander(input=global_val)
    global_out = await _safe_aget(global_out, "global_topic_expander_output")
    global_parsed = extract_outputs(["global_summary", "central_page_ids"], global_out)

    global_summary = _guard_input(global_parsed.get("global_summary", ""))
    central_page_ids = _guard_input(global_parsed.get("central_page_ids", ""))

    # (2) timeline_refiner_1 (DB + LLM)
    timeline_db_str = await _call_db("timeline_refiner_1", TIMELINE_QUERIES, {"user_query": uq})
    timeline1_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "global_summary": global_summary,
            "central_page_ids": central_page_ids,
        },
        [timeline_db_str],
    )
    timeline1_val = P.variable("user_query", content=timeline1_prompt)
    coarse_out = timeline_refiner_1(input=timeline1_val)
    coarse_out = await _safe_aget(coarse_out, "timeline_refiner_1_output")
    coarse_parsed = extract_outputs(["coarse_timeline"], coarse_out)
    coarse_timeline = _guard_input(coarse_parsed.get("coarse_timeline", ""))

    # (3) timeline_refiner_2 (LLM-only)
    timeline2_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "coarse_timeline": coarse_timeline},
        [],
    )
    timeline2_val = P.variable("user_query", content=timeline2_prompt)
    refined_out = timeline_refiner_2(input=timeline2_val)
    refined_out = await _safe_aget(refined_out, "timeline_refiner_2_output")
    refined_parsed = extract_outputs(["refined_timeline"], refined_out)
    refined_timeline = _guard_input(refined_parsed.get("refined_timeline", ""))

    # (4) category_overview (DB + LLM)
    category_db_str = await _call_db("category_overview", CATEGORY_QUERIES, {"user_query": uq})
    category_prompt = build_halo_prompt(None, {"user_query": uq}, [category_db_str])
    category_val = P.variable("user_query", content=category_prompt)
    cat_over_out = category_overview(input=category_val)
    cat_over_out = await _safe_aget(cat_over_out, "category_overview_output")
    cat_over_parsed = extract_outputs(["category_overview"], cat_over_out)
    category_overview_txt = _guard_input(cat_over_parsed.get("category_overview", ""))

    # (5) category_drilldown (LLM-only)
    drill_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "category_overview": category_overview_txt},
        [],
    )
    drill_val = P.variable("user_query", content=drill_prompt)
    cat_details_out = category_drilldown(input=drill_val)
    cat_details_out = await _safe_aget(cat_details_out, "category_drilldown_output")
    cat_details_parsed = extract_outputs(["category_details"], cat_details_out)
    category_details = _guard_input(cat_details_parsed.get("category_details", ""))

    # (6) outbound_link_view (DB + LLM)
    outbound_db_str = await _call_db("outbound_link_view", OUTBOUND_QUERIES, {"user_query": uq})
    outbound_prompt = build_halo_prompt(None, {"user_query": uq}, [outbound_db_str])
    outbound_val = P.variable("user_query", content=outbound_prompt)
    outbound_out = outbound_link_view(input=outbound_val)
    outbound_out = await _safe_aget(outbound_out, "outbound_link_view_output")
    outbound_parsed = extract_outputs(["outbound_links_overview"], outbound_out)
    outbound_links_overview = _guard_input(outbound_parsed.get("outbound_links_overview", ""))

    # (7) inbound_link_view (LLM-only)
    inbound_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "outbound_links_overview": outbound_links_overview,
            "global_summary": global_summary,
        },
        [],
    )
    inbound_val = P.variable("user_query", content=inbound_prompt)
    inbound_out = inbound_link_view(input=inbound_val)
    inbound_out = await _safe_aget(inbound_out, "inbound_link_view_output")
    inbound_parsed = extract_outputs(["inbound_links_overview"], inbound_out)
    inbound_links_overview = _guard_input(inbound_parsed.get("inbound_links_overview", ""))

    # (8) narrow_focus_rewriter (LLM-only)
    narrow_prompt = build_halo_prompt(None, {"user_query": uq}, [])
    narrow_val = P.variable("user_query", content=narrow_prompt)
    narrow_out = narrow_focus_rewriter(input=narrow_val)
    narrow_out = await _safe_aget(narrow_out, "narrow_focus_rewriter_output")
    narrow_parsed = extract_outputs(["narrowed_query"], narrow_out)
    narrowed_query = _guard_input(narrow_parsed.get("narrowed_query", ""))

    # (9) final_answer_summarizer (DB + LLM)
    final_db_str = await _call_db("final_answer_summarizer", FINAL_SNIPPET_QUERIES, {"user_query": uq})
    final_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "global_summary": global_summary,
            "refined_timeline": refined_timeline,
            "category_details": category_details,
            "outbound_links_overview": outbound_links_overview,
            "inbound_links_overview": inbound_links_overview,
            "narrowed_query": narrowed_query,
        },
        [final_db_str],
    )
    final_val = P.variable("user_query", content=final_prompt)
    final_out = final_answer_summarizer(input=final_val)
    final_out = await _safe_aget(final_out, "final_answer_summarizer_output")
    final_parsed = extract_outputs(["final_answer"], final_out)
    final_answer = _guard_input(final_parsed.get("final_answer", ""))

    if idx is not None and total is not None:
        print(f"\n===== Query #{idx}/{total} =====")
        print(user_query)
    else:
        print("\n===== Query =====")
        print(user_query)
    print("\n=== Final Answer ===")
    print(final_answer)


async def _run_query_async_wrapper(q: str, idx: int, total: int, start: float, metrics: BaselineMetrics):
    await run_one_query_async(q, idx=idx, total=total)
    metrics.record_query(time.perf_counter() - start)


def _load_queries_from_dataset(args: argparse.Namespace) -> SampleResult:
    return sample_queries(
        dataset=args.dataset_name,
        subset=args.dataset_subset,
        split=args.dataset_split,
        sample_count=max(1, args.sample_count),
        seed=args.sample_seed,
        pool_size=args.sample_pool_size,
        extract_query=default_extract_query,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FineWiki long-chain branches Parrot (Halo-aligned).")
    parser.add_argument("--query", type=str, default=None, help="单条用户查询；若提供则忽略数据集采样。")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceFW/finewiki", help="HuggingFace 数据集名称。")
    parser.add_argument("--dataset-subset", type=str, default="en", help="数据集子集/语言。")
    parser.add_argument("--dataset-split", type=str, default="train", help="数据集 split。")
    parser.add_argument("--sample-count", type=int, default=1024, help="要执行的样本数。")
    parser.add_argument("--sample-seed", type=int, default=44, help="采样随机种子。")
    parser.add_argument(
        "--sample-pool-size",
        type=int,
        default=DEFAULT_SAMPLE_POOL_SIZE,
        help="流式采样时最多扫描的池大小。",
    )
    args = parser.parse_args(argv)

    if args.query:
        queries = [args.query.strip()]
        sample_info: SampleResult | None = None
    else:
        sample_info = _load_queries_from_dataset(args)
        queries = sample_info.queries
        if not queries:
            print("No queries sampled from dataset.")
            return 1

    metrics = BaselineMetrics()
    metrics.start(run_name="finewiki_qa_agent_longchain_branches_parrot", total_queries=len(queries))
    metrics.add_metadata("runner", "parrot")
    if sample_info is not None:
        metrics.add_metadata("dataset", args.dataset_name)
        metrics.add_metadata("dataset_subset", args.dataset_subset)
        metrics.add_metadata("dataset_split", args.dataset_split)
        metrics.add_metadata("sample_count", args.sample_count)
        metrics.add_metadata("rows_scanned", sample_info.rows_scanned)

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
