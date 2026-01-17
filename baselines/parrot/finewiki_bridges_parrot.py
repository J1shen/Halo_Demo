"""Parrot workflow for the FineWiki bridge topology (templates/finewiki_bridges.yaml).

- Mirrors: finewiki_qa_agent_bridge_topology_9nodes_3models
- Keeps baseline sequencing (no optimizer tweaks).
- Uses shared DB worker (gunicorn FastAPI) for SQL.
- IMDb-aligned: every semantic function uses ONLY (input, output).
- Dataset sampling logic preserved (sample_queries).
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

# Ensure project root on path for shared utils (query sampler, etc.)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from parrot import P
from parrot.frontend.pfunc.semantic_variable import SemanticVariable
from baselines.metrics import BaselineMetrics, get_global_metrics, set_global_metrics
from baselines.parrot.parrot_utils import (
    build_halo_prompt,
    extract_outputs,
    guard_input as _guard_input,
    safe_aget as _safe_aget,
)
from run.query_sampler import (
    DEFAULT_SAMPLE_POOL_SIZE,
    SampleResult,
    default_extract_query,
    sample_queries,
)

# ---- Runtime wiring ----
CORE_HTTP_ADDR = os.getenv("PARROT_CORE_HTTP", "http://localhost:9000")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))
MAX_FIELD_CHARS = 2048
MAX_ROWS = 5

# ---- Model routing (match engines you run) ----
MODEL_A = os.getenv("FINEWIKI_MODEL_A", "openai/gpt-oss-20b")  # A0/A1/A2
MODEL_B = os.getenv("FINEWIKI_MODEL_B", "Qwen/Qwen3-32B")      # B1/B2
MODEL_C = os.getenv("FINEWIKI_MODEL_C", "Qwen/Qwen3-14B")      # C1/C2/S0/FINAL

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


def _unwrap(val: Any) -> Any:
    if isinstance(val, SemanticVariable):
        if not val.is_registered:
            val.assign_id(vm.register_semantic_variable_handler(val.name))
        if val.content is not None:
            return val.content
        return val.get()
    return val


async def _call_db(node_id: str, queries: Sequence[Dict[str, Any]], context: Dict[str, Any]) -> str:
    """Async DB worker call; returns a sanitized string payload."""
    def _normalize(payload: Any) -> str:
        if payload is None:
            return "{}"

        # Try JSON first
        try:
            if isinstance(payload, str):
                data = json.loads(payload)
            else:
                data = payload
        except Exception:
            text = str(payload)
            lines = text.splitlines() if text else []
            trimmed = [ln[:MAX_FIELD_CHARS] for ln in lines[:MAX_ROWS]]
            return "\n".join(trimmed) if trimmed else "{}"

        try:
            text = json.dumps(data, ensure_ascii=False)
        except Exception:
            return "{}"

        lines = text.splitlines() if text else []
        trimmed = [ln[:MAX_FIELD_CHARS] for ln in lines[:MAX_ROWS]]
        return "\n".join(trimmed) if trimmed else "{}"

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
        first = outputs[0] if isinstance(outputs, list) and outputs else outputs
        return _normalize(first)

    return await asyncio.to_thread(_do_call)


# ---- DB query payloads (mirror template) ----
FW_GLOBAL_PASS_A0_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "fw_global_pass_a0_main",
        "sql": """
            SELECT page_id, title, url, wikitext
            FROM pages
            WHERE in_language = 'en'
              AND to_tsvector('english', coalesce(wikitext,''))
                  @@ plainto_tsquery('english', :keyword)
            LIMIT 30;
        """,
        "parameters": {"keyword": "{{ user_query }}"},
        "param_types": {"keyword": "text"},
    }
]

FW_CATEGORY_BRIDGE_B1_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "fw_category_bridge_b1_main",
        "sql": """
            SELECT page_id, title, url, wikitext
            FROM pages
            WHERE in_language = 'en'
              AND LOWER(title) LIKE LOWER('Category:%%' || :keyword || '%%')
            LIMIT 20;
        """,
        "parameters": {"keyword": "{{ user_query }}"},
        "param_types": {"keyword": "text"},
    }
]

FW_LANGUAGE_BRIDGE_C1_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "fw_language_bridge_c1_main",
        "sql": """
            SELECT page_id, title, url, wikitext, in_language
            FROM pages
            WHERE in_language <> 'en'
              AND to_tsvector('simple', coalesce(wikitext,''))
                  @@ plainto_tsquery('simple', :keyword)
            LIMIT 40;
        """,
        "parameters": {"keyword": "{{ user_query }}"},
        "param_types": {"keyword": "text"},
    }
]


# ---- Semantic functions: IMDb-aligned (input/output only), prompts mirror template ----
@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def fw_global_pass_a0(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """First global pass over FineWiki for the topic.
    Use fw_global_pass_a0_main as your retrieval pool.
    Respond ONLY with:
    {
      "a0_summary": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def fw_global_pass_a1(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Refine a0_summary into a structured outline.
    Respond ONLY with:
    {
      "a1_outline": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def fw_global_pass_a2(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Expand a1_outline into a longer narrative suitable for
    downstream bridges and final answering.
    Respond ONLY with:
    {
      "a2_narrative": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def fw_category_bridge_b1(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Category-based bridge: connect a1_outline with category pages
    from fw_category_bridge_b1_main.
    Respond ONLY with:
    {
      "b1_bridge_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def fw_category_bridge_b2(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Refine b1_bridge_notes into 3–5 key subtopics that are useful
    for guiding the final answer.
    Respond ONLY with:
    {
      "b2_key_subtopics": ["string", ...]
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def fw_language_bridge_c1(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Non-English language bridge for the topic.
    Use fw_language_bridge_c1_main to understand how the topic
    appears in other languages.
    Respond ONLY with:
    {
      "c1_language_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def fw_language_bridge_c2(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Refine c1_language_notes into 2–4 concise cross-lingual
    insights that can influence the final answer.
    Respond ONLY with:
    {
      "c2_crosslingual_insights": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def fw_light_sampler_s0(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Light sampler for the bridge topology.
    Condense a2_narrative, b2_key_subtopics, and
    c2_crosslingual_insights into short hints that are easy for
    the final answer head to consume.
    Respond ONLY with:
    {
      "s0_hints": ["string", ...]
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def fw_final_answer_bridge(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Final answer assistant for the bridge topology.
    Combine a2_narrative and s0_hints into a single coherent,
    well-structured answer to user_query.
    Respond ONLY with:
    {
      "final_answer": "string"
    }

    {{input}}
    {{output}}
    """


async def run_one_query_async(user_query: str, idx: int | None = None, total: int | None = None) -> None:
    uq = _guard_input(user_query)

    # ---- fw_global_pass_a0 (DB + LLM) ----
    a0_db_str = await _call_db("fw_global_pass_a0", FW_GLOBAL_PASS_A0_QUERIES, {"user_query": uq})
    a0_prompt = build_halo_prompt(None, {"user_query": uq}, [_guard_input(a0_db_str or "{}")])
    a0_in = P.variable("user_query", content=a0_prompt)
    a0_out = fw_global_pass_a0(input=a0_in)
    a0_raw = await _safe_aget(a0_out, "a0_summary")
    a0_summary = _guard_input(extract_outputs(["a0_summary"], a0_raw or "{}").get("a0_summary", ""))

    # ---- fw_global_pass_a1 ----
    a1_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "a0_summary": a0_summary},
        [],
    )
    a1_in = P.variable("user_query", content=a1_prompt)
    a1_out = fw_global_pass_a1(input=a1_in)
    a1_raw = await _safe_aget(a1_out, "a1_outline")
    a1_outline = _guard_input(extract_outputs(["a1_outline"], a1_raw or "{}").get("a1_outline", ""))

    # ---- fw_global_pass_a2 ----
    a2_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "a1_outline": a1_outline},
        [],
    )
    a2_in = P.variable("user_query", content=a2_prompt)
    a2_out = fw_global_pass_a2(input=a2_in)
    a2_raw = await _safe_aget(a2_out, "a2_narrative")
    a2_narrative = _guard_input(extract_outputs(["a2_narrative"], a2_raw or "{}").get("a2_narrative", ""))

    # ---- fw_category_bridge_b1 (DB + LLM) off a1 ----
    b1_db_str = await _call_db("fw_category_bridge_b1", FW_CATEGORY_BRIDGE_B1_QUERIES, {"user_query": uq})
    b1_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "a1_outline": a1_outline},
        [_guard_input(b1_db_str or "{}")],
    )
    b1_in = P.variable("user_query", content=b1_prompt)
    b1_out = fw_category_bridge_b1(input=b1_in)
    b1_raw = await _safe_aget(b1_out, "b1_bridge_notes")
    b1_bridge_notes = _guard_input(extract_outputs(["b1_bridge_notes"], b1_raw or "{}").get("b1_bridge_notes", ""))

    # ---- fw_category_bridge_b2 ----
    b2_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "b1_bridge_notes": b1_bridge_notes},
        [],
    )
    b2_in = P.variable("user_query", content=b2_prompt)
    b2_out = fw_category_bridge_b2(input=b2_in)
    b2_raw = await _safe_aget(b2_out, "b2_key_subtopics")
    b2_key_subtopics = _guard_input(
        extract_outputs(["b2_key_subtopics"], b2_raw or "{}").get("b2_key_subtopics", "")
    )

    # ---- fw_language_bridge_c1 (DB + LLM) off a0 ----
    c1_db_str = await _call_db("fw_language_bridge_c1", FW_LANGUAGE_BRIDGE_C1_QUERIES, {"user_query": uq})
    c1_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "a0_summary": a0_summary},
        [_guard_input(c1_db_str or "{}")],
    )
    c1_in = P.variable("user_query", content=c1_prompt)
    c1_out = fw_language_bridge_c1(input=c1_in)
    c1_raw = await _safe_aget(c1_out, "c1_language_notes")
    c1_language_notes = _guard_input(
        extract_outputs(["c1_language_notes"], c1_raw or "{}").get("c1_language_notes", "")
    )

    # ---- fw_language_bridge_c2 ----
    c2_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "c1_language_notes": c1_language_notes},
        [],
    )
    c2_in = P.variable("user_query", content=c2_prompt)
    c2_out = fw_language_bridge_c2(input=c2_in)
    c2_raw = await _safe_aget(c2_out, "c2_crosslingual_insights")
    c2_crosslingual_insights = _guard_input(
        extract_outputs(["c2_crosslingual_insights"], c2_raw or "{}").get("c2_crosslingual_insights", "")
    )

    # ---- fw_light_sampler_s0 merges A2 + B2 + C2 ----
    s0_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "a2_narrative": a2_narrative,
            "b2_key_subtopics": b2_key_subtopics,
            "c2_crosslingual_insights": c2_crosslingual_insights,
        },
        [],
    )
    s0_in = P.variable("user_query", content=s0_prompt)
    s0_out = fw_light_sampler_s0(input=s0_in)
    s0_raw = await _safe_aget(s0_out, "s0_hints")
    s0_hints = _guard_input(extract_outputs(["s0_hints"], s0_raw or "{}").get("s0_hints", ""))

    # ---- fw_final_answer_bridge depends on A2 + S0 ----
    final_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "a2_narrative": a2_narrative, "s0_hints": s0_hints},
        [],
    )
    final_in = P.variable("user_query", content=final_prompt)
    final_out = fw_final_answer_bridge(input=final_in)
    final_raw = await _safe_aget(final_out, "final_answer")
    final_answer = _guard_input(extract_outputs(["final_answer"], final_raw or "{}").get("final_answer", ""))

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
    # Keep original dataset sampling logic unchanged
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
    parser = argparse.ArgumentParser(description="FineWiki bridges Parrot (Halo-aligned).")
    parser.add_argument("--query", type=str, default=None, help="单条用户查询；若提供则忽略数据集采样。")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceFW/finewiki", help="HuggingFace 数据集名称。")
    parser.add_argument("--dataset-subset", type=str, default="en", help="数据集子集/语言。")
    parser.add_argument("--dataset-split", type=str, default="train", help="数据集 split。")
    parser.add_argument("--sample-count", type=int, default=1024, help="要执行的样本查询数。")
    parser.add_argument("--sample-seed", type=int, default=44, help="采样随机种子。")
    parser.add_argument(
        "--sample-pool-size",
        type=int,
        default=DEFAULT_SAMPLE_POOL_SIZE,
        help="从数据流中最多查看多少行后停止采样。",
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
    metrics.start(run_name="finewiki_bridges_parrot", total_queries=len(queries))
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
