"""Baseline Parrot workflow for the TPC-H trident template (template-aligned).

- Mirrors the nodes from templates/tpch_trident.yaml (tpch_qbench_trident_rules_v1).
- Keeps baseline behavior: sequential orchestration in Python (per query).
- Uses the shared DB worker (gunicorn FastAPI) for all SQL.
- Semantic functions use ONLY (input, output) variables (trident-style).
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
from baselines.metrics import BaselineMetrics, get_global_metrics, set_global_metrics
from baselines.parrot.parrot_utils import (
    build_halo_prompt,
    extract_outputs,
    guard_input as _guard_input,
    prepare_result_for_prompt as _prepare_result_for_prompt,
    safe_aget as _safe_aget,
)
from baselines.tpch_utils import extract_tpch_params

# ---- Runtime wiring ----
CORE_HTTP_ADDR = os.getenv("PARROT_CORE_HTTP", "http://localhost:9000")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))

MAX_FIELD_CHARS = 2048
MAX_ROWS = 5

# Model routing: must match the template model assignment.
MODEL_A = os.getenv("TPCH_MODEL_A", "openai/gpt-oss-20b")
MODEL_B = os.getenv("TPCH_MODEL_B", "Qwen/Qwen3-14B")
MODEL_C = os.getenv("TPCH_MODEL_C", "Qwen/Qwen3-32B")

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


# ---- DB query payloads (must match template names/sql/params) ----
ENTRY_Q1_Q6_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "tpch_q1_pricing_summary",
        "sql": """
            SELECT
              l_returnflag, l_linestatus,
              SUM(l_quantity) AS sum_qty,
              SUM(l_extendedprice) AS sum_base_price,
              SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
              SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
              AVG(l_quantity) AS avg_qty,
              AVG(l_extendedprice) AS avg_price,
              AVG(l_discount) AS avg_disc,
              COUNT(*) AS count_order
            FROM lineitem
            WHERE l_shipdate <= (COALESCE(NULLIF(:date::text, 'sample')::date, DATE '1995-03-15') - INTERVAL '90 day')
            GROUP BY l_returnflag, l_linestatus
            ORDER BY l_returnflag, l_linestatus;
        """,
        "parameters": {"date": "{{ tpch_params.date }}"},
        "param_types": {"date": "date"},
    },
    {
        "name": "tpch_q6_revenue_change",
        "sql": """
            SELECT SUM(l_extendedprice * l_discount) AS revenue
            FROM lineitem
            WHERE l_shipdate >= make_date(:year::int, 1, 1)
              AND l_shipdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
              AND l_discount BETWEEN :discount_lo::numeric AND :discount_hi::numeric
              AND l_quantity < :qty::numeric;
        """,
        "parameters": {
            "year": "{{ tpch_params.year }}",
            "discount_lo": "{{ tpch_params.discount_lo }}",
            "discount_hi": "{{ tpch_params.discount_hi }}",
            "qty": "{{ tpch_params.qty }}",
        },
        "param_types": {
            "year": "int",
            "discount_lo": "numeric",
            "discount_hi": "numeric",
            "qty": "numeric",
        },
    },
]

ENTRY_Q3_Q5_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "tpch_q3_shipping_priority",
        "sql": """
            SELECT
              l.l_orderkey,
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
              o.o_orderdate,
              o.o_shippriority
            FROM customer c
            JOIN orders o ON o.o_custkey = c.c_custkey
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            WHERE c.c_mktsegment = :segment
              AND o.o_orderdate < COALESCE(NULLIF(:date::text, 'sample')::date, DATE '1995-03-15')
              AND l.l_shipdate > COALESCE(NULLIF(:date::text, 'sample')::date, DATE '1995-03-15')
            GROUP BY l.l_orderkey, o.o_orderdate, o.o_shippriority
            ORDER BY revenue DESC, o.o_orderdate
            LIMIT 20;
        """,
        "parameters": {"segment": "{{ tpch_params.segment }}", "date": "{{ tpch_params.date }}"},
        "param_types": {"segment": "text", "date": "date"},
    },
    {
        "name": "tpch_q5_local_supplier_volume",
        "sql": """
            SELECT n.n_name AS nation,
                   SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
            FROM customer c
            JOIN orders o ON o.o_custkey = c.c_custkey
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            JOIN supplier s ON s.s_suppkey = l.l_suppkey
            JOIN nation n ON n.n_nationkey = s.s_nationkey
            JOIN region r ON r.r_regionkey = n.n_regionkey
            WHERE r.r_name = :region
              AND o.o_orderdate >= make_date(:year::int, 1, 1)
              AND o.o_orderdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
            GROUP BY n.n_name
            ORDER BY revenue DESC
            LIMIT 25;
        """,
        "parameters": {"region": "{{ tpch_params.region }}", "year": "{{ tpch_params.year }}"},
        "param_types": {"region": "text", "year": "int"},
    },
]

ENTRY_Q2_Q9_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "tpch_q2_min_cost_supplier",
        "sql": """
            SELECT s.s_acctbal, s.s_name, n.n_name AS nation,
                   p.p_partkey, p.p_mfgr, ps.ps_supplycost
            FROM part p
            JOIN partsupp ps ON ps.ps_partkey = p.p_partkey
            JOIN supplier s ON s.s_suppkey = ps.ps_suppkey
            JOIN nation n ON n.n_nationkey = s.s_nationkey
            JOIN region r ON r.r_regionkey = n.n_regionkey
            WHERE r.r_name = :region
              AND p.p_size = :size::int
              AND p.p_type ILIKE '%' || :type || '%'
              AND ps.ps_supplycost = (
                SELECT MIN(ps2.ps_supplycost)
                FROM partsupp ps2
                JOIN supplier s2 ON s2.s_suppkey = ps2.ps_suppkey
                JOIN nation n2 ON n2.n_nationkey = s2.s_nationkey
                JOIN region r2 ON r2.r_regionkey = n2.n_regionkey
                WHERE r2.r_name = :region AND ps2.ps_partkey = p.p_partkey
              )
            ORDER BY s.s_acctbal DESC, n.n_name, s.s_name
            LIMIT 20;
        """,
        "parameters": {
            "region": "{{ tpch_params.region }}",
            "size": "{{ tpch_params.size }}",
            "type": "{{ tpch_params.type }}",
        },
        "param_types": {"region": "text", "size": "int", "type": "text"},
    },
    {
        "name": "tpch_q9_profit",
        "sql": """
            SELECT n.n_name AS nation,
                   EXTRACT(YEAR FROM o.o_orderdate)::int AS year,
                   SUM(l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity) AS profit
            FROM part p
            JOIN lineitem l ON l.l_partkey = p.p_partkey
            JOIN orders o ON o.o_orderkey = l.l_orderkey
            JOIN partsupp ps ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
            JOIN supplier s ON s.s_suppkey = l.l_suppkey
            JOIN nation n ON n.n_nationkey = s.s_nationkey
            WHERE p.p_brand = :brand
            GROUP BY n.n_name, year
            ORDER BY nation, year
            LIMIT 60;
        """,
        "parameters": {"brand": "{{ tpch_params.brand }}"},
        "param_types": {"brand": "text"},
    },
]

HUB_OPS_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "tpch_q12_shipmode_priority",
        "sql": """
            SELECT l.l_shipmode,
                   SUM(CASE WHEN o.o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
                   SUM(CASE WHEN o.o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
            FROM orders o
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            WHERE l.l_shipmode IN (:shipmode, :shipmode2)
              AND l.l_commitdate < l.l_receiptdate
              AND l.l_shipdate < l.l_commitdate
              AND l.l_receiptdate >= make_date(:year::int, 1, 1)
              AND l.l_receiptdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
            GROUP BY l.l_shipmode
            ORDER BY l.l_shipmode;
        """,
        "parameters": {
            "shipmode": "{{ tpch_params.shipmode }}",
            "shipmode2": "{{ tpch_params.shipmode2 }}",
            "year": "{{ tpch_params.year }}",
        },
        "param_types": {"shipmode": "text", "shipmode2": "text", "year": "int"},
    }
]

HUB_FULL_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "tpch_q14_promo_revenue",
        "sql": """
            SELECT
              100.00 * SUM(CASE WHEN p.p_type LIKE 'PROMO%' THEN l.l_extendedprice * (1 - l.l_discount) ELSE 0 END)
              / SUM(l.l_extendedprice * (1 - l.l_discount)) AS promo_revenue
            FROM lineitem l
            JOIN part p ON p.p_partkey = l.l_partkey
            WHERE l.l_shipdate >= make_date(:year::int, 9, 1)
              AND l.l_shipdate <  make_date(:year::int, 10, 1);
        """,
        "parameters": {"year": "{{ tpch_params.year }}"},
        "param_types": {"year": "int"},
    }
]

FINAL_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "tpch_q15_top_supplier",
        "sql": """
            WITH revenue0 AS (
              SELECT l.l_suppkey AS supplier_no,
                     SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
              FROM lineitem l
              WHERE l.l_shipdate >= make_date(:year::int, 1, 1)
                AND l.l_shipdate <  make_date(:year::int, 4, 1)
              GROUP BY l.l_suppkey
            )
            SELECT s.s_suppkey, s.s_name, r.total_revenue
            FROM supplier s
            JOIN revenue0 r ON r.supplier_no = s.s_suppkey
            WHERE r.total_revenue = (SELECT MAX(total_revenue) FROM revenue0)
            ORDER BY s.s_suppkey;
        """,
        "parameters": {"year": "{{ tpch_params.year }}"},
        "param_types": {"year": "int"},
    }
]


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def entry_q1_q6_lineitem_qwen14b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Entry A: Summarize Q1 + Q6 outputs and connect them to user_query.
    Respond ONLY with:
    {
      "entry_lineitem_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def entry_q3_q5_revenue_openai20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Entry B: Summarize Q3 + Q5 outputs and connect them to user_query.
    Respond ONLY with:
    {
      "entry_revenue_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def entry_q2_q9_supply_qwen32b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Entry C: Summarize Q2 + Q9 outputs and connect them to user_query.
    Respond ONLY with:
    {
      "entry_supply_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def hub_ops_openai20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Hub1: fuse entry_lineitem_notes + entry_revenue_notes with Q12 anchor.
    Respond ONLY with:
    {
      "hub_ops_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def hub_full_context_qwen14b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Hub2: combine hub_ops_notes + entry_supply_notes with Q14 promo anchor.
    Respond ONLY with:
    {
      "hub_full_context": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def tail_focus_qwen32b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Tail: based on hub_full_context, recommend 4-6 next TPC-H queries to run
    and which params to refine (year/date/region/nation/segment/brand/shipmode/discount/qty/size/type).
    Respond ONLY with:
    {
      "tail_focus_notes": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def tpch_final_answer(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Final: use tpch_params + hub_full_context + tail_focus_notes + Q15 output
    to produce final_answer grounded in the DB results.
    Respond ONLY with:
    {
      "final_answer": "string"
    }

    {{input}}
    {{output}}
    """


async def _call_db(node_id: str, queries: Sequence[Dict[str, Any]], tpch_params: Dict[str, Any]) -> str:
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
            "contexts": [{"tpch_params": tpch_params}],
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
    tpch_params = extract_tpch_params(uq)

    # ===== 1) entry fan-out =====
    entry_lineitem_db = await _call_db("entry_q1_q6_lineitem", ENTRY_Q1_Q6_QUERIES, tpch_params)
    entry_lineitem_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params},
        [_guard_input(entry_lineitem_db or "{}")],
    )
    el_in = P.variable("user_query", content=entry_lineitem_prompt)
    el_out_var = entry_q1_q6_lineitem_qwen14b(input=el_in)

    entry_revenue_db = await _call_db("entry_q3_q5_revenue", ENTRY_Q3_Q5_QUERIES, tpch_params)
    entry_revenue_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params},
        [_guard_input(entry_revenue_db or "{}")],
    )
    er_in = P.variable("user_query", content=entry_revenue_prompt)
    er_out_var = entry_q3_q5_revenue_openai20b(input=er_in)

    entry_supply_db = await _call_db("entry_q2_q9_supply", ENTRY_Q2_Q9_QUERIES, tpch_params)
    entry_supply_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params},
        [_guard_input(entry_supply_db or "{}")],
    )
    es_in = P.variable("user_query", content=entry_supply_prompt)
    es_out_var = entry_q2_q9_supply_qwen32b(input=es_in)

    # collect entry outputs
    el_raw = await _safe_aget(el_out_var, "entry_lineitem_notes")
    er_raw = await _safe_aget(er_out_var, "entry_revenue_notes")
    es_raw = await _safe_aget(es_out_var, "entry_supply_notes")

    entry_lineitem_notes = extract_outputs(["entry_lineitem_notes"], el_raw or "{}").get("entry_lineitem_notes", "")
    entry_revenue_notes = extract_outputs(["entry_revenue_notes"], er_raw or "{}").get("entry_revenue_notes", "")
    entry_supply_notes = extract_outputs(["entry_supply_notes"], es_raw or "{}").get("entry_supply_notes", "")

    # ===== 2) hub ops =====
    hub_ops_db = await _call_db("hub_ops_openai20b", HUB_OPS_QUERIES, tpch_params)
    hub_ops_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "tpch_params": tpch_params,
            "entry_lineitem_notes": entry_lineitem_notes,
            "entry_revenue_notes": entry_revenue_notes,
        },
        [_guard_input(hub_ops_db or "{}")],
    )
    ho_in = P.variable("user_query", content=hub_ops_prompt)
    ho_out_var = hub_ops_openai20b(input=ho_in)
    hub_ops_raw = await _safe_aget(ho_out_var, "hub_ops_notes")
    hub_ops_notes = extract_outputs(["hub_ops_notes"], hub_ops_raw or "{}").get("hub_ops_notes", "")

    # ===== 3) hub full context =====
    hub_full_db = await _call_db("hub_full_context_qwen14b", HUB_FULL_QUERIES, tpch_params)
    hub_full_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "tpch_params": tpch_params,
            "hub_ops_notes": hub_ops_notes,
            "entry_supply_notes": entry_supply_notes,
        },
        [_guard_input(hub_full_db or "{}")],
    )
    hf_in = P.variable("user_query", content=hub_full_prompt)
    hf_out_var = hub_full_context_qwen14b(input=hf_in)
    hub_full_raw = await _safe_aget(hf_out_var, "hub_full_context")
    hub_full_context = extract_outputs(["hub_full_context"], hub_full_raw or "{}").get("hub_full_context", "")

    # ===== 4) tail =====
    tail_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "hub_full_context": hub_full_context},
        [],
    )
    tf_in = P.variable("user_query", content=tail_prompt)
    tf_out_var = tail_focus_qwen32b(input=tf_in)
    tail_raw = await _safe_aget(tf_out_var, "tail_focus_notes")
    tail_focus_notes = extract_outputs(["tail_focus_notes"], tail_raw or "{}").get("tail_focus_notes", "")

    # ===== 5) final =====
    final_db = await _call_db("tpch_final_answer", FINAL_QUERIES, tpch_params)
    final_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "tpch_params": tpch_params,
            "hub_full_context": hub_full_context,
            "tail_focus_notes": tail_focus_notes,
        },
        [_guard_input(final_db or "{}")],
    )
    fa_in = P.variable("user_query", content=final_prompt)
    fa_out_var = tpch_final_answer(input=fa_in)
    final_raw = await _safe_aget(fa_out_var, "final_answer")
    final_answer = extract_outputs(["final_answer"], final_raw or "{}").get("final_answer", "")

    print(f"\n===== Query {idx}/{total} =====")
    print(uq)
    print("\n=== Final Answer ===")
    print(final_answer)
    print("\n--- Tail Focus ---")
    print(tail_focus_notes)
    print("\n" + "=" * 60)


async def _run_query_async_wrapper(
    query: str,
    idx: int,
    total: int,
    start: float,
    metrics: BaselineMetrics,
) -> None:
    await run_one_query_async(query, idx, total)
    metrics.record_query(time.perf_counter() - start)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TPC-H trident Parrot workflow (template-aligned).")
    parser.add_argument("--query", type=str, default=None, help="Single query; if set, ignores file.")
    parser.add_argument("--file", type=str, default="data/tpch_user_inputs.txt", help="Input file, one query per line.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N queries.")
    parser.add_argument("--count", type=int, default=None, help="Process only the first count queries.")
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
    metrics.start(run_name="tpch_trident_parrot", total_queries=len(queries))
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
