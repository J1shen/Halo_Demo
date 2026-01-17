"""Baseline Parrot workflow for the TPC-H layerd template (template-aligned).

- Mirrors the nodes from templates/tpch_layerd.yaml (tpch_qbench_layered_rules_v1).
- Keeps baseline behavior: sequential orchestration in Python (per query).
- Uses the shared DB worker (gunicorn FastAPI) for all SQL.
- Semantic functions use ONLY (input, output) variables (layered-style).
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
TEMPLATE_PATH = Path("templates/tpch_layerd.yaml")

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
L1_OPS_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "tpch_q4_order_priority_check",
        "sql": """
            SELECT
              o.o_orderpriority,
              COUNT(*) AS order_count
            FROM orders o
            WHERE o.o_orderdate >= make_date(:year::int, 1, 1)
              AND o.o_orderdate <  make_date(:year::int, 1, 1) + INTERVAL '3 month'
              AND EXISTS (
                SELECT 1
                FROM lineitem l
                WHERE l.l_orderkey = o.o_orderkey
                  AND l.l_commitdate < l.l_receiptdate
              )
            GROUP BY o.o_orderpriority
            ORDER BY o.o_orderpriority;
        """,
        "parameters": {"year": "{{ tpch_params.year }}"},
        "param_types": {"year": "int"},
    }
]

L1_TRADE_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "tpch_q7_volume_between_two_nations",
        "sql": """
            SELECT
              supp_nation.n_name AS supp_nation,
              cust_nation.n_name AS cust_nation,
              EXTRACT(YEAR FROM l.l_shipdate)::int AS l_year,
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
            FROM supplier s
            JOIN nation supp_nation ON supp_nation.n_nationkey = s.s_nationkey
            JOIN lineitem l ON l.l_suppkey = s.s_suppkey
            JOIN orders o ON o.o_orderkey = l.l_orderkey
            JOIN customer c ON c.c_custkey = o.o_custkey
            JOIN nation cust_nation ON cust_nation.n_nationkey = c.c_nationkey
            WHERE (
                (supp_nation.n_name = :nation AND cust_nation.n_name = :nation2)
             OR (supp_nation.n_name = :nation2 AND cust_nation.n_name = :nation)
            )
              AND l.l_shipdate >= make_date(:year::int, 1, 1)
              AND l.l_shipdate <  make_date(:year::int, 1, 1) + INTERVAL '2 year'
            GROUP BY supp_nation, cust_nation, l_year
            ORDER BY supp_nation, cust_nation, l_year;
        """,
        "parameters": {
            "nation": "{{ tpch_params.nation }}",
            "nation2": "{{ tpch_params.nation2 }}",
            "year": "{{ tpch_params.year }}",
        },
        "param_types": {"nation": "text", "nation2": "text", "year": "int"},
    }
]

L1_DEMAND_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "tpch_q10_top_customers_returned",
        "sql": """
            SELECT
              c.c_custkey,
              c.c_name,
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
            FROM customer c
            JOIN orders o ON o.o_custkey = c.c_custkey
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            WHERE l.l_returnflag = 'R'
              AND o.o_orderdate >= make_date(:year::int, 1, 1)
              AND o.o_orderdate <  make_date(:year::int, 1, 1) + INTERVAL '3 month'
            GROUP BY c.c_custkey, c.c_name
            ORDER BY revenue DESC
            LIMIT 20;
        """,
        "parameters": {"year": "{{ tpch_params.year }}"},
        "param_types": {"year": "int"},
    },
    {
        "name": "tpch_q18_large_orders",
        "sql": """
            SELECT
              o.o_orderkey,
              SUM(l.l_quantity) AS sum_qty
            FROM orders o
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            GROUP BY o.o_orderkey
            HAVING SUM(l.l_quantity) > :qty::numeric
            ORDER BY sum_qty DESC
            LIMIT 20;
        """,
        "parameters": {"qty": "{{ tpch_params.qty }}"},
        "param_types": {"qty": "numeric"},
    },
]

L2_SHIP_QUERIES: List[Dict[str, Any]] = [
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
    },
    {
        "name": "tpch_q21_late_shipments_proxy",
        "sql": """
            SELECT COUNT(*) AS late_line_cnt
            FROM lineitem l
            WHERE l.l_receiptdate > l.l_commitdate
              AND l.l_shipdate >= make_date(:year::int, 1, 1)
              AND l.l_shipdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year';
        """,
        "parameters": {"year": "{{ tpch_params.year }}"},
        "param_types": {"year": "int"},
    },
]

L2_MIX_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "tpch_q16_parts_brand_type_size",
        "sql": """
            SELECT
              p.p_brand,
              p.p_type,
              p.p_size,
              COUNT(*) AS part_cnt
            FROM part p
            WHERE p.p_brand = :brand
              AND p.p_type ILIKE '%' || :type || '%'
              AND p.p_size = :size::int
            GROUP BY p.p_brand, p.p_type, p.p_size
            ORDER BY part_cnt DESC;
        """,
        "parameters": {
            "brand": "{{ tpch_params.brand }}",
            "type": "{{ tpch_params.type }}",
            "size": "{{ tpch_params.size }}",
        },
        "param_types": {"brand": "text", "type": "text", "size": "int"},
    },
    {
        "name": "tpch_q19_discounted_revenue_proxy",
        "sql": """
            SELECT
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS disc_revenue
            FROM lineitem l
            JOIN part p ON p.p_partkey = l.l_partkey
            WHERE p.p_brand = :brand
              AND l.l_discount BETWEEN :discount_lo::numeric AND :discount_hi::numeric
              AND l.l_quantity < :qty::numeric;
        """,
        "parameters": {
            "brand": "{{ tpch_params.brand }}",
            "discount_lo": "{{ tpch_params.discount_lo }}",
            "discount_hi": "{{ tpch_params.discount_hi }}",
            "qty": "{{ tpch_params.qty }}",
        },
        "param_types": {
            "brand": "text",
            "discount_lo": "numeric",
            "discount_hi": "numeric",
            "qty": "numeric",
        },
    },
    {
        "name": "tpch_q11_partsupp_value_region_proxy",
        "sql": """
            SELECT
              SUM(ps.ps_supplycost * ps.ps_availqty) AS total_value
            FROM partsupp ps
            JOIN supplier s ON s.s_suppkey = ps.ps_suppkey
            JOIN nation n ON n.n_nationkey = s.s_nationkey
            JOIN region r ON r.r_regionkey = n.n_regionkey
            WHERE r.r_name = :region;
        """,
        "parameters": {"region": "{{ tpch_params.region }}"},
        "param_types": {"region": "text"},
    },
]

FINAL_QUERIES: List[Dict[str, Any]] = [
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


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def l1_ops_backlog_q4_32b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Layer1-A (Ops backlog, Q4-like):
    Summarize backlog risk from order priority distribution. Be numeric and concise.
    Respond ONLY with: { "l1_ops_notes": "string" }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def l1_tradeflow_q7_20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Layer1-B (Trade flow, Q7-like):
    Summarize revenue-by-year trade flow between the two nations.
    Respond ONLY with: { "l1_trade_notes": "string" }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def l1_demand_q10q18_20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Layer1-C (Demand concentration, Q10/Q18-like):
    Summarize top returned-revenue customers and identify large orders above qty threshold.
    Respond ONLY with: { "l1_demand_notes": "string" }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def m1_synth_14b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Merge1:
    Combine l1_ops_notes, l1_trade_notes, and l1_demand_notes into a coherent mid-level brief.
    Output should state the dominant driver (ops / trade / demand) and a hypothesis to test next.
    Respond ONLY with: { "m1_brief": "string" }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def l2_shipping_q12q21_20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Layer2-A (Shipping diagnostics, Q12/Q21-like):
    Explain shipping-mode priority distribution and late-shipment pressure; relate back to m1_brief.
    Respond ONLY with: { "l2_ship_notes": "string" }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def l2_mix_q16q19q11_20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Layer2-B (Product & supply mix, Q16/Q19/Q11-like):
    Interpret brand/type/size supply signals, discounted revenue proxy, and regional supply value.
    Respond ONLY with: { "l2_mix_notes": "string" }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def m2_full_context_20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Merge2:
    Fuse l2_ship_notes and l2_mix_notes with m1_brief into a final structured report:
      (i) key metrics, (ii) likely bottleneck/driver, (iii) 3 parameter refinements,
      (iv) 3 follow-up TPC-H queries (by Q#) and why.
    Respond ONLY with: { "hub_full_context": "string" }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def tpch_final_answer_20b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Final:
    Use hub_full_context plus the Q14 promo_revenue anchor to produce a concise final_answer.
    Respond ONLY with: { "final_answer": "string" }

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
    tpch_params = extract_tpch_params(uq, TEMPLATE_PATH)

    # ===== Layer 1 =====
    l1_ops_db = await _call_db("l1_ops_backlog_q4_32b", L1_OPS_QUERIES, tpch_params)
    l1_ops_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params},
        [_guard_input(l1_ops_db or "{}")],
    )
    l1_ops_in = P.variable("user_query", content=l1_ops_prompt)
    l1_ops_out_var = l1_ops_backlog_q4_32b(input=l1_ops_in)

    l1_trade_db = await _call_db("l1_tradeflow_q7_20b", L1_TRADE_QUERIES, tpch_params)
    l1_trade_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params},
        [_guard_input(l1_trade_db or "{}")],
    )
    l1_trade_in = P.variable("user_query", content=l1_trade_prompt)
    l1_trade_out_var = l1_tradeflow_q7_20b(input=l1_trade_in)

    l1_demand_db = await _call_db("l1_demand_q10q18_20b", L1_DEMAND_QUERIES, tpch_params)
    l1_demand_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params},
        [_guard_input(l1_demand_db or "{}")],
    )
    l1_demand_in = P.variable("user_query", content=l1_demand_prompt)
    l1_demand_out_var = l1_demand_q10q18_20b(input=l1_demand_in)

    l1_ops_raw = await _safe_aget(l1_ops_out_var, "l1_ops_notes")
    l1_trade_raw = await _safe_aget(l1_trade_out_var, "l1_trade_notes")
    l1_demand_raw = await _safe_aget(l1_demand_out_var, "l1_demand_notes")

    l1_ops_notes = extract_outputs(["l1_ops_notes"], l1_ops_raw or "{}").get("l1_ops_notes", "")
    l1_trade_notes = extract_outputs(["l1_trade_notes"], l1_trade_raw or "{}").get("l1_trade_notes", "")
    l1_demand_notes = extract_outputs(["l1_demand_notes"], l1_demand_raw or "{}").get("l1_demand_notes", "")

    # ===== Merge 1 =====
    m1_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "tpch_params": tpch_params,
            "l1_ops_notes": l1_ops_notes,
            "l1_trade_notes": l1_trade_notes,
            "l1_demand_notes": l1_demand_notes,
        },
        [],
    )
    m1_in = P.variable("user_query", content=m1_prompt)
    m1_out_var = m1_synth_14b(input=m1_in)
    m1_raw = await _safe_aget(m1_out_var, "m1_brief")
    m1_brief = extract_outputs(["m1_brief"], m1_raw or "{}").get("m1_brief", "")

    # ===== Layer 2 =====
    l2_ship_db = await _call_db("l2_shipping_q12q21_20b", L2_SHIP_QUERIES, tpch_params)
    l2_ship_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params, "m1_brief": m1_brief},
        [_guard_input(l2_ship_db or "{}")],
    )
    l2_ship_in = P.variable("user_query", content=l2_ship_prompt)
    l2_ship_out_var = l2_shipping_q12q21_20b(input=l2_ship_in)

    l2_mix_db = await _call_db("l2_mix_q16q19q11_20b", L2_MIX_QUERIES, tpch_params)
    l2_mix_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params, "m1_brief": m1_brief},
        [_guard_input(l2_mix_db or "{}")],
    )
    l2_mix_in = P.variable("user_query", content=l2_mix_prompt)
    l2_mix_out_var = l2_mix_q16q19q11_20b(input=l2_mix_in)

    l2_ship_raw = await _safe_aget(l2_ship_out_var, "l2_ship_notes")
    l2_mix_raw = await _safe_aget(l2_mix_out_var, "l2_mix_notes")
    l2_ship_notes = extract_outputs(["l2_ship_notes"], l2_ship_raw or "{}").get("l2_ship_notes", "")
    l2_mix_notes = extract_outputs(["l2_mix_notes"], l2_mix_raw or "{}").get("l2_mix_notes", "")

    # ===== Merge 2 =====
    m2_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "tpch_params": tpch_params,
            "m1_brief": m1_brief,
            "l2_ship_notes": l2_ship_notes,
            "l2_mix_notes": l2_mix_notes,
        },
        [],
    )
    m2_in = P.variable("user_query", content=m2_prompt)
    m2_out_var = m2_full_context_20b(input=m2_in)
    m2_raw = await _safe_aget(m2_out_var, "hub_full_context")
    hub_full_context = extract_outputs(["hub_full_context"], m2_raw or "{}").get("hub_full_context", "")

    # ===== Final =====
    final_db = await _call_db("tpch_final_answer_20b", FINAL_QUERIES, tpch_params)
    final_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params, "hub_full_context": hub_full_context},
        [_guard_input(final_db or "{}")],
    )
    final_in = P.variable("user_query", content=final_prompt)
    final_out_var = tpch_final_answer_20b(input=final_in)
    final_raw = await _safe_aget(final_out_var, "final_answer")
    final_answer = extract_outputs(["final_answer"], final_raw or "{}").get("final_answer", "")

    print(f"\n===== Query {idx}/{total} =====")
    print(uq)
    print("\n=== Final Answer ===")
    print(final_answer)
    print("\n--- Hub Full Context ---")
    print(hub_full_context)
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
    parser = argparse.ArgumentParser(description="TPC-H layerd Parrot workflow (template-aligned).")
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
    metrics.start(run_name="tpch_layerd_parrot", total_queries=len(queries))
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
