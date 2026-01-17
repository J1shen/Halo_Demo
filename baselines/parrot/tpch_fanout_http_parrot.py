"""Baseline Parrot workflow for the TPC-H fanout HTTP template (template-aligned).

- Mirrors the nodes from templates/tpch_fanout_http.yaml.
- Keeps baseline behavior: sequential orchestration in Python (per query).
- Uses the shared DB worker (gunicorn FastAPI) for all SQL.
- Uses Halo's HTTP node executor for HTTP stand-ins.
- Semantic functions use ONLY (input, output) variables (fanout-style).
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
from halo_dev.executor import HTTPNodeExecutor
from halo_dev.models import Node

# ---- Runtime wiring ----
CORE_HTTP_ADDR = os.getenv("PARROT_CORE_HTTP", "http://localhost:9000")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))
HTTP_DEFAULT_SLEEP_S = float(os.getenv("HALO_HTTP_DEFAULT_SLEEP_S", "0.0"))
HTTP_CONCURRENCY = int(os.getenv("HALO_HTTP_CONCURRENCY", "1"))

MAX_FIELD_CHARS = 2048
MAX_ROWS = 5
TEMPLATE_PATH = Path("templates/tpch_fanout_http.yaml")

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

HTTP_EXECUTOR = HTTPNodeExecutor(
    http_concurrency=HTTP_CONCURRENCY,
    default_sleep_s=HTTP_DEFAULT_SLEEP_S,
)

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
S1_Q5_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "s1_q5_nation_revenue_light_v3",
        "sql": """
            SELECT
              n.n_name AS nation,
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
            FROM orders o
            JOIN customer c  ON c.c_custkey = o.o_custkey
            JOIN nation n    ON n.n_nationkey = c.c_nationkey
            JOIN lineitem l  ON l.l_orderkey = o.o_orderkey
            WHERE c.c_mktsegment = :segment
              AND o.o_orderdate >= make_date(:year::int, 1, 1)
              AND o.o_orderdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
            GROUP BY n.n_name
            ORDER BY revenue DESC, nation
            LIMIT 12;
        """,
        "parameters": {"year": "{{ tpch_params.year }}", "segment": "{{ tpch_params.segment }}"},
        "param_types": {"year": "int", "segment": "str"},
    },
    {
        "name": "s1_q5_brand_volume_share_light_v3",
        "sql": """
            SELECT
              EXTRACT(YEAR FROM o.o_orderdate)::int AS o_year,
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_volume,
              SUM(CASE WHEN p.p_brand = :brand
                       THEN l.l_extendedprice * (1 - l.l_discount) ELSE 0 END) AS brand_volume
            FROM orders o
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            JOIN part p     ON p.p_partkey = l.l_partkey
            WHERE o.o_orderdate >= make_date(:year::int, 1, 1)
              AND o.o_orderdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
            GROUP BY o_year
            ORDER BY o_year;
        """,
        "parameters": {"year": "{{ tpch_params.year }}", "brand": "{{ tpch_params.brand }}"},
        "param_types": {"year": "int", "brand": "str"},
    },
    {
        "name": "s1_q5_return_revenue_by_nation_light_v3",
        "sql": """
            SELECT
              n.n_name AS nation,
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
              COUNT(*) AS line_cnt
            FROM customer c
            JOIN nation n   ON n.n_nationkey = c.c_nationkey
            JOIN orders o   ON o.o_custkey = c.c_custkey
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            WHERE c.c_mktsegment = :segment
              AND o.o_orderdate >= make_date(:year::int, 1, 1)
              AND o.o_orderdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
              AND l.l_returnflag = 'R'
            GROUP BY n.n_name
            ORDER BY revenue DESC, nation
            LIMIT 12;
        """,
        "parameters": {"year": "{{ tpch_params.year }}", "segment": "{{ tpch_params.segment }}"},
        "param_types": {"year": "int", "segment": "str"},
    },
    {
        "name": "s1_q5_brand_inventory_by_supplier_nation_light_v3",
        "sql": """
            SELECT
              n.n_name AS nation,
              SUM(ps.ps_supplycost * ps.ps_availqty) AS inv_cost,
              COUNT(DISTINCT s.s_suppkey) AS supplier_cnt
            FROM part p
            JOIN partsupp ps ON ps.ps_partkey = p.p_partkey
            JOIN supplier s  ON s.s_suppkey = ps.ps_suppkey
            JOIN nation n    ON n.n_nationkey = s.s_nationkey
            WHERE p.p_brand = :brand
            GROUP BY n.n_name
            ORDER BY inv_cost DESC, nation
            LIMIT 12;
        """,
        "parameters": {"brand": "{{ tpch_params.brand }}"},
        "param_types": {"brand": "str"},
    },
]

# HTTP stand-ins (must match template node ids/outputs).
S1_Q3_HTTP_NODES: List[Dict[str, Any]] = [
    {
        "id": "http_s1_q3_q3_segment_revenue",
        "output": "http_s1_q3_q3_segment_revenue",
        "sleep_s": 2.0,
    },
    {
        "id": "http_s1_q3_q12_shipmode_priority",
        "output": "http_s1_q3_q12_shipmode_priority",
        "sleep_s": 2.0,
    },
]

S1_Q1_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "s1_q1_discounted_revenue",
        "sql": """
            SELECT
              SUM(l.l_extendedprice * l.l_discount) AS total_price,
              COUNT(*) AS line_cnt,
              AVG(l.l_discount) AS avg_discount
            FROM lineitem l
            WHERE l.l_shipdate >= make_date(:year::int, 1, 1)
              AND l.l_shipdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
              AND l.l_discount BETWEEN :discount_lo AND :discount_hi
              AND l.l_quantity < :qty
              AND l.l_returnflag IN ('R','A');
        """,
        "parameters": {
            "year": "{{ tpch_params.year }}",
            "qty": "{{ tpch_params.qty }}",
            "discount_lo": "{{ tpch_params.discount_lo }}",
            "discount_hi": "{{ tpch_params.discount_hi }}",
        },
        "param_types": {"year": "int", "qty": "int", "discount_lo": "float", "discount_hi": "float"},
    }
]

S2_Q5_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "s2_q5_nation_revenue_light_v3",
        "sql": """
            SELECT
              n.n_name AS nation,
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
            FROM orders o
            JOIN customer c  ON c.c_custkey = o.o_custkey
            JOIN nation n    ON n.n_nationkey = c.c_nationkey
            JOIN lineitem l  ON l.l_orderkey = o.o_orderkey
            WHERE c.c_mktsegment = :segment
              AND o.o_orderdate >= make_date(:year::int, 1, 1)
              AND o.o_orderdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
            GROUP BY n.n_name
            ORDER BY revenue DESC, nation
            LIMIT 12;
        """,
        "parameters": {"year": "{{ tpch_params.year }}", "segment": "{{ tpch_params.segment }}"},
        "param_types": {"year": "int", "segment": "str"},
    },
    {
        "name": "s2_q5_brand_volume_share_light_v3",
        "sql": """
            SELECT
              EXTRACT(YEAR FROM o.o_orderdate)::int AS o_year,
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_volume,
              SUM(CASE WHEN p.p_brand = :brand
                       THEN l.l_extendedprice * (1 - l.l_discount) ELSE 0 END) AS brand_volume
            FROM orders o
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            JOIN part p     ON p.p_partkey = l.l_partkey
            WHERE o.o_orderdate >= make_date(:year::int, 1, 1)
              AND o.o_orderdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
            GROUP BY o_year
            ORDER BY o_year;
        """,
        "parameters": {"year": "{{ tpch_params.year }}", "brand": "{{ tpch_params.brand }}"},
        "param_types": {"year": "int", "brand": "str"},
    },
    {
        "name": "s2_q5_return_revenue_by_nation_light_v3",
        "sql": """
            SELECT
              n.n_name AS nation,
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
              COUNT(*) AS line_cnt
            FROM customer c
            JOIN nation n   ON n.n_nationkey = c.c_nationkey
            JOIN orders o   ON o.o_custkey = c.c_custkey
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            WHERE c.c_mktsegment = :segment
              AND o.o_orderdate >= make_date(:year::int, 1, 1)
              AND o.o_orderdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
              AND l.l_returnflag = 'R'
            GROUP BY n.n_name
            ORDER BY revenue DESC, nation
            LIMIT 12;
        """,
        "parameters": {"year": "{{ tpch_params.year }}", "segment": "{{ tpch_params.segment }}"},
        "param_types": {"year": "int", "segment": "str"},
    },
    {
        "name": "s2_q5_brand_inventory_by_supplier_nation_light_v3",
        "sql": """
            SELECT
              n.n_name AS nation,
              SUM(ps.ps_supplycost * ps.ps_availqty) AS inv_cost,
              COUNT(DISTINCT s.s_suppkey) AS supplier_cnt
            FROM part p
            JOIN partsupp ps ON ps.ps_partkey = p.p_partkey
            JOIN supplier s  ON s.s_suppkey = ps.ps_suppkey
            JOIN nation n    ON n.n_nationkey = s.s_nationkey
            WHERE p.p_brand = :brand
            GROUP BY n.n_name
            ORDER BY inv_cost DESC, nation
            LIMIT 12;
        """,
        "parameters": {"brand": "{{ tpch_params.brand }}"},
        "param_types": {"brand": "str"},
    },
]

S2_Q1_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "s2_q1_discounted_revenue",
        "sql": """
            SELECT
              SUM(l.l_extendedprice * l.l_discount) AS total_price,
              COUNT(*) AS line_cnt,
              AVG(l.l_discount) AS avg_discount
            FROM lineitem l
            WHERE l.l_shipdate >= make_date(:year::int, 1, 1)
              AND l.l_shipdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
              AND l.l_discount BETWEEN :discount_lo AND :discount_hi
              AND l.l_quantity < :qty
              AND l.l_returnflag IN ('R','A');
        """,
        "parameters": {
            "year": "{{ tpch_params.year }}",
            "qty": "{{ tpch_params.qty }}",
            "discount_lo": "{{ tpch_params.discount_lo }}",
            "discount_hi": "{{ tpch_params.discount_hi }}",
        },
        "param_types": {"year": "int", "qty": "int", "discount_lo": "float", "discount_hi": "float"},
    }
]


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def stage1_econ_rollup_q5(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are a TPC-H analyst agent (Stage 1 / Q5-pack).

    Inputs:
    - user_query
    - tpch_params
    - results of EXACTLY FOUR DB queries

    Do:
    - For each query, extract 2 key numeric facts (leaders/totals/shares).
    - Provide 1 cross-query connection.
    - Provide 1 warning about interpretation risk (skew, sparsity, time window).

    Output JSON only:
    {
      "s1_q5_brief": "one dense paragraph",
      "per_query_facts": [{"query":"string","facts":["string","string"]}],
      "cross_link": "string",
      "risk": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_B], model_type="text", remove_pure_fill=False)
def stage1_seg_ship_q3(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are a TPC-H reasoning agent (Stage 1 / Q3-pack).

    You will receive EXACTLY TWO API payloads:
    - http_s1_q3_q3_segment_revenue
    - http_s1_q3_q12_shipmode_priority

    Requirements:
    - Use numbers: cite totals and at least one breakdown item.
    - Provide 2 insights and 2 sanity checks.

    Output JSON only:
    {
      "s1_q3_brief": "one dense paragraph",
      "insights": ["string","string"],
      "sanity": ["string","string"]
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def stage1_disc_band_q1(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Numeric summarizer (Stage 1 / Q1-pack).

    Interpret the discount-band revenue query:
    - explain what it measures
    - extract key metrics (total, count, avg_discount)
    - provide one interpretation linked to tpch_params

    Output JSON only:
    {
      "s1_q1_brief": "one paragraph",
      "metrics": {"total_price":"number or null","line_cnt":"int or null","avg_discount":"number or null"},
      "interpretation": "string"
    }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def stage1_hypothesis_q0a(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Stage 1 quick planner (no DB queries).

    Using user_query + tpch_params:
    - propose 2 hypotheses
    - propose 2 checks to validate DB results

    Output JSON only:
    { "s1_q0a_brief": "string", "hypotheses": ["string","string"], "checks": ["string","string"] }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def stage1_altangle_q0b(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Stage 1 alternative angle (no DB queries).

    Provide:
    - 2 alternative explanations that might mislead aggregates
    - 2 questions for stage 2

    Output JSON only:
    { "s1_q0b_brief": "string", "alt_explanations": ["string","string"], "questions": ["string","string"] }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def stage2_econ_rollup_q5(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """You are a TPC-H analyst agent (Stage 2 / Q5-pack).

    Inputs:
    - user_query, tpch_params
    - stage1 briefs: s1_q5_brief, s1_q3_brief, s1_q1_brief, s1_q0a_brief, s1_q0b_brief
    - results of EXACTLY FOUR DB queries

    Tasks:
    - Validate at least 1 stage1 hypothesis/check using concrete numbers.
    - Provide 2 delta-aware statements (consistent vs uncertain).
    - Produce a compact brief.

    Output JSON only:
    { "s2_q5_brief": "one dense paragraph", "deltas": ["string","string"] }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_A], model_type="text", remove_pure_fill=False)
def stage2_disc_band_q1(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Stage 2 numeric summarizer (Q1-pack).

    Use stage1 briefs to contextualize the discount-band revenue:
    - Does it align with other patterns?
    - Note 1 caveat.

    Output JSON only:
    { "s2_q1_brief": "one paragraph", "notes": ["string","string"] }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def stage2_narrative_q0a(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Stage 2 narrative synthesizer (no DB queries).

    Provide:
    - 2 candidate narratives
    - 2 caveats

    Output JSON only:
    { "s2_q0a_brief": "string", "narratives": ["string","string"], "caveats": ["string","string"] }

    {{input}}
    {{output}}
    """


@P.semantic_function(formatter=P.allowing_newline, models=[MODEL_C], model_type="text", remove_pure_fill=False)
def stage2_final_merge(input: P.Input, output: P.Output(SAMPLING_CONFIG)):
    """Final merger.

    Inputs: s2_q5_brief, s2_q1_brief, s2_q0a_brief
    Write a coherent final answer that:
    - includes at least 3 concrete numeric facts or ranked statements
    - includes 1 caveat
    - ends with 2 follow-up questions

    Output JSON only:
    { "final_answer": "string" }

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


async def _call_http(
    node_id: str,
    output_name: str,
    *,
    sleep_s: float,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    def _do_call() -> Dict[str, Any]:
        node = Node(
            id=node_id,
            type="http",
            engine="http",
            outputs=(output_name,),
            raw={"sleep_s": sleep_s},
        )
        outputs = HTTP_EXECUTOR.execute(node, context)
        return outputs.get(output_name, {})

    return await asyncio.to_thread(_do_call)


async def run_one_query_async(user_query: str, idx: int | None = None, total: int | None = None) -> None:
    uq = _guard_input(user_query)
    tpch_params = extract_tpch_params(uq, TEMPLATE_PATH)

    # ===== Stage 1 =====
    s1_q5_db = await _call_db("stage1_econ_rollup_q5", S1_Q5_QUERIES, tpch_params)
    s1_q5_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params},
        [_guard_input(s1_q5_db or "{}")],
    )
    s1_q5_in = P.variable("user_query", content=s1_q5_prompt)
    s1_q5_out_var = stage1_econ_rollup_q5(input=s1_q5_in)

    s1_q3_payloads: Dict[str, Any] = {}
    for spec in S1_Q3_HTTP_NODES:
        payload = await _call_http(
            spec["id"],
            spec["output"],
            sleep_s=spec["sleep_s"],
            context={"user_query": uq, "tpch_params": tpch_params},
        )
        s1_q3_payloads[spec["output"]] = payload
    s1_q3_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "tpch_params": tpch_params,
            **s1_q3_payloads,
        },
        [],
    )
    s1_q3_in = P.variable("user_query", content=s1_q3_prompt)
    s1_q3_out_var = stage1_seg_ship_q3(input=s1_q3_in)

    s1_q1_db = await _call_db("stage1_disc_band_q1", S1_Q1_QUERIES, tpch_params)
    s1_q1_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params},
        [_guard_input(s1_q1_db or "{}")],
    )
    s1_q1_in = P.variable("user_query", content=s1_q1_prompt)
    s1_q1_out_var = stage1_disc_band_q1(input=s1_q1_in)

    s1_q0a_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params},
        [],
    )
    s1_q0a_in = P.variable("user_query", content=s1_q0a_prompt)
    s1_q0a_out_var = stage1_hypothesis_q0a(input=s1_q0a_in)

    s1_q0b_prompt = build_halo_prompt(
        None,
        {"user_query": uq, "tpch_params": tpch_params},
        [],
    )
    s1_q0b_in = P.variable("user_query", content=s1_q0b_prompt)
    s1_q0b_out_var = stage1_altangle_q0b(input=s1_q0b_in)

    s1_q5_raw = await _safe_aget(s1_q5_out_var, "s1_q5_brief")
    s1_q3_raw = await _safe_aget(s1_q3_out_var, "s1_q3_brief")
    s1_q1_raw = await _safe_aget(s1_q1_out_var, "s1_q1_brief")
    s1_q0a_raw = await _safe_aget(s1_q0a_out_var, "s1_q0a_brief")
    s1_q0b_raw = await _safe_aget(s1_q0b_out_var, "s1_q0b_brief")

    s1_q5_brief = extract_outputs(["s1_q5_brief"], s1_q5_raw or "{}").get("s1_q5_brief", "")
    s1_q3_brief = extract_outputs(["s1_q3_brief"], s1_q3_raw or "{}").get("s1_q3_brief", "")
    s1_q1_brief = extract_outputs(["s1_q1_brief"], s1_q1_raw or "{}").get("s1_q1_brief", "")
    s1_q0a_brief = extract_outputs(["s1_q0a_brief"], s1_q0a_raw or "{}").get("s1_q0a_brief", "")
    s1_q0b_brief = extract_outputs(["s1_q0b_brief"], s1_q0b_raw or "{}").get("s1_q0b_brief", "")

    # ===== Stage 2 =====
    s2_q5_db = await _call_db("stage2_econ_rollup_q5", S2_Q5_QUERIES, tpch_params)
    s2_q5_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "tpch_params": tpch_params,
            "s1_q5_brief": s1_q5_brief,
            "s1_q3_brief": s1_q3_brief,
            "s1_q1_brief": s1_q1_brief,
            "s1_q0a_brief": s1_q0a_brief,
            "s1_q0b_brief": s1_q0b_brief,
        },
        [_guard_input(s2_q5_db or "{}")],
    )
    s2_q5_in = P.variable("user_query", content=s2_q5_prompt)
    s2_q5_out_var = stage2_econ_rollup_q5(input=s2_q5_in)

    s2_q1_db = await _call_db("stage2_disc_band_q1", S2_Q1_QUERIES, tpch_params)
    s2_q1_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "tpch_params": tpch_params,
            "s1_q5_brief": s1_q5_brief,
            "s1_q3_brief": s1_q3_brief,
            "s1_q1_brief": s1_q1_brief,
            "s1_q0a_brief": s1_q0a_brief,
            "s1_q0b_brief": s1_q0b_brief,
        },
        [_guard_input(s2_q1_db or "{}")],
    )
    s2_q1_in = P.variable("user_query", content=s2_q1_prompt)
    s2_q1_out_var = stage2_disc_band_q1(input=s2_q1_in)

    s2_q0a_prompt = build_halo_prompt(
        None,
        {
            "user_query": uq,
            "tpch_params": tpch_params,
            "s1_q5_brief": s1_q5_brief,
            "s1_q3_brief": s1_q3_brief,
            "s1_q1_brief": s1_q1_brief,
            "s1_q0a_brief": s1_q0a_brief,
            "s1_q0b_brief": s1_q0b_brief,
        },
        [],
    )
    s2_q0a_in = P.variable("user_query", content=s2_q0a_prompt)
    s2_q0a_out_var = stage2_narrative_q0a(input=s2_q0a_in)

    s2_q5_raw = await _safe_aget(s2_q5_out_var, "s2_q5_brief")
    s2_q1_raw = await _safe_aget(s2_q1_out_var, "s2_q1_brief")
    s2_q0a_raw = await _safe_aget(s2_q0a_out_var, "s2_q0a_brief")

    s2_q5_brief = extract_outputs(["s2_q5_brief"], s2_q5_raw or "{}").get("s2_q5_brief", "")
    s2_q1_brief = extract_outputs(["s2_q1_brief"], s2_q1_raw or "{}").get("s2_q1_brief", "")
    s2_q0a_brief = extract_outputs(["s2_q0a_brief"], s2_q0a_raw or "{}").get("s2_q0a_brief", "")

    # ===== Final =====
    final_prompt = build_halo_prompt(
        None,
        {
            "s2_q5_brief": s2_q5_brief,
            "s2_q1_brief": s2_q1_brief,
            "s2_q0a_brief": s2_q0a_brief,
        },
        [],
    )
    final_in = P.variable("user_query", content=final_prompt)
    final_out_var = stage2_final_merge(input=final_in)
    final_raw = await _safe_aget(final_out_var, "final_answer")
    final_answer = extract_outputs(["final_answer"], final_raw or "{}").get("final_answer", "")

    print(f"\n===== Query {idx}/{total} =====")
    print(uq)
    print("\n=== Final Answer ===")
    print(final_answer)
    print("\n--- Stage2 Q5 Brief ---")
    print(s2_q5_brief)
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
    parser = argparse.ArgumentParser(
        description="TPC-H fanout HTTP Parrot workflow (template-aligned)."
    )
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
    metrics.start(run_name="tpch_fanout_http_parrot", total_queries=len(queries))
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
