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
from baselines.tpch_utils import extract_tpch_params
from halo_dev.utils import render_template

# Defaults aligned with local vLLM workers (ports 9101-9104)
API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
MODEL_A = os.getenv("TPCH_MODEL_A", "openai/gpt-oss-20b")
MODEL_B = os.getenv("TPCH_MODEL_B", "Qwen/Qwen3-14B")
MODEL_C = os.getenv("TPCH_MODEL_C", "Qwen/Qwen3-32B")
BASE_A = os.getenv("MODEL_A_BASE_URL", "http://localhost:9101/v1")
BASE_B = os.getenv("MODEL_B_BASE_URL", "http://localhost:9102/v1")
BASE_C = os.getenv("MODEL_C_BASE_URL", "http://localhost:9103/v1")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))
MAX_FIELD_CHARS = 2000
MAX_ROWS = 5
TEMPLATE_PATH = Path("templates/tpch_layerd.yaml")


class State(TypedDict, total=False):
    user_query: str
    tpch_params: Dict[str, Any]
    l1_ops_notes: str
    l1_trade_notes: str
    l1_demand_notes: str
    m1_brief: str
    l2_ship_notes: str
    l2_mix_notes: str
    hub_full_context: str
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
        return [_sanitize_value(v) for v in value]
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


# DB query definitions (from templates/tpch_layerd.yaml)
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


def rule_param_extractor(state: State) -> Dict[str, Any]:
    params = extract_tpch_params(state["user_query"], TEMPLATE_PATH)
    return {"tpch_params": params}


def l1_ops(state: State, model_c: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Layer1-A (Ops backlog, Q4-like):\n"
        "Summarize backlog risk from order priority distribution. Be numeric and concise.\n"
        'Respond ONLY with: { "l1_ops_notes": "string" }'
    )
    entry_db = call_db_worker_sync(
        "l1_ops_backlog_q4_32b",
        L1_OPS_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(entry_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "tpch_params": state["tpch_params"]},
        [prepared],
    )
    resp = invoke_llm(model_c, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["l1_ops_notes"], resp.content)
    return {"l1_ops_notes": parsed.get("l1_ops_notes", "")}


def l1_trade(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Layer1-B (Trade flow, Q7-like):\n"
        "Summarize revenue-by-year trade flow between the two nations.\n"
        'Respond ONLY with: { "l1_trade_notes": "string" }'
    )
    entry_db = call_db_worker_sync(
        "l1_tradeflow_q7_20b",
        L1_TRADE_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(entry_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "tpch_params": state["tpch_params"]},
        [prepared],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["l1_trade_notes"], resp.content)
    return {"l1_trade_notes": parsed.get("l1_trade_notes", "")}


def l1_demand(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Layer1-C (Demand concentration, Q10/Q18-like):\n"
        "Summarize top returned-revenue customers and identify large orders above qty threshold.\n"
        'Respond ONLY with: { "l1_demand_notes": "string" }'
    )
    entry_db = call_db_worker_sync(
        "l1_demand_q10q18_20b",
        L1_DEMAND_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(entry_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "tpch_params": state["tpch_params"]},
        [prepared],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["l1_demand_notes"], resp.content)
    return {"l1_demand_notes": parsed.get("l1_demand_notes", "")}


def m1_synth(state: State, model_b: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Merge1:\n"
        "Combine l1_ops_notes, l1_trade_notes, and l1_demand_notes into a coherent mid-level brief.\n"
        "Output should state the dominant driver (ops / trade / demand) and a hypothesis to test next.\n"
        'Respond ONLY with: { "m1_brief": "string" }'
    )
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "tpch_params": state["tpch_params"],
            "l1_ops_notes": state.get("l1_ops_notes", ""),
            "l1_trade_notes": state.get("l1_trade_notes", ""),
            "l1_demand_notes": state.get("l1_demand_notes", ""),
        },
        [],
    )
    resp = invoke_llm(model_b, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["m1_brief"], resp.content)
    return {"m1_brief": parsed.get("m1_brief", "")}


def l2_ship(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Layer2-A (Shipping diagnostics, Q12/Q21-like):\n"
        "Explain shipping-mode priority distribution and late-shipment pressure; relate back to m1_brief.\n"
        'Respond ONLY with: { "l2_ship_notes": "string" }'
    )
    entry_db = call_db_worker_sync(
        "l2_shipping_q12q21_20b",
        L2_SHIP_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(entry_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "tpch_params": state["tpch_params"],
            "m1_brief": state.get("m1_brief", ""),
        },
        [prepared],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["l2_ship_notes"], resp.content)
    return {"l2_ship_notes": parsed.get("l2_ship_notes", "")}


def l2_mix(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Layer2-B (Product & supply mix, Q16/Q19/Q11-like):\n"
        "Interpret brand/type/size supply signals, discounted revenue proxy, and regional supply value.\n"
        'Respond ONLY with: { "l2_mix_notes": "string" }'
    )
    entry_db = call_db_worker_sync(
        "l2_mix_q16q19q11_20b",
        L2_MIX_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(entry_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "tpch_params": state["tpch_params"],
            "m1_brief": state.get("m1_brief", ""),
        },
        [prepared],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["l2_mix_notes"], resp.content)
    return {"l2_mix_notes": parsed.get("l2_mix_notes", "")}


def m2_full_context(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Merge2:\n"
        "Fuse l2_ship_notes and l2_mix_notes with m1_brief into a final structured report:\n"
        "  (i) key metrics, (ii) likely bottleneck/driver, (iii) 3 parameter refinements,\n"
        "  (iv) 3 follow-up TPC-H queries (by Q#) and why.\n"
        'Respond ONLY with: { "hub_full_context": "string" }'
    )
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "tpch_params": state["tpch_params"],
            "m1_brief": state.get("m1_brief", ""),
            "l2_ship_notes": state.get("l2_ship_notes", ""),
            "l2_mix_notes": state.get("l2_mix_notes", ""),
        },
        [],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["hub_full_context"], resp.content)
    return {"hub_full_context": parsed.get("hub_full_context", "")}


def final_answer(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Final:\n"
        "Use hub_full_context plus the Q14 promo_revenue anchor to produce a concise final_answer.\n"
        'Respond ONLY with: { "final_answer": "string" }'
    )
    final_db = call_db_worker_sync(
        "tpch_final_answer_20b",
        FINAL_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(final_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "tpch_params": state["tpch_params"],
            "hub_full_context": state.get("hub_full_context", ""),
        },
        [prepared],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["final_answer"], resp.content)
    return {"final_answer": parsed.get("final_answer", "")}


def build_graph(model_a: ChatOpenAI, model_b: ChatOpenAI, model_c: ChatOpenAI) -> StateGraph[State]:
    g = StateGraph(State)
    g.add_node("rule_param_extractor", rule_param_extractor)
    g.add_node("l1_ops", lambda s: l1_ops(s, model_c))
    g.add_node("l1_trade", lambda s: l1_trade(s, model_a))
    g.add_node("l1_demand", lambda s: l1_demand(s, model_a))
    g.add_node("m1_synth", lambda s: m1_synth(s, model_b))
    g.add_node("l2_ship", lambda s: l2_ship(s, model_a))
    g.add_node("l2_mix", lambda s: l2_mix(s, model_a))
    g.add_node("m2_full_context", lambda s: m2_full_context(s, model_a))
    g.add_node("final_answer", lambda s: final_answer(s, model_a))

    g.add_edge(START, "rule_param_extractor")
    g.add_edge("rule_param_extractor", "l1_ops")
    g.add_edge("rule_param_extractor", "l1_trade")
    g.add_edge("rule_param_extractor", "l1_demand")
    g.add_edge("l1_ops", "m1_synth")
    g.add_edge("l1_trade", "m1_synth")
    g.add_edge("l1_demand", "m1_synth")
    g.add_edge("m1_synth", "l2_ship")
    g.add_edge("m1_synth", "l2_mix")
    g.add_edge("l2_ship", "m2_full_context")
    g.add_edge("l2_mix", "m2_full_context")
    g.add_edge("m2_full_context", "final_answer")
    g.add_edge("final_answer", END)
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


async def process_one_query(query: str, models: Dict[str, ChatOpenAI], workflow) -> tuple[str, State]:
    full_state: State = {"user_query": query}
    final_state: State = await asyncio.get_event_loop().run_in_executor(None, workflow.invoke, full_state)
    return query, final_state


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TPC-H layerd LangGraph workflow (Halo-aligned prompts).")
    parser.add_argument("--query", type=str, default=None, help="Single query; if set, ignores file.")
    parser.add_argument("--file", type=str, default="data/tpch_user_inputs.txt", help="Input file, one query per line.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N queries.")
    parser.add_argument("--count", type=int, default=None, help="Process only the first count queries.")
    args = parser.parse_args(argv)

    metrics = BaselineMetrics()
    models = {
        "a": make_model(MODEL_A, BASE_A),
        "b": make_model(MODEL_B, BASE_B),
        "c": make_model(MODEL_C, BASE_C),
    }

    graph = build_graph(models["a"], models["b"], models["c"]).compile()

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
    metrics.start(run_name="tpch_layerd_langgraph", total_queries=len(queries))
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
            idx, q, final_state, elapsed = await coro
            metrics.record_query(elapsed)
            print(f"\n===== Query #{idx}/{total} =====\n{q}")
            print("\n=== Final Answer ===")
            print(final_state.get("final_answer", ""))
            print("\n--- Hub Full Context ---")
            print(final_state.get("hub_full_context", ""))
            print("\n" + "=" * 60)

    set_global_metrics(metrics)
    asyncio.run(runner())
    set_global_metrics(None)
    metrics.finish()
    print("\n" + metrics.format_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
