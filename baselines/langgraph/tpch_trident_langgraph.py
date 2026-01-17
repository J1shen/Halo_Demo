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


class State(TypedDict, total=False):
    user_query: str
    tpch_params: Dict[str, Any]
    entry_lineitem_notes: str
    entry_revenue_notes: str
    entry_supply_notes: str
    hub_ops_notes: str
    hub_full_context: str
    tail_focus_notes: str
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


# DB query definitions (from templates/tpch_trident.yaml)
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


def rule_param_extractor(state: State) -> Dict[str, Any]:
    params = extract_tpch_params(state["user_query"])
    return {"tpch_params": params}


def entry_lineitem(state: State, model_b: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Entry A: Summarize Q1 + Q6 outputs and connect them to user_query.\n"
        'Respond ONLY with: { "entry_lineitem_notes": "string" }'
    )
    entry_db = call_db_worker_sync(
        "entry_q1_q6_lineitem",
        ENTRY_Q1_Q6_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(entry_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "tpch_params": state["tpch_params"]},
        [prepared],
    )
    resp = invoke_llm(model_b, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["entry_lineitem_notes"], resp.content)
    return {"entry_lineitem_notes": parsed.get("entry_lineitem_notes", "")}


def entry_revenue(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Entry B: Summarize Q3 + Q5 outputs and connect them to user_query.\n"
        'Respond ONLY with: { "entry_revenue_notes": "string" }'
    )
    entry_db = call_db_worker_sync(
        "entry_q3_q5_revenue",
        ENTRY_Q3_Q5_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(entry_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "tpch_params": state["tpch_params"]},
        [prepared],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["entry_revenue_notes"], resp.content)
    return {"entry_revenue_notes": parsed.get("entry_revenue_notes", "")}


def entry_supply(state: State, model_c: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Entry C: Summarize Q2 + Q9 outputs and connect them to user_query.\n"
        'Respond ONLY with: { "entry_supply_notes": "string" }'
    )
    entry_db = call_db_worker_sync(
        "entry_q2_q9_supply",
        ENTRY_Q2_Q9_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(entry_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "tpch_params": state["tpch_params"]},
        [prepared],
    )
    resp = invoke_llm(model_c, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["entry_supply_notes"], resp.content)
    return {"entry_supply_notes": parsed.get("entry_supply_notes", "")}


def hub_ops(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Hub1: fuse entry_lineitem_notes + entry_revenue_notes with Q12 anchor.\n"
        'Respond ONLY with: { "hub_ops_notes": "string" }'
    )
    hub_db = call_db_worker_sync(
        "hub_ops_openai20b",
        HUB_OPS_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(hub_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "tpch_params": state["tpch_params"],
            "entry_lineitem_notes": state.get("entry_lineitem_notes", ""),
            "entry_revenue_notes": state.get("entry_revenue_notes", ""),
        },
        [prepared],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["hub_ops_notes"], resp.content)
    return {"hub_ops_notes": parsed.get("hub_ops_notes", "")}


def hub_full_context(state: State, model_b: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Hub2: combine hub_ops_notes + entry_supply_notes with Q14 promo anchor.\n"
        'Respond ONLY with: { "hub_full_context": "string" }'
    )
    hub_db = call_db_worker_sync(
        "hub_full_context_qwen14b",
        HUB_FULL_QUERIES,
        {"tpch_params": state["tpch_params"]},
    )
    prepared = json.dumps(_prepare_result_for_prompt(json.loads(hub_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "tpch_params": state["tpch_params"],
            "hub_ops_notes": state.get("hub_ops_notes", ""),
            "entry_supply_notes": state.get("entry_supply_notes", ""),
        },
        [prepared],
    )
    resp = invoke_llm(model_b, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["hub_full_context"], resp.content)
    return {"hub_full_context": parsed.get("hub_full_context", "")}


def tail_focus(state: State, model_c: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Tail: based on hub_full_context, recommend 4-6 next TPC-H queries to run\n"
        "and which params to refine (year/date/region/nation/segment/brand/shipmode/discount/qty/size/type).\n"
        'Respond ONLY with: { "tail_focus_notes": "string" }'
    )
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "hub_full_context": state.get("hub_full_context", "")},
        [],
    )
    resp = invoke_llm(model_c, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["tail_focus_notes"], resp.content)
    return {"tail_focus_notes": parsed.get("tail_focus_notes", "")}


def final_answer(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Final: use tpch_params + hub_full_context + tail_focus_notes + Q15 output\n"
        "to produce final_answer grounded in the DB results.\n"
        'Respond ONLY with: { "final_answer": "string" }'
    )
    final_db = call_db_worker_sync(
        "tpch_final_answer",
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
            "tail_focus_notes": state.get("tail_focus_notes", ""),
        },
        [prepared],
    )
    resp = invoke_llm(model_a, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["final_answer"], resp.content)
    return {"final_answer": parsed.get("final_answer", "")}


def build_graph(model_a: ChatOpenAI, model_b: ChatOpenAI, model_c: ChatOpenAI) -> StateGraph[State]:
    g = StateGraph(State)
    g.add_node("rule_param_extractor", rule_param_extractor)
    g.add_node("entry_lineitem", lambda s: entry_lineitem(s, model_b))
    g.add_node("entry_revenue", lambda s: entry_revenue(s, model_a))
    g.add_node("entry_supply", lambda s: entry_supply(s, model_c))
    g.add_node("hub_ops", lambda s: hub_ops(s, model_a))
    g.add_node("hub_full_context", lambda s: hub_full_context(s, model_b))
    g.add_node("tail_focus", lambda s: tail_focus(s, model_c))
    g.add_node("final_answer", lambda s: final_answer(s, model_a))

    g.add_edge(START, "rule_param_extractor")
    g.add_edge("rule_param_extractor", "entry_lineitem")
    g.add_edge("rule_param_extractor", "entry_revenue")
    g.add_edge("rule_param_extractor", "entry_supply")
    g.add_edge("entry_lineitem", "hub_ops")
    g.add_edge("entry_revenue", "hub_ops")
    g.add_edge("hub_ops", "hub_full_context")
    g.add_edge("entry_supply", "hub_full_context")
    g.add_edge("hub_full_context", "tail_focus")
    g.add_edge("hub_full_context", "final_answer")
    g.add_edge("tail_focus", "final_answer")
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
    parser = argparse.ArgumentParser(description="TPC-H trident LangGraph workflow (Halo-aligned prompts).")
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
    metrics.start(run_name="tpch_trident_langgraph", total_queries=len(queries))
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
            print("\n--- Tail Focus ---")
            print(final_state.get("tail_focus_notes", ""))
            print("\n" + "=" * 60)

    set_global_metrics(metrics)
    asyncio.run(runner())
    set_global_metrics(None)
    metrics.finish()
    print("\n" + metrics.format_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
