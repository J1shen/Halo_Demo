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
from baselines.tpch_utils import extract_tpch_params
from halo_dev.utils import render_template
from openai import APITimeoutError

# Defaults aligned with Halo
API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
MODEL_A = os.getenv("TPCH_MODEL_A", "openai/gpt-oss-20b")
MODEL_B = os.getenv("TPCH_MODEL_B", "Qwen/Qwen3-14B")
MODEL_C = os.getenv("TPCH_MODEL_C", "Qwen/Qwen3-32B")
BASE_A = os.getenv("MODEL_A_BASE_URL", "http://localhost:9101/v1")
BASE_B = os.getenv("MODEL_B_BASE_URL", "http://localhost:9102/v1")
BASE_C = os.getenv("MODEL_C_BASE_URL", "http://localhost:9103/v1")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))
LLM_TIMEOUT = float(os.getenv("HALO_LLM_TIMEOUT", "1200"))
MAX_FIELD_CHARS = 2048
MAX_ROWS = 5
METRICS: BaselineMetrics | None = None
TEMPLATE_PATH = Path("templates/tpch_fanout.yaml")


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
    return outputs


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


# DB query definitions (from templates/tpch_fanout.yaml)
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

S1_Q3_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "s1_q3_q3_segment_revenue",
        "sql": """
            WITH oiy AS (
              SELECT o.o_orderkey, o.o_custkey, o.o_orderdate, o.o_orderpriority
              FROM orders o
              WHERE o.o_orderdate >= make_date(:year::int, 1, 1)
                AND o.o_orderdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
            )
            SELECT
              SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
              COUNT(*) AS line_cnt,
              SUM(CASE WHEN oiy.o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_pri_lines,
              SUM(CASE WHEN oiy.o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_pri_lines
            FROM oiy
            JOIN customer c ON c.c_custkey = oiy.o_custkey
            JOIN lineitem l ON l.l_orderkey = oiy.o_orderkey
            WHERE c.c_mktsegment = :segment
              AND l.l_shipdate > oiy.o_orderdate;
        """,
        "parameters": {"year": "{{ tpch_params.year }}", "segment": "{{ tpch_params.segment }}"},
        "param_types": {"year": "int", "segment": "str"},
    },
    {
        "name": "s1_q3_q12_shipmode_priority",
        "sql": """
            WITH base AS (
              SELECT
                l.l_shipmode,
                o.o_orderpriority
              FROM orders o
              JOIN lineitem l ON l.l_orderkey = o.o_orderkey
              WHERE l.l_shipmode IN (:shipmode, :shipmode2)
                AND l.l_receiptdate >= make_date(:year::int, 1, 1)
                AND l.l_receiptdate <  make_date(:year::int, 1, 1) + INTERVAL '1 year'
                AND l.l_commitdate < l.l_receiptdate
                AND l.l_shipdate < l.l_commitdate
            )
            SELECT
              l_shipmode,
              COUNT(*) AS cnt,
              SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_cnt,
              SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_cnt
            FROM base
            GROUP BY l_shipmode
            ORDER BY l_shipmode;
        """,
        "parameters": {
            "year": "{{ tpch_params.year }}",
            "shipmode": "{{ tpch_params.shipmode }}",
            "shipmode2": "{{ tpch_params.shipmode2 }}",
        },
        "param_types": {"year": "int", "shipmode": "str", "shipmode2": "str"},
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


async def run_one_query(user_query: str, models: Dict[str, OpenAIChatModel]) -> tuple[str, str]:
    tpch_params = extract_tpch_params(user_query, TEMPLATE_PATH)

    s1_q5_system = (
        "You are a TPC-H analyst agent (Stage 1 / Q5-pack).\n\n"
        "Inputs:\n"
        "- user_query\n"
        "- tpch_params\n"
        "- results of EXACTLY FOUR DB queries\n\n"
        "Do:\n"
        "- For each query, extract 2 key numeric facts (leaders/totals/shares).\n"
        "- Provide 1 cross-query connection.\n"
        "- Provide 1 warning about interpretation risk (skew, sparsity, time window).\n\n"
        "Output JSON only:\n"
        "{\n"
        "  \"s1_q5_brief\": \"one dense paragraph\",\n"
        "  \"per_query_facts\": [{\"query\":\"string\",\"facts\":[\"string\",\"string\"]}],\n"
        "  \"cross_link\": \"string\",\n"
        "  \"risk\": \"string\"\n"
        "}"
    )
    s1_q5_db = await call_db_worker(
        "stage1_econ_rollup_q5",
        S1_Q5_QUERIES,
        {"tpch_params": tpch_params},
    )
    s1_q5_db = json.dumps(_prepare_result_for_prompt(json.loads(s1_q5_db)), ensure_ascii=False)
    s1_q5_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "tpch_params": tpch_params},
        [s1_q5_db],
    )
    s1_q5_resp = await chat(
        models["c"],
        [{"role": "system", "content": s1_q5_system}, {"role": "user", "content": s1_q5_prompt}],
    )
    s1_q5_brief = extract_outputs(["s1_q5_brief"], s1_q5_resp).get("s1_q5_brief", "")

    s1_q3_system = (
        "You are a TPC-H reasoning agent (Stage 1 / Q3-pack).\n\n"
        "You will receive EXACTLY TWO DB results:\n"
        "- segment/year revenue summary\n"
        "- shipmode/priority pipeline breakdown\n\n"
        "Requirements:\n"
        "- Use numbers: cite totals and at least one breakdown item.\n"
        "- Provide 2 insights and 2 sanity checks.\n\n"
        "Output JSON only:\n"
        "{\n"
        "  \"s1_q3_brief\": \"one dense paragraph\",\n"
        "  \"insights\": [\"string\",\"string\"],\n"
        "  \"sanity\": [\"string\",\"string\"]\n"
        "}"
    )
    s1_q3_db = await call_db_worker(
        "stage1_seg_ship_q3",
        S1_Q3_QUERIES,
        {"tpch_params": tpch_params},
    )
    s1_q3_db = json.dumps(_prepare_result_for_prompt(json.loads(s1_q3_db)), ensure_ascii=False)
    s1_q3_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "tpch_params": tpch_params},
        [s1_q3_db],
    )
    s1_q3_resp = await chat(
        models["b"],
        [{"role": "system", "content": s1_q3_system}, {"role": "user", "content": s1_q3_prompt}],
    )
    s1_q3_brief = extract_outputs(["s1_q3_brief"], s1_q3_resp).get("s1_q3_brief", "")

    s1_q1_system = (
        "Numeric summarizer (Stage 1 / Q1-pack).\n\n"
        "Interpret the discount-band revenue query:\n"
        "- explain what it measures\n"
        "- extract key metrics (total, count, avg_discount)\n"
        "- provide one interpretation linked to tpch_params\n\n"
        "Output JSON only:\n"
        "{\n"
        "  \"s1_q1_brief\": \"one paragraph\",\n"
        "  \"metrics\": {\"total_price\":\"number or null\",\"line_cnt\":\"int or null\",\"avg_discount\":\"number or null\"},\n"
        "  \"interpretation\": \"string\"\n"
        "}"
    )
    s1_q1_db = await call_db_worker(
        "stage1_disc_band_q1",
        S1_Q1_QUERIES,
        {"tpch_params": tpch_params},
    )
    s1_q1_db = json.dumps(_prepare_result_for_prompt(json.loads(s1_q1_db)), ensure_ascii=False)
    s1_q1_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "tpch_params": tpch_params},
        [s1_q1_db],
    )
    s1_q1_resp = await chat(
        models["a"],
        [{"role": "system", "content": s1_q1_system}, {"role": "user", "content": s1_q1_prompt}],
    )
    s1_q1_brief = extract_outputs(["s1_q1_brief"], s1_q1_resp).get("s1_q1_brief", "")

    s1_q0a_system = (
        "Stage 1 quick planner (no DB queries).\n\n"
        "Using user_query + tpch_params:\n"
        "- propose 2 hypotheses\n"
        "- propose 2 checks to validate DB results\n\n"
        "Output JSON only:\n"
        "{ \"s1_q0a_brief\": \"string\", \"hypotheses\": [\"string\",\"string\"], \"checks\": [\"string\",\"string\"] }"
    )
    s1_q0a_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "tpch_params": tpch_params},
        [],
    )
    s1_q0a_resp = await chat(
        models["c"],
        [{"role": "system", "content": s1_q0a_system}, {"role": "user", "content": s1_q0a_prompt}],
    )
    s1_q0a_brief = extract_outputs(["s1_q0a_brief"], s1_q0a_resp).get("s1_q0a_brief", "")

    s1_q0b_system = (
        "Stage 1 alternative angle (no DB queries).\n\n"
        "Provide:\n"
        "- 2 alternative explanations that might mislead aggregates\n"
        "- 2 questions for stage 2\n\n"
        "Output JSON only:\n"
        "{ \"s1_q0b_brief\": \"string\", \"alt_explanations\": [\"string\",\"string\"], \"questions\": [\"string\",\"string\"] }"
    )
    s1_q0b_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "tpch_params": tpch_params},
        [],
    )
    s1_q0b_resp = await chat(
        models["c"],
        [{"role": "system", "content": s1_q0b_system}, {"role": "user", "content": s1_q0b_prompt}],
    )
    s1_q0b_brief = extract_outputs(["s1_q0b_brief"], s1_q0b_resp).get("s1_q0b_brief", "")

    s2_q5_system = (
        "You are a TPC-H analyst agent (Stage 2 / Q5-pack).\n\n"
        "Inputs:\n"
        "- user_query, tpch_params\n"
        "- stage1 briefs: s1_q5_brief, s1_q3_brief, s1_q1_brief, s1_q0a_brief, s1_q0b_brief\n"
        "- results of EXACTLY FOUR DB queries\n\n"
        "Tasks:\n"
        "- Validate at least 1 stage1 hypothesis/check using concrete numbers.\n"
        "- Provide 2 delta-aware statements (consistent vs uncertain).\n"
        "- Produce a compact brief.\n\n"
        "Output JSON only:\n"
        "{ \"s2_q5_brief\": \"one dense paragraph\", \"deltas\": [\"string\",\"string\"] }"
    )
    s2_q5_db = await call_db_worker(
        "stage2_econ_rollup_q5",
        S2_Q5_QUERIES,
        {"tpch_params": tpch_params},
    )
    s2_q5_db = json.dumps(_prepare_result_for_prompt(json.loads(s2_q5_db)), ensure_ascii=False)
    s2_q5_prompt = build_halo_prompt(
        None,
        {
            "user_query": user_query,
            "tpch_params": tpch_params,
            "s1_q5_brief": s1_q5_brief,
            "s1_q3_brief": s1_q3_brief,
            "s1_q1_brief": s1_q1_brief,
            "s1_q0a_brief": s1_q0a_brief,
            "s1_q0b_brief": s1_q0b_brief,
        },
        [s2_q5_db],
    )
    s2_q5_resp = await chat(
        models["c"],
        [{"role": "system", "content": s2_q5_system}, {"role": "user", "content": s2_q5_prompt}],
    )
    s2_q5_brief = extract_outputs(["s2_q5_brief"], s2_q5_resp).get("s2_q5_brief", "")

    s2_q1_system = (
        "Stage 2 numeric summarizer (Q1-pack).\n\n"
        "Use stage1 briefs to contextualize the discount-band revenue:\n"
        "- Does it align with other patterns?\n"
        "- Note 1 caveat.\n\n"
        "Output JSON only:\n"
        "{ \"s2_q1_brief\": \"one paragraph\", \"notes\": [\"string\",\"string\"] }"
    )
    s2_q1_db = await call_db_worker(
        "stage2_disc_band_q1",
        S2_Q1_QUERIES,
        {"tpch_params": tpch_params},
    )
    s2_q1_db = json.dumps(_prepare_result_for_prompt(json.loads(s2_q1_db)), ensure_ascii=False)
    s2_q1_prompt = build_halo_prompt(
        None,
        {
            "user_query": user_query,
            "tpch_params": tpch_params,
            "s1_q5_brief": s1_q5_brief,
            "s1_q3_brief": s1_q3_brief,
            "s1_q1_brief": s1_q1_brief,
            "s1_q0a_brief": s1_q0a_brief,
            "s1_q0b_brief": s1_q0b_brief,
        },
        [s2_q1_db],
    )
    s2_q1_resp = await chat(
        models["a"],
        [{"role": "system", "content": s2_q1_system}, {"role": "user", "content": s2_q1_prompt}],
    )
    s2_q1_brief = extract_outputs(["s2_q1_brief"], s2_q1_resp).get("s2_q1_brief", "")

    s2_q0a_system = (
        "Stage 2 narrative synthesizer (no DB queries).\n\n"
        "Provide:\n"
        "- 2 candidate narratives\n"
        "- 2 caveats\n\n"
        "Output JSON only:\n"
        "{ \"s2_q0a_brief\": \"string\", \"narratives\": [\"string\",\"string\"], \"caveats\": [\"string\",\"string\"] }"
    )
    s2_q0a_prompt = build_halo_prompt(
        None,
        {
            "user_query": user_query,
            "tpch_params": tpch_params,
            "s1_q5_brief": s1_q5_brief,
            "s1_q3_brief": s1_q3_brief,
            "s1_q1_brief": s1_q1_brief,
            "s1_q0a_brief": s1_q0a_brief,
            "s1_q0b_brief": s1_q0b_brief,
        },
        [],
    )
    s2_q0a_resp = await chat(
        models["c"],
        [{"role": "system", "content": s2_q0a_system}, {"role": "user", "content": s2_q0a_prompt}],
    )
    s2_q0a_brief = extract_outputs(["s2_q0a_brief"], s2_q0a_resp).get("s2_q0a_brief", "")

    final_system = (
        "Final merger.\n\n"
        "Inputs: s2_q5_brief, s2_q1_brief, s2_q0a_brief\n"
        "Write a coherent final answer that:\n"
        "- includes at least 3 concrete numeric facts or ranked statements\n"
        "- includes 1 caveat\n"
        "- ends with 2 follow-up questions\n\n"
        "Output JSON only:\n"
        "{ \"final_answer\": \"string\" }"
    )
    final_prompt = build_halo_prompt(
        None,
        {
            "s2_q5_brief": s2_q5_brief,
            "s2_q1_brief": s2_q1_brief,
            "s2_q0a_brief": s2_q0a_brief,
        },
        [],
    )
    final_resp = await chat(
        models["c"],
        [{"role": "system", "content": final_system}, {"role": "user", "content": final_prompt}],
    )
    final_answer = extract_outputs(["final_answer"], final_resp).get("final_answer", "")

    return final_answer, s2_q5_brief


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


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TPC-H fanout workflow with OpenAIChatModel + DB worker.")
    parser.add_argument("--query", type=str, default=None, help="Single query; if set, ignores file.")
    parser.add_argument("--file", type=str, default="data/tpch_user_inputs.txt", help="Input file, one query per line.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N queries.")
    parser.add_argument("--count", type=int, default=None, help="Process only the first count queries.")
    args = parser.parse_args(argv)

    global METRICS
    METRICS = BaselineMetrics()
    models = {
        "a": make_model(MODEL_A, BASE_A),
        "b": make_model(MODEL_B, BASE_B),
        "c": make_model(MODEL_C, BASE_C),
    }

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

    METRICS.start(run_name="tpch_fanout_agentscope", total_queries=len(queries))
    METRICS.add_metadata("runner", "agentscope")
    METRICS.add_metadata("input_source", "inline" if args.query else args.file)
    if args.limit is not None:
        METRICS.add_metadata("limit", args.limit)
    if args.count is not None:
        METRICS.add_metadata("count", args.count)

    async def runner():
        total = len(queries)

        async def timed_task(idx: int, query: str):
            start = time.perf_counter()
            final_answer, s2_q5_brief = await run_one_query(query, models)
            elapsed = time.perf_counter() - start
            return idx, query, final_answer, s2_q5_brief, elapsed

        tasks = [asyncio.create_task(timed_task(idx, q)) for idx, q in enumerate(queries, 1)]
        for coro in asyncio.as_completed(tasks):
            idx, q, final_answer, s2_q5_brief, elapsed = await coro
            METRICS.record_query(elapsed)
            print(f"\n===== Query #{idx}/{total} =====\n{q}")
            print("\n=== Final Answer ===")
            print(final_answer)
            print("\n--- Stage2 Q5 Brief ---")
            print(s2_q5_brief)
            print("\n" + "=" * 60)

    asyncio.run(runner())
    METRICS.finish()
    print("\n" + METRICS.format_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
