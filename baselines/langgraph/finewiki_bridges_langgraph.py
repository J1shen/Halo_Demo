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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.metrics import BaselineMetrics, get_global_metrics, set_global_metrics
from halo_dev.utils import render_template
from run.query_sampler import (
    DEFAULT_SAMPLE_POOL_SIZE,
    SampleResult,
    default_extract_query,
    sample_queries,
)

API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
MODEL_A = os.getenv("FINEWIKI_MODEL_A", "openai/gpt-oss-20b")
MODEL_B = os.getenv("FINEWIKI_MODEL_B", "Qwen/Qwen3-32B")
MODEL_C = os.getenv("FINEWIKI_MODEL_C", "Qwen/Qwen3-14B")
BASE_A = os.getenv("MODEL_A_BASE_URL", "http://localhost:9101/v1")
BASE_B = os.getenv("MODEL_B_BASE_URL", "http://localhost:9103/v1")
BASE_C = os.getenv("MODEL_C_BASE_URL", "http://localhost:9102/v1")
DB_WORKER_URL = os.getenv("HALO_DB_WORKER_URL", "http://localhost:9104")
DB_TIMEOUT = float(os.getenv("HALO_DB_TIMEOUT", "1200"))
MAX_FIELD_CHARS = 2000
MAX_ROWS = 5


class State(TypedDict, total=False):
    user_query: str
    a0_summary: str
    a1_outline: str
    a2_narrative: str
    b1_bridge_notes: str
    b2_key_subtopics: Any
    c1_language_notes: str
    c2_crosslingual_insights: str
    s0_hints: Any
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


# DB queries (from templates/finewiki_bridges.yaml)
A0_QUERIES = [
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
B1_QUERIES = [
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
C1_QUERIES = [
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


# LangGraph node functions
def node_a0(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "First global pass over FineWiki for the topic.\n"
        "Use fw_global_pass_a0_main as your retrieval pool.\n"
        "Respond ONLY with:\n"
        '{\n  "a0_summary": "string"\n}'
    )
    a0_db = call_db_worker_sync(
        "fw_global_pass_a0",
        A0_QUERIES,
        {"user_query": state["user_query"]},
    )
    prepared_a0_db = json.dumps(_prepare_result_for_prompt(json.loads(a0_db)), ensure_ascii=False)
    prompt = build_halo_prompt(None, {"user_query": state["user_query"]}, [prepared_a0_db])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["a0_summary"], resp.content)
    return {"a0_summary": parsed.get("a0_summary", "")}


def node_a1(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Refine a0_summary into a structured outline.\n"
        "Respond ONLY with:\n"
        '{\n  "a1_outline": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"], "a0_summary": state.get("a0_summary", "")}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["a1_outline"], resp.content)
    return {"a1_outline": parsed.get("a1_outline", "")}


def node_a2(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Expand a1_outline into a longer narrative suitable for\n"
        "downstream bridges and final answering.\n"
        "Respond ONLY with:\n"
        '{\n  "a2_narrative": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"], "a1_outline": state.get("a1_outline", "")}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["a2_narrative"], resp.content)
    return {"a2_narrative": parsed.get("a2_narrative", "")}


def node_b1(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Category-based bridge: connect a1_outline with category pages\n"
        "from fw_category_bridge_b1_main.\n"
        "Respond ONLY with:\n"
        '{\n  "b1_bridge_notes": "string"\n}'
    )
    b1_db = call_db_worker_sync(
        "fw_category_bridge_b1",
        B1_QUERIES,
        {"user_query": state["user_query"]},
    )
    prepared_b1_db = json.dumps(_prepare_result_for_prompt(json.loads(b1_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "a1_outline": state.get("a1_outline", "")},
        [prepared_b1_db],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["b1_bridge_notes"], resp.content)
    return {"b1_bridge_notes": parsed.get("b1_bridge_notes", "")}


def node_b2(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Refine b1_bridge_notes into 3–5 key subtopics that are useful\n"
        "for guiding the final answer.\n"
        "Respond ONLY with:\n"
        '{\n  "b2_key_subtopics": ["string", ...]\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"], "b1_bridge_notes": state.get("b1_bridge_notes", "")}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["b2_key_subtopics"], resp.content)
    return {"b2_key_subtopics": parsed.get("b2_key_subtopics", "")}


def node_c1(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Non-English language bridge for the topic.\n"
        "Use fw_language_bridge_c1_main to understand how the topic\n"
        "appears in other languages.\n"
        "Respond ONLY with:\n"
        '{\n  "c1_language_notes": "string"\n}'
    )
    c1_db = call_db_worker_sync(
        "fw_language_bridge_c1",
        C1_QUERIES,
        {"user_query": state["user_query"]},
    )
    prepared_c1_db = json.dumps(_prepare_result_for_prompt(json.loads(c1_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "a0_summary": state.get("a0_summary", "")},
        [prepared_c1_db],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["c1_language_notes"], resp.content)
    return {"c1_language_notes": parsed.get("c1_language_notes", "")}


def node_c2(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Refine c1_language_notes into 2–4 concise cross-lingual\n"
        "insights that can influence the final answer.\n"
        "Respond ONLY with:\n"
        '{\n  "c2_crosslingual_insights": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"], "c1_language_notes": state.get("c1_language_notes", "")}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["c2_crosslingual_insights"], resp.content)
    return {"c2_crosslingual_insights": parsed.get("c2_crosslingual_insights", "")}


def node_s0(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Light sampler for the bridge topology.\n"
        "Condense a2_narrative, b2_key_subtopics, and\n"
        "c2_crosslingual_insights into short hints that are easy for\n"
        "the final answer head to consume.\n"
        "Respond ONLY with:\n"
        '{\n  "s0_hints": ["string", ...]\n}'
    )
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "a2_narrative": state.get("a2_narrative", ""),
            "b2_key_subtopics": state.get("b2_key_subtopics", ""),
            "c2_crosslingual_insights": state.get("c2_crosslingual_insights", ""),
        },
        [],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["s0_hints"], resp.content)
    return {"s0_hints": parsed.get("s0_hints", "")}


def node_final(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "Final answer assistant for the bridge topology.\n"
        "Combine a2_narrative and s0_hints into a single coherent,\n"
        "well-structured answer to user_query.\n"
        "Respond ONLY with:\n"
        '{\n  "final_answer": "string"\n}'
    )
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "a2_narrative": state.get("a2_narrative", ""), "s0_hints": state.get("s0_hints", "")},
        [],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["final_answer"], resp.content)
    return {"final_answer": parsed.get("final_answer", "")}


def build_graph(models: Dict[str, ChatOpenAI]) -> StateGraph[State]:
    g = StateGraph(State)
    g.add_node("a0", lambda s: node_a0(s, models["a"]))
    g.add_node("a1", lambda s: node_a1(s, models["a"]))
    g.add_node("a2", lambda s: node_a2(s, models["a"]))
    g.add_node("b1", lambda s: node_b1(s, models["b"]))
    g.add_node("b2", lambda s: node_b2(s, models["b"]))
    g.add_node("c1", lambda s: node_c1(s, models["c"]))
    g.add_node("c2", lambda s: node_c2(s, models["c"]))
    g.add_node("s0", lambda s: node_s0(s, models["c"]))
    g.add_node("final", lambda s: node_final(s, models["c"]))

    g.add_edge(START, "a0")
    g.add_edge("a0", "a1")
    g.add_edge("a1", "a2")
    g.add_edge("a1", "b1")
    g.add_edge("b1", "b2")
    g.add_edge("a0", "c1")
    g.add_edge("c1", "c2")
    g.add_edge("a2", "s0")
    g.add_edge("b2", "s0")
    g.add_edge("c2", "s0")
    g.add_edge("a2", "final")
    g.add_edge("s0", "final")
    g.add_edge("final", END)
    return g


def _sample_queries(args: argparse.Namespace) -> SampleResult:
    return sample_queries(
        dataset=args.dataset_name,
        subset=args.dataset_subset,
        split=args.dataset_split,
        sample_count=max(1, args.sample_count),
        seed=args.sample_seed,
        pool_size=args.sample_pool_size,
        extract_query=default_extract_query,
    )


async def process_one_query(query: str, models: Dict[str, ChatOpenAI], workflow) -> tuple[str, State]:
    base_state: State = {"user_query": query}
    loop = asyncio.get_event_loop()
    final_state: State = await loop.run_in_executor(None, workflow.invoke, base_state)
    return query, final_state


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FineWiki bridge-topology LangGraph baseline (Halo-aligned).")
    parser.add_argument("--query", type=str, default=None, help="单条 user_query；若提供则忽略数据集采样。")
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

    metrics = BaselineMetrics()
    models = {
        "a": make_model(MODEL_A, BASE_A),
        "b": make_model(MODEL_B, BASE_B),
        "c": make_model(MODEL_C, BASE_C),
    }

    graph = build_graph(models).compile()

    if args.query:
        queries = [args.query.strip()]
        sample_info: SampleResult | None = None
    else:
        sample_info = _sample_queries(args)
        queries = sample_info.queries
        if not queries:
            print("No queries sampled from dataset.")
            return 1
    metrics.start(run_name="finewiki_bridges_langgraph", total_queries=len(queries))
    metrics.add_metadata("runner", "langgraph")
    metrics.add_metadata("input_source", "inline" if args.query else "dataset")
    if sample_info is not None:
        metrics.add_metadata("dataset", args.dataset_name)
        metrics.add_metadata("dataset_subset", args.dataset_subset)
        metrics.add_metadata("dataset_split", args.dataset_split)
        metrics.add_metadata("sample_count", args.sample_count)
        metrics.add_metadata("rows_scanned", sample_info.rows_scanned)

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
            print("\n" + "=" * 60)

    set_global_metrics(metrics)
    asyncio.run(runner())
    set_global_metrics(None)
    metrics.finish()
    print("\n" + metrics.format_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
