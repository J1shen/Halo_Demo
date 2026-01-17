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
    global_summary: str
    central_page_ids: Any
    coarse_timeline: str
    refined_timeline: str
    category_overview: str
    category_details: Any
    outbound_links_overview: str
    inbound_links_overview: str
    narrowed_query: str
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
        rendered_system = render_template(system_prompt.strip(), inputs)
        parts.append(rendered_system)
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


# DB queries (from templates/finewiki_long_chain.yaml)
GLOBAL_QUERIES = [
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

TIMELINE_QUERIES = [
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

CATEGORY_QUERIES = [
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

OUTBOUND_QUERIES = [
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

FINAL_SNIPPET_QUERIES = [
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


# LangGraph node functions
def node_global(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You are a global topic expander over a Wikipedia-like corpus.\n"
        "Use fw_global_wikitext_fulltext and fw_global_title_keyword to build\n"
        "a broad but coherent summary of the topic and propose a small list\n"
        "of central candidate page_ids (as plain integers in text).\n"
        "Respond ONLY with:\n"
        '{\n  "global_summary": "string",\n  "central_page_ids": [integer, ...]\n}'
    )
    global_db = call_db_worker_sync(
        "global_topic_expander",
        GLOBAL_QUERIES,
        {"user_query": state["user_query"]},
    )
    prepared_global_db = json.dumps(_prepare_result_for_prompt(json.loads(global_db)), ensure_ascii=False)
    prompt = build_halo_prompt(None, {"user_query": state["user_query"]}, [prepared_global_db])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["global_summary", "central_page_ids"], resp.content)
    return {
        "global_summary": parsed.get("global_summary", ""),
        "central_page_ids": parsed.get("central_page_ids", ""),
    }


def node_timeline1(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You are a timeline-focused refiner.\n"
        "Given global_summary and (textual) central_page_ids, and using\n"
        "fw_timeline_from_keyword as context, construct a coarse\n"
        "chronological narrative of the topic.\n"
        "Respond ONLY with:\n"
        '{\n  "coarse_timeline": "string"\n}'
    )
    timeline_db = call_db_worker_sync(
        "timeline_refiner_1",
        TIMELINE_QUERIES,
        {"user_query": state["user_query"]},
    )
    prepared_timeline_db = json.dumps(_prepare_result_for_prompt(json.loads(timeline_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "global_summary": state.get("global_summary", ""), "central_page_ids": state.get("central_page_ids", "")},
        [prepared_timeline_db],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["coarse_timeline"], resp.content)
    return {"coarse_timeline": parsed.get("coarse_timeline", "")}


def node_timeline2(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You are a second-pass timeline refiner.\n"
        "Take the coarse_timeline and improve clarity and ordering, and\n"
        "highlight 3–5 key milestones.\n"
        "Respond ONLY with:\n"
        '{\n  "refined_timeline": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"], "coarse_timeline": state.get("coarse_timeline", "")}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["refined_timeline"], resp.content)
    return {"refined_timeline": parsed.get("refined_timeline", "")}


def node_category_overview(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You are a category-focused summarizer.\n"
        "Use fw_category_pages to provide a high-level taxonomy of the topic.\n"
        "Respond ONLY with:\n"
        '{\n  "category_overview": "string"\n}'
    )
    category_db = call_db_worker_sync(
        "category_overview",
        CATEGORY_QUERIES,
        {"user_query": state["user_query"]},
    )
    prepared_category_db = json.dumps(_prepare_result_for_prompt(json.loads(category_db)), ensure_ascii=False)
    prompt = build_halo_prompt(None, {"user_query": state["user_query"]}, [prepared_category_db])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["category_overview"], resp.content)
    return {"category_overview": parsed.get("category_overview", "")}


def node_category_drilldown(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You are a drill-down assistant for category structures.\n"
        "Take category_overview and extract 3–5 important subtopics that\n"
        "should be mentioned in a final answer.\n"
        "Respond ONLY with:\n"
        '{\n  "category_details": ["string", ...]\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"], "category_overview": state.get("category_overview", "")}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["category_details"], resp.content)
    return {"category_details": parsed.get("category_details", "")}


def node_outbound(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You are an outbound link summarizer.\n"
        "Given pages whose titles match user_query, summarize which topics\n"
        "they tend to link out to (based on their wikitext).\n"
        "Respond ONLY with:\n"
        '{\n  "outbound_links_overview": "string"\n}'
    )
    outbound_db = call_db_worker_sync(
        "outbound_link_view",
        OUTBOUND_QUERIES,
        {"user_query": state["user_query"]},
    )
    prepared_outbound_db = json.dumps(_prepare_result_for_prompt(json.loads(outbound_db)), ensure_ascii=False)
    prompt = build_halo_prompt(None, {"user_query": state["user_query"]}, [prepared_outbound_db])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["outbound_links_overview"], resp.content)
    return {"outbound_links_overview": parsed.get("outbound_links_overview", "")}


def node_inbound(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You summarize who links into this topic.\n"
        "Given outbound_links_overview and global_summary, infer which\n"
        "other pages or domains frequently point to this topic and why\n"
        "that matters for understanding it.\n"
        "Respond ONLY with:\n"
        '{\n  "inbound_links_overview": "string"\n}'
    )
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "outbound_links_overview": state.get("outbound_links_overview", ""),
            "global_summary": state.get("global_summary", ""),
        },
        [],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["inbound_links_overview"], resp.content)
    return {"inbound_links_overview": parsed.get("inbound_links_overview", "")}


def node_narrow(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You rewrite the user_query into a narrower focus question\n"
        "suitable for a final answer over this topic.\n"
        "Respond ONLY with:\n"
        '{\n  "narrowed_query": "string"\n}'
    )
    prompt = build_halo_prompt(None, {"user_query": state["user_query"]}, [])
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["narrowed_query"], resp.content)
    return {"narrowed_query": parsed.get("narrowed_query", "")}


def node_final(state: State, model: ChatOpenAI) -> Dict[str, Any]:
    system = (
        "You are the final answer assistant.\n"
        "Combine global_summary, refined_timeline, category_details,\n"
        "outbound_links_overview, inbound_links_overview, and\n"
        "narrowed_query (plus fw_supporting_snippets_global as loose\n"
        "background) into a single clear, grounded answer.\n"
        "Respond ONLY with:\n"
        '{\n  "final_answer": "string"\n}'
    )
    final_db = call_db_worker_sync(
        "final_answer_summarizer",
        FINAL_SNIPPET_QUERIES,
        {"user_query": state["user_query"]},
    )
    prepared_final_db = json.dumps(_prepare_result_for_prompt(json.loads(final_db)), ensure_ascii=False)
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "global_summary": state.get("global_summary", ""),
            "refined_timeline": state.get("refined_timeline", ""),
            "category_details": state.get("category_details", ""),
            "outbound_links_overview": state.get("outbound_links_overview", ""),
            "inbound_links_overview": state.get("inbound_links_overview", ""),
            "narrowed_query": state.get("narrowed_query", ""),
        },
        [prepared_final_db],
    )
    resp = invoke_llm(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    parsed = extract_outputs(["final_answer"], resp.content)
    return {"final_answer": parsed.get("final_answer", "")}


def build_graph(models: Dict[str, ChatOpenAI]) -> StateGraph[State]:
    g = StateGraph(State)
    g.add_node("global_topic_expander", lambda s: node_global(s, models["a"]))
    g.add_node("timeline_refiner_1", lambda s: node_timeline1(s, models["a"]))
    g.add_node("timeline_refiner_2", lambda s: node_timeline2(s, models["a"]))
    g.add_node("category_overview", lambda s: node_category_overview(s, models["b"]))
    g.add_node("category_drilldown", lambda s: node_category_drilldown(s, models["b"]))
    g.add_node("outbound_link_view", lambda s: node_outbound(s, models["c"]))
    g.add_node("inbound_link_view", lambda s: node_inbound(s, models["c"]))
    g.add_node("narrow_focus_rewriter", lambda s: node_narrow(s, models["a"]))
    # Final head uses model C (Qwen/Qwen3-14B) to match template.
    g.add_node("final_answer_summarizer", lambda s: node_final(s, models["c"]))

    g.add_edge(START, "global_topic_expander")
    g.add_edge(START, "category_overview")
    g.add_edge(START, "outbound_link_view")
    g.add_edge(START, "narrow_focus_rewriter")
    g.add_edge("global_topic_expander", "timeline_refiner_1")
    g.add_edge("timeline_refiner_1", "timeline_refiner_2")
    g.add_edge("category_overview", "category_drilldown")
    g.add_edge("outbound_link_view", "inbound_link_view")
    g.add_edge("global_topic_expander", "inbound_link_view")
    g.add_edge("global_topic_expander", "final_answer_summarizer")
    g.add_edge("timeline_refiner_2", "final_answer_summarizer")
    g.add_edge("category_drilldown", "final_answer_summarizer")
    g.add_edge("outbound_link_view", "final_answer_summarizer")
    g.add_edge("inbound_link_view", "final_answer_summarizer")
    g.add_edge("narrow_focus_rewriter", "final_answer_summarizer")
    g.add_edge("final_answer_summarizer", END)
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
    base_state: State = {"user_query": query}
    loop = asyncio.get_event_loop()
    final_state: State = await loop.run_in_executor(None, workflow.invoke, base_state)
    return query, final_state


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FineWiki long-chain LangGraph baseline (Halo-aligned).")
    parser.add_argument("--query", type=str, default=None, help="单条 user_query；若提供则忽略数据集采样。")
    parser.add_argument("--file", type=str, default=None, help="批量查询文件，每行一条（优先于数据集采样）。")
    parser.add_argument("--limit", type=int, default=None, help="从文件最多读取前 N 条。")
    parser.add_argument("--count", type=int, default=None, help="仅处理前 count 条（早于 limit 截断）。")
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
        input_source = "inline"
    elif args.file:
        path = Path(args.file)
        queries = load_queries(path, args.limit)
        if args.count is not None:
            queries = queries[: args.count]
        if not queries:
            print("No queries found in file.")
            return 1
        sample_info = None
        input_source = "file"
    else:
        sample_info = _sample_queries(args)
        queries = sample_info.queries
        if not queries:
            print("No queries sampled from dataset.")
            return 1
        input_source = "dataset"
    metrics.start(run_name="finewiki_long_chain_langgraph", total_queries=len(queries))
    metrics.add_metadata("runner", "langgraph")
    metrics.add_metadata("input_source", input_source)
    if input_source == "file":
        metrics.add_metadata("input_file", args.file)
        if args.limit is not None:
            metrics.add_metadata("limit", args.limit)
        if args.count is not None:
            metrics.add_metadata("count", args.count)
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
