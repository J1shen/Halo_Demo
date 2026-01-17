from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from agentscope.model import OpenAIChatModel
from baselines.metrics import BaselineMetrics
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
LLM_TIMEOUT = float(os.getenv("HALO_LLM_TIMEOUT", "1200"))
MAX_FIELD_CHARS = 2048
MAX_ROWS = 5


def make_model(model_name: str, base_url: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=model_name,
        api_key=API_KEY,
        client_kwargs={"base_url": base_url, "timeout": LLM_TIMEOUT},
        stream=False,
        generate_kwargs={"temperature": 0.2, "top_p": 0.9, "max_tokens": 1024},
    )


async def chat(model: OpenAIChatModel, messages: List[Dict[str, Any]]) -> str:
    response = await model(messages)
    text_parts: List[str] = []
    for block in response.content:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "thinking":
            text_parts.append(block.get("thinking", ""))
        else:
            text_parts.append(str(block))
    return "\n".join(text_parts).strip()


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
    if "search_keyword" in outputs and isinstance(outputs["search_keyword"], str):
        first = outputs["search_keyword"].split(",")[0].strip()
        if first:
            outputs["search_keyword"] = first
    return outputs


def call_db_worker_sync(node_id: str, queries: List[Dict[str, Any]], context: Mapping[str, Any]) -> str:
    payload = {"node_id": node_id, "queries": queries, "contexts": [dict(context)]}
    resp = requests.post(f"{DB_WORKER_URL}/execute_batch", json=payload, timeout=DB_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    outputs = data.get("outputs") or [{}]
    return json.dumps(outputs[0], ensure_ascii=False)


async def call_db_worker(node_id: str, queries: List[Dict[str, Any]], context: Mapping[str, Any]) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, call_db_worker_sync, node_id, queries, context)


# DB queries mirroring templates/finewiki_long_chain.yaml
GLOBAL_EXPANDER_QUERIES = [
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


async def run_one_query(user_query: str, models: Dict[str, OpenAIChatModel]) -> str:
    # global_topic_expander
    global_system = (
        "You are a global topic expander over a Wikipedia-like corpus.\n"
        "Use fw_global_wikitext_fulltext and fw_global_title_keyword to build\n"
        "a broad but coherent summary of the topic and propose a small list\n"
        "of central candidate page_ids (as plain integers in text).\n"
        "Respond ONLY with:\n"
        '{\n  "global_summary": "string",\n  "central_page_ids": [integer, ...]\n}'
    )
    global_db = await call_db_worker("global_topic_expander", GLOBAL_EXPANDER_QUERIES, {"user_query": user_query})
    global_db = json.dumps(_prepare_result_for_prompt(json.loads(global_db)), ensure_ascii=False)
    global_prompt = build_halo_prompt(None, {"user_query": user_query}, [global_db])
    global_resp = await chat(models["a"], [{"role": "system", "content": global_system}, {"role": "user", "content": global_prompt}])
    parsed = extract_outputs(["global_summary", "central_page_ids"], global_resp)
    global_summary = parsed.get("global_summary", "")
    central_page_ids = parsed.get("central_page_ids", "")

    # timeline_refiner_1
    timeline1_system = (
        "You are a timeline-focused refiner.\n"
        "Given global_summary and (textual) central_page_ids, and using\n"
        "fw_timeline_from_keyword as context, construct a coarse\n"
        "chronological narrative of the topic.\n"
        "Respond ONLY with:\n"
        '{\n  "coarse_timeline": "string"\n}'
    )
    timeline_db = await call_db_worker("timeline_refiner_1", TIMELINE_QUERIES, {"user_query": user_query})
    timeline_db = json.dumps(_prepare_result_for_prompt(json.loads(timeline_db)), ensure_ascii=False)
    timeline1_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "global_summary": global_summary, "central_page_ids": central_page_ids},
        [timeline_db],
    )
    timeline1_resp = await chat(models["a"], [{"role": "system", "content": timeline1_system}, {"role": "user", "content": timeline1_prompt}])
    coarse_timeline = extract_outputs(["coarse_timeline"], timeline1_resp).get("coarse_timeline", "")

    # timeline_refiner_2
    timeline2_system = (
        "You are a second-pass timeline refiner.\n"
        "Take the coarse_timeline and improve clarity and ordering, and\n"
        "highlight 3–5 key milestones.\n"
        "Respond ONLY with:\n"
        '{\n  "refined_timeline": "string"\n}'
    )
    timeline2_prompt = build_halo_prompt(None, {"user_query": user_query, "coarse_timeline": coarse_timeline}, [])
    timeline2_resp = await chat(models["a"], [{"role": "system", "content": timeline2_system}, {"role": "user", "content": timeline2_prompt}])
    refined_timeline = extract_outputs(["refined_timeline"], timeline2_resp).get("refined_timeline", "")

    # category_overview
    category_system = (
        "You are a category-focused summarizer.\n"
        "Use fw_category_pages to provide a high-level taxonomy of the topic.\n"
        "Respond ONLY with:\n"
        '{\n  "category_overview": "string"\n}'
    )
    category_db = await call_db_worker("category_overview", CATEGORY_QUERIES, {"user_query": user_query})
    category_db = json.dumps(_prepare_result_for_prompt(json.loads(category_db)), ensure_ascii=False)
    category_prompt = build_halo_prompt(None, {"user_query": user_query}, [category_db])
    category_resp = await chat(models["b"], [{"role": "system", "content": category_system}, {"role": "user", "content": category_prompt}])
    category_overview = extract_outputs(["category_overview"], category_resp).get("category_overview", "")

    # category_drilldown
    drill_system = (
        "You are a drill-down assistant for category structures.\n"
        "Take category_overview and extract 3–5 important subtopics that\n"
        "should be mentioned in a final answer.\n"
        "Respond ONLY with:\n"
        '{\n  "category_details": ["string", ...]\n}'
    )
    drill_prompt = build_halo_prompt(None, {"user_query": user_query, "category_overview": category_overview}, [])
    drill_resp = await chat(models["b"], [{"role": "system", "content": drill_system}, {"role": "user", "content": drill_prompt}])
    category_details = extract_outputs(["category_details"], drill_resp).get("category_details", "")

    # outbound_link_view
    outbound_system = (
        "You are an outbound link summarizer.\n"
        "Given pages whose titles match user_query, summarize which topics\n"
        "they tend to link out to (based on their wikitext).\n"
        "Respond ONLY with:\n"
        '{\n  "outbound_links_overview": "string"\n}'
    )
    outbound_db = await call_db_worker("outbound_link_view", OUTBOUND_QUERIES, {"user_query": user_query})
    outbound_db = json.dumps(_prepare_result_for_prompt(json.loads(outbound_db)), ensure_ascii=False)
    outbound_prompt = build_halo_prompt(None, {"user_query": user_query}, [outbound_db])
    outbound_resp = await chat(models["c"], [{"role": "system", "content": outbound_system}, {"role": "user", "content": outbound_prompt}])
    outbound_links_overview = extract_outputs(["outbound_links_overview"], outbound_resp).get("outbound_links_overview", "")

    # inbound_link_view
    inbound_system = (
        "You summarize who links into this topic.\n"
        "Given outbound_links_overview and global_summary, infer which\n"
        "other pages or domains frequently point to this topic and why\n"
        "that matters for understanding it.\n"
        "Respond ONLY with:\n"
        '{\n  "inbound_links_overview": "string"\n}'
    )
    inbound_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "outbound_links_overview": outbound_links_overview, "global_summary": global_summary},
        [],
    )
    inbound_resp = await chat(models["c"], [{"role": "system", "content": inbound_system}, {"role": "user", "content": inbound_prompt}])
    inbound_links_overview = extract_outputs(["inbound_links_overview"], inbound_resp).get("inbound_links_overview", "")

    # narrow_focus_rewriter
    narrow_system = (
        "You rewrite the user_query into a narrower focus question\n"
        "suitable for a final answer over this topic.\n"
        "Respond ONLY with:\n"
        '{\n  "narrowed_query": "string"\n}'
    )
    narrow_prompt = build_halo_prompt(None, {"user_query": user_query}, [])
    narrow_resp = await chat(models["a"], [{"role": "system", "content": narrow_system}, {"role": "user", "content": narrow_prompt}])
    narrowed_query = extract_outputs(["narrowed_query"], narrow_resp).get("narrowed_query", "")

    # final_answer_summarizer
    final_system = (
        "You are the final answer assistant.\n"
        "Combine global_summary, refined_timeline, category_details,\n"
        "outbound_links_overview, inbound_links_overview, and\n"
        "narrowed_query (plus fw_supporting_snippets_global as loose\n"
        "background) into a single clear, grounded answer.\n"
        "Respond ONLY with:\n"
        '{\n  "final_answer": "string"\n}'
    )
    final_db = await call_db_worker("final_answer_summarizer", FINAL_SNIPPET_QUERIES, {"user_query": user_query})
    final_db = json.dumps(_prepare_result_for_prompt(json.loads(final_db)), ensure_ascii=False)
    final_prompt = build_halo_prompt(
        None,
        {
            "user_query": user_query,
            "global_summary": global_summary,
            "refined_timeline": refined_timeline,
            "category_details": category_details,
            "outbound_links_overview": outbound_links_overview,
            "inbound_links_overview": inbound_links_overview,
            "narrowed_query": narrowed_query,
        },
        [final_db],
    )
    # Final head runs on model C (Qwen/Qwen3-14B) to match template.
    final_resp = await chat(models["c"], [{"role": "system", "content": final_system}, {"role": "user", "content": final_prompt}])
    final_answer = extract_outputs(["final_answer"], final_resp).get("final_answer", "")
    return str(final_answer)


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


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FineWiki long-chain agentscope baseline (Halo-aligned).")
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

    models = {
        "a": make_model(MODEL_A, BASE_A),
        "b": make_model(MODEL_B, BASE_B),
        "c": make_model(MODEL_C, BASE_C),
    }
    metrics = BaselineMetrics()

    if args.query:
        queries = [args.query.strip()]
        sample_info: SampleResult | None = None
    else:
        sample_info = _sample_queries(args)
        queries = sample_info.queries
        if not queries:
            print("No queries sampled from dataset.")
            return 1

    metrics.start()
    if sample_info is not None:
        metrics.add_metadata("dataset", args.dataset_name)
        metrics.add_metadata("dataset_subset", args.dataset_subset)
        metrics.add_metadata("dataset_split", args.dataset_split)
        metrics.add_metadata("sample_count", args.sample_count)
        metrics.add_metadata("rows_scanned", sample_info.rows_scanned)

    async def worker(q: str, idx: int) -> tuple[int, str, str, float]:
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        ans = await run_one_query(q, models)
        dur = loop.time() - t0
        metrics.record_query(dur)
        return idx, q, ans, dur

    async def run_all() -> int:
        total = len(queries)
        tasks = [asyncio.create_task(worker(q, idx)) for idx, q in enumerate(queries, 1)]
        for coro in asyncio.as_completed(tasks):
            idx, q, ans, dur = await coro
            print(f"\n===== Query #{idx}/{total} =====")
            print(q)
            print("\n=== Final Answer ===")
            print(ans)
            print(f"\n(latency: {dur:.3f}s)")
            print("\n" + "=" * 60)
        metrics.finish()
        print("\n" + metrics.format_summary())
        return 0

    return asyncio.run(run_all())


if __name__ == "__main__":
    sys.exit(main())
