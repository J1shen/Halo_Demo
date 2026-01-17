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

from run.query_sampler import (
    DEFAULT_SAMPLE_POOL_SIZE,
    SampleResult,
    default_extract_query,
    sample_queries,
)

import requests
from agentscope.model import OpenAIChatModel
from baselines.metrics import BaselineMetrics
from halo_dev.utils import render_template

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
MAX_FIELD_CHARS = 2000
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


# DB queries
FW_A0_QUERIES = [
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

FW_B1_QUERIES = [
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

FW_C1_QUERIES = [
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


async def run_one_query(user_query: str, models: Dict[str, OpenAIChatModel]) -> str:
    # a0
    a0_system = (
        "First global pass over FineWiki for the topic.\n"
        "Use fw_global_pass_a0_main as your retrieval pool.\n"
        "Respond ONLY with:\n"
        '{\n  "a0_summary": "string"\n}'
    )
    a0_db = await call_db_worker("fw_global_pass_a0", FW_A0_QUERIES, {"user_query": user_query})
    a0_db = json.dumps(_prepare_result_for_prompt(json.loads(a0_db)), ensure_ascii=False)
    a0_prompt = build_halo_prompt(None, {"user_query": user_query}, [a0_db])
    a0_resp = await chat(models["a"], [{"role": "system", "content": a0_system}, {"role": "user", "content": a0_prompt}])
    a0_summary = extract_outputs(["a0_summary"], a0_resp).get("a0_summary", "")

    # a1
    a1_system = (
        "Refine a0_summary into a structured outline.\n"
        "Respond ONLY with:\n"
        '{\n  "a1_outline": "string"\n}'
    )
    a1_prompt = build_halo_prompt(None, {"user_query": user_query, "a0_summary": a0_summary}, [])
    a1_resp = await chat(models["a"], [{"role": "system", "content": a1_system}, {"role": "user", "content": a1_prompt}])
    a1_outline = extract_outputs(["a1_outline"], a1_resp).get("a1_outline", "")

    # a2
    a2_system = (
        "Expand a1_outline into a longer narrative suitable for\n"
        "downstream bridges and final answering.\n"
        "Respond ONLY with:\n"
        '{\n  "a2_narrative": "string"\n}'
    )
    a2_prompt = build_halo_prompt(None, {"user_query": user_query, "a1_outline": a1_outline}, [])
    a2_resp = await chat(models["a"], [{"role": "system", "content": a2_system}, {"role": "user", "content": a2_prompt}])
    a2_narrative = extract_outputs(["a2_narrative"], a2_resp).get("a2_narrative", "")

    # b1
    b1_system = (
        "Category-based bridge: connect a1_outline with category pages\n"
        "from fw_category_bridge_b1_main.\n"
        "Respond ONLY with:\n"
        '{\n  "b1_bridge_notes": "string"\n}'
    )
    b1_db = await call_db_worker("fw_category_bridge_b1", FW_B1_QUERIES, {"user_query": user_query})
    b1_db = json.dumps(_prepare_result_for_prompt(json.loads(b1_db)), ensure_ascii=False)
    b1_prompt = build_halo_prompt(None, {"user_query": user_query, "a1_outline": a1_outline}, [b1_db])
    b1_resp = await chat(models["b"], [{"role": "system", "content": b1_system}, {"role": "user", "content": b1_prompt}])
    b1_bridge_notes = extract_outputs(["b1_bridge_notes"], b1_resp).get("b1_bridge_notes", "")

    # b2
    b2_system = (
        "Refine b1_bridge_notes into 3–5 key subtopics that are useful\n"
        "for guiding the final answer.\n"
        "Respond ONLY with:\n"
        '{\n  "b2_key_subtopics": ["string", ...]\n}'
    )
    b2_prompt = build_halo_prompt(None, {"user_query": user_query, "b1_bridge_notes": b1_bridge_notes}, [])
    b2_resp = await chat(models["b"], [{"role": "system", "content": b2_system}, {"role": "user", "content": b2_prompt}])
    b2_key_subtopics = extract_outputs(["b2_key_subtopics"], b2_resp).get("b2_key_subtopics", "")

    # c1
    c1_system = (
        "Non-English language bridge for the topic.\n"
        "Use fw_language_bridge_c1_main to understand how the topic\n"
        "appears in other languages.\n"
        "Respond ONLY with:\n"
        '{\n  "c1_language_notes": "string"\n}'
    )
    c1_db = await call_db_worker("fw_language_bridge_c1", FW_C1_QUERIES, {"user_query": user_query})
    c1_db = json.dumps(_prepare_result_for_prompt(json.loads(c1_db)), ensure_ascii=False)
    c1_prompt = build_halo_prompt(None, {"user_query": user_query, "a0_summary": a0_summary}, [c1_db])
    c1_resp = await chat(models["c"], [{"role": "system", "content": c1_system}, {"role": "user", "content": c1_prompt}])
    c1_language_notes = extract_outputs(["c1_language_notes"], c1_resp).get("c1_language_notes", "")

    # c2
    c2_system = (
        "Refine c1_language_notes into 2–4 concise cross-lingual\n"
        "insights that can influence the final answer.\n"
        "Respond ONLY with:\n"
        '{\n  "c2_crosslingual_insights": "string"\n}'
    )
    c2_prompt = build_halo_prompt(None, {"user_query": user_query, "c1_language_notes": c1_language_notes}, [])
    c2_resp = await chat(models["c"], [{"role": "system", "content": c2_system}, {"role": "user", "content": c2_prompt}])
    c2_crosslingual_insights = extract_outputs(["c2_crosslingual_insights"], c2_resp).get("c2_crosslingual_insights", "")

    # s0
    s0_system = (
        "Light sampler for the bridge topology.\n"
        "Condense a2_narrative, b2_key_subtopics, and\n"
        "c2_crosslingual_insights into short hints that are easy for\n"
        "the final answer head to consume.\n"
        "Respond ONLY with:\n"
        '{\n  "s0_hints": ["string", ...]\n}'
    )
    s0_prompt = build_halo_prompt(
        None,
        {
            "user_query": user_query,
            "a2_narrative": a2_narrative,
            "b2_key_subtopics": b2_key_subtopics,
            "c2_crosslingual_insights": c2_crosslingual_insights,
        },
        [],
    )
    s0_resp = await chat(models["c"], [{"role": "system", "content": s0_system}, {"role": "user", "content": s0_prompt}])
    s0_hints = extract_outputs(["s0_hints"], s0_resp).get("s0_hints", "")

    # final
    final_system = (
        "Final answer assistant for the bridge topology.\n"
        "Combine a2_narrative and s0_hints into a single coherent,\n"
        "well-structured answer to user_query.\n"
        "Respond ONLY with:\n"
        '{\n  "final_answer": "string"\n}'
    )
    final_prompt = build_halo_prompt(
        None,
        {"user_query": user_query, "a2_narrative": a2_narrative, "s0_hints": s0_hints},
        [],
    )
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
    parser = argparse.ArgumentParser(description="FineWiki bridge topology Agentscope baseline (Halo-aligned).")
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
    from pathlib import Path

    sys.exit(main())
