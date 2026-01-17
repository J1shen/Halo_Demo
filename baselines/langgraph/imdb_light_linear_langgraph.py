from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.metrics import BaselineMetrics, get_global_metrics, set_global_metrics
from halo_dev.utils import render_template

API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
MODEL_STAGE1 = os.getenv("IMDB_LIGHT_MODEL_A", "Qwen/Qwen3-0.6B")
MODEL_STAGE2 = os.getenv("IMDB_LIGHT_MODEL_B", "Qwen/Qwen3-1.7B")
MODEL_STAGE3 = os.getenv("IMDB_LIGHT_MODEL_C", "Qwen/Qwen3-4B")
BASE_A = os.getenv("MODEL_A_BASE_URL", "http://localhost:9101/v1")
BASE_B = os.getenv("MODEL_B_BASE_URL", "http://localhost:9102/v1")
BASE_C = os.getenv("MODEL_C_BASE_URL", "http://localhost:9103/v1")

STAGE1_SYSTEM = """Stage 1 (Planner).
Extract intent + key entities and make a short outline.
Do not answer the question.

Output JSON only:
{
  "intent": "director | filmography | starring | top_rated | released_in_year | other",
  "entities": {
    "person": "string_or_null",
    "title": "string_or_null",
    "genre": "string_or_null",
    "year": "int_or_null"
  },
  "constraints": ["..."],
  "outline": ["step1", "step2"]
}
"""

STAGE2_SYSTEM = """Stage 2 (Draft).
Follow stage1_plan_json to write a helpful draft.
If facts are unknown without a database, say so briefly and give a reasonable way to proceed.

Output JSON only:
{
  "draft_answer": "string",
  "assumptions": ["..."]
}
"""

STAGE3_SYSTEM = """Stage 3 (Finalize).
Improve clarity and concision, remove repetition, keep the answer direct.

Output JSON only:
{
  "final_answer": "string"
}
"""


class State(TypedDict, total=False):
    user_query: str
    stage1_plan_json: Any
    stage2_draft_json: Any
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


def _render_input_value(value: Any) -> str:
    try:
        if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
            return json.dumps(value, ensure_ascii=False)
    except Exception:
        pass
    return str(value)


def build_halo_prompt(system_prompt: str | None, inputs: Mapping[str, Any]) -> str:
    parts: List[str] = []
    if system_prompt:
        rendered_system = render_template(system_prompt.strip(), inputs)
        parts.append(rendered_system)
    for name, value in inputs.items():
        rendered = _render_input_value(value)
        parts.append(f"[Input::{name}]\n{rendered}")
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


def _coerce_dict(parsed: Any, fallback_key: str) -> Dict[str, Any]:
    if isinstance(parsed, dict):
        return parsed
    return {fallback_key: parsed}


def invoke_llm(model: ChatOpenAI, messages: List[Mapping[str, str]]):
    start = time.perf_counter()
    resp = model.invoke(messages)
    metrics = get_global_metrics()
    if metrics:
        metrics.record_llm(time.perf_counter() - start, call_count=1, prompt_count=1)
    return resp


def stage1_plan(state: State, model_a: ChatOpenAI) -> Dict[str, Any]:
    prompt = build_halo_prompt(None, {"user_query": state["user_query"]})
    resp = invoke_llm(model_a, [{"role": "system", "content": STAGE1_SYSTEM}, {"role": "user", "content": prompt}])
    parsed = maybe_parse_structured_response(resp.content)
    return {"stage1_plan_json": _coerce_dict(parsed, "raw_plan")}


def stage2_draft(state: State, model_b: ChatOpenAI) -> Dict[str, Any]:
    prompt = build_halo_prompt(
        None,
        {"user_query": state["user_query"], "stage1_plan_json": state.get("stage1_plan_json", {})},
    )
    resp = invoke_llm(model_b, [{"role": "system", "content": STAGE2_SYSTEM}, {"role": "user", "content": prompt}])
    parsed = maybe_parse_structured_response(resp.content)
    return {"stage2_draft_json": _coerce_dict(parsed, "draft_answer")}


def stage3_finalize(state: State, model_c: ChatOpenAI) -> Dict[str, Any]:
    prompt = build_halo_prompt(
        None,
        {
            "user_query": state["user_query"],
            "stage1_plan_json": state.get("stage1_plan_json", {}),
            "stage2_draft_json": state.get("stage2_draft_json", {}),
        },
    )
    resp = invoke_llm(model_c, [{"role": "system", "content": STAGE3_SYSTEM}, {"role": "user", "content": prompt}])
    parsed = maybe_parse_structured_response(resp.content)
    if isinstance(parsed, dict) and "final_answer" in parsed:
        final_answer = parsed["final_answer"]
    elif isinstance(parsed, (dict, list)):
        final_answer = json.dumps(parsed, ensure_ascii=False)
    else:
        final_answer = str(parsed)
    return {"final_answer": final_answer}


def build_graph(model_a: ChatOpenAI, model_b: ChatOpenAI, model_c: ChatOpenAI) -> StateGraph[State]:
    g: StateGraph[State] = StateGraph(State)
    g.add_node("stage1_plan", lambda s: stage1_plan(s, model_a))
    g.add_node("stage2_draft", lambda s: stage2_draft(s, model_b))
    g.add_node("stage3_finalize", lambda s: stage3_finalize(s, model_c))
    g.add_edge(START, "stage1_plan")
    g.add_edge("stage1_plan", "stage2_draft")
    g.add_edge("stage2_draft", "stage3_finalize")
    g.add_edge("stage3_finalize", END)
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


def _format_state_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IMDb light linear LangGraph workflow (LLM only).")
    parser.add_argument("--query", type=str, default=None, help="Single user query; overrides --file.")
    parser.add_argument("--file", type=str, default="data/imdb_input.txt", help="Batch query file.")
    parser.add_argument("--limit", type=int, default=None, help="Read at most N lines from the file.")
    parser.add_argument("--count", type=int, default=None, help="Process only the first N queries.")
    args = parser.parse_args(argv)

    metrics = BaselineMetrics()
    models = {
        "a": make_model(MODEL_STAGE1, BASE_A),
        "b": make_model(MODEL_STAGE2, BASE_B),
        "c": make_model(MODEL_STAGE3, BASE_C),
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

    metrics.start(run_name="imdb_light_linear_langgraph", total_queries=len(queries))
    metrics.add_metadata("runner", "langgraph")
    metrics.add_metadata("template", "imdb_light_linear")
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
            print("\n--- Stage 1 Plan ---")
            print(_format_state_value(final_state.get("stage1_plan_json", "")))
            print("\n--- Stage 2 Draft ---")
            print(_format_state_value(final_state.get("stage2_draft_json", "")))
            print("\n" + "=" * 60)

    set_global_metrics(metrics)
    asyncio.run(runner())
    set_global_metrics(None)
    metrics.finish()
    print("\n" + metrics.format_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
