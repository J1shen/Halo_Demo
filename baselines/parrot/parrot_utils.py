"""Shared helpers for Parrot baseline scripts (prompt building, parsing, guards)."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

from parrot.frontend.pfunc.semantic_variable import SemanticVariable
from baselines.metrics import get_global_metrics

# 从环境变量读取超时时间（秒），默认 1200 秒（20 分钟）
PARROT_AGET_TIMEOUT = float(os.getenv("PARROT_AGET_TIMEOUT", "1200"))


def guard_input(val: Any) -> str:
    """Normalize any input to string; None -> empty string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


def sanitize_value(value: Any, max_chars: int = 2000) -> Any:
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars] + "...[truncated]"
        return value
    if isinstance(value, dict):
        return {k: sanitize_value(v, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(v, max_chars) for v in value]
    return value


def prepare_result_for_prompt(result: Dict[str, Any], max_rows: int = 5, max_chars: int = 2000) -> Dict[str, Any]:
    prepared = dict(result)
    rows = prepared.get("rows")
    if isinstance(rows, list):
        trimmed = []
        for row in rows[:max_rows]:
            if isinstance(row, dict):
                trimmed.append({k: sanitize_value(v, max_chars) for k, v in row.items()})
            else:
                trimmed.append(sanitize_value(row, max_chars))
        prepared["rows"] = trimmed
        if len(rows) > max_rows:
            prepared["rows_truncated"] = f"{len(rows) - max_rows} rows omitted"
    return prepared


def render_input_value(value: Any, max_rows: int = 5, max_chars: int = 2000) -> str:
    if isinstance(value, dict) and "rows" in value:
        value = prepare_result_for_prompt(value, max_rows=max_rows, max_chars=max_chars)
    try:
        if isinstance(value, (dict, list)) and not isinstance(value, (str, bytes, bytearray)):
            return json.dumps(value, ensure_ascii=False)
    except Exception:
        pass
    return str(value)


def parse_db_payload(payload: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            if "query" in data:
                return [data]
            results: List[Dict[str, Any]] = []
            for val in data.values():
                if isinstance(val, dict):
                    results.append(val)
            return results
    except Exception:
        return []
    return []


def build_halo_prompt(
    system_prompt: str | None, inputs: Dict[str, Any], db_payloads: List[str], max_rows: int = 5, max_chars: int = 2000
) -> str:
    parts: List[str] = []
    if system_prompt:
        from halo_dev.utils import render_template
        parts.append(render_template(system_prompt.strip(), inputs))
    for name, value in inputs.items():
        rendered = render_input_value(value, max_rows=max_rows, max_chars=max_chars)
        parts.append(f"[Input::{name}]\n{rendered}")
    for payload in db_payloads:
        for result in parse_db_payload(payload):
            prepared = prepare_result_for_prompt(result, max_rows=max_rows, max_chars=max_chars)
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


def extract_outputs(expected_keys: List[str], response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        parsed = response
    else:
        parsed = maybe_parse_structured_response(str(response))
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


def prepare_db_payload_for_prompt(payload: str, max_rows: int = 5, max_chars: int = 2000) -> str:
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return json.dumps(prepare_result_for_prompt(data, max_rows=max_rows, max_chars=max_chars), ensure_ascii=False)
    except Exception:
        pass
    return guard_input(payload)


async def safe_aget(var: SemanticVariable, name: str, default: str = "{}", timeout: float | None = None) -> Any:
    """Async get with timeout support.
    
    Args:
        var: SemanticVariable to fetch
        name: Variable name (for logging)
        default: Default value if fetch fails
        timeout: Timeout in seconds (uses PARROT_AGET_TIMEOUT env var if not provided)
    """
    import asyncio
    
    actual_timeout = timeout if timeout is not None else PARROT_AGET_TIMEOUT
    start = time.perf_counter()
    try:
        res = await asyncio.wait_for(var.aget(), timeout=actual_timeout)
    except asyncio.TimeoutError as e:
        print(f"[WARN] {name} .aget() timeout: {e}. Using default.")
        return default
    except Exception as e:
        print(f"[WARN] {name} .aget() failed: {e}. Using default.")
        return default
    finally:
        metrics = get_global_metrics()
        if metrics:
            metrics.record_llm(time.perf_counter() - start, call_count=1, prompt_count=1)
    
    if res is None or res == "":
        print(f"[WARN] {name} returned empty content. Using default.")
        return default
    return res
