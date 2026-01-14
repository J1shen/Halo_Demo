from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, List, Mapping, Optional

from datasets import load_dataset


DEFAULT_SAMPLE_POOL_SIZE = 2_048


@dataclass(slots=True)
class SampleResult:
    queries: List[str]
    rows_scanned: int


def sample_queries(
    *,
    dataset: str,
    subset: str,
    split: str,
    sample_count: int,
    seed: int | None,
    pool_size: int | None = DEFAULT_SAMPLE_POOL_SIZE,
    extract_query: Optional[Callable[[Mapping[str, Any]], str | None]] = None,
) -> SampleResult:
    """Stream a HuggingFace dataset and return `sample_count` queries.

    Parameters mirror the original logic from FineWiki runner but are reusable.
    `extract_query` lets callers customize how each dataset row becomes a query.
    """

    if sample_count <= 0:
        return SampleResult(queries=[], rows_scanned=0)

    extractor = extract_query or default_extract_query
    rng = random.Random(seed)
    stream = load_dataset(dataset, subset, split=split, streaming=True)

    samples: List[str] = []
    seen = 0

    for row in stream:
        query = extractor(row)
        if not query:
            continue
        seen += 1
        if pool_size is not None and seen > pool_size:
            break

        if len(samples) < sample_count:
            samples.append(query)
        else:
            j = rng.randint(0, seen - 1)
            if j < sample_count:
                samples[j] = query

    if samples:
        rng.shuffle(samples)

    return SampleResult(queries=samples, rows_scanned=seen)


def default_extract_query(row: Mapping[str, Any]) -> str | None:
    title = row.get("title") or row.get("id")
    if not title:
        return None
    return str(title).strip() or None


__all__ = [
    "DEFAULT_SAMPLE_POOL_SIZE",
    "sample_queries",
    "SampleResult",
    "default_extract_query",
]
