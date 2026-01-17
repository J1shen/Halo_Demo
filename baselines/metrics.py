from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BaselineMetrics:
    """Lightweight shared metrics collector for baseline runs.

    Mirrors Halo's MetricsTracker so baseline runs emit comparable summaries.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    run_name: str | None = None
    total_queries: int = 0
    start_time: float | None = None
    end_time: float | None = None
    query_count: int = 0
    total_query_time: float = 0.0
    max_query_time: float = 0.0
    min_query_time: float = 0.0
    db_exec_time_sum: float = 0.0
    db_exec_count: int = 0
    llm_exec_time_sum: float = 0.0
    llm_exec_count: int = 0
    llm_prompt_count: int = 0
    model_init_time_sum: float = 0.0
    model_init_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self, *, run_name: str | None = None, total_queries: int | None = None) -> None:
        """Begin a run; resets all counters."""
        with self._lock:
            self.run_name = run_name or self.run_name
            self.total_queries = max(0, total_queries or 0)
            self.start_time = time.perf_counter()
            self.end_time = None
            self.query_count = 0
            self.total_query_time = 0.0
            self.max_query_time = 0.0
            self.min_query_time = 0.0
            self.db_exec_time_sum = 0.0
            self.db_exec_count = 0
            self.llm_exec_time_sum = 0.0
            self.llm_exec_count = 0
            self.llm_prompt_count = 0
            self.model_init_time_sum = 0.0
            self.model_init_count = 0
            self.metadata.clear()

    def record_query(self, duration: float, *, count: int = 1) -> None:
        """Record a single or aggregated query latency (seconds)."""
        if duration < 0:
            duration = 0.0
        effective_count = max(count, 1)
        with self._lock:
            self.query_count += effective_count
            self.total_query_time += duration * effective_count
            self.total_queries = max(self.total_queries, self.query_count)
            # Preserve min/max for single-query recordings
            if effective_count == 1:
                if self.query_count == 1:
                    self.max_query_time = duration
                    self.min_query_time = duration
                else:
                    if duration > self.max_query_time:
                        self.max_query_time = duration
                    if duration < self.min_query_time:
                        self.min_query_time = duration

    def record_db(self, duration: float, *, count: int = 1) -> None:
        """Record DB execution time in seconds."""
        if duration < 0:
            duration = 0.0
        with self._lock:
            self.db_exec_time_sum += duration
            self.db_exec_count += max(count, 1)

    def record_llm(self, duration: float, *, call_count: int = 1, prompt_count: int = 1) -> None:
        """Record LLM execution time in seconds."""
        if duration < 0:
            duration = 0.0
        with self._lock:
            self.llm_exec_time_sum += duration
            self.llm_exec_count += max(call_count, 1)
            self.llm_prompt_count += max(prompt_count, 0)

    def record_model_init(self, duration: float, *, count: int = 1) -> None:
        """Record model initialization time in seconds."""
        if duration < 0:
            duration = 0.0
        with self._lock:
            self.model_init_time_sum += duration
            self.model_init_count += max(count, 1)

    def add_metadata(self, key: str, value: Any) -> None:
        with self._lock:
            self.metadata[key] = value

    def finish(self) -> None:
        with self._lock:
            self.end_time = time.perf_counter()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            end = self.end_time or time.perf_counter()
            total_run = max(0.0, end - self.start_time) if self.start_time else 0.0
            total_queries = self.total_queries or self.query_count
            avg_query = (
                (self.total_query_time / self.query_count) if self.query_count else 0.0
            )
            throughput = (total_queries / total_run) if total_run > 0 and total_queries else 0.0
            avg_db = (
                (self.db_exec_time_sum / self.db_exec_count) if self.db_exec_count else None
            )
            avg_llm = (
                (self.llm_exec_time_sum / self.llm_exec_count) if self.llm_exec_count else None
            )
            avg_llm_prompt = (
                (self.llm_exec_time_sum / self.llm_prompt_count) if self.llm_prompt_count else None
            )
            avg_model_init = (
                (self.model_init_time_sum / self.model_init_count) if self.model_init_count else None
            )

            return {
                "run_name": self.run_name or "baseline",
                "total_run_seconds": total_run,
                "queries": float(total_queries),
                "avg_query_seconds": avg_query,
                "max_query_seconds": self.max_query_time if self.query_count else 0.0,
                "min_query_seconds": self.min_query_time if self.query_count else 0.0,
                "throughput_qps": throughput,
                "db_count": float(self.db_exec_count),
                "db_total_seconds": self.db_exec_time_sum,
                "db_avg_seconds": avg_db or 0.0,
                "llm_calls": float(self.llm_exec_count),
                "llm_prompts": float(self.llm_prompt_count),
                "llm_total_seconds": self.llm_exec_time_sum,
                "llm_avg_call_seconds": avg_llm or 0.0,
                "llm_avg_prompt_seconds": avg_llm_prompt or 0.0,
                "model_init_count": float(self.model_init_count),
                "model_init_total_seconds": self.model_init_time_sum,
                "model_init_avg_seconds": avg_model_init or 0.0,
                "metadata": dict(self.metadata),
            }

    def format_summary(self) -> str:
        stats = self.summary()
        lines = [
            "===== Baseline Metrics =====",
            f"Run: {stats['run_name']}",
            f"Total queries: {int(stats['queries'])}",
            f"E2E latency: {stats['total_run_seconds']:.2f}s",
        ]
        if stats["avg_query_seconds"] > 0:
            lines.append(f"Avg per-query latency: {stats['avg_query_seconds']:.2f}s")
        if stats["throughput_qps"] > 0:
            lines.append(f"Throughput: {stats['throughput_qps']:.2f} query/s")
        if stats["db_count"]:
            avg_db = stats["db_avg_seconds"]
            avg_db_str = f"{avg_db:.2f}s" if avg_db else "n/a"
            lines.append(
                f"DB queries: count={int(stats['db_count'])} "
                f"total={stats['db_total_seconds']:.2f}s avg={avg_db_str}"
            )
        if stats["llm_calls"]:
            avg_llm_call = stats["llm_avg_call_seconds"]
            avg_llm_prompt = stats["llm_avg_prompt_seconds"]
            lines.append(
                f"LLM exec: calls={int(stats['llm_calls'])} prompts={int(stats['llm_prompts'])} "
                f"total={stats['llm_total_seconds']:.2f}s "
                f"avg_call={avg_llm_call:.2f}s avg_prompt={avg_llm_prompt:.2f}s"
            )
        if stats["model_init_count"]:
            avg_init = stats["model_init_avg_seconds"]
            lines.append(
                f"Model init: count={int(stats['model_init_count'])} "
                f"total={stats['model_init_total_seconds']:.2f}s avg={avg_init:.2f}s"
            )
        else:
            lines.append(
                "Model init: count=0 total=0.00s avg=n/a"
            )
        metadata = stats.get("metadata") or {}
        if metadata:
            lines.append("Metadata:")
            for key, value in metadata.items():
                lines.append(f"  - {key}: {value}")
        lines.append("============================")
        return "\n".join(lines)


_GLOBAL_METRICS: BaselineMetrics | None = None


def set_global_metrics(metrics: BaselineMetrics | None) -> None:
    global _GLOBAL_METRICS
    _GLOBAL_METRICS = metrics


def get_global_metrics() -> BaselineMetrics | None:
    return _GLOBAL_METRICS
