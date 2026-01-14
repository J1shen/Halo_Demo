from __future__ import annotations
import argparse
import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Sequence

from halo_dev import metrics
from halo_dev.db import PostgresDatabaseExecutor, PostgresPlanExplainer
from halo_dev.models import ExecutionPlan, GraphSpec, PlanMetric


def print_planning_timings(plan: ExecutionPlan) -> None:
    metadata = plan.metadata or {}
    total = metadata.get("optimize_duration_s")
    breakdown = metadata.get("planning_breakdown_s") or {}
    eval_t = breakdown.get("plan_evaluation")
    sched_t = breakdown.get("scheduling")
    dp_stats = metadata.get("dp_stats") or {}

    has_any = any(isinstance(x, (int, float)) for x in (total, eval_t, sched_t))
    if not has_any:
        return

    print("\n[Planning] Optimizer timings:")
    if isinstance(total, (int, float)):
        print(f"  total: {total * 1000:.2f} ms")
    if isinstance(eval_t, (int, float)):
        print(f"  plan_evaluation: {eval_t * 1000:.2f} ms")
    if isinstance(sched_t, (int, float)):
        print(f"  scheduling: {sched_t * 1000:.2f} ms")
    if dp_stats:
        states = dp_stats.get("states_explored")
        memo_entries = dp_stats.get("memo_entries")
        memo_hits = dp_stats.get("memo_hits")
        best_cost = dp_stats.get("best_cost")
        print("  dp_stats:")
        if states is not None:
            print(f"    states_explored: {states}")
        if memo_entries is not None:
            print(f"    memo_entries: {memo_entries}")
        if memo_hits is not None:
            print(f"    memo_hits: {memo_hits}")
        if isinstance(best_cost, (int, float)):
            print(f"    best_cost: {best_cost:.4f}")


def print_phase_one_summary(plan: ExecutionPlan) -> None:
    print("\n[Phase 1] DB Plan Profiling")

    metadata = getattr(plan, "metadata", {}) or {}
    optimize_duration = metadata.get("optimize_duration_s")
    if isinstance(optimize_duration, (int, float)):
        print(f"Optimize time: {optimize_duration * 1000:.2f} ms")
    if not plan.selected_query_plans:
        print("No query plan profiling data available.")
        return

    for (node_id, query_name), choice in sorted(plan.selected_query_plans.items()):
        cost_str = f"{choice.cost:.2f}" if choice.cost is not None else "n/a"
        runtime = _avg_metric(choice.samples, "runtime_ms")
        hit_blocks = _avg_metric(choice.samples, "shared_hit_blocks")
        read_blocks = _avg_metric(choice.samples, "shared_read_blocks")
        explain_summary = _summarize_explain(choice.explain_json)
        pg_cost_str = f"{choice.raw_cost:.2f}" if choice.raw_cost is not None else "n/a"
        print(f"  - node={node_id} query={query_name}")
        print(
            f"      plan={choice.plan_id:<14} cost={cost_str:<8} pg_cost={pg_cost_str:<10} desc={choice.description}"
        )
        runtime_str = f"{runtime:.2f}ms" if runtime is not None else "n/a"
        hits = f"hits={hit_blocks:.1f}" if hit_blocks is not None else "hits=n/a"
        reads = f"reads={read_blocks:.1f}" if read_blocks is not None else "reads=n/a"
        print(f"        runtime={runtime_str} {hits} {reads}")
        if explain_summary:
            print(f"        explain={explain_summary}")
        footprint_summary = _format_footprints(choice.footprints, max_items=6)
        if footprint_summary:
            print(f"        order_features=pages[{footprint_summary}]")
        else:
            print("        order_features=pages[n/a]")


def print_phase_two_summary(plan: ExecutionPlan, graph: GraphSpec | None = None) -> None:
    print("\n[Phase 2] Worker Assignment")
    if not plan.tasks:
        print("No tasks in execution plan.")
        return
    def _worker_sort_key(worker_id: str | Sequence[str]) -> tuple:
        if isinstance(worker_id, (list, tuple)):
            return tuple(str(w) for w in worker_id)
        return (str(worker_id),)

    tasks = sorted(plan.tasks, key=lambda t: (t.epoch, _worker_sort_key(t.worker_id), t.node_id))
    if graph is None:
        for task in tasks:
            print(
                f"  • node={task.node_id:<32} worker={str(task.worker_id):<12} epoch={task.epoch}"
            )
    else:
        llm_tasks: List[Any] = []
        db_tasks: List[Any] = []
        other_tasks: List[Any] = []
        for task in tasks:
            node = graph.nodes.get(task.node_id)
            engine = getattr(node, "engine", None)
            model = getattr(node, "model", None)
            if engine == "vllm":
                llm_tasks.append((task, model))
            elif engine == "db":
                db_tasks.append((task, model))
            else:
                other_tasks.append((task, model))

        if llm_tasks:
            print("  LLM tasks:")
            for task, model in llm_tasks:
                print(
                    f"    • node={task.node_id:<40} worker={str(task.worker_id):<14} epoch={task.epoch:<3} model={model or 'n/a'}"
                )
        if db_tasks:
            print("  DB tasks:")
            for task, model in db_tasks:
                print(
                    f"    • node={task.node_id:<40} worker={str(task.worker_id):<14} epoch={task.epoch:<3} model={model or 'n/a'}"
                )
        if other_tasks:
            print("  Other tasks:")
            for task, model in other_tasks:
                print(
                    f"    • node={task.node_id:<40} worker={str(task.worker_id):<14} epoch={task.epoch:<3} model={model or 'n/a'}"
                )
    max_epoch = max(task.epoch for task in plan.tasks)
    print(f"Total epochs: {max_epoch + 1} (0 through {max_epoch})")
    metrics.add_metadata("max_epoch", max_epoch)

    # MILP continuous-time schedule details (if available).
    metadata = getattr(plan, "metadata", {}) or {}
    milp_meta = metadata.get("milp") if isinstance(metadata, dict) else None
    if isinstance(metadata, dict):
        cpu_load_weight = metadata.get("dp_cpu_load_cost_weight")
        if isinstance(cpu_load_weight, (int, float)):
            metrics.add_metadata("dp_cpu_load_cost_weight", cpu_load_weight)
        cpu_cost_mode = metadata.get("dp_cpu_cost_mode")
        if isinstance(cpu_cost_mode, str) and cpu_cost_mode:
            metrics.add_metadata("dp_cpu_cost_mode", cpu_cost_mode)
        disable_cpu_load = metadata.get("dp_disable_cpu_load_cost")
        if isinstance(disable_cpu_load, bool):
            metrics.add_metadata("dp_disable_cpu_load_cost", disable_cpu_load)
    node_times = milp_meta.get("node_times_s") if isinstance(milp_meta, dict) else None
    if isinstance(node_times, dict) and node_times:
        print("\n  MILP node times (continuous seconds):")
        task_by_node = {t.node_id: t for t in plan.tasks}
        # Display in increasing start time for readability.
        def _start_key(item: tuple[str, Any]) -> float:
            try:
                val = item[1].get("start_s") if isinstance(item[1], dict) else None
                return float(val) if val is not None else 0.0
            except Exception:
                return 0.0

        for node_id, times in sorted(node_times.items(), key=_start_key):
            if not isinstance(times, dict):
                continue
            start_s = times.get("start_s")
            end_s = times.get("end_s")
            dur_s = times.get("dur_s")
            task = task_by_node.get(node_id)
            epoch = task.epoch if task else "n/a"
            worker = str(task.worker_id) if task else "n/a"
            try:
                start_f = float(start_s) if start_s is not None else 0.0
                end_f = float(end_s) if end_s is not None else 0.0
                dur_f = float(dur_s) if dur_s is not None else max(0.0, end_f - start_f)
                print(
                    f"    • node={node_id:<40} worker={worker:<14} epoch={epoch:<3} "
                    f"start={start_f:.3f}s end={end_f:.3f}s dur={dur_f:.3f}s"
                )
            except Exception:
                print(f"    • node={node_id:<40} worker={worker:<14} epoch={epoch:<3} times={times}")


def print_run_outputs(
    final_context: Dict[str, Any],
    initial_context: Dict[str, Any],
    *,
    show_context: bool,
) -> None:
    produced = {
        key: value
        for key, value in final_context.items()
        if key not in initial_context or final_context[key] != initial_context[key]
    }

    if not produced:
        print("No new keys were produced.")
    else:
        print("Produced context keys:")
        _print_context_pairs(produced)

    if show_context:
        print("\nFull context:")
        _print_context_pairs(final_context)


def _avg_metric(samples: Sequence[PlanMetric], attr: str) -> float | None:
    values: List[float] = []
    for sample in samples:
        value = getattr(sample, attr, None)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _summarize_explain(payload: Any) -> str:
    if not payload:
        return ""
    try:
        entry = payload[0] if isinstance(payload, list) and payload else payload
        plan = entry.get("Plan") if isinstance(entry, dict) else None
        if not plan:
            return ""
        parts = []
        node_type = plan.get("Node Type")
        if node_type:
            parts.append(node_type)
        rel = plan.get("Relation Name")
        if rel:
            parts.append(f"rel={rel}")
        total_cost = plan.get("Total Cost")
        if total_cost is not None:
            parts.append(f"cost={total_cost}")
        actual_rows = plan.get("Actual Rows")
        if actual_rows is not None:
            parts.append(f"rows={actual_rows}")
        return " ".join(parts)
    except Exception:
        return ""


def _format_footprints(
    footprints: Mapping[str, int] | None,
    *,
    max_items: int = 5,
) -> str:
    if not footprints:
        return ""
    top_items = sorted(footprints.items(), key=lambda item: item[1], reverse=True)[
        :max(1, max_items)
    ]
    return ", ".join(f"{name}({weight})" for name, weight in top_items)


def _print_context_pairs(items: Dict[str, Any]) -> None:
    for key, value in items.items():
        serialized_value = _sanitize_for_json(value)
        if isinstance(serialized_value, (dict, list)):
            serialized = json.dumps(serialized_value, ensure_ascii=False, indent=2)
            print(f"{key}:\n{serialized}")
        else:
            print(f"{key}: {serialized_value}")


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        try:
            return float(value)
        except (ValueError, OverflowError):
            return str(value)
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    return value

def record_run_metadata(args: argparse.Namespace, plan: ExecutionPlan) -> None:
    """Attach run configuration to metrics for easier comparisons."""
    metrics.add_metadata("plan_mode", args.plan_mode)

    # Handle runner-specific args
    if hasattr(args, "input_file"):
        metrics.add_metadata("input_file", str(args.input_file))
    if hasattr(args, "template"):
        metrics.add_metadata("template", str(args.template))
    if hasattr(args, "dataset_name"):
        metrics.add_metadata("dataset", args.dataset_name)
    if hasattr(args, "dataset_subset"):
        metrics.add_metadata("dataset_subset", args.dataset_subset)
    if hasattr(args, "dataset_split"):
        metrics.add_metadata("dataset_split", args.dataset_split)

    metrics.add_metadata("sample_count", args.sample_count)
    metrics.add_metadata("plan_sample_count", args.plan_sample_count)
    metrics.add_metadata("max_batch_size", args.max_batch_size)
    metrics.add_metadata("db_concurrency", args.db_concurrency)
    if hasattr(args, "enable_result_reuse"):
        metrics.add_metadata("enable_result_reuse", args.enable_result_reuse)
    if hasattr(args, "enforce_epoch_barrier"):
        metrics.add_metadata("enforce_epoch_barrier", args.enforce_epoch_barrier)
    metrics.add_metadata("scheduler_mode", args.scheduler_mode)
    metrics.add_metadata("processor_mode", getattr(args, "processor_mode", "mp"))
    if getattr(args, "processor_mode", "mp") == "opwise":
        metrics.add_metadata("opwise_parallel_mode", getattr(args, "opwise_parallel_mode", "tensor"))
        metrics.add_metadata("opwise_parallel_size", getattr(args, "opwise_parallel_size", 0))

    gpu_workers = [worker for worker in plan.workers.values() if worker.kind == "gpu"]
    metrics.add_metadata("gpus_detected", len(gpu_workers))
    cpu_workers = [worker for worker in plan.workers.values() if worker.kind != "gpu"]
    metrics.add_metadata("cpu_workers", len(cpu_workers))

    metadata = getattr(plan, "metadata", {}) or {}
    milp_meta = metadata.get("milp") if isinstance(metadata, dict) else None
    if isinstance(milp_meta, dict):
        makespan = milp_meta.get("objective_makespan")
        if isinstance(makespan, (int, float)):
            metrics.add_metadata("milp_makespan", makespan)
        status = milp_meta.get("status")
        if isinstance(status, str):
            metrics.add_metadata("milp_status", status)
        quantum = milp_meta.get("epoch_quantum_s")
        if isinstance(quantum, (int, float)):
            metrics.add_metadata("milp_epoch_quantum_s", quantum)
        buckets = milp_meta.get("epoch_buckets")
        if isinstance(buckets, int):
            metrics.add_metadata("milp_epoch_buckets", buckets)
        # Optional: include full per-node times in metrics metadata (can be large).
        # Enable with HALO_METRICS_INCLUDE_MILP_NODE_TIMES=1.
        import os
        if os.getenv("HALO_METRICS_INCLUDE_MILP_NODE_TIMES", "0").lower() not in ("", "0", "false", "no"):
            node_times = milp_meta.get("node_times_s")
            if isinstance(node_times, dict):
                metrics.add_metadata("milp_node_times_s", node_times)

    if plan.tasks:
        max_epoch = max(task.epoch for task in plan.tasks)
        metrics.add_metadata("max_epoch", max_epoch)
