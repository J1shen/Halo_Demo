from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from halo_dev import metrics
from halo_dev.db import PostgresDatabaseExecutor
from halo_dev.executor import DBNodeExecutor
from halo_dev.models import DBQuery, ExecutionPlan, GraphSpec
from halo_dev.optimizer import GraphOptimizer
from halo_dev.parser import GraphTemplateParser
from halo_dev.query_planner import QueryPlanEvaluator
from run.runner_utils import (
    print_phase_one_summary,
    print_phase_two_summary,
    print_planning_timings,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbenchmark DB nodes in plan order with EXPLAIN ANALYZE (BUFFERS)."
    )
    parser.add_argument(
        "-t",
        "--template",
        type=Path,
        required=True,
        help="Path to the YAML workflow template.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Input file (text: one user_query per line; jsonl: per-line JSON context).",
    )
    parser.add_argument(
        "--input-format",
        choices=["text", "jsonl"],
        default="text",
        help="Input file format.",
    )
    parser.add_argument(
        "--context-key",
        type=str,
        default="user_query",
        help="Context key to use for text inputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of contexts to run (default: all).",
    )
    parser.add_argument(
        "--plan-sample-count",
        type=int,
        default=4,
        help="Number of contexts used for plan optimization.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Input query count for planner cost scaling (default: len(run contexts)).",
    )
    parser.add_argument(
        "--plan-mode",
        choices=["profiled", "baseline", "default"],
        default="default",
        help=(
            "Query plan strategy: profiled tests multiple hints, "
            "default runs only the Postgres default plan (EXPLAIN), "
            "baseline skips profiling."
        ),
    )
    parser.add_argument(
        "-s",
        "--scheduler-mode",
        choices=[
            "auto",
            "dp",
            "rr_topo",
            "random",
            "random_topo",
            "model_first",
            "greedy",
            "minswitch",
            "milp",
            "opwise",
        ],
        default="auto",
        help="Scheduler mode for the optimizer.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Override detected GPU count for the optimizer.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="Number of CPU workers for DB nodes.",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default=None,
        help="Postgres database name.",
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default=None,
        help="Postgres user.",
    )
    parser.add_argument(
        "--db-host",
        type=str,
        default=None,
        help="Postgres host or socket directory.",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=None,
        help="Postgres port.",
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default=None,
        help="Postgres password.",
    )
    parser.add_argument(
        "--db-pool-size",
        type=int,
        default=1,
        help="Connection pool size for DB executor.",
    )
    parser.add_argument(
        "--db-concurrency",
        type=int,
        default=1,
        help="Threaded DB query concurrency for DB executor.",
    )
    parser.add_argument(
        "--db-explain-mode",
        choices=["wrap", "replace"],
        default="replace",
        help="DB explain mode: wrap (execute + explain) or replace (explain only).",
    )
    parser.add_argument(
        "--db-explain-sample-rate",
        type=float,
        default=1.0,
        help="Sampling rate in [0,1] for DB explain collection (default: 1.0).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of benchmark runs to execute.",
    )
    parser.add_argument(
        "--cold-mode",
        choices=["none", "restart", "evict"],
        default="evict",
        help="Cold-ish mode: restart Postgres, evict via table scan, or skip.",
    )
    parser.add_argument(
        "--restart-cmd",
        type=str,
        default=None,
        help="Command to restart Postgres (used when --cold-mode=restart).",
    )
    parser.add_argument(
        "--restart-wait-s",
        type=float,
        default=2.0,
        help="Seconds to wait after restarting Postgres.",
    )
    parser.add_argument(
        "--evict-sql",
        type=str,
        default=None,
        help="SQL to run for cache eviction (used when --cold-mode=evict).",
    )
    parser.add_argument(
        "--evict-table",
        type=str,
        default=None,
        help="Table name to scan for eviction (used when --cold-mode=evict).",
    )
    parser.add_argument(
        "--evict-repeat",
        type=int,
        default=1,
        help="Repeat eviction query this many times.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path for the summary metrics.",
    )
    return parser.parse_args(argv)


def _load_contexts(
    input_file: Path | None,
    *,
    input_format: str,
    context_key: str,
    limit: int | None,
) -> List[Dict[str, Any]]:
    if input_file is None:
        return []
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    contexts: List[Dict[str, Any]] = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if input_format == "jsonl":
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("JSONL inputs must be objects (one context per line).")
                contexts.append(payload)
            else:
                contexts.append({context_key: raw})
            if limit is not None and len(contexts) >= limit:
                break
    return contexts


def _build_connect_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    host = args.db_host or os.getenv("HALO_PG_HOST")
    if host:
        kwargs["host"] = host
    port = args.db_port if args.db_port is not None else os.getenv("HALO_PG_PORT")
    if port:
        try:
            kwargs["port"] = int(port)
        except (TypeError, ValueError):
            pass
    dbname = args.db_name or os.getenv("HALO_PG_DBNAME")
    if dbname:
        kwargs["dbname"] = dbname
    user = args.db_user or os.getenv("HALO_PG_USER") or os.getenv("USER")
    if user:
        kwargs["user"] = user
    password = args.db_password or os.getenv("HALO_PG_PASSWORD")
    if password:
        kwargs["password"] = password
    return kwargs


def _build_plan(
    graph: GraphSpec,
    args: argparse.Namespace,
    plan_contexts: Sequence[Mapping[str, Any]],
    input_query_count: int,
    db_connect_kwargs: Dict[str, Any],
) -> ExecutionPlan:
    plan_pool_size = max(1, int(args.plan_sample_count or 1))
    plan_evaluator = QueryPlanEvaluator(
        explainer_connect_kwargs=db_connect_kwargs,
        explainer_pool_size=plan_pool_size,
    )
    optimizer = GraphOptimizer(
        num_gpus=args.gpus,
        num_cpu_workers=args.cpu_workers,
        plan_mode=args.plan_mode,
        scheduler_mode=args.scheduler_mode,
        plan_evaluator=plan_evaluator,
        enable_post_refine=False,
    )
    plan = optimizer.build_plan(
        graph,
        sample_contexts=plan_contexts or [{}],
        input_query_count=input_query_count,
    )
    plan_evaluator.cleanup()
    return plan


def _db_task_order(plan: ExecutionPlan, graph: GraphSpec) -> List[str]:
    order: List[str] = []
    for task in plan.tasks:
        node = graph.nodes.get(task.node_id)
        if node and node.engine == "db":
            order.append(task.node_id)
    return order


def _run_cold_step(
    *,
    cold_mode: str,
    restart_cmd: str | None,
    restart_wait_s: float,
    db_executor: PostgresDatabaseExecutor | None,
    evict_sql: str | None,
    evict_table: str | None,
    evict_repeat: int,
) -> None:
    if cold_mode == "none":
        return
    if cold_mode == "restart":
        if not restart_cmd:
            raise ValueError("--restart-cmd is required when --cold-mode=restart.")
        subprocess.run(restart_cmd, shell=True, check=True)
        if restart_wait_s > 0:
            time.sleep(restart_wait_s)
        return
    if db_executor is None:
        raise ValueError("Eviction requires a Postgres executor.")
    sql = evict_sql
    if not sql and evict_table:
        sql = f"SELECT count(*) FROM {evict_table}"
    if not sql:
        logging.warning("Cold eviction requested but no --evict-sql or --evict-table provided; skipping.")
        return
    evict_repeat = max(1, int(evict_repeat))
    query = DBQuery(name="__evict__", sql=sql, parameters={})
    for _ in range(evict_repeat):
        db_executor.run(query, {}, node_id="__evict__")


def _summarize_explain() -> Dict[str, Any]:
    tracker = metrics._TRACKER  # intentionally using internal aggregator for summary
    explain_by_query = tracker.db_explain_by_query
    total_calls = sum(agg.calls for agg in explain_by_query.values())
    runtime_sum = sum(agg.runtime_ms_sum for agg in explain_by_query.values())
    cost_sum = sum(agg.planned_total_cost_sum for agg in explain_by_query.values())
    cost_count = sum(agg.planned_total_cost_count for agg in explain_by_query.values())
    total_hit = sum(agg.shared_hit_blocks_sum for agg in explain_by_query.values())
    total_read = sum(agg.shared_read_blocks_sum for agg in explain_by_query.values())
    denom = total_hit + total_read
    hit_rate = (total_hit / denom) if denom > 0 else None
    read_rate = (total_read / denom) if denom > 0 else None
    avg_latency_ms = (runtime_sum / total_calls) if total_calls > 0 else None
    avg_planned_cost = (cost_sum / cost_count) if cost_count > 0 else None
    return {
        "explain_calls": total_calls,
        "avg_latency_ms": avg_latency_ms,
        "avg_planned_cost": avg_planned_cost,
        "shared_hit_blocks": total_hit,
        "shared_read_blocks": total_read,
        "hit_rate": hit_rate,
        "read_rate": read_rate,
    }


def _run_db_only(
    plan: ExecutionPlan,
    graph: GraphSpec,
    contexts: Sequence[Mapping[str, Any]],
    db_executor: PostgresDatabaseExecutor,
    *,
    db_concurrency: int,
    db_explain_mode: str,
    db_explain_sample_rate: float,
) -> List[Dict[str, Any]]:
    db_node_executor = DBNodeExecutor(
        db_executor=db_executor,
        db_concurrency=max(1, int(db_concurrency)),
        enable_result_cache=False,
        enable_db_explain_analyze=True,
        db_explain_mode=db_explain_mode,
        db_explain_sample_rate=db_explain_sample_rate,
    )
    order = _db_task_order(plan, graph)
    if not order:
        raise RuntimeError("No DB nodes found in the execution plan.")
    final_contexts: List[Dict[str, Any]] = []
    for ctx in contexts:
        context = dict(ctx)
        for node_id in order:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            outputs = db_node_executor.execute(node, context)
            stats = db_node_executor.consume_stats()
            db_time = float(stats.get("db_time") or 0.0)
            db_calls = int(stats.get("db_calls") or 0)
            if db_calls > 0:
                metrics.record_query_execution_time(db_time, count=db_calls)
            if outputs:
                context.update(outputs)
        final_contexts.append(context)
    return final_contexts


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if not args.template.exists():
        raise FileNotFoundError(f"Template not found: {args.template}")

    graph = GraphTemplateParser(args.template).parse()
    contexts = _load_contexts(
        args.input_file,
        input_format=args.input_format,
        context_key=args.context_key,
        limit=args.limit,
    )
    if not contexts:
        contexts = [{}]

    plan_contexts = contexts[: max(1, int(args.plan_sample_count or 1))]
    input_query_count = args.sample_count if args.sample_count is not None else len(contexts)

    db_connect_kwargs = _build_connect_kwargs(args)
    plan = _build_plan(
        graph,
        args,
        plan_contexts,
        max(1, int(input_query_count)),
        db_connect_kwargs,
    )

    print(f"Graph: {graph.name} — {graph.description.strip()}")
    print(f"Plan mode: {args.plan_mode}")
    print_planning_timings(plan)
    print_phase_one_summary(plan)
    print_phase_two_summary(plan, graph)

    all_summaries: List[Dict[str, Any]] = []
    for run_idx in range(max(1, int(args.runs))):
        if args.cold_mode == "restart":
            _run_cold_step(
                cold_mode=args.cold_mode,
                restart_cmd=args.restart_cmd,
                restart_wait_s=float(args.restart_wait_s),
                db_executor=None,
                evict_sql=args.evict_sql,
                evict_table=args.evict_table,
                evict_repeat=args.evict_repeat,
            )
            db_executor = PostgresDatabaseExecutor(
                connect_kwargs=db_connect_kwargs,
                pool_size=max(1, int(args.db_pool_size or 1)),
            )
        else:
            db_executor = PostgresDatabaseExecutor(
                connect_kwargs=db_connect_kwargs,
                pool_size=max(1, int(args.db_pool_size or 1)),
            )
            _run_cold_step(
                cold_mode=args.cold_mode,
                restart_cmd=args.restart_cmd,
                restart_wait_s=float(args.restart_wait_s),
                db_executor=db_executor,
                evict_sql=args.evict_sql,
                evict_table=args.evict_table,
                evict_repeat=args.evict_repeat,
            )
        metrics.start_run(f"db_microbench_{run_idx + 1}", len(contexts))
        metrics.start_batch()
        _run_db_only(
            plan,
            graph,
            contexts,
            db_executor,
            db_concurrency=args.db_concurrency,
            db_explain_mode=args.db_explain_mode,
            db_explain_sample_rate=args.db_explain_sample_rate,
        )
        batch_duration = metrics.end_batch()
        metrics.end_run()
        if batch_duration is not None and contexts:
            metrics.record_query_latency(batch_duration, count=len(contexts))
        summary = _summarize_explain()
        summary["run_index"] = run_idx + 1
        all_summaries.append(summary)
        avg_latency = summary.get("avg_latency_ms")
        avg_cost = summary.get("avg_planned_cost")
        hit_rate = summary.get("hit_rate")
        read_rate = summary.get("read_rate")
        print(
            "DB summary "
            f"(run {run_idx + 1}/{args.runs}): "
            f"avg_latency_ms={avg_latency if avg_latency is not None else 'n/a'} "
            f"avg_planned_cost={avg_cost if avg_cost is not None else 'n/a'} "
            f"hit_rate={hit_rate if hit_rate is not None else 'n/a'} "
            f"read_rate={read_rate if read_rate is not None else 'n/a'} "
            f"shared_hit={summary.get('shared_hit_blocks')} "
            f"shared_read={summary.get('shared_read_blocks')}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "runs": all_summaries,
            "template": str(args.template),
            "input_file": str(args.input_file) if args.input_file else None,
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
