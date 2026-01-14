from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence
import logging

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from halo_dev import metrics
from halo_dev.models import ExecutionPlan
from halo_dev.db import PostgresPlanExplainer, make_peer_postgres_executor
from halo_dev.optimizer import GraphOptimizer
from halo_dev.parser import GraphTemplateParser
from halo_dev.processors import MultiProcessGraphProcessor, OpwiseGraphProcessor, SerialGraphProcessor
from halo_dev.query_planner import QueryPlanEvaluator
from run.query_sampler import (
    DEFAULT_SAMPLE_POOL_SIZE,
    default_extract_query,
    sample_queries,
)
from run.runner_utils import (
    print_phase_one_summary,
    print_phase_two_summary,
    print_planning_timings,
    print_run_outputs,
    record_run_metadata,
)


DEFAULT_DATASET = "HuggingFaceFW/finewiki"
HALO_PG_DBNAME = "finewiki"
HALO_PG_USER = os.getenv("USER") or "postgres"
HALO_PG_HOST = "/var/run/postgresql"
HALO_PG_PORT = 4032


def main(argv: List[str] | None = None) -> None:
    os.environ.setdefault("HALO_METRICS_NODE_STATS", "0")
    os.environ.setdefault("HALO_MONITOR_ENABLE", "0")
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    _log("Parsing workflow template...")

    graph = GraphTemplateParser(args.template).parse()

    plan_pool_size = max(1, args.plan_sample_count)
    db_pool_size = max(1, args.db_concurrency)
    db_connect_kwargs = {
        "dbname": HALO_PG_DBNAME,
        "user": HALO_PG_USER,
        "host": HALO_PG_HOST,
        "port": HALO_PG_PORT,
    }

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
        enable_post_refine=True,
    )

    plan_samples, run_queries = _prepare_queries(args)
    _log("Building execution plan with sampled contexts...")
    plan_contexts = plan_samples or [{}]
    plan_build_start = time.perf_counter()
    plan = optimizer.build_plan(graph, sample_contexts=plan_contexts, input_query_count=args.sample_count)
    plan_build_duration = time.perf_counter() - plan_build_start

    print(f"Graph: {graph.name} — {graph.description.strip()}")
    print(f"Plan mode: {args.plan_mode}")
    print_planning_timings(plan)
    print_phase_one_summary(plan)
    print_phase_two_summary(plan, graph)
    plan_evaluator.cleanup()

    if not args.run:
        print("\nPass --run to execute the plan over sampled queries.")
        return

    queries = run_queries
    if not queries:
        raise RuntimeError("No queries available for execution after planning.")

    processor_mode = getattr(args, "processor_mode", "mp")
    if processor_mode == "serial":
        processor = SerialGraphProcessor(
            db_connect_kwargs=db_connect_kwargs,
            db_pool_size=db_pool_size,
            db_concurrency=args.db_concurrency,
            enable_result_cache=args.enable_result_reuse,
            enable_db_explain_analyze=args.db_explain_analyze,
            db_explain_mode=args.db_explain_mode,
            db_explain_sample_rate=args.db_explain_sample_rate,
        )
    elif processor_mode == "opwise":
        parallel_mode = getattr(args, "opwise_parallel_mode", "tensor")
        tp_size_arg = getattr(args, "opwise_parallel_size", 0)
        if parallel_mode == "dp":
            tp_enabled = False
            tp_size = None
            dp_size = None if tp_size_arg is None or tp_size_arg <= 0 else tp_size_arg
        else:
            tp_enabled = tp_size_arg not in (0, None)
            tp_size = None if tp_size_arg == -1 else tp_size_arg if tp_size_arg and tp_size_arg > 0 else None
            dp_size = None
        processor = OpwiseGraphProcessor(
            max_batch_size=args.max_batch_size,
            db_connect_kwargs=db_connect_kwargs,
            db_pool_size=db_pool_size,
            db_concurrency=args.db_concurrency,
            enable_result_cache=args.enable_result_reuse,
            enable_db_explain_analyze=args.db_explain_analyze,
            db_explain_mode=args.db_explain_mode,
            db_explain_sample_rate=args.db_explain_sample_rate,
            enable_tensor_parallel=tp_enabled,
            tensor_parallel_size=tp_size,
            parallel_mode=parallel_mode,
            parallel_size=dp_size,
        )
    else:
        executor_kwargs: Dict[str, Any] = {"db_concurrency": args.db_concurrency}
        executor_kwargs["enable_result_cache"] = args.enable_result_reuse
        if args.db_explain_analyze is not None:
            executor_kwargs["enable_db_explain_analyze"] = args.db_explain_analyze
        if args.db_explain_mode is not None:
            executor_kwargs["db_explain_mode"] = args.db_explain_mode
        if args.db_explain_sample_rate is not None:
            executor_kwargs["db_explain_sample_rate"] = args.db_explain_sample_rate
        processor = MultiProcessGraphProcessor(
            db_connect_kwargs=db_connect_kwargs,
            db_pool_size=db_pool_size,
            max_batch_size=args.max_batch_size,
            executor_kwargs=executor_kwargs,
            enforce_epoch_barrier=args.enforce_epoch_barrier,
        )
    _log("Starting workflow execution...")
    initial_contexts = [{"user_query": query} for query in queries]
    _log(f"Dispatching batch of {len(initial_contexts)} querie(s).")
    metrics.start_run("finewiki", len(queries))
    record_run_metadata(args, plan)
    metrics.start_batch()
    final_contexts = processor.run_batch(plan, graph, initial_contexts)
    batch_duration = metrics.end_batch()
    metrics.end_run()
    if batch_duration is not None and queries:
        metrics.record_query_latency(batch_duration, count=len(queries))

    for idx, (user_query, initial_context, final_context) in enumerate(
        zip(queries, initial_contexts, final_contexts), start=1
    ):
        if args.print_run_outputs or args.show_context:
            print_run_outputs(final_context, initial_context, show_context=args.show_context)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    cli = argparse.ArgumentParser(description="Run FineWiki workflow over sampled dataset queries.")
    cli.add_argument(
        "-t",
        "--template",
        type=Path,
        default=Path("templates/finewiki_two_stage.yaml"),
        help="Path to the YAML workflow template.",
    )
    cli.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Override detected GPU count for the optimizer.",
    )
    cli.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="Number of CPU workers for DB nodes.",
    )
    cli.add_argument(
        "--run",
        action="store_true",
        help="Execute the workflow after planning.",
    )
    cli.add_argument(
        "--show-context",
        action="store_true",
        help="Print the final context dictionary after each execution.",
    )
    cli.add_argument(
        "--print-run-outputs",
        action="store_true",
        help="Print produced context keys after each query execution.",
    )
    cli.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset on HuggingFace Hub to sample queries from.",
    )
    cli.add_argument(
        "--dataset-subset",
        type=str,
        default="en",
        help="Dataset subset/language to use when sampling.",
    )
    cli.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use when sampling.",
    )
    cli.add_argument(
        "-n",
        "--sample-count",
        type=int,
        default=1024,
        help="Number of user queries to sample and execute per run.",
    )
    cli.add_argument(
        "--sample-seed",
        type=int,
        default=44,
        help="Optional random seed for deterministic sampling.",
    )
    cli.add_argument(
        "--plan-sample-count",
        type=int,
        default=4,
        help="Number of sampled queries to use during plan optimization.",
    )
    cli.add_argument(
        "--max-batch-size",
        type=int,
        default=256,
        help="Max contexts per dispatch to a worker (GPU/CPU).",
    )
    cli.add_argument(
        "--enforce-epoch-barrier",
        action="store_true",
        help="Strictly enforce epoch barriers during execution.",
    )
    cli.add_argument(
        "--db-concurrency",
        type=int,
        default=32,
        help="Threaded DB query concurrency per vLLM call.",
    )
    cli.add_argument(
        "--enable-result-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DB result cache reuse across identical queries within a run.",
    )
    cli.add_argument(
        "--db-explain-analyze",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, run DB queries with EXPLAIN ANALYZE (BUFFERS) and collect per-query buffer/cost stats.",
    )
    cli.add_argument(
        "--db-explain-mode",
        choices=["wrap", "replace"],
        default=None,
        help="DB explain mode: wrap (execute + explain) or replace (explain only, no rows).",
    )
    cli.add_argument(
        "--db-explain-sample-rate",
        type=float,
        default=None,
        help="Optional sampling rate in [0,1] for DB explain collection (default: 1.0).",
    )
    cli.add_argument(
        "--plan-mode",
        choices=["profiled", "baseline", "default"],
        default="default",
        help=(
            "Query plan strategy: profiled tests multiple hints, "
            "default runs only the Postgres default plan (EXPLAIN), "
            "baseline skips profiling."
        ),
    )
    cli.add_argument(
        "-s",
        "--scheduler-mode",
        choices=["auto", "dp", "rr_topo", "random", "random_topo", "model_first", "greedy", "minswitch", "milp", "opwise"],
        default="auto",
        help=(
            "Scheduler: auto (sample_count<512: greedy, else dp), dp (stateful DP), rr_topo (structure-only), "
            "random/random_topo "
            "(topo order + random worker), model_first "
            "(model-aware), greedy (cost-based), minswitch (reuse-friendly), "
            "opwise (LLM data-parallel baseline), or milp oracle (small graphs only)."
        ),
    )
    cli.add_argument(
        "--processor-mode",
        choices=["mp", "serial", "opwise"],
        default="mp",
        help="Processor: mp (multi-process, batched), serial (query-by-query), or opwise (topo-ordered, optional tensor/data parallel).",
    )
    cli.add_argument(
        "--opwise-parallel-size",
        type=int,
        default=0,
        help=(
            "Parallel size for opwise processor. Tensor mode uses this as TP size "
            "(<=0 disables, -1 uses all visible GPUs). DP mode uses this as worker count "
            "(<=0 uses all visible GPUs)."
        ),
    )
    cli.add_argument(
        "--opwise-parallel-mode",
        choices=["tensor", "dp"],
        default="tensor",
        help="Opwise parallel mode: tensor (default) or dp (data-parallel across GPUs).",
    )
    return cli.parse_args(argv)


def _prepare_queries(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[str]]:
    plan_needed = max(args.plan_sample_count, 1)
    run_needed = args.sample_count if args.run else 0
    total_needed = max(plan_needed, run_needed, 1)

    sample_result = sample_queries(
        dataset=args.dataset_name,
        subset=args.dataset_subset,
        split=args.dataset_split,
        pool_size=DEFAULT_SAMPLE_POOL_SIZE,
        sample_count=total_needed,
        seed=args.sample_seed,
        extract_query=default_extract_query,
    )
    queries = sample_result.queries
    if len(queries) < total_needed:
        raise RuntimeError(
            f"Needed {total_needed} queries for planning/execution, only got {len(queries)}."
        )
    plan_contexts = [{"user_query": query} for query in queries[:plan_needed]]
    run_queries = queries[: run_needed or len(queries)]
    _log(
        "Sampled "
        f"{len(queries)} query(ies) after scanning {sample_result.rows_scanned} rows."
    )
    return plan_contexts, run_queries


def _log(message: str) -> None:
    print(f"[FineWikiRunner] {message}")


if __name__ == "__main__":
    main()
