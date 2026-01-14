from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
import time
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from ..models import GraphSpec, Node, QueryPlanChoice, Worker
from .topo_utils import topological_order

try:
    from ._dp_core_rs import solve as _solve_rs  # type: ignore
    _HAS_DP_RUST = True
except Exception:
    _solve_rs = None
    _HAS_DP_RUST = False


# 单个 DB query 的签名，用于 cache multiplier
@dataclass(frozen=True, slots=True)
class QuerySignature:
    node_id: str
    query_name: str
    plan_id: str
    footprints: Tuple[Tuple[str, int], ...] = tuple()


# worker 的局部状态：
#   - GPU worker: 维护 last_model / last_node
#   - CPU worker: 仅用于状态去重
@dataclass(frozen=True, slots=True)
class WorkerState:
    worker_idx: int
    last_model_id: int
    last_node_id: int


class DPSolver:
    """DP 调度器：只优化 GPU 侧批次与 worker 映射，CPU 节点按 GPU 规划结果自动填充。

    每个 epoch：
      1. GPU 侧：从 ready 的 LLM 节点中选一批（可行子图），枚举 GPU worker 映射，
         计算 LLM cost（exec_cost * llm_cache_bonus + model_init）。
      2. CPU 侧：根据 GPU 规划结果，自动补齐本 epoch 内可执行且必要的 CPU 节点（无 cost、无容量上限）。
      3. cpu load cost = 所选 GPU 节点的 DB queries 依次累加的 cost
         （EXPLAIN raw_cost/cost 经 _raw_cost_scale * cache_multiplier），
         以及 CPU 节点中的 HTTP sleep 代价。
      4. Total cost per epoch = global_epoch_penalty + gpu_cost
         + cpu_load_weight(epoch) * cpu_load_cost (cpu_load_cost 已包含顺序影响).
      5. 选中 GPU 节点的子图拓扑深度越深，epoch cost 额外加深度惩罚。
    """

    def __init__(
        self,
        *,
        graph: GraphSpec,
        dependencies: Mapping[str, Sequence[str]],
        workers: Dict[str, Worker],
        worker_ids: Tuple[str, ...],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]],
        window_size: int,
        exec_cost_fn,
        cache_multiplier_fn: Callable[[Tuple[QuerySignature, ...], QueryPlanChoice], float],
        model_init_cost_fn: Callable[[Node, str | None], float] | None = None,
        llm_cache_bonus_fn: Callable[[Node, str | None, Sequence[str]], float] | None = None,
        node_ids: Sequence[str] | None = None,
        epoch_penalty_fn: Callable[[int], float] | None = None,
        cpu_load_cost_weight: float | None = None,
        cpu_load_early_weight: float | None = None,
        cpu_cost_mode: str | None = None,
        switch_penalty_weight: float | None = None,
        gpu_cost_max_weight: float | None = None,
        gpu_cost_sum_weight: float | None = None,
        disable_epoch_batch_cost: bool | None = None,
        disable_cpu_load_cost: bool | None = None,
        node_worker_options: Mapping[str, Sequence[str]] | None = None,
        gpu_worker_ids: Tuple[str, ...] | None = None,
        cpu_worker_ids: Tuple[str, ...] | None = None,
        http_latency_s: Mapping[str, float] | None = None,

        # 优化选项
        enable_batch_shape_pruning: bool = True,
        gpu_batch_slack: int = 1,
        enable_lower_bound_pruning: bool = True,
        enable_worker_symmetry: bool = True,
        lower_bound_cost_factor: float = 0.2,
        debug_log: bool = True,
        debug_every: int = 100000,
        prefer_rust: bool | None = None,
    ) -> None:
        self.graph = graph
        self.dependencies = dependencies
        self.workers = workers
        self.worker_ids = worker_ids
        self.exec_cost_fn = exec_cost_fn
        self.plan_choices = plan_choices or {}
        self.window_size = window_size
        self.cache_multiplier_fn = cache_multiplier_fn
        self.model_init_cost_fn = model_init_cost_fn or (lambda node, last_model: 0.0)
        self.llm_cache_bonus_fn = llm_cache_bonus_fn or (
            lambda node, last_node, parents: 1.0
        )
        self._http_latency_s = dict(http_latency_s or {})
        # 每-epoch 惩罚可随深度增长
        self._epoch_penalty_fn = epoch_penalty_fn or (lambda epoch: 1.0)

        env_weight = os.getenv("HALO_DP_CPU_LOAD_COST_WEIGHT")
        if env_weight is not None and env_weight.strip() != "":
            try:
                cpu_load_cost_weight = float(env_weight)
            except ValueError:
                cpu_load_cost_weight = 1.0
        elif cpu_load_cost_weight is None:
            cpu_load_cost_weight = 1.0
        self.cpu_load_cost_weight = max(0.0, float(cpu_load_cost_weight))
        env_weight = os.getenv("HALO_DP_CPU_EARLY_WEIGHT")
        if env_weight is not None and env_weight.strip() != "":
            try:
                cpu_load_early_weight = float(env_weight)
            except ValueError:
                cpu_load_early_weight = 1.0
        elif cpu_load_early_weight is None:
            cpu_load_early_weight = 2.0
        self.cpu_load_early_weight = max(0.0, float(cpu_load_early_weight))
        self.disable_epoch_batch_cost = bool(disable_epoch_batch_cost)
        self.disable_cpu_load_cost = bool(disable_cpu_load_cost)
        self.cpu_cost_mode = self._resolve_cpu_cost_mode(cpu_cost_mode)
        max_weight = gpu_cost_max_weight
        sum_weight = gpu_cost_sum_weight
        if max_weight is None and sum_weight is None:
            env_max = (os.getenv("HALO_DP_GPU_COST_MAX_WEIGHT") or "").strip()
            env_sum = (os.getenv("HALO_DP_GPU_COST_SUM_WEIGHT") or "").strip()
            if env_max:
                try:
                    max_weight = float(env_max)
                except ValueError:
                    max_weight = None
            if env_sum:
                try:
                    sum_weight = float(env_sum)
                except ValueError:
                    sum_weight = None
        if max_weight is None and sum_weight is None:
            max_weight = 0.9
            sum_weight = 0.1
        if max_weight is None:
            max_weight = 0.0
        if sum_weight is None:
            sum_weight = 0.0
        total = float(max_weight) + float(sum_weight)
        if total <= 0:
            max_weight = 0.9
            sum_weight = 0.1
        else:
            max_weight = float(max_weight) / total
            sum_weight = float(sum_weight) / total
        self.gpu_cost_max_weight = max(0.0, float(max_weight))
        self.gpu_cost_sum_weight = max(0.0, float(sum_weight))
        if switch_penalty_weight is None:
            env_weight = os.getenv("HALO_DP_SWITCH_PENALTY_WEIGHT")
            if env_weight is not None and env_weight.strip() != "":
                try:
                    switch_penalty_weight = float(env_weight)
                except ValueError:
                    switch_penalty_weight = 0.1
            else:
                switch_penalty_weight = 0.1
        self.switch_penalty_weight = max(0.0, float(switch_penalty_weight))
        self.enable_batch_shape_pruning = enable_batch_shape_pruning
        self.gpu_batch_slack = max(0, gpu_batch_slack)
        self.enable_lower_bound_pruning = enable_lower_bound_pruning
        self.enable_worker_symmetry = enable_worker_symmetry
        self.lower_bound_cost_factor = max(0.0, float(lower_bound_cost_factor))
        self.debug_log = bool(debug_log)
        self.debug_every = max(1, int(debug_every))
        self._progress_enabled = os.getenv("HALO_DP_PROGRESS", "1").lower() not in ("", "0", "false", "no")
        self._progress_every = max(1, int(os.getenv("HALO_DP_PROGRESS_EVERY", str(self.debug_every))))
        self._progress_start = time.perf_counter()
        self._progress_last = self._progress_start
        if prefer_rust is None:
            env_flag = os.getenv("HALO_DP_USE_RUST", "0").lower()
            self._prefer_rust = env_flag not in ("", "0", "false", "no")
        else:
            self._prefer_rust = bool(prefer_rust)
        self._rust_available = _HAS_DP_RUST

        node_worker_options = node_worker_options or {}
        if node_ids is not None:
            self.node_ids = tuple(node_ids)
        else:
            self.node_ids = tuple(sorted(graph.nodes))
        self.node_worker_options = {
            node_id: tuple(node_worker_options.get(node_id, worker_ids))
            for node_id in self.node_ids
        }
        self._gpu_node_ids = tuple(nid for nid in self.node_ids if self._is_gpu_node(nid))
        self._db_node_ids = tuple(
            nid for nid in self.node_ids if not self._is_gpu_node(nid)
        )

        if gpu_worker_ids is not None and cpu_worker_ids is not None:
            self.gpu_worker_ids = gpu_worker_ids
            self.cpu_worker_ids = cpu_worker_ids
        else:
            self.gpu_worker_ids = tuple(
                wid for wid in worker_ids if workers[wid].kind == "gpu"
            )
            self.cpu_worker_ids = tuple(
                wid for wid in worker_ids if workers[wid].kind != "gpu"
            )
        if not self.cpu_worker_ids:
            # 没有显式 CPU worker 时兜底使用全体 worker。
            self.cpu_worker_ids = worker_ids

        # 记录 CPU 节点的拓扑顺序，便于自动批量填充。
        try:
            topo = topological_order(self.dependencies, self.node_ids)
        except Exception:
            topo = list(self.node_ids)
        self._db_topo_order = tuple(nid for nid in topo if not self._is_gpu_node(nid))

        self.node_index = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        self.all_mask = (1 << len(self.node_ids)) - 1
        self.worker_index = {wid: idx for idx, wid in enumerate(worker_ids)}
        self._worker_id_by_idx = tuple(worker_ids)

        # EXPLAIN raw_cost 缩放因子，沿用原逻辑
        optimizer = getattr(exec_cost_fn, "__self__", None)
        self._raw_cost_scale = getattr(optimizer, "_raw_cost_scale", 3.65e-6)

        # fallback plan（某些 query 没有 plan_choices 时使用）
        self._fallback_choice = QueryPlanChoice(
            plan_id="default",
            description="fallback",
            cost=1.0,
            raw_cost=None,
            explain_json=None,
            samples=tuple(),
            footprints={},
        )
        env_depth_weight = os.getenv("HALO_DP_GPU_DEPTH_WEIGHT")
        if env_depth_weight is not None and env_depth_weight.strip() != "":
            try:
                depth_weight = float(env_depth_weight)
            except ValueError:
                depth_weight = 1.0
        else:
            depth_weight = 1.0
        self._gpu_depth_cost_weight = max(0.0, float(depth_weight))
        
        # id 压缩：将 node/model/query/plan/footprint key 统一映射为 int，减少哈希开销
        self._none_id = -1
        self._node_id_to_int = dict(self.node_index)
        self._node_int_to_id = tuple(self.node_ids)
        self._parents_mask = self._build_parents_mask(self.dependencies)
        self._gpu_parents_mask = self._build_parents_mask(self.dependencies, gpu_only=True)

        all_models = sorted(
            {n.model for n in graph.nodes.values() if getattr(n, "model", None)}
        )
        self._model_to_int = {name: i for i, name in enumerate(all_models)}
        self._model_int_to_name = tuple(all_models)

        all_query_names: Set[str] = set()
        for node in graph.nodes.values():
            for q in getattr(node, "db_queries", []) or []:
                all_query_names.add(q.name)
        self._query_to_int = {name: i for i, name in enumerate(sorted(all_query_names))}
        self._query_int_to_name = tuple(sorted(all_query_names))

        all_plan_ids: Set[str] = {"default"}
        for choice_list in self.plan_choices.values():
            for choice in choice_list:
                all_plan_ids.add(choice.plan_id)
        self._plan_to_int = {pid: i for i, pid in enumerate(sorted(all_plan_ids))}
        self._plan_int_to_id = tuple(sorted(all_plan_ids))

        all_fp_keys: Set[str] = set()
        for choice_list in self.plan_choices.values():
            for choice in choice_list:
                all_fp_keys.update((choice.footprints or {}).keys())
        self._fp_to_int = {k: i for i, k in enumerate(sorted(all_fp_keys))}
        self._fp_int_to_key = tuple(sorted(all_fp_keys))

        self._signature_to_id: Dict[
            tuple[int, int, int, Tuple[Tuple[int, int], ...]], int
        ] = {}
        self._id_to_signature: List[QuerySignature] = []
        self._node_min_cost: Dict[str, float] = {}

        # cache: (node_id_int, enter_window) -> 所有 query plan 组合
        self._query_plan_cache: Dict[
            tuple[int, Tuple[int, ...]],
            Tuple[
                Tuple[
                    float,
                    Tuple[int, ...],
                    Tuple[Tuple[str, QueryPlanChoice], ...],
                ],
                ...
            ],
        ] = {}
        self._lower_bound_cache: Dict[tuple[int, int], float] = {}
        self._window_sig_cache: Dict[Tuple[int, ...], Tuple[QuerySignature, ...]] = {}
        self._node_min_cost = self._precompute_node_min_cost()
        self._solve_calls = 0
        self._global_best_cost = float("inf")
        self._memo_hits = 0

        # memo: (done_mask, worker_states, epoch_idx) -> (cost, schedule_rel, plans)
        self._memo: Dict[
            Tuple[int, Tuple[WorkerState, ...], int],
            Tuple[
                float,
                List[tuple[int, str, str]],
                Dict[tuple[str, str], QueryPlanChoice],
            ],
        ] = {}

        self._cpu_dep_counts = self._precompute_cpu_dep_counts()

    # === 对外接口 ===

    def solve(
        self,
        initial_worker_states: Tuple[WorkerState, ...] | None = None,
    ) -> tuple[
        float,
        List[tuple[int, str, str]],
        Dict[tuple[str, str], QueryPlanChoice],
    ]:
        """返回:
        - best_cost: float
        - schedule: List[(epoch, worker_id, node_id)]
        - best_plans: {(node_id, query_name) -> QueryPlanChoice}
        """
        if initial_worker_states is None:
            initial_worker_states = tuple(
                WorkerState(
                    worker_idx=self.worker_index[wid],
                    last_model_id=-1,
                    last_node_id=-1,
                )
                for wid in self.worker_ids
            )

        if self._rust_available and self._prefer_rust and self.cpu_cost_mode == "default":
            try:
                return self._solve_with_rust(initial_worker_states)
            except Exception as exc:  # pragma: no cover
                if self.debug_log:
                    print(f"[DP][rust] failed: {exc!r}; fallback to Python solver")
        best_cost, schedule_rel, plans = self._solve(
            done_mask=0,
            worker_states=initial_worker_states,
            epoch_idx=0,
        )
        return best_cost, schedule_rel, plans

    def _plan_base_cost(self, choice: QueryPlanChoice) -> float:
        if choice.raw_cost is not None:
            return max(0.05, self._raw_cost_scale * float(choice.raw_cost))
        if choice.cost is not None:
            return float(choice.cost)
        return 1.0

    def _build_rust_input(self, initial_worker_states: Tuple[WorkerState, ...]) -> Dict[str, object]:
        if not _HAS_DP_RUST:
            raise RuntimeError("Rust DP backend is not available")
        node_dicts: List[Dict[str, object]] = []
        for node_id in self.node_ids:
            node = self.graph.nodes[node_id]
            is_gpu = self._is_gpu_node(node_id)
            model_id = self._model_to_id(getattr(node, "model", None), fallback=self._none_id)
            queries: List[Dict[str, object]] = []
            for query in getattr(node, "db_queries", []) or []:
                choices = self.plan_choices.get((node.id, query.name), ())
                if not choices:
                    choices = (self._fallback_choice,)
                plans: List[Dict[str, object]] = []
                for choice in choices:
                    base_cost = self._plan_base_cost(choice)
                    sig_id = self._signature_id(node.id, query.name, choice)
                    plans.append(
                        {
                            "plan_id": choice.plan_id,
                            "base_cost": float(base_cost),
                            "sig_id": int(sig_id),
                            "choice_obj": choice,
                        }
                    )
                queries.append({"name": query.name, "plans": plans})
            node_dicts.append(
                {
                    "id": node_id,
                    "is_gpu": is_gpu,
                    "model_id": int(model_id),
                    "queries": queries,
                }
            )

        worker_ids = list(self.worker_ids)
        worker_kinds = ["gpu" if self.workers[wid].kind == "gpu" else "cpu" for wid in worker_ids]

        exec_costs: List[List[float]] = [
            [float(self._exec_cost_cached(node_id, wid)) for wid in worker_ids]
            for node_id in self.node_ids
        ]

        model_ids = [self._none_id] + list(range(len(self._model_int_to_name)))
        model_init_costs: List[List[float]] = [
            [float(self._model_init_cost_cached(node_id, mid)) for mid in model_ids]
            for node_id in self.node_ids
        ]

        last_node_ids = [self._none_id] + [self._node_id_to_int[nid] for nid in self.node_ids]
        llm_bonus: List[List[float]] = [
            [
                float(self._llm_bonus_cached(node_id, last_nid, tuple(self.dependencies.get(node_id, ()))))
                for last_nid in last_node_ids
            ]
            for node_id in self.node_ids
        ]

        node_worker_options: List[List[int]] = [
            [self.worker_index[wid] for wid in self.node_worker_options[node_id]]
            for node_id in self.node_ids
        ]

        gpu_worker_indices = [self.worker_index[wid] for wid in self.gpu_worker_ids]
        cpu_worker_indices = [self.worker_index[wid] for wid in self.cpu_worker_ids]

        gpu_node_indices = [self.node_index[nid] for nid in self._gpu_node_ids]
        db_node_indices = [self.node_index[nid] for nid in self._db_node_ids]
        epoch_penalties = [
            float(self._epoch_penalty_fn(epoch)) for epoch in range(len(self.node_ids) + 1)
        ]
        node_min_cost = [float(self._node_min_cost.get(nid, 0.0)) for nid in self.node_ids]

        initial_states_payload = [
            {
                "worker_idx": st.worker_idx,
                "last_model_id": st.last_model_id,
                "last_node_id": st.last_node_id,
            }
            for st in initial_worker_states
        ]

        payload: Dict[str, object] = {
            "nodes": node_dicts,
            "worker_ids": worker_ids,
            "worker_kinds": worker_kinds,
            "gpu_worker_indices": gpu_worker_indices,
            "cpu_worker_indices": cpu_worker_indices,
            "node_worker_options": node_worker_options,
            "gpu_node_indices": gpu_node_indices,
            "db_node_indices": db_node_indices,
            "parents_mask": list(self._parents_mask),
            "gpu_parents_mask": list(self._gpu_parents_mask),
            "exec_costs": exec_costs,
            "model_init_costs": model_init_costs,
            "llm_bonus": llm_bonus,
            "node_min_cost": node_min_cost,
            "epoch_penalties": epoch_penalties,
            "initial_worker_states": initial_states_payload,
            "window_size": int(self.window_size),
            "none_id": int(self._none_id),
            "raw_cost_scale": float(self._raw_cost_scale),
            "cpu_load_cost_weight": float(self.cpu_load_cost_weight),
            "gpu_batch_slack": int(self.gpu_batch_slack),
            "enable_batch_shape_pruning": bool(self.enable_batch_shape_pruning),
            "enable_lower_bound_pruning": bool(self.enable_lower_bound_pruning),
            "lower_bound_cost_factor": float(self.lower_bound_cost_factor),
            "cache_multiplier_fn": self.cache_multiplier_fn,
            "id_to_signature": list(self._id_to_signature),
        }
        return payload

    def _solve_with_rust(
        self,
        initial_worker_states: Tuple[WorkerState, ...],
    ) -> tuple[
        float,
        List[tuple[int, str, str]],
        Dict[tuple[str, str], QueryPlanChoice],
    ]:
        if not _HAS_DP_RUST:
            raise RuntimeError("Rust DP backend is not available")
        payload = self._build_rust_input(initial_worker_states)
        best_cost, schedule, plans = _solve_rs(payload)
        best_plans: Dict[tuple[str, str], QueryPlanChoice] = {}
        for (node_id, query_name), choice in plans.items():
            best_plans[(node_id, query_name)] = choice
        return best_cost, schedule, best_plans

    # === DP 主过程：状态不含 epoch，epoch 深度通过递归隐式体现 ===

    def _solve(
        self,
        *,
        done_mask: int,
        worker_states: Tuple[WorkerState, ...],
        epoch_idx: int,
        allow_relax: bool = True,
    ) -> tuple[
        float,
        List[tuple[int, str, str]],
        Dict[tuple[str, str], QueryPlanChoice],
    ]:
        self._solve_calls += 1
        if self.debug_log and self._solve_calls % self.debug_every == 0:
            done_cnt = done_mask.bit_count()
            print(
                f"[DP][state {self._solve_calls}] done={done_cnt}/{len(self.node_ids)} "
                f"memo={len(self._memo)}"
            )
        if self._progress_enabled and self._solve_calls % self._progress_every == 0:
            now = time.perf_counter()
            elapsed = now - self._progress_start
            since_last = now - self._progress_last
            rate = self._progress_every / since_last if since_last > 0 else 0.0
            best = self._global_best_cost if self._global_best_cost < float("inf") else None
            print(
                f"[DP][progress] states={self._solve_calls} memo={len(self._memo)} "
                f"best={'{:.4f}'.format(best) if best is not None else 'n/a'} "
                f"elapsed={elapsed:.1f}s rate={rate:.1f}/s"
            )
            self._progress_last = now
        if done_mask == self.all_mask:
            return 0.0, [], {}

        key = (done_mask, worker_states, epoch_idx)
        if key in self._memo:
            self._memo_hits += 1
            return self._memo[key]

        best_cost = float("inf")
        best_schedule_rel: List[tuple[int, str, str]] = []
        best_plans: Dict[tuple[str, str], QueryPlanChoice] = {}

        gpu_batches = self._enumerate_gpu_batches(done_mask)

        for gpu_nodes in gpu_batches:
            gpu_depths = self._batch_gpu_depths(gpu_nodes)
            if gpu_depths:
                gpu_nodes = sorted(gpu_nodes, key=lambda nid: gpu_depths.get(nid, 0))
            cpu_nodes = self._auto_cpu_batch(done_mask, gpu_nodes)
            cpu_nodes = self._order_cpu_nodes_by_parent_depth(cpu_nodes, gpu_depths)
            if not gpu_nodes and not cpu_nodes:
                continue  # 至少要选一个节点

            combined = set(gpu_nodes) | set(cpu_nodes)
            if not self._batch_feasible(combined, done_mask):
                continue
            new_done_mask = done_mask
            for node_id in combined:
                new_done_mask |= 1 << self.node_index[node_id]
            epoch_penalty = 0.0 if self.disable_epoch_batch_cost else float(self._epoch_penalty_fn(epoch_idx))
            depth_penalty = 0.0 if self.disable_epoch_batch_cost else self._batch_gpu_depth_penalty(gpu_nodes, gpu_depths)
            cpu_load_weight = self.cpu_load_cost_weight
            if self.disable_cpu_load_cost:
                cpu_load_weight = 0.0
            elif self.cpu_load_early_weight > 0:
                cpu_load_weight *= 1.0 + self.cpu_load_early_weight / (1.0 + float(epoch_idx))
            naive_cpu_cost = None
            if self.cpu_cost_mode == "naive":
                naive_cpu_cost = self._naive_cpu_cost(gpu_nodes)

            gpu_assignments = list(self._gpu_assignments(gpu_nodes))
            if self.enable_worker_symmetry:
                gpu_assignments = self._dedup_worker_assignments(
                    gpu_assignments, worker_states
                )

            cpu_assign = self._cpu_assignments(cpu_nodes)

            for g_assign in gpu_assignments:
                if not g_assign and not cpu_assign:
                    continue

                for (
                    gpu_cost_max,
                    gpu_cost_sum,
                    cpu_load_cost,
                    next_states,
                    batch_plans,
                ) in self._assignment_outcomes(g_assign, cpu_nodes, worker_states):
                    if naive_cpu_cost is not None:
                        cpu_load_cost = naive_cpu_cost
                    gpu_cost = (
                        self.gpu_cost_max_weight * float(gpu_cost_max)
                        + self.gpu_cost_sum_weight * float(gpu_cost_sum)
                    )
                    epoch_cost = (
                        epoch_penalty
                        + gpu_cost
                        + cpu_load_weight * cpu_load_cost
                        + depth_penalty
                    )

                    if (
                        self.enable_lower_bound_pruning
                        and best_cost < float("inf")
                    ):
                        remaining_lb = self._lower_bound_remaining(new_done_mask, epoch_idx + 1)
                        if epoch_cost + remaining_lb >= best_cost:
                            continue

                    sub_cost, sub_sched_rel, sub_plans = self._solve(
                        done_mask=new_done_mask,
                        worker_states=next_states,
                        epoch_idx=epoch_idx + 1,
                        allow_relax=allow_relax,
                    )

                    total_cost = epoch_cost + sub_cost
                    if total_cost < best_cost:
                        if self.debug_log and total_cost < self._global_best_cost:
                            print(
                                f"[DP][best] cost={total_cost:.4f} "
                                f"epochs={len(sub_sched_rel)+1} "
                                f"done={new_done_mask.bit_count()}/{len(self.node_ids)}"
                            )
                        self._global_best_cost = min(self._global_best_cost, total_cost)
                        best_cost = total_cost
                        # 当前 epoch 记为 0，子 schedule 的 epoch 全部 +1
                        this_epoch_sched = [(0, wid, nid) for (wid, nid) in g_assign]
                        this_epoch_sched += [(0, wid, nid) for (wid, nid) in cpu_assign]
                        shifted_sub_sched = [
                            (e + 1, wid, nid) for (e, wid, nid) in sub_sched_rel
                        ]
                        best_schedule_rel = this_epoch_sched + shifted_sub_sched

                        # 合并 plan/order：DB node 只执行一次，直接覆盖即可
                        best_plans = dict(sub_plans)
                        best_plans.update(batch_plans)

        if best_cost == float("inf"):
            if allow_relax and self.enable_batch_shape_pruning:
                # 若剪枝过于激进导致无解，回退到未剪枝（保留对称消除）再尝试一次。
                print("DP relaxation: retrying without batch shape pruning (symmetry preserved)...")
                orig_batch_prune = self.enable_batch_shape_pruning
                self.enable_batch_shape_pruning = False
                try:
                    return self._solve(
                        done_mask=done_mask,
                        worker_states=worker_states,
                        epoch_idx=epoch_idx,
                        allow_relax=False,
                    )
                finally:
                    self.enable_batch_shape_pruning = orig_batch_prune
            elif self.debug_log:
                print(
                    f"[DP][fail] No feasible schedule from state done={done_mask.bit_count()}/"
                    f"{len(self.node_ids)}; memo={len(self._memo)}"
                )
            raise RuntimeError("DP failed to find a valid schedule; check DAG dependencies.")

        self._memo[key] = (best_cost, best_schedule_rel, best_plans)
        return self._memo[key]

    # === 简单下界（用于 branch-and-bound 剪枝） ===

    def _lower_bound_remaining(self, done_mask: int, epoch_idx: int) -> float:
        """基于剩余 GPU / DB 节点的最少 epoch 数，计算一个乐观的剩余成本下界。"""
        if not self.enable_lower_bound_pruning:
            return 0.0

        cached = self._lower_bound_cache.get((done_mask, epoch_idx))
        if cached is not None:
            return cached

        remaining_gpu = sum(
            1 for node_id in self._gpu_node_ids if not self._is_done(done_mask, node_id)
        )
        remaining_cpu = sum(
            1 for node_id in self._db_node_ids if not self._is_done(done_mask, node_id)
        )
        if remaining_gpu == 0 and remaining_cpu == 0:
            self._lower_bound_cache[(done_mask, epoch_idx)] = 0.0
            return 0.0

        max_gpu_per_epoch = max(1, len(self.gpu_worker_ids))
        min_epoch_gpu = math.ceil(remaining_gpu / max_gpu_per_epoch) if remaining_gpu else 0
        min_epoch_cpu = 1 if remaining_cpu else 0
        min_epochs = max(min_epoch_gpu, min_epoch_cpu)
        # 粗略 cost 下界：按资源容量分摊的最小负载，再乘折扣因子，避免对并行度估计过高。
        remaining_gpu_nodes = [
            nid for nid in self._gpu_node_ids if not self._is_done(done_mask, nid)
        ]
        gpu_min_sum = sum(self._node_min_cost.get(nid, 0.0) for nid in remaining_gpu_nodes)
        gpu_capacity = max(1, len(self.gpu_worker_ids))
        load_lb = gpu_min_sum / gpu_capacity if gpu_min_sum > 0 else 0.0
        if self.disable_epoch_batch_cost:
            penalty_lb = 0.0
        else:
            penalty_lb = sum(float(self._epoch_penalty_fn(epoch_idx + i)) for i in range(min_epochs))
        lb = penalty_lb + self.lower_bound_cost_factor * load_lb
        self._lower_bound_cache[(done_mask, epoch_idx)] = lb
        return lb

    def _resolve_cpu_cost_mode(self, mode: str | None) -> str:
        raw = (mode or os.getenv("HALO_DP_CPU_COST_MODE") or "default").strip().lower()
        if raw in ("default", "naive"):
            return raw
        raise ValueError(f"Unsupported cpu_cost_mode '{mode}'. Expected 'default' or 'naive'.")

    def _precompute_cpu_dep_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node_id in self._gpu_node_ids:
            counts[node_id] = len(self._cpu_needed_for_gpu([node_id]))
        return counts

    def _precompute_node_min_cost(self) -> Dict[str, float]:
        """预估每个节点的最小单机 cost（GPU 节点含 cpu load cost 下界）。"""
        res: Dict[str, float] = {}
        for node_id in self.node_ids:
            node = self.graph.nodes[node_id]
            if not self._is_gpu_node(node_id):
                res[node_id] = 0.0
                continue
            # 基础 exec cost（按最小 worker capacity 假设）
            min_worker_cost = float("inf")
            allowed_workers = self.node_worker_options.get(node_id, self.worker_ids)
            for wid in allowed_workers:
                worker = self.workers[wid]
                try:
                    cost = float(self.exec_cost_fn(node, worker))
                except Exception:
                    cost = float("inf")
                min_worker_cost = min(min_worker_cost, cost)
            if min_worker_cost == float("inf"):
                min_worker_cost = 0.0
            # 最便宜的 plan cost（忽略 cache multiplier）
            min_plan_cost = 0.0
            for q in getattr(node, "db_queries", []) or []:
                choices = self.plan_choices.get((node.id, q.name), ())
                if not choices:
                    choices = (self._fallback_choice,)
                best = float("inf")
                for choice in choices:
                    if choice.raw_cost is not None:
                        base = max(0.05, self._raw_cost_scale * float(choice.raw_cost))
                    elif choice.cost is not None:
                        base = float(choice.cost)
                    else:
                        base = 1.0
                    best = min(best, base)
                if best == float("inf"):
                    best = 0.0
                min_plan_cost += best
            if self.disable_cpu_load_cost or self.cpu_cost_mode == "naive":
                min_plan_cost = 0.0
            else:
                min_plan_cost *= self.cpu_load_cost_weight
            res[node_id] = min_worker_cost + min_plan_cost
        return res

    # === assignment -> cost & state 更新（GPU cost + cpu load cost 分开算） ===

    def _assignment_outcomes(
        self,
        assign: Sequence[tuple[str, str]],
        cpu_nodes: Sequence[str],
        worker_states: Tuple[WorkerState, ...],
    ) -> Iterable[
        tuple[
            float,
            float,
            float,
            Tuple[WorkerState, ...],
            Dict[tuple[str, str], QueryPlanChoice],
        ]
    ]:
        """给定一组 (worker_id, node_id)，枚举:
          - GPU cost（LLM makespan）
          - GPU cost sum（LLM 总时长）
          - cpu load cost（GPU 节点 DB queries 累加 + CPU 节点 HTTP sleep）
          - 更新后的 worker_states
          - 本 batch 内的 plan 选择
        """

        @lru_cache(maxsize=None)
        def compute(
            assign_key: Tuple[tuple[str, str], ...],
            worker_states_key: Tuple[WorkerState, ...],
            cpu_key: Tuple[str, ...],
        ) -> Tuple[
            Tuple[
                float,
                float,
                float,
                Tuple[WorkerState, ...],
                Dict[tuple[str, str], QueryPlanChoice],
            ],
            ...
        ]:
            results: List[
                Tuple[
                    float,
                    float,
                    float,
                    Tuple[WorkerState, ...],
                    Dict[tuple[str, str], QueryPlanChoice],
                ]
            ] = []

            def dfs(
                idx: int,
                current_states: List[WorkerState],
                worker_costs: List[float],
                acc_plans: Dict[tuple[str, str], QueryPlanChoice],
                cpu_load_cost: float,
                window: Tuple[int, ...],
            ) -> None:
                if idx >= len(assign_key):
                    cpu_load_cost_extra, cpu_plans = self._apply_cpu_load(cpu_key, window)
                    gpu_costs = [
                        worker_costs[self.worker_index[wid]] for wid in self.gpu_worker_ids
                    ]
                    gpu_cost_max = max(gpu_costs) if gpu_costs else 0.0
                    gpu_cost_sum = sum(gpu_costs) if gpu_costs else 0.0
                    merged_plans = dict(acc_plans)
                    merged_plans.update(cpu_plans)
                    results.append(
                        (
                            gpu_cost_max,
                            gpu_cost_sum,
                            float(cpu_load_cost + cpu_load_cost_extra),
                            tuple(current_states),
                            merged_plans,
                        )
                    )
                    return

                worker_id, node_id = assign_key[idx]
                worker_idx = self.worker_index[worker_id]
                state = current_states[worker_idx]

                for (
                    node_cost,
                    node_cpu_cost,
                    updated_state,
                    plan_map,
                    exit_window,
                ) in self._execute_node(node_id, state, window):
                    current_states[worker_idx] = updated_state

                    new_worker_costs = list(worker_costs)
                    new_worker_costs[worker_idx] += node_cost

                    merged_plans = dict(acc_plans)
                    merged_plans.update(plan_map)

                    dfs(
                        idx + 1,
                        current_states,
                        new_worker_costs,
                        merged_plans,
                        cpu_load_cost + node_cpu_cost,
                        exit_window,
                    )

                    # 回溯
                    current_states[worker_idx] = state

            dfs(
                0,
                list(worker_states_key),
                [0.0 for _ in self.worker_ids],
                {},
                0.0,
                tuple(),
            )
            return tuple(results)

        yield from compute(tuple(assign), worker_states, tuple(cpu_nodes))

    # === 单个节点在某个 worker 上的所有执行方式（LLM or DB） ===

    def _execute_node(
        self,
        node_id: str,
        state: WorkerState,
        enter_window: Tuple[int, ...],
    ) -> Iterable[
        tuple[
            float,
            float,
            WorkerState,
            Dict[tuple[str, str], QueryPlanChoice],
            Tuple[int, ...],
        ]
    ]:
        node = self.graph.nodes[node_id]
        parents = tuple(self.dependencies.get(node_id, []))

        worker_id = self._worker_id_by_idx[state.worker_idx]
        worker = self.workers[worker_id]
        last_model = self._model_from_id(state.last_model_id)
        last_node = self._node_from_id(state.last_node_id)

        # LLM / DB 都使用 exec_cost_fn；DB 的 estimate cost 由 query plan 单独提供。
        exec_cost = self._exec_cost_cached(node_id, worker_id)
        model_cost = self._model_init_cost_cached(node_id, state.last_model_id)
        bonus_multiplier = self._llm_bonus_cached(node_id, state.last_node_id, parents)
        if bonus_multiplier <= 0:
            bonus_multiplier = 1.0

        if self._is_gpu_node(node_id):
            # LLM 节点：LLM cost 与 cpu load cost 分开累计
            base_cost = exec_cost * bonus_multiplier + model_cost
            queries = getattr(node, "db_queries", []) or []
            if not queries:
                new_state = WorkerState(
                    worker_idx=state.worker_idx,
                    last_model_id=self._model_to_id(node.model, fallback=state.last_model_id),
                    last_node_id=self._node_id_to_int[node.id],
                )
                yield base_cost, 0.0, new_state, {}, enter_window
                return

            options = self._query_plan_options(node, enter_window)
            for query_cost, exit_window, plan_seq in options:
                new_state = WorkerState(
                    worker_idx=state.worker_idx,
                    last_model_id=self._model_to_id(node.model, fallback=state.last_model_id),
                    last_node_id=self._node_id_to_int[node.id],
                )
                plan_map = {(node.id, q_name): choice for q_name, choice in plan_seq}
                yield base_cost, query_cost, new_state, plan_map, exit_window
            return

        # 非 GPU 节点：不计 cost、不更新 window（由 CPU 自动填充）
        new_state = WorkerState(
            worker_idx=state.worker_idx,
            last_model_id=state.last_model_id,
            last_node_id=self._node_id_to_int[node.id],
        )
        yield 0.0, 0.0, new_state, {}, enter_window

    # === DB queries：遍历 plan 组合（当前假设单 query 节点），window 仅在 epoch 内生效 ===

    def _query_plan_options(
        self,
        node: Node,
        enter_window: Tuple[int, ...],
    ) -> Tuple[
        Tuple[float, Tuple[int, ...], Tuple[Tuple[str, QueryPlanChoice], ...]],
        ...,
    ]:
        cache_key = (self._node_id_to_int[node.id], enter_window)
        if cache_key in self._query_plan_cache:
            return self._query_plan_cache[cache_key]

        queries = list(getattr(node, "db_queries", []) or [])
        if not queries:
            # 没有 DB query，则 query cost 为 0，window 不变
            self._query_plan_cache[cache_key] = ((0.0, enter_window, tuple()),)
            return self._query_plan_cache[cache_key]
        # 现在每个 DB node 只有一个 query；若后续有多个 query，则按声明顺序累乘 plan 组合，不再枚举顺序。
        outcomes: List[
            Tuple[float, Tuple[int, ...], Tuple[Tuple[str, QueryPlanChoice], ...]]
        ] = [(0.0, enter_window, tuple())]
        for query in queries:
            choices = self.plan_choices.get((node.id, query.name), ())
            if not choices:
                choices = (self._fallback_choice,)
            next_outcomes: List[
                Tuple[float, Tuple[int, ...], Tuple[Tuple[str, QueryPlanChoice], ...]]
            ] = []
            for base_cost, window, plan_seq in outcomes:
                for choice in choices:
                    q_cost = self._query_cost(window, choice)
                    next_window = self._append_window(window, node.id, query.name, choice)
                    next_outcomes.append(
                        (base_cost + q_cost, next_window, plan_seq + ((query.name, choice),))
                    )
            outcomes = next_outcomes

        results = tuple(outcomes)
        self._query_plan_cache[cache_key] = results
        return results

    def _query_cost(
        self,
        window: Tuple[int, ...],
        choice: QueryPlanChoice,
    ) -> float:
        if choice.raw_cost is not None:
            base = max(0.05, self._raw_cost_scale * float(choice.raw_cost))
        elif choice.cost is not None:
            base = float(choice.cost)
        else:
            base = 1.0
        return base * self.cache_multiplier_fn(self._window_signatures(window), choice)

    def _append_window(
        self,
        window: Tuple[int, ...],
        node_id: str,
        query_name: str,
        choice: QueryPlanChoice,
    ) -> Tuple[int, ...]:
        if self.window_size <= 0:
            return window
        sig_id = self._signature_id(node_id, query_name, choice)
        if not window:
            if self.window_size == 1:
                return (sig_id,)
            return (self._none_id,) * (self.window_size - 1) + (sig_id,)
        new_window = window + (sig_id,)
        if len(new_window) > self.window_size:
            new_window = new_window[-self.window_size :]
        return new_window

    def _batch_gpu_depths(self, gpu_nodes: Sequence[str]) -> Dict[str, int]:
        """计算当前 epoch 选中 GPU 子图内的拓扑深度（根=0）。"""
        if not gpu_nodes:
            return {}
        selected_list = list(gpu_nodes)
        selected = set(selected_list)
        indeg: Dict[str, int] = {nid: 0 for nid in selected_list}
        children: Dict[str, List[str]] = {nid: [] for nid in selected_list}
        for nid in selected_list:
            for parent in self.dependencies.get(nid, ()):
                if parent not in selected:
                    continue
                indeg[nid] += 1
                children[parent].append(nid)

        queue = [nid for nid in selected_list if indeg[nid] == 0]
        depths: Dict[str, int] = {nid: 0 for nid in queue}
        while queue:
            nid = queue.pop(0)
            base = depths.get(nid, 0)
            for child in children.get(nid, []):
                depths[child] = max(depths.get(child, 0), base + 1)
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)
        for nid in selected_list:
            depths.setdefault(nid, 0)
        return depths

    def _batch_gpu_depth_penalty(
        self, gpu_nodes: Sequence[str], gpu_depths: Mapping[str, int] | None = None
    ) -> float:
        """基于当前 epoch 选中的 GPU 子图深度计算惩罚。"""
        if self._gpu_depth_cost_weight <= 0 or not gpu_nodes:
            return 0.0
        depths = gpu_depths if gpu_depths is not None else self._batch_gpu_depths(gpu_nodes)
        total_depth = sum(depths.get(nid, 0) for nid in gpu_nodes)
        return self._gpu_depth_cost_weight * float(total_depth)

    def _apply_cpu_load(
        self,
        cpu_nodes: Sequence[str],
        enter_window: Tuple[int, ...],
    ) -> Tuple[float, Dict[tuple[str, str], QueryPlanChoice]]:
        """按既定 CPU 顺序累计 load cost（DB plan + HTTP sleep），window 仅在 epoch 内有效。"""
        if not cpu_nodes:
            return 0.0, {}
        window = enter_window
        total_cost = 0.0
        plan_map: Dict[tuple[str, str], QueryPlanChoice] = {}
        for node_id in cpu_nodes:
            node = self.graph.nodes[node_id]
            if node.engine == "http":
                total_cost += self._http_sleep_cost(node)
                continue
            if not getattr(node, "db_queries", None):
                continue
            options = self._query_plan_options(node, window)
            best_cost = float("inf")
            best_window = window
            best_seq: Tuple[Tuple[str, QueryPlanChoice], ...] = tuple()
            for cost, exit_window, plan_seq in options:
                if cost < best_cost:
                    best_cost = cost
                    best_window = exit_window
                    best_seq = plan_seq
            if best_cost == float("inf"):
                continue
            total_cost += float(best_cost)
            for q_name, choice in best_seq:
                plan_map[(node.id, q_name)] = choice
            window = best_window
        return total_cost, plan_map

    def _http_sleep_cost(self, node: Node) -> float:
        if self._http_latency_s:
            profiled = self._http_latency_s.get(node.id)
            if profiled is not None:
                try:
                    return max(0.0, float(profiled))
                except (TypeError, ValueError):
                    pass
        raw = node.raw if isinstance(node.raw, dict) else {}
        for key, scale in (
            ("sleep_s", 1.0),
            ("sleep_ms", 0.001),
            ("latency_s", 1.0),
            ("latency_ms", 0.001),
            ("timeout_s", 1.0),
            ("timeout_ms", 0.001),
        ):
            if key not in raw:
                continue
            value = raw.get(key)
            if isinstance(value, str):
                value = value.strip()
            try:
                seconds = float(value) * scale
            except (TypeError, ValueError):
                continue
            return max(0.0, seconds)
        return 0.0

    def _order_cpu_nodes_by_parent_depth(
        self, cpu_nodes: Sequence[str], gpu_depths: Mapping[str, int]
    ) -> List[str]:
        if not cpu_nodes or not gpu_depths:
            return list(cpu_nodes)

        def key(node_id: str) -> Tuple[bool, int]:
            node = self.graph.nodes.get(node_id)
            parent = None
            if node is not None and isinstance(node.raw, dict):
                parent = node.raw.get("parent")
            depth = gpu_depths.get(parent) if parent is not None else None
            return (depth is None, depth or 0)

        return sorted(cpu_nodes, key=key)

    # === id 映射 & window/signature 辅助 ===

    def _model_to_id(self, model: str | None, fallback: int | None = None) -> int:
        if model is None:
            return self._none_id if fallback is None else fallback
        return self._model_to_int.get(model, self._none_id if fallback is None else fallback)

    def _model_from_id(self, model_id: int) -> str | None:
        if model_id == self._none_id:
            return None
        if 0 <= model_id < len(self._model_int_to_name):
            return self._model_int_to_name[model_id]
        return None

    def _node_from_id(self, node_id: int) -> str | None:
        if node_id == self._none_id:
            return None
        if 0 <= node_id < len(self._node_int_to_id):
            return self._node_int_to_id[node_id]
        return None

    def _signature_id(self, node_id: str, query_name: str, choice: QueryPlanChoice) -> int:
        fp_tuple: Tuple[Tuple[int, int], ...] = tuple(
            sorted(
                (self._fp_to_int.get(k, self._none_id), int(v))
                for k, v in (choice.footprints or {}).items()
            )
        )
        key = (
            self._node_id_to_int[node_id],
            self._query_to_int.get(query_name, self._none_id),
            self._plan_to_int.get(choice.plan_id, self._none_id),
            fp_tuple,
        )
        cached = self._signature_to_id.get(key)
        if cached is not None:
            return cached
        sig = QuerySignature(
            node_id=node_id,
            query_name=query_name,
            plan_id=choice.plan_id,
            footprints=tuple(sorted((choice.footprints or {}).items())),
        )
        sig_id = len(self._id_to_signature)
        self._signature_to_id[key] = sig_id
        self._id_to_signature.append(sig)
        return sig_id

    def _window_signatures(self, window: Tuple[int, ...]) -> Tuple[QuerySignature, ...]:
        cached = self._window_sig_cache.get(window)
        if cached is not None:
            return cached
        sigs = tuple(self._id_to_signature[sig_id] for sig_id in window if sig_id != self._none_id)
        self._window_sig_cache[window] = sigs
        return sigs

    def _build_parents_mask(
        self, deps: Mapping[str, Sequence[str]], gpu_only: bool = False
    ) -> Tuple[int, ...]:
        masks = [0] * len(self.node_ids)
        for nid, idx in self.node_index.items():
            mask = 0
            for parent in deps.get(nid, []):
                if parent not in self.node_index:
                    continue
                if gpu_only and not self._is_gpu_node(parent):
                    continue
                pidx = self.node_index[parent]
                mask |= 1 << pidx
            masks[idx] = mask
        return tuple(masks)

    @lru_cache(maxsize=None)
    def _exec_cost_cached(self, node_id: str, worker_id: str) -> float:
        node = self.graph.nodes[node_id]
        worker = self.workers[worker_id]
        return float(self.exec_cost_fn(node, worker))

    @lru_cache(maxsize=None)
    def _model_init_cost_cached(self, node_id: str, last_model_id: int) -> float:
        node = self.graph.nodes[node_id]
        last_model = self._model_from_id(last_model_id)
        return float(self.model_init_cost_fn(node, last_model))

    @lru_cache(maxsize=None)
    def _llm_bonus_cached(
        self,
        node_id: str,
        last_node_id: int,
        parents: Tuple[str, ...],
    ) -> float:
        node = self.graph.nodes[node_id]
        last_node = self._node_from_id(last_node_id)
        return float(self.llm_cache_bonus_fn(node, last_node, parents))

    # === worker 对称性消除 ===

    def _worker_signature(self, worker_idx: int, state: WorkerState) -> Tuple:
        worker = self.workers[self._worker_id_by_idx[worker_idx]]
        return (
            worker.kind,
            float(worker.capacity),
            state.last_model_id,
            state.last_node_id,
        )

    def _dedup_worker_assignments(
        self,
        assignments: List[List[tuple[str, str]]],
        worker_states: Tuple[WorkerState, ...],
    ) -> List[List[tuple[str, str]]]:
        """消除同构 worker 间的对称映射，减少重复 assignment。"""
        seen: Set[Tuple] = set()
        uniq: List[List[tuple[str, str]]] = []
        for assign in assignments:
            key = self._assignment_signature(assign, worker_states)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(assign)
        return uniq

    def _assignment_signature(
        self, assign: List[tuple[str, str]], worker_states: Tuple[WorkerState, ...]
    ) -> Tuple:
        buckets: Dict[Tuple, List[str]] = {}
        for wid, node_id in assign:
            idx = self.worker_index[wid]
            sig = self._worker_signature(idx, worker_states[idx])
            buckets.setdefault(sig, []).append(node_id)
        items = []
        for sig, nodes in buckets.items():
            items.append((sig, tuple(sorted(nodes))))
        items.sort(key=lambda x: x[0])
        return tuple(items)

    # === GPU 批次枚举 ===

    def _enumerate_gpu_batches(self, done_mask: int) -> List[List[str]]:
        """枚举当前 epoch GPU 侧的可行子图（受 GPU 数量限制，可为空）。"""
        max_gpu = max(1, len(self.gpu_worker_ids))
        pending = [
            node_id
            for node_id in self.node_ids
            if not self._is_done(done_mask, node_id) and self._is_gpu_node(node_id)
        ]
        if not pending:
            return [[]]

        # 先做 GPU 子图的拓扑排序，保证父节点在前，枚举时只生成依赖闭包。
        pending_list = list(pending)
        pending_set = set(pending_list)
        in_deg = {nid: 0 for nid in pending_list}
        for nid in pending_list:
            for parent in self.dependencies.get(nid, []):
                if parent in pending_set:
                    in_deg[nid] += 1
        queue = [nid for nid in pending_list if in_deg[nid] == 0]
        topo: List[str] = []
        while queue:
            nid = queue.pop(0)
            topo.append(nid)
            for child in pending_list:
                if nid in self.dependencies.get(child, []):
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)
        if len(topo) != len(pending_set):
            # 若有环或异常依赖，退化为原顺序以避免崩溃。
            topo = pending_list

        target = min(max_gpu, len(pending))
        min_r = 1
        max_r = min(max_gpu, len(pending))
        if self.enable_batch_shape_pruning:
            min_r = max(1, target - self.gpu_batch_slack)
            max_r = target

        results: List[List[str]] = [[]]

        def dfs(idx: int, selected: List[str], selected_mask: int) -> None:
            if idx >= len(topo):
                size = len(selected)
                if size >= min_r and size <= max_r:
                    results.append(list(selected))
                return

            # 不选当前节点
            dfs(idx + 1, selected, selected_mask)

            # 尝试选择当前节点（前置依赖必须已完成或已被选中）
            if len(selected) >= max_r:
                return
            node_id = topo[idx]
            node_idx = self.node_index[node_id]
            gpu_parents_mask = self._gpu_parents_mask[node_idx]
            if gpu_parents_mask & ~(done_mask | selected_mask):
                dfs(idx + 1, selected, selected_mask)
                return

            selected.append(node_id)
            dfs(idx + 1, selected, selected_mask | (1 << node_idx))
            selected.pop()

        dfs(0, [], 0)
        if pending and len(results) == 1 and results[0] == []:
            print("[DP] Warning: GPU batch enumeration empty despite pending nodes:", pending)
        return results

    def _auto_cpu_batch(self, done_mask: int, gpu_nodes: Sequence[str]) -> List[str]:
        """在不考虑容量/成本的前提下，自动填充当前 epoch 必要的 CPU 节点。"""
        if not self._db_node_ids:
            return []
        batch_mask = done_mask
        for node_id in gpu_nodes:
            batch_mask |= 1 << self.node_index[node_id]

        needed = self._cpu_needed_for_gpu(gpu_nodes)
        needed = self._expand_cpu_parents(needed)
        pending = {
            nid
            for nid in self._db_node_ids
            if not self._is_done(done_mask, nid) and nid in needed
        }
        if not pending:
            return []

        selected: List[str] = []
        progressed = True
        while progressed:
            progressed = False
            for nid in self._db_topo_order:
                if nid not in pending:
                    continue
                idx = self.node_index[nid]
                pmask = self._parents_mask[idx]
                if pmask & ~batch_mask:
                    continue
                selected.append(nid)
                pending.remove(nid)
                batch_mask |= 1 << idx
                progressed = True
        return selected

    def _cpu_needed_for_gpu(self, gpu_nodes: Sequence[str]) -> Set[str]:
        if not gpu_nodes:
            return set()
        needed: Set[str] = set()
        stack = list(gpu_nodes)
        visited: Set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            for parent in self.dependencies.get(node_id, ()):
                if parent not in visited:
                    stack.append(parent)
                if parent in self._db_node_ids:
                    needed.add(parent)
        return needed

    def _expand_cpu_parents(self, cpu_nodes: Set[str]) -> Set[str]:
        if not cpu_nodes:
            return set()
        expanded = set(cpu_nodes)
        stack = list(cpu_nodes)
        while stack:
            node_id = stack.pop()
            for parent in self.dependencies.get(node_id, ()):
                if parent in self._db_node_ids and parent not in expanded:
                    expanded.add(parent)
                    stack.append(parent)
        return expanded

    def _cpu_assignments(self, cpu_nodes: Sequence[str]) -> List[tuple[str, str]]:
        if not cpu_nodes:
            return []
        workers = self.cpu_worker_ids or self.worker_ids
        if not workers:
            return []
        seq: List[tuple[str, str]] = []
        for idx, node_id in enumerate(cpu_nodes):
            wid = workers[idx % len(workers)]
            seq.append((wid, node_id))
        return seq

    def _naive_cpu_cost(self, gpu_nodes: Sequence[str]) -> float:
        if not gpu_nodes:
            return 0.0
        return float(sum(self._cpu_dep_counts.get(node_id, 0) for node_id in gpu_nodes))

    # === worker assignment（GPU） ===
    def _gpu_assignments(
        self,
        gpu_nodes: Sequence[str],
    ) -> Iterable[List[tuple[str, str]]]:
        """GPU 节点到 GPU worker 的所有映射（每个 worker 至多一个 LLM node）。"""
        if not gpu_nodes:
            return [[]]

        results: List[List[tuple[str, str]]] = []

        def dfs(idx: int, used: Set[str], current: List[tuple[str, str]]) -> None:
            if idx >= len(gpu_nodes):
                results.append(list(current))
                return
            node_id = gpu_nodes[idx]
            allowed = self.node_worker_options.get(node_id, self.gpu_worker_ids)
            for wid in self.gpu_worker_ids:
                if wid not in allowed:
                    continue
                if wid in used:
                    continue
                used.add(wid)
                current.append((wid, node_id))
                dfs(idx + 1, used, current)
                current.pop()
                used.remove(wid)

        dfs(0, set(), [])
        return results

    # === 工具函数 ===

    def _batch_feasible(self, nodes: Set[str], done_mask: int) -> bool:
        """判断一个 epoch 内的节点集合是否满足依赖（父节点已完成或同批次执行）。"""
        batch_mask = done_mask
        for node_id in nodes:
            batch_mask |= 1 << self.node_index[node_id]
        for node_id in nodes:
            idx = self.node_index[node_id]
            pmask = self._parents_mask[idx]
            if pmask & ~batch_mask:
                return False
        return True

    def _is_done(self, done_mask: int, node_id: str) -> bool:
        idx = self.node_index[node_id]
        return bool(done_mask & (1 << idx))

    def _is_gpu_node(self, node_id: str) -> bool:
        node = self.graph.nodes[node_id]
        return node.engine == "vllm"
