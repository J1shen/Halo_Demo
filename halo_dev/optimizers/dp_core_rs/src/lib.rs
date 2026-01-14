use hashbrown::{HashMap, HashSet};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkerKind {
    Gpu,
    Cpu,
}

#[derive(Clone, Debug)]
struct WorkerSpec {
    id: String,
    kind: WorkerKind,
}

#[derive(Clone, Debug)]
struct PlanChoice {
    plan_id: String,
    base_cost: f64,
    sig_id: i32,
    choice_obj: Py<PyAny>,
}

#[derive(Clone, Debug)]
struct QuerySpec {
    name: String,
    plans: Vec<PlanChoice>,
}

#[derive(Clone, Debug)]
struct NodeSpec {
    id: String,
    is_gpu: bool,
    model_id: i32,
    queries: Vec<QuerySpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WorkerStateKey {
    last_model: i32,
    last_node: i32,
}

#[derive(Clone, Debug)]
struct WorkerState {
    worker_idx: usize,
    last_model: i32,
    last_node: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SolverKey {
    done_mask: u64,
    epoch_idx: u32,
    workers: Vec<WorkerStateKey>,
}

#[derive(Clone, Debug)]
struct EpochPlan {
    cost: f64,
    schedule: Vec<(usize, usize)>, // (worker_idx, node_idx)
    plans: HashMap<(usize, String), Py<PyAny>>, // (node_idx, query_name) -> plan choice
    states: Vec<WorkerState>,
    done_mask: u64,
}

#[derive(Clone, Debug)]
struct RustInput {
    nodes: Vec<NodeSpec>,
    workers: Vec<WorkerSpec>,
    gpu_workers: Vec<usize>,
    cpu_workers: Vec<usize>,
    node_worker_options: Vec<Vec<usize>>,
    gpu_node_indices: Vec<usize>,
    db_node_indices: Vec<usize>,
    parents_mask: Vec<u64>,
    gpu_parents_mask: Vec<u64>,
    exec_costs: Vec<Vec<f64>>,
    model_init_costs: Vec<Vec<f64>>,
    llm_bonus: Vec<Vec<f64>>,
    node_min_cost: Vec<f64>,
    epoch_penalties: Vec<f64>,
    gpu_batch_factors: Vec<f64>,
    initial_states: Vec<WorkerState>,
    window_size: usize,
    none_id: i32,
    raw_cost_scale: f64,
    cpu_load_cost_weight: f64,
    cpu_defer_weight: f64,
    enable_batch_shape_pruning: bool,
    gpu_batch_slack: usize,
    enable_lower_bound_pruning: bool,
    lower_bound_cost_factor: f64,
    cache_multiplier_fn: Py<PyAny>,
    id_to_signature: Vec<Py<PyAny>>,
}

fn bool_from_str(s: &str) -> Option<bool> {
    match s {
        "gpu" => Some(true),
        "cpu" => Some(false),
        _ => None,
    }
}

fn parse_worker_kind(s: &str) -> PyResult<WorkerKind> {
    match s {
        "gpu" => Ok(WorkerKind::Gpu),
        "cpu" => Ok(WorkerKind::Cpu),
        other => Err(PyRuntimeError::new_err(format!(
            "Unknown worker kind '{}'",
            other
        ))),
    }
}

fn to_offset(id: i32, none_id: i32) -> usize {
    if id == none_id {
        0
    } else if id >= 0 {
        (id as usize) + 1
    } else {
        0
    }
}

fn read_vec_f64(obj: &PyAny) -> PyResult<Vec<f64>> {
    obj.extract()
}

fn read_vec_i32(obj: &PyAny) -> PyResult<Vec<i32>> {
    obj.extract()
}

fn read_vec_usize(obj: &PyAny) -> PyResult<Vec<usize>> {
    obj.extract()
}

fn read_bool(obj: &PyAny) -> PyResult<bool> {
    obj.extract()
}

fn read_nodes(py: Python<'_>, dict: &PyDict) -> PyResult<Vec<NodeSpec>> {
    let nodes_obj = dict
        .get_item("nodes")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'nodes' in Rust DP input"))?;
    let nodes_list: &PyList = nodes_obj.downcast().map_err(|_| {
        PyRuntimeError::new_err("Expected 'nodes' to be a list of node specs")
    })?;
    let mut nodes = Vec::with_capacity(nodes_list.len());
    for item in nodes_list.iter() {
        let nd: &PyDict = item.downcast().map_err(|_| {
            PyRuntimeError::new_err("Node spec must be a dict")
        })?;
        let id: String = nd
            .get_item("id")
            .ok_or_else(|| PyRuntimeError::new_err("Node spec missing 'id'"))?
            .extract()?;
        let is_gpu: bool = nd
            .get_item("is_gpu")
            .ok_or_else(|| PyRuntimeError::new_err("Node spec missing 'is_gpu'"))?
            .extract()?;
        let model_id: i32 = nd
            .get_item("model_id")
            .ok_or_else(|| PyRuntimeError::new_err("Node spec missing 'model_id'"))?
            .extract()?;

        let queries_obj = nd
            .get_item("queries")
            .ok_or_else(|| PyRuntimeError::new_err("Node spec missing 'queries'"))?;
        let queries_list: &PyList = queries_obj.downcast().map_err(|_| {
            PyRuntimeError::new_err("Node queries must be a list")
        })?;
        let mut queries: Vec<QuerySpec> = Vec::with_capacity(queries_list.len());
        for q_item in queries_list.iter() {
            let qd: &PyDict = q_item.downcast().map_err(|_| {
                PyRuntimeError::new_err("Query spec must be a dict")
            })?;
            let name: String = qd
                .get_item("name")
                .ok_or_else(|| PyRuntimeError::new_err("Query spec missing 'name'"))?
                .extract()?;
            let plans_obj = qd
                .get_item("plans")
                .ok_or_else(|| PyRuntimeError::new_err("Query spec missing 'plans'"))?;
            let plans_list: &PyList = plans_obj.downcast().map_err(|_| {
                PyRuntimeError::new_err("Plans must be a list")
            })?;
            let mut plans: Vec<PlanChoice> = Vec::with_capacity(plans_list.len());
            for p_item in plans_list.iter() {
                let pd: &PyDict = p_item.downcast().map_err(|_| {
                    PyRuntimeError::new_err("Plan choice must be a dict")
                })?;
                let plan_id: String = pd
                    .get_item("plan_id")
                    .ok_or_else(|| PyRuntimeError::new_err("Plan choice missing 'plan_id'"))?
                    .extract()?;
                let base_cost: f64 = pd
                    .get_item("base_cost")
                    .ok_or_else(|| PyRuntimeError::new_err("Plan choice missing 'base_cost'"))?
                    .extract()?;
                let sig_id: i32 = pd
                    .get_item("sig_id")
                    .ok_or_else(|| PyRuntimeError::new_err("Plan choice missing 'sig_id'"))?
                    .extract()?;
                let choice_obj = pd
                    .get_item("choice_obj")
                    .ok_or_else(|| PyRuntimeError::new_err("Plan choice missing 'choice_obj'"))?;
                plans.push(PlanChoice {
                    plan_id,
                    base_cost,
                    sig_id,
                    choice_obj: choice_obj.into_py(py),
                });
            }
            queries.push(QuerySpec { name, plans });
        }
        nodes.push(NodeSpec {
            id,
            is_gpu,
            model_id,
            queries,
        });
    }
    Ok(nodes)
}

fn parse_workers(dict: &PyDict) -> PyResult<Vec<WorkerSpec>> {
    let ids_obj = dict
        .get_item("worker_ids")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'worker_ids'"))?;
    let ids: Vec<String> = ids_obj.extract()?;
    let kinds_obj = dict
        .get_item("worker_kinds")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'worker_kinds'"))?;
    let kinds: Vec<String> = kinds_obj.extract()?;
    if ids.len() != kinds.len() {
        return Err(PyRuntimeError::new_err(
            "worker_ids and worker_kinds length mismatch",
        ));
    }
    let mut workers = Vec::with_capacity(ids.len());
    for (id, kind_str) in ids.into_iter().zip(kinds.into_iter()) {
        let kind = parse_worker_kind(&kind_str)?;
        workers.push(WorkerSpec { id, kind });
    }
    Ok(workers)
}

fn parse_initial_states(dict: &PyDict) -> PyResult<Vec<WorkerState>> {
    let states_obj = dict
        .get_item("initial_worker_states")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'initial_worker_states'"))?;
    let states_list: &PyList = states_obj.downcast().map_err(|_| {
        PyRuntimeError::new_err("initial_worker_states must be a list")
    })?;
    let mut states = Vec::with_capacity(states_list.len());
    for item in states_list.iter() {
        let sd: &PyDict = item.downcast().map_err(|_| {
            PyRuntimeError::new_err("worker state must be a dict")
        })?;
        let worker_idx: usize = sd
            .get_item("worker_idx")
            .ok_or_else(|| PyRuntimeError::new_err("state missing worker_idx"))?
            .extract()?;
        let last_model: i32 = sd
            .get_item("last_model_id")
            .ok_or_else(|| PyRuntimeError::new_err("state missing last_model_id"))?
            .extract()?;
        let last_node: i32 = sd
            .get_item("last_node_id")
            .ok_or_else(|| PyRuntimeError::new_err("state missing last_node_id"))?
            .extract()?;
        states.push(WorkerState {
            worker_idx,
            last_model,
            last_node,
        });
    }
    Ok(states)
}

fn parse_vec_vec_f64(dict: &PyDict, key: &str) -> PyResult<Vec<Vec<f64>>> {
    let obj = dict
        .get_item(key)
        .ok_or_else(|| PyRuntimeError::new_err(format!("Missing '{}'", key)))?;
    obj.extract()
}

fn parse_vec_vec_usize(dict: &PyDict, key: &str) -> PyResult<Vec<Vec<usize>>> {
    let obj = dict
        .get_item(key)
        .ok_or_else(|| PyRuntimeError::new_err(format!("Missing '{}'", key)))?;
    obj.extract()
}

fn parse_vec_vec_i32(dict: &PyDict, key: &str) -> PyResult<Vec<Vec<i32>>> {
    let obj = dict
        .get_item(key)
        .ok_or_else(|| PyRuntimeError::new_err(format!("Missing '{}'", key)))?;
    obj.extract()
}

fn parse_py_list(dict: &PyDict, key: &str) -> PyResult<&PyList> {
    let obj = dict
        .get_item(key)
        .ok_or_else(|| PyRuntimeError::new_err(format!("Missing '{}'", key)))?;
    obj.downcast().map_err(|_| {
        PyRuntimeError::new_err(format!("Expected '{}' to be a list", key))
    })
}

fn parse_input(py: Python<'_>, dict: &PyDict) -> PyResult<RustInput> {
    let nodes = read_nodes(py, dict)?;
    let workers = parse_workers(dict)?;
    let parents_mask: Vec<u64> = dict
        .get_item("parents_mask")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'parents_mask'"))?
        .extract()?;
    let gpu_parents_mask: Vec<u64> = dict
        .get_item("gpu_parents_mask")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'gpu_parents_mask'"))?
        .extract()?;
    let gpu_nodes: Vec<usize> = dict
        .get_item("gpu_node_indices")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'gpu_node_indices'"))?
        .extract()?;
    let db_nodes: Vec<usize> = dict
        .get_item("db_node_indices")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'db_node_indices'"))?
        .extract()?;
    let node_worker_options = parse_vec_vec_usize(dict, "node_worker_options")?;
    let exec_costs = parse_vec_vec_f64(dict, "exec_costs")?;
    let model_init_costs = parse_vec_vec_f64(dict, "model_init_costs")?;
    let llm_bonus = parse_vec_vec_f64(dict, "llm_bonus")?;
    let epoch_penalties: Vec<f64> = dict
        .get_item("epoch_penalties")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'epoch_penalties'"))?
        .extract()?;
    let gpu_batch_factors: Vec<f64> = dict
        .get_item("gpu_batch_factors")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'gpu_batch_factors'"))?
        .extract()?;
    let node_min_cost: Vec<f64> = dict
        .get_item("node_min_cost")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'node_min_cost'"))?
        .extract()?;
    let initial_states = parse_initial_states(dict)?;
    let window_size: usize = dict
        .get_item("window_size")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'window_size'"))?
        .extract()?;
    let none_id: i32 = dict
        .get_item("none_id")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'none_id'"))?
        .extract()?;
    let raw_cost_scale: f64 = dict
        .get_item("raw_cost_scale")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'raw_cost_scale'"))?
        .extract()?;
    let cpu_load_cost_weight: f64 = if let Some(value) = dict.get_item("cpu_load_cost_weight") {
        value.extract()?
    } else if let Some(value) = dict.get_item("db_load_cost_weight") {
        value.extract()?
    } else {
        return Err(PyRuntimeError::new_err(
            "Missing 'cpu_load_cost_weight' (or legacy 'db_load_cost_weight')",
        ));
    };
    let cpu_defer_weight: f64 = if let Some(value) = dict.get_item("cpu_defer_weight") {
        value.extract()?
    } else if let Some(value) = dict.get_item("db_defer_weight") {
        value.extract()?
    } else {
        0.0
    };
    let gpu_batch_slack: usize = dict
        .get_item("gpu_batch_slack")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'gpu_batch_slack'"))?
        .extract()?;
    let enable_batch_shape_pruning: bool = dict
        .get_item("enable_batch_shape_pruning")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'enable_batch_shape_pruning'"))?
        .extract()?;
    let enable_lower_bound_pruning: bool = dict
        .get_item("enable_lower_bound_pruning")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'enable_lower_bound_pruning'"))?
        .extract()?;
    let lower_bound_cost_factor: f64 = dict
        .get_item("lower_bound_cost_factor")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'lower_bound_cost_factor'"))?
        .extract()?;
    let gpu_workers: Vec<usize> = dict
        .get_item("gpu_worker_indices")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'gpu_worker_indices'"))?
        .extract()?;
    let cpu_workers: Vec<usize> = dict
        .get_item("cpu_worker_indices")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'cpu_worker_indices'"))?
        .extract()?;
    let cache_multiplier_fn = dict
        .get_item("cache_multiplier_fn")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'cache_multiplier_fn'"))?
        .into_py(py);
    let id_to_sig_list: &PyList = dict
        .get_item("id_to_signature")
        .ok_or_else(|| PyRuntimeError::new_err("Missing 'id_to_signature'"))?
        .downcast()
        .map_err(|_| PyRuntimeError::new_err("'id_to_signature' must be a list"))?;
    let mut id_to_signature: Vec<Py<PyAny>> = Vec::with_capacity(id_to_sig_list.len());
    for item in id_to_sig_list.iter() {
        id_to_signature.push(item.into_py(py));
    }
    Ok(RustInput {
        nodes,
        workers,
        gpu_workers,
        cpu_workers,
        node_worker_options,
        gpu_node_indices: gpu_nodes,
        db_node_indices: db_nodes,
        parents_mask,
        gpu_parents_mask,
        exec_costs,
        model_init_costs,
        llm_bonus,
        node_min_cost,
        epoch_penalties,
        gpu_batch_factors,
        initial_states,
        window_size,
        none_id,
        raw_cost_scale,
        cpu_load_cost_weight,
        cpu_defer_weight,
        enable_batch_shape_pruning,
        gpu_batch_slack,
        enable_lower_bound_pruning,
        lower_bound_cost_factor,
        cache_multiplier_fn,
        id_to_signature,
    })
}

struct Solver<'py> {
    input: RustInput,
    all_mask: u64,
    memo: HashMap<SolverKey, (f64, Vec<(usize, usize, usize)>, HashMap<(usize, String), Py<PyAny>>)>,
    global_best: f64,
    window_cache: HashMap<(Vec<i32>, i32), f64>, // (window, sig_id) -> multiplier
    py: Python<'py>,
}

impl<'py> Solver<'py> {
    fn new(py: Python<'py>, input: RustInput) -> PyResult<Self> {
        if input.nodes.len() > 63 {
            return Err(PyRuntimeError::new_err(
                "Rust DP solver currently supports up to 63 nodes (bitmask).",
            ));
        }
        let all_mask = if input.nodes.is_empty() {
            0
        } else {
            (1u64 << input.nodes.len()) - 1
        };
        Ok(Self {
            input,
            all_mask,
            memo: HashMap::new(),
            global_best: f64::INFINITY,
            window_cache: HashMap::new(),
            py,
        })
    }

    fn solve(mut self) -> PyResult<(f64, Vec<(usize, String, String)>, HashMap<(String, String), Py<PyAny>>)> {
        let init_states = self.input.initial_states.clone();
        let (cost, schedule, plans) = self._solve(0, init_states, 0)?;
        // convert schedule to (epoch, worker_id, node_id)
        let mut schedule_out: Vec<(usize, String, String)> = Vec::with_capacity(schedule.len());
        for (epoch, wid_idx, node_idx) in schedule {
            let wid = self.input.workers[wid_idx].id.clone();
            let nid = self.input.nodes[node_idx].id.clone();
            schedule_out.push((epoch, wid, nid));
        }
        let mut plans_out: HashMap<(String, String), Py<PyAny>> = HashMap::with_capacity(plans.len());
        for ((node_idx, qname), choice) in plans {
            let nid = self.input.nodes[node_idx].id.clone();
            plans_out.insert((nid, qname), choice);
        }
        Ok((cost, schedule_out, plans_out))
    }

    fn _solve(
        &mut self,
        done_mask: u64,
        worker_states: Vec<WorkerState>,
        epoch_idx: usize,
    ) -> PyResult<(f64, Vec<(usize, usize, usize)>, HashMap<(usize, String), Py<PyAny>>)> {
        if done_mask == self.all_mask {
            return Ok((0.0, Vec::new(), HashMap::new()));
        }
        let key = SolverKey {
            done_mask,
            epoch_idx: epoch_idx as u32,
            workers: worker_states
                .iter()
                .map(|s| WorkerStateKey {
                    last_model: s.last_model,
                    last_node: s.last_node,
                })
                .collect(),
        };
        if let Some((cost, sched, plans)) = self.memo.get(&key) {
            return Ok((cost.clone(), sched.clone(), plans.clone()));
        }

        let mut best_cost = f64::INFINITY;
        let mut best_schedule: Vec<(usize, usize, usize)> = Vec::new();
        let mut best_plans: HashMap<(usize, String), Py<PyAny>> = HashMap::new();

        let gpu_batches = self.enumerate_gpu_batches(done_mask)?;

        for gpu_nodes in gpu_batches.iter() {
            let cpu_nodes = self.auto_cpu_batch(done_mask, gpu_nodes);
            if gpu_nodes.is_empty() && cpu_nodes.is_empty() {
                continue;
            }
            let mut combined: HashSet<usize> = HashSet::new();
            combined.extend(gpu_nodes.iter().copied());
            combined.extend(cpu_nodes.iter().copied());
            if !self.batch_feasible(&combined, done_mask) {
                continue;
            }
            let mut gpu_assignments = self.gpu_assignments(gpu_nodes);
            if self.input.enable_batch_shape_pruning {
                gpu_assignments = self.dedup_worker_assignments(gpu_assignments, &worker_states);
            }
            let cpu_assign = self.cpu_assignments(&cpu_nodes);
            for g_assign in gpu_assignments.iter() {
                if g_assign.is_empty() && cpu_assign.is_empty() {
                    continue;
                }
                for (gpu_cost, cpu_load_cost, next_states, batch_plans) in
                    self.assignment_outcomes(g_assign, &worker_states)?
                {
                    let mut new_done = done_mask;
                    for node_idx in combined.iter() {
                        new_done |= 1u64 << node_idx;
                    }
                    let epoch_penalty = if epoch_idx < self.input.epoch_penalties.len() {
                        self.input.epoch_penalties[epoch_idx]
                    } else {
                        *self.input.epoch_penalties.last().unwrap_or(&1.0)
                    };
                    let gpu_batch_size = g_assign.len();
                    let gpu_factor = if gpu_batch_size < self.input.gpu_batch_factors.len() {
                        self.input.gpu_batch_factors[gpu_batch_size]
                    } else {
                        *self.input.gpu_batch_factors.last().unwrap_or(&1.0)
                    };
                    let cpu_load_penalty_weight =
                        self.input.cpu_load_cost_weight + self.input.cpu_defer_weight;
                    let epoch_cost = epoch_penalty
                        + gpu_cost * gpu_factor
                        + cpu_load_penalty_weight * cpu_load_cost;
                    if self.input.enable_lower_bound_pruning && best_cost.is_finite() {
                        let lb = self.lower_bound_remaining(new_done);
                        if epoch_cost + lb >= best_cost {
                            continue;
                        }
                    }
                    let (sub_cost, sub_sched, sub_plans) =
                        self._solve(new_done, next_states.clone(), epoch_idx + 1)?;
                    let total = epoch_cost + sub_cost;
                    if total < best_cost {
                        best_cost = total;
                        // epoch 0 is current
                        let mut this_epoch: Vec<(usize, usize, usize)> =
                            g_assign.iter().map(|(w, n)| (epoch_idx, *w, *n)).collect();
                        for (w, n) in cpu_assign.iter() {
                            this_epoch.push((epoch_idx, *w, *n));
                        }
                        let mut shifted = Vec::with_capacity(this_epoch.len() + sub_sched.len());
                        shifted.append(&mut this_epoch);
                        for (e, w, n) in sub_sched {
                            shifted.push((e, w, n));
                        }
                        best_schedule = shifted;
                        let mut merged = sub_plans;
                        for ((nid, q), choice) in batch_plans.iter() {
                            merged.insert((*nid, q.clone()), choice.clone());
                        }
                        best_plans = merged;
                        if total < self.global_best {
                            self.global_best = total;
                        }
                    }
                }
            }
        }

        if !best_cost.is_finite() {
            return Err(PyRuntimeError::new_err(
                "Rust DP failed to find schedule (no feasible state)",
            ));
        }

        self.memo.insert(
            key,
            (best_cost, best_schedule.clone(), best_plans.clone()),
        );
        Ok((best_cost, best_schedule, best_plans))
    }

    fn is_done(&self, done_mask: u64, node_idx: usize) -> bool {
        (done_mask & (1u64 << node_idx)) != 0
    }

    fn is_gpu_node(&self, node_idx: usize) -> bool {
        self.input.nodes[node_idx].is_gpu
    }

    fn batch_feasible(&self, nodes: &HashSet<usize>, done_mask: u64) -> bool {
        let mut batch_mask = done_mask;
        for node_idx in nodes {
            batch_mask |= 1u64 << node_idx;
        }
        for node_idx in nodes {
            let pmask = self.input.parents_mask[*node_idx];
            if (pmask & !batch_mask) != 0 {
                return false;
            }
        }
        true
    }

    fn enumerate_gpu_batches(&self, done_mask: u64) -> PyResult<Vec<Vec<usize>>> {
        let max_gpu = std::cmp::max(1, self.input.gpu_workers.len());
        let mut pending: Vec<usize> = self
            .input
            .nodes
            .iter()
            .enumerate()
            .filter(|(idx, n)| !self.is_done(done_mask, *idx) && n.is_gpu)
            .map(|(idx, _)| idx)
            .collect();
        if pending.is_empty() {
            return Ok(vec![Vec::new()]);
        }

        let pending_set: HashSet<usize> = pending.iter().copied().collect();
        let mut in_deg: HashMap<usize, usize> = HashMap::new();
        for nid in pending.iter() {
            in_deg.insert(*nid, 0);
        }
        for nid in pending.iter() {
            let pmask = self.input.gpu_parents_mask[*nid];
            let mut deg = 0usize;
            for p in pending.iter() {
                let bit = 1u64 << p;
                if (pmask & bit) != 0 {
                    deg += 1;
                }
            }
            in_deg.insert(*nid, deg);
        }
        let mut queue: Vec<usize> = in_deg
            .iter()
            .filter_map(|(nid, deg)| if *deg == 0 { Some(*nid) } else { None })
            .collect();
        let mut topo: Vec<usize> = Vec::with_capacity(pending.len());
        while let Some(nid) = queue.pop() {
            topo.push(nid);
            for child in pending_set.iter() {
                let pmask = self.input.gpu_parents_mask[*child];
                if (pmask & (1u64 << nid)) != 0 {
                    let entry = in_deg.get_mut(child).unwrap();
                    if *entry > 0 {
                        *entry -= 1;
                        if *entry == 0 {
                            queue.push(*child);
                        }
                    }
                }
            }
        }
        if topo.len() != pending.len() {
            topo = pending.clone();
        }

        let target = std::cmp::min(max_gpu, pending.len());
        let mut min_r = 1usize;
        let mut max_r = std::cmp::min(max_gpu, pending.len());
        if self.input.enable_batch_shape_pruning {
            min_r = std::cmp::max(1, target.saturating_sub(self.input.gpu_batch_slack));
            max_r = target;
        }
        let mut results: Vec<Vec<usize>> = vec![Vec::new()];
        fn dfs(
            solver: &Solver,
            topo: &Vec<usize>,
            idx: usize,
            selected: &mut Vec<usize>,
            selected_mask: u64,
            done_mask: u64,
            min_r: usize,
            max_r: usize,
            results: &mut Vec<Vec<usize>>,
        ) {
            if idx >= topo.len() {
                let size = selected.len();
                if size >= min_r && size <= max_r {
                    results.push(selected.clone());
                }
                return;
            }
            dfs(
                solver,
                topo,
                idx + 1,
                selected,
                selected_mask,
                min_r,
                max_r,
                results,
            );
            if selected.len() >= max_r {
                return;
            }
            let node_idx = topo[idx];
            let gpu_parents_mask = solver.input.gpu_parents_mask[node_idx];
            if (gpu_parents_mask & !(selected_mask | done_mask)) != 0 {
                dfs(
                    solver,
                    topo,
                    idx + 1,
                    selected,
                    selected_mask,
                    done_mask,
                    min_r,
                    max_r,
                    results,
                );
                return;
            }
            selected.push(node_idx);
            dfs(
                solver,
                topo,
                idx + 1,
                selected,
                selected_mask | (1u64 << node_idx),
                done_mask,
                min_r,
                max_r,
                results,
            );
            selected.pop();
        }
        dfs(
            self,
            &topo,
            0,
            &mut Vec::new(),
            0,
            done_mask,
            min_r,
            max_r,
            &mut results,
        );
        Ok(results)
    }

    fn auto_cpu_batch(&self, done_mask: u64, gpu_nodes: &Vec<usize>) -> Vec<usize> {
        if self.input.db_node_indices.is_empty() {
            return Vec::new();
        }
        let mut batch_mask = done_mask;
        for node_idx in gpu_nodes.iter() {
            batch_mask |= 1u64 << node_idx;
        }
        let needed = self.cpu_needed_for_gpu(gpu_nodes);
        let mut pending: HashSet<usize> = self
            .input
            .db_node_indices
            .iter()
            .filter(|idx| !self.is_done(done_mask, **idx) && needed.contains(idx))
            .cloned()
            .collect();
        if pending.is_empty() {
            return Vec::new();
        }
        let mut selected: Vec<usize> = Vec::new();
        let mut progressed = true;
        while progressed {
            progressed = false;
            for node_idx in self.input.db_node_indices.iter() {
                if !pending.contains(node_idx) {
                    continue;
                }
                let pmask = self.input.parents_mask[*node_idx];
                if (pmask & !batch_mask) != 0 {
                    continue;
                }
                selected.push(*node_idx);
                pending.remove(node_idx);
                batch_mask |= 1u64 << node_idx;
                progressed = true;
            }
        }
        selected
    }

    fn cpu_needed_for_gpu(&self, gpu_nodes: &Vec<usize>) -> HashSet<usize> {
        let mut needed: HashSet<usize> = HashSet::new();
        if gpu_nodes.is_empty() {
            return needed;
        }
        let mut stack = gpu_nodes.clone();
        let mut visited: HashSet<usize> = HashSet::new();
        while let Some(node_idx) = stack.pop() {
            if !visited.insert(node_idx) {
                continue;
            }
            let mut pmask = self.input.parents_mask[node_idx];
            while pmask != 0 {
                let parent_idx = pmask.trailing_zeros() as usize;
                let bit = 1u64 << parent_idx;
                pmask &= !bit;
                if !visited.contains(&parent_idx) {
                    stack.push(parent_idx);
                }
                if !self.input.nodes[parent_idx].is_gpu {
                    needed.insert(parent_idx);
                }
            }
        }
        needed
    }

    fn cpu_assignments(&self, cpu_nodes: &Vec<usize>) -> Vec<(usize, usize)> {
        if cpu_nodes.is_empty() {
            return Vec::new();
        }
        let workers = if self.input.cpu_workers.is_empty() {
            &self.input.gpu_workers
        } else {
            &self.input.cpu_workers
        };
        if workers.is_empty() {
            return Vec::new();
        }
        let mut seq: Vec<(usize, usize)> = Vec::with_capacity(cpu_nodes.len());
        for (idx, node_idx) in cpu_nodes.iter().enumerate() {
            let wid = workers[idx % workers.len()];
            seq.push((wid, *node_idx));
        }
        seq
    }

    fn gpu_assignments(&self, gpu_nodes: &Vec<usize>) -> Vec<Vec<(usize, usize)>> {
        if gpu_nodes.is_empty() {
            return vec![Vec::new()];
        }
        let mut results: Vec<Vec<(usize, usize)>> = Vec::new();
        fn dfs(
            solver: &Solver,
            idx: usize,
            gpu_nodes: &Vec<usize>,
            used: &mut HashSet<usize>,
            current: &mut Vec<(usize, usize)>,
            results: &mut Vec<Vec<(usize, usize)>>,
        ) {
            if idx >= gpu_nodes.len() {
                results.push(current.clone());
                return;
            }
            let node_idx = gpu_nodes[idx];
            let allowed = &solver.input.node_worker_options[node_idx];
            for wid in solver.input.gpu_workers.iter() {
                if !allowed.contains(wid) {
                    continue;
                }
                if used.contains(wid) {
                    continue;
                }
                used.insert(*wid);
                current.push((*wid, node_idx));
                dfs(solver, idx + 1, gpu_nodes, used, current, results);
                current.pop();
                used.remove(wid);
            }
        }
        dfs(
            self,
            0,
            gpu_nodes,
            &mut HashSet::new(),
            &mut Vec::new(),
            &mut results,
        );
        results
    }

    fn worker_signature(&self, worker_idx: usize, state: &WorkerState) -> (u8, i32, i32) {
        let worker = &self.input.workers[worker_idx];
        let kind = match worker.kind {
            WorkerKind::Gpu => 1u8,
            WorkerKind::Cpu => 0u8,
        };
        (kind, state.last_model, state.last_node)
    }

    fn dedup_worker_assignments(
        &self,
        assignments: Vec<Vec<(usize, usize)>>,
        worker_states: &Vec<WorkerState>,
    ) -> Vec<Vec<(usize, usize)>> {
        let mut seen: HashSet<Vec<((u8, i32, i32), Vec<usize>)>> = HashSet::new();
        let mut uniq: Vec<Vec<(usize, usize)>> = Vec::new();
        for assign in assignments.into_iter() {
            let mut buckets: HashMap<(u8, i32, i32), Vec<usize>> = HashMap::new();
            for (wid, nid) in assign.iter() {
                let sig = self.worker_signature(*wid, &worker_states[*wid]);
                buckets.entry(sig).or_insert_with(Vec::new).push(*nid);
            }
            let mut items: Vec<((u8, i32, i32), Vec<usize>)> = buckets
                .into_iter()
                .map(|(sig, mut nodes)| {
                    nodes.sort_unstable();
                    (sig, nodes)
                })
                .collect();
            items.sort_by(|a, b| a.0.cmp(&b.0));
            if seen.insert(items) {
                uniq.push(assign);
            }
        }
        uniq
    }

    fn assignment_outcomes(
        &mut self,
        assign: &Vec<(usize, usize)>,
        worker_states: &Vec<WorkerState>,
    ) -> PyResult<Vec<(f64, f64, Vec<WorkerState>, HashMap<(usize, String), Py<PyAny>>)>> {
        let mut results: Vec<(f64, f64, Vec<WorkerState>, HashMap<(usize, String), Py<PyAny>>)> =
            Vec::new();
        fn dfs(
            solver: &mut Solver,
            assign: &Vec<(usize, usize)>,
            idx: usize,
            states: &mut Vec<WorkerState>,
            worker_costs: &mut Vec<f64>,
            acc_plans: &mut HashMap<(usize, String), Py<PyAny>>,
            cpu_load_cost: f64,
            window: Vec<i32>,
            results: &mut Vec<(f64, f64, Vec<WorkerState>, HashMap<(usize, String), Py<PyAny>>)>,
        ) -> PyResult<()> {
            if idx >= assign.len() {
                let gpu_costs: Vec<f64> = solver
                    .input
                    .gpu_workers
                    .iter()
                    .map(|wid| worker_costs[*wid])
                    .collect();
                let gpu_cost = gpu_costs.iter().cloned().fold(0.0, f64::max);
                results.push((gpu_cost, cpu_load_cost, states.clone(), acc_plans.clone()));
                return Ok(());
            }
            let (worker_idx, node_idx) = assign[idx];
            let state = states[worker_idx].clone();
            let node = &solver.input.nodes[node_idx];
            if node.is_gpu {
                let exec_cost = solver.input.exec_costs[node_idx][worker_idx];
                let model_cost_idx = to_offset(state.last_model, solver.input.none_id);
                let model_cost = if model_cost_idx < solver.input.model_init_costs[node_idx].len() {
                    solver.input.model_init_costs[node_idx][model_cost_idx]
                } else {
                    *solver.input.model_init_costs[node_idx].last().unwrap_or(&0.0)
                };
                let bonus_idx = to_offset(state.last_node, solver.input.none_id);
                let bonus = if bonus_idx < solver.input.llm_bonus[node_idx].len() {
                    solver.input.llm_bonus[node_idx][bonus_idx]
                } else {
                    1.0
                };
                let base_cost = exec_cost * bonus + model_cost;
                if node.queries.is_empty() {
                    states[worker_idx].last_model = node.model_id;
                    states[worker_idx].last_node = node_idx as i32;
                    worker_costs[worker_idx] += base_cost;
                    dfs(
                        solver,
                        assign,
                        idx + 1,
                        states,
                        worker_costs,
                        acc_plans,
                        cpu_load_cost,
                        window.clone(),
                        results,
                    )?;
                    // backtrack
                    states[worker_idx] = state;
                    worker_costs[worker_idx] -= base_cost;
                    return Ok(());
                }
                let options = solver.query_plan_options(node_idx, &window)?;
                for (q_cost, exit_window, plan_seq) in options {
                    states[worker_idx].last_model = node.model_id;
                    states[worker_idx].last_node = node_idx as i32;
                    let prev_cost = worker_costs[worker_idx];
                    worker_costs[worker_idx] += base_cost;
                    for (qname, choice) in plan_seq.iter() {
                        acc_plans.insert((node_idx, qname.clone()), choice.clone());
                    }
                    dfs(
                        solver,
                        assign,
                        idx + 1,
                        states,
                        worker_costs,
                        acc_plans,
                        cpu_load_cost + q_cost,
                        exit_window,
                        results,
                    )?;
                    for (qname, _) in plan_seq.iter() {
                        acc_plans.remove(&(node_idx, qname.clone()));
                    }
                    worker_costs[worker_idx] = prev_cost;
                    states[worker_idx] = state.clone();
                }
                return Ok(());
            }
            // CPU node: cost ignored
            dfs(
                solver,
                assign,
                idx + 1,
                states,
                worker_costs,
                acc_plans,
                cpu_load_cost,
                window,
                results,
            )?;
            Ok(())
        }
        dfs(
            self,
            assign,
            0,
            &mut worker_states.clone(),
            &mut vec![0.0f64; self.input.workers.len()],
            &mut HashMap::new(),
            0.0,
            Vec::new(),
            &mut results,
        )?;
        Ok(results)
    }

    fn append_window(&self, window: &Vec<i32>, sig_id: i32) -> Vec<i32> {
        if self.input.window_size == 0 {
            return window.clone();
        }
        if window.is_empty() {
            if self.input.window_size == 1 {
                return vec![sig_id];
            }
            let mut res = vec![self.input.none_id; self.input.window_size - 1];
            res.push(sig_id);
            return res;
        }
        let mut new_window = window.clone();
        new_window.push(sig_id);
        if new_window.len() > self.input.window_size {
            new_window = new_window[new_window.len() - self.input.window_size..].to_vec();
        }
        new_window
    }

    fn window_signatures(&mut self, window: &Vec<i32>) -> PyResult<Py<PyAny>> {
        let mut items: Vec<PyObject> = Vec::new();
        for sig_id in window.iter() {
            if *sig_id == self.input.none_id {
                continue;
            }
            let idx = *sig_id as usize;
            if idx < self.input.id_to_signature.len() {
                items.push(self.input.id_to_signature[idx].clone().into_py(self.py));
            }
        }
        let tuple = PyTuple::new(self.py, items);
        Ok(tuple.into_py(self.py))
    }

    fn cache_multiplier(&mut self, window: &Vec<i32>, choice: &PlanChoice) -> PyResult<f64> {
        let key = (window.clone(), choice.sig_id);
        if let Some(val) = self.window_cache.get(&key) {
            return Ok(*val);
        }
        let window_obj = self.window_signatures(window)?;
        let choice_obj = choice.choice_obj.as_ref(self.py);
        let mult_obj = self
            .input
            .cache_multiplier_fn
            .call1(self.py, (window_obj, choice_obj))?;
        let mult: f64 = mult_obj.extract(self.py)?;
        self.window_cache.insert((window.clone(), choice.sig_id), mult);
        Ok(mult)
    }

    fn query_plan_options(
        &mut self,
        node_idx: usize,
        enter_window: &Vec<i32>,
    ) -> PyResult<Vec<(f64, Vec<i32>, Vec<(String, Py<PyAny>)>)>> {
        let node = &self.input.nodes[node_idx];
        if node.queries.is_empty() {
            return Ok(vec![(0.0, enter_window.clone(), Vec::new())]);
        }
        let mut outcomes: Vec<(f64, Vec<i32>, Vec<(String, Py<PyAny>)>)> =
            vec![(0.0, enter_window.clone(), Vec::new())];
        for query in node.queries.iter() {
            let mut next_outcomes: Vec<(f64, Vec<i32>, Vec<(String, Py<PyAny>)>)> =
                Vec::new();
            for (base_cost, window, plan_seq) in outcomes.into_iter() {
                for choice in query.plans.iter() {
                    let mult = self.cache_multiplier(&window, choice)?;
                    let q_cost = choice.base_cost * mult;
                    let next_window = self.append_window(&window, choice.sig_id);
                    let mut new_seq = plan_seq.clone();
                    new_seq.push((query.name.clone(), choice.choice_obj.clone()));
                    next_outcomes.push((base_cost + q_cost, next_window, new_seq));
                }
            }
            outcomes = next_outcomes;
        }
        Ok(outcomes)
    }

    fn lower_bound_remaining(&self, done_mask: u64) -> f64 {
        if !self.input.enable_lower_bound_pruning {
            return 0.0;
        }
        let remaining_gpu: Vec<usize> = self
            .input
            .gpu_node_indices
            .iter()
            .filter(|idx| !self.is_done(done_mask, **idx))
            .cloned()
            .collect();
        let remaining_cpu: Vec<usize> = self
            .input
            .db_node_indices
            .iter()
            .filter(|idx| !self.is_done(done_mask, **idx))
            .cloned()
            .collect();
        if remaining_gpu.is_empty() && remaining_cpu.is_empty() {
            return 0.0;
        }
        let max_gpu_per_epoch = std::cmp::max(1, self.input.gpu_workers.len());
        let min_epoch_gpu = if remaining_gpu.is_empty() {
            0
        } else {
            (remaining_gpu.len() + max_gpu_per_epoch - 1) / max_gpu_per_epoch
        };
        let min_epoch_cpu = if remaining_cpu.is_empty() { 0 } else { 1 };
        let min_epochs = std::cmp::max(min_epoch_gpu, min_epoch_cpu);
        let gpu_min_sum: f64 = remaining_gpu
            .iter()
            .map(|idx| self.input.node_min_cost[*idx])
            .sum();
        let gpu_capacity = std::cmp::max(1, self.input.gpu_workers.len());
        let load_lb = if gpu_min_sum > 0.0 {
            gpu_min_sum / (gpu_capacity as f64)
        } else {
            0.0
        };
        let mut penalty_lb = 0.0;
        for i in 0..min_epochs {
            if i < self.input.epoch_penalties.len() {
                penalty_lb += self.input.epoch_penalties[i];
            } else if let Some(last) = self.input.epoch_penalties.last() {
                penalty_lb += *last;
            }
        }
        penalty_lb + self.input.lower_bound_cost_factor * load_lb
    }
}

#[pyfunction]
fn solve(py: Python<'_>, input: &PyAny) -> PyResult<(f64, Vec<(usize, String, String)>, HashMap<(String, String), Py<PyAny>>)> {
    let dict: &PyDict = input.downcast().map_err(|_| {
        PyRuntimeError::new_err("Rust DP solver expects a dict input")
    })?;
    let parsed = parse_input(py, dict)?;
    let solver = Solver::new(py, parsed)?;
    solver.solve()
}

#[pymodule]
fn _dp_core_rs(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
