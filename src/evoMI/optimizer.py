import os
import gc
import json
import time
import re
import glob
import shutil
import colorsys
import concurrent.futures
import threading
import inspect
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.fit import fit_gpytorch_mll as botorch_fit_gpytorch_mll
from src.evoMI.runtime_artifacts import build_eval_metadata


def _task_color(task_type, task_key=None):
    if str(task_type) != "full":
        return "#2ca02c"
    if task_key is None:
        return "#2ca02c"
    key = int(task_key)
    hue = (key * 0.618033988749895) % 1.0
    sat = 0.55 + 0.10 * (key % 3)
    val = 0.80 + 0.08 * (key % 2)
    return colorsys.hsv_to_rgb(min(max(hue, 0.0), 1.0), min(sat, 0.95), min(val, 0.98))


def _normalize_score_tensor(values):
    if values.numel() == 0:
        return values
    finite_mask = torch.isfinite(values)
    if not finite_mask.any():
        return torch.full_like(values, 0.5)
    finite_vals = values[finite_mask]
    min_v = finite_vals.min()
    max_v = finite_vals.max()
    normalized = torch.full_like(values, 0.5)
    if (max_v - min_v).abs().item() < 1e-8:
        normalized[finite_mask] = 0.5
        return normalized
    normalized[finite_mask] = (finite_vals - min_v) / (max_v - min_v)
    return torch.clamp(normalized, 0.0, 1.0)


def _is_non_dominated(points):
    n = points.shape[0]
    mask = torch.ones(n, dtype=torch.bool, device=points.device)
    for i in range(n):
        if not mask[i]:
            continue
        dominates_i = torch.all(points >= points[i], dim=1) & torch.any(points > points[i], dim=1)
        dominates_i[i] = False
        if torch.any(dominates_i):
            mask[i] = False
    return mask


def _compute_gap_reward(candidate_obj_mean, observed_obj):
    if observed_obj is None or observed_obj.shape[0] < 2 or candidate_obj_mean.shape[0] == 0:
        return torch.zeros(candidate_obj_mean.shape[0], device=candidate_obj_mean.device, dtype=candidate_obj_mean.dtype)
    pareto_mask = _is_non_dominated(observed_obj)
    pareto_obj = observed_obj[pareto_mask]
    if pareto_obj.shape[0] < 2:
        return torch.zeros(candidate_obj_mean.shape[0], device=candidate_obj_mean.device, dtype=candidate_obj_mean.dtype)
    order = torch.argsort(pareto_obj[:, 0])
    pareto_obj = pareto_obj[order]
    gap_vectors = pareto_obj[1:] - pareto_obj[:-1]
    gap_sizes = torch.linalg.norm(gap_vectors, dim=1)
    if gap_sizes.numel() == 0:
        return torch.zeros(candidate_obj_mean.shape[0], device=candidate_obj_mean.device, dtype=candidate_obj_mean.dtype)
    gap_midpoints = 0.5 * (pareto_obj[1:] + pareto_obj[:-1])
    obj_range = torch.clamp(pareto_obj.max(dim=0).values - pareto_obj.min(dim=0).values, min=1e-6)
    sigma = torch.clamp(torch.linalg.norm(obj_range) / max(1, pareto_obj.shape[0] - 1), min=1e-3)
    distances = torch.cdist(candidate_obj_mean, gap_midpoints)
    gap_weights = gap_sizes / torch.clamp(gap_sizes.max(), min=1e-8)
    reward_matrix = torch.exp(-0.5 * (distances / sigma) ** 2) * gap_weights.unsqueeze(0)
    return reward_matrix.max(dim=1).values


def _compute_proximity_penalty(candidates, avoid_points):
    if avoid_points is None or avoid_points.numel() == 0 or candidates.shape[0] == 0:
        return torch.zeros(candidates.shape[0], device=candidates.device, dtype=candidates.dtype)
    min_distances = torch.cdist(candidates, avoid_points).min(dim=1).values
    return torch.exp(-4.0 * min_distances)


def _select_gap_aware_indices(
    candidate_repr,
    candidate_values,
    candidate_obj_mean,
    observed_obj,
    q,
    pending_repr=None,
    gap_reward_weight=0.25,
    pending_penalty_weight=0.15,
):
    total = int(candidate_repr.shape[0])
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=candidate_repr.device), []
    select_count = min(int(max(1, q)), total)
    base_scores = _normalize_score_tensor(candidate_values.reshape(-1))
    gap_scores = _normalize_score_tensor(_compute_gap_reward(candidate_obj_mean, observed_obj))
    remaining = list(range(total))
    selected = []
    selected_repr = None
    selection_details = []
    for selection_rank in range(select_count):
        idx_tensor = torch.as_tensor(remaining, dtype=torch.long, device=candidate_repr.device)
        current_base_scores = base_scores[idx_tensor]
        current_gap_scores = gap_scores[idx_tensor]
        proximity_penalty = torch.zeros_like(current_base_scores)
        joint_score = current_base_scores + float(gap_reward_weight) * current_gap_scores
        avoid_points = []
        if pending_repr is not None and pending_repr.numel() > 0:
            avoid_points.append(pending_repr)
        if selected_repr is not None and selected_repr.numel() > 0:
            avoid_points.append(selected_repr)
        if len(avoid_points) > 0 and float(pending_penalty_weight) > 0:
            avoid_tensor = torch.cat(avoid_points, dim=0)
            proximity_penalty = _normalize_score_tensor(
                _compute_proximity_penalty(candidate_repr[idx_tensor], avoid_tensor)
            )
            joint_score = joint_score - float(pending_penalty_weight) * proximity_penalty
        best_local = int(torch.argmax(joint_score).item())
        best_global = remaining.pop(best_local)
        selected.append(best_global)
        selection_details.append(
            {
                "selection_rank": int(selection_rank),
                "candidate_index": int(best_global),
                "base_score": float(current_base_scores[best_local].item()),
                "gap_score": float(current_gap_scores[best_local].item()),
                "pending_penalty": float(proximity_penalty[best_local].item()),
                "final_score": float(joint_score[best_local].item()),
            }
        )
        best_repr = candidate_repr[best_global : best_global + 1]
        selected_repr = best_repr if selected_repr is None else torch.cat([selected_repr, best_repr], dim=0)
    return torch.as_tensor(selected, dtype=torch.long, device=candidate_repr.device), selection_details


def prior_bo_optimizer(
    objective_function,
    dim=3,
    num_objectives=2,
    bounds=None,
    BATCH_SIZE=5,
    NUM_RESTARTS=20,
    RAW_SAMPLES=512,
    MC_SAMPLES=128,
    N_BATCH=40,
    verbose=True,
    device="cpu",
    dtype=torch.double,
    initial_samples=10,
    noise_level=0.01,
    iteration_callback=None,
    ref_point=-1.1,
    run_id=None,
    checkpoint_dir="./checkpoints",
    custom_initial_solutions=None,
    seed=42,
    m_prior=None,
    u_prior=None,
    rho=0.5,
    topk=6,
    n_groups=4,
    enable_grouping=True,
    scheduler_gpu_count=4,
    max_evaluations=None,
    async_mode=False,
    wait_for_completion_threshold=0.15,
    enable_gap_aware_postprocess=False,
    gap_reward_weight=0.25,
    gap_pending_penalty_weight=0.15,
    gap_candidate_pool_multiplier=3,
    full_eval_limits=None,
    eval_profile="aime_gpqa",
    eval_repeats=None,
    eval_setting_id=None,
    eval_metadata=None,
    shared_initial_dataset=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tkwargs = {"dtype": dtype, "device": torch.device(device)}

    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_dir = os.path.join(checkpoint_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    if bounds is None:
        bounds = torch.zeros(2, dim, **tkwargs)
        bounds[1] = 1
    else:
        bounds = bounds.to(**tkwargs)

    z_bounds = torch.full((2, dim), -1.0, **tkwargs)
    z_bounds[1] = 1.0

    standard_bounds = torch.zeros(2, dim, **tkwargs)
    standard_bounds[1] = 1

    NOISE_SE = torch.full((num_objectives,), noise_level, **tkwargs)

    if type(ref_point) == torch.Tensor:
        problem_ref_point = ref_point.to(**tkwargs)
    else:
        problem_ref_point = torch.full((num_objectives,), ref_point, **tkwargs)

    lower = bounds[0]
    upper = bounds[1]
    span = (upper - lower).clamp_min(1e-8)

    if m_prior is None:
        m_prior_t = (lower + upper) * 0.5
    else:
        m_prior_t = torch.tensor(m_prior, **tkwargs).flatten()
        if m_prior_t.numel() != dim:
            raise ValueError(f"m_prior长度应为{dim}，当前为{m_prior_t.numel()}")
        m_prior_t = m_prior_t.clamp(lower, upper)

    if u_prior is None:
        u_prior_t = span * 0.5
        importance_for_group = torch.ones(dim, **tkwargs)
    else:
        u_prior_raw = torch.tensor(u_prior, **tkwargs).flatten()
        if u_prior_raw.numel() != dim:
            raise ValueError(f"u_prior长度应为{dim}，当前为{u_prior_raw.numel()}")
        if torch.min(u_prior_raw) >= 0 and torch.max(u_prior_raw) <= 1.0:
            u_prior_t = (u_prior_raw * span).clamp_min(1e-8)
        else:
            u_prior_t = u_prior_raw.clamp_min(1e-8)
        importance_for_group = u_prior_raw

    if enable_grouping:
        topk = max(1, min(topk, dim))
        sorted_idx = torch.argsort(importance_for_group, descending=True).cpu().numpy().tolist()
        top_layers = sorted_idx[:topk]
        rest = sorted_idx[topk:]
        groups = [g.tolist() for g in np.array_split(np.array(rest, dtype=np.int64), n_groups) if len(g) > 0]
    else:
        if u_prior is None:
            top_layers = list(range(dim))
        else:
            top_layers = torch.argsort(importance_for_group, descending=True).cpu().numpy().tolist()
        groups = []

    z_dim = len(top_layers) + len(groups)
    z_bounds_eff = torch.full((2, z_dim), -1.0, **tkwargs)
    z_bounds_eff[1] = 1.0
    standard_bounds_eff = torch.zeros(2, z_dim, **tkwargs)
    standard_bounds_eff[1] = 1.0

    def decode_z_to_x(z_batch):
        n = z_batch.shape[0]
        x = m_prior_t.unsqueeze(0).repeat(n, 1)
        ptr = 0
        for l in top_layers:
            x[:, l] = m_prior_t[l] + rho * u_prior_t[l] * z_batch[:, ptr]
            ptr += 1
        for grp in groups:
            zg = z_batch[:, ptr].unsqueeze(1)
            idx = torch.tensor(grp, device=z_batch.device, dtype=torch.long)
            x[:, idx] = m_prior_t[idx].unsqueeze(0) + rho * u_prior_t[idx].unsqueeze(0) * zg
            ptr += 1
        return x.clamp(lower.unsqueeze(0), upper.unsqueeze(0))

    def encode_x_to_z(x_batch):
        z = torch.zeros((x_batch.shape[0], z_dim), **tkwargs)
        ptr = 0
        denom = (rho * u_prior_t).clamp_min(1e-8)
        for l in top_layers:
            z[:, ptr] = ((x_batch[:, l] - m_prior_t[l]) / denom[l]).clamp(-1, 1)
            ptr += 1
        for grp in groups:
            idx = torch.tensor(grp, device=x_batch.device, dtype=torch.long)
            z_grp = ((x_batch[:, idx] - m_prior_t[idx].unsqueeze(0)) / denom[idx].unsqueeze(0)).mean(dim=1)
            z[:, ptr] = z_grp.clamp(-1, 1)
            ptr += 1
        return z

    if full_eval_limits is None:
        full_eval_limits = {"aime25": 5, "gpqa_diamond": 60}
    if eval_metadata is None:
        eval_metadata = build_eval_metadata(
            eval_profile=eval_profile,
            eval_limits=full_eval_limits,
            eval_repeats=eval_repeats,
            eval_setting_id=eval_setting_id,
        )
    if max_evaluations is None:
        max_evaluations = int(initial_samples + N_BATCH * BATCH_SIZE)
    objective_signature = inspect.signature(objective_function)
    objective_supports_estimated_tokens = "estimated_tokens" in objective_signature.parameters
    idle_gpu_count_fn = getattr(objective_function, "get_idle_gpu_count", None)

    def evaluate_from_z(z_batch, eval_limit=None, eval_mode="full", estimated_tokens=None):
        def collect_cleanup_paths(info_list):
            cleanup = []
            if not isinstance(info_list, list):
                return cleanup
            for item in info_list:
                if not isinstance(item, dict):
                    continue
                item_cleanup = item.pop("_cleanup_model_dirs", [])
                if isinstance(item_cleanup, list):
                    cleanup.extend([p for p in item_cleanup if isinstance(p, str) and len(p) > 0])
            if len(cleanup) == 0:
                return cleanup
            return list(dict.fromkeys(cleanup))

        x_batch = decode_z_to_x(z_batch)
        if objective_supports_estimated_tokens:
            result = objective_function(x_batch, eval_limit=eval_limit, eval_mode=eval_mode, estimated_tokens=estimated_tokens)
        else:
            result = objective_function(x_batch, eval_limit=eval_limit, eval_mode=eval_mode)
        cleanup_paths = []
        if isinstance(result, tuple) and len(result) == 3:
            obj_true, info, cleanup_paths = result
        elif isinstance(result, tuple) and len(result) == 2:
            obj_true, info = result
        else:
            obj_true = result
            info = [{} for _ in range(z_batch.shape[0])]
        if not isinstance(cleanup_paths, list):
            cleanup_paths = []
        cleanup_paths.extend(collect_cleanup_paths(info))
        if len(cleanup_paths) > 0:
            cleanup_paths = list(dict.fromkeys(cleanup_paths))
        obj_true = obj_true.to(**tkwargs)
        noise_scale = NOISE_SE.to(device=obj_true.device, dtype=obj_true.dtype)
        obj = obj_true + torch.randn_like(obj_true) * noise_scale
        return x_batch, obj, obj_true, info, cleanup_paths

    def save_scheduler_reports():
        if len(hv_curve) > 0:
            hv_curve_path = os.path.join(run_dir, "hypervolume_curve.json")
            with open(hv_curve_path, "w") as f:
                json.dump(hv_curve, f, indent=2)
            hv_x = [int(item["evaluations"]) for item in hv_curve]
            hv_y = [float(item["hypervolume"]) for item in hv_curve]
            fig_hv, ax_hv = plt.subplots(figsize=(10, 4))
            ax_hv.plot(hv_x, hv_y, marker="o")
            ax_hv.set_xlabel("Evaluations")
            ax_hv.set_ylabel("Hypervolume")
            ax_hv.set_title("Hypervolume Curve")
            ax_hv.grid(True, alpha=0.3)
            fig_hv.tight_layout()
            fig_hv.savefig(os.path.join(run_dir, "hypervolume_curve.png"), dpi=160)
            plt.close(fig_hv)
        if len(scheduler_history) == 0:
            return
        report_path = os.path.join(run_dir, "scheduler_usage.json")
        with open(report_path, "w") as f:
            json.dump(scheduler_history, f, indent=2)
        iterations = [int(rec["iteration"]) for rec in scheduler_history]
        wall_times = [float(rec["wall_time_sec"]) for rec in scheduler_history]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(iterations, wall_times, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Wall Time (s)")
        ax.set_title("Iteration Wall Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "iteration_wall_time.png"), dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 5))
        full_labeled = False
        idle_labeled = False

        def is_initialization_record(record):
            if not isinstance(record, dict):
                return False
            mode = str(record.get("mode", "")).lower()
            if mode == "init":
                return True
            iteration = record.get("iteration", None)
            return isinstance(iteration, int) and int(iteration) == 0

        plot_scheduler_history = [record for record in scheduler_history if not is_initialization_record(record)]
        init_task_count = sum(
            len(record.get("tasks", [])) for record in scheduler_history if is_initialization_record(record)
        )
        total_task_count = sum(
            1
            for record in scheduler_history
            for task in record.get("tasks", [])
            if isinstance(task, dict)
        )
        scheduler_task_count = sum(
            1
            for record in plot_scheduler_history
            for task in record.get("tasks", [])
            if isinstance(task, dict)
        )

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        global_real_stats_path = os.path.join(project_root, "output", "vllm_gpu_usage_stats.json")
        run_real_stats_path = os.path.join(run_dir, "vllm_gpu_usage_stats.json")
        if os.path.exists(global_real_stats_path):
            try:
                shutil.copy2(global_real_stats_path, run_real_stats_path)
            except Exception:
                pass
        real_stats_candidates = [
            run_real_stats_path,
            global_real_stats_path,
        ]
        log_path_candidates = []
        exact_log_path = os.path.join(project_root, "output", f"{run_id}.log")
        if os.path.exists(exact_log_path):
            log_path_candidates.append(exact_log_path)
        for pattern in [
            os.path.join(project_root, "output", f"*{run_id}*.log"),
            os.path.join(project_root, "output", f"*{run_id}*nohup*.log"),
        ]:
            for candidate in sorted(glob.glob(pattern)):
                if candidate not in log_path_candidates:
                    log_path_candidates.append(candidate)
        run_label = os.path.basename(run_dir)
        run_label_prefix = f"{run_id}_"
        if run_label.startswith(run_label_prefix):
            run_label = run_label[len(run_label_prefix) :]
        required_real_runtime_tasks = (
            max(1, int(np.ceil(float(scheduler_task_count) * 0.8))) if scheduler_task_count > 0 else 1
        )

        def infer_checkpoint_time_window():
            candidates = []
            for pattern in ["checkpoint_iter_*.json", "checkpoint_iter_*.pt", "checkpoint_latest.pt"]:
                candidates.extend(glob.glob(os.path.join(run_dir, pattern)))
            mtimes = []
            for path in candidates:
                try:
                    mtimes.append(float(os.path.getmtime(path)))
                except Exception:
                    continue
            if len(mtimes) < 2:
                return None
            mtimes.sort()
            if len(mtimes) >= 10:
                start = float(mtimes[int(len(mtimes) * 0.1)])
                end = float(mtimes[int(len(mtimes) * 0.9)])
            else:
                start = float(mtimes[0])
                end = float(mtimes[-1])
            span = float(end - start)
            if not np.isfinite(span) or span <= 0.0:
                return None
            return {"start": start, "end": end, "span": span}

        checkpoint_window = infer_checkpoint_time_window()

        def build_reconstructed_runtime_payload(tasks, source_path):
            normalized_tasks = sorted(
                [
                    task
                    for task in tasks
                    if isinstance(task, dict)
                    and isinstance(task.get("start"), (int, float))
                    and isinstance(task.get("end"), (int, float))
                    and isinstance(task.get("duration"), (int, float))
                ],
                key=lambda item: (float(item["start"]), int(item["gpu"])),
            )
            if len(normalized_tasks) == 0:
                return None
            available_gpus = sorted(
                {
                    int(task.get("gpu", 0))
                    for record in plot_scheduler_history
                    for task in record.get("tasks", [])
                    if isinstance(task, dict)
                }
                | {int(task.get("gpu", 0)) for task in normalized_tasks}
            )
            if len(available_gpus) == 0:
                available_gpus = sorted({int(task.get("gpu", 0)) for task in normalized_tasks})
            series_start = min(float(task["start"]) for task in normalized_tasks)
            series_end = max(float(task["end"]) for task in normalized_tasks)
            window_sec = max(series_end - series_start, 1e-9)
            gpu_busy = {gpu_id: 0.0 for gpu_id in available_gpus}
            gpu_count = {gpu_id: 0 for gpu_id in available_gpus}
            gpu_task_records = []
            task_runtime_records = []
            for idx, task in enumerate(normalized_tasks):
                gpu_id = int(task["gpu"])
                start_ts = float(task["start"])
                end_ts = float(task["end"])
                duration = max(float(task["duration"]), 0.0)
                if gpu_id not in gpu_busy:
                    gpu_busy[gpu_id] = 0.0
                    gpu_count[gpu_id] = 0
                gpu_busy[gpu_id] += duration
                gpu_count[gpu_id] += 1
                task_id = str(
                    task.get("task_id")
                    or (
                        f"task_{task['model_id']}"
                        if isinstance(task.get("model_id"), str) and len(str(task["model_id"])) > 0
                        else f"reconstructed_task_{idx:04d}"
                    )
                )
                server_port = int(task["gpu"]) if isinstance(task.get("gpu"), (int, float)) else -1
                task_runtime_records.append(
                    {
                        "task_id": task_id,
                        "status": "completed",
                        "gpu_ids": [gpu_id],
                        "start_time": start_ts,
                        "end_time": end_ts,
                        "runtime_sec": duration,
                        "server_port": server_port,
                        "estimated_tokens": float(task.get("estimated_tokens", 0.0)),
                    }
                )
                gpu_task_records.append(
                    {
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "status": "completed",
                        "start_time": start_ts,
                        "end_time": end_ts,
                        "runtime_sec": duration,
                        "server_port": server_port,
                    }
                )
            gpu_summary = []
            for gpu_id in available_gpus:
                busy = float(gpu_busy.get(gpu_id, 0.0))
                gpu_summary.append(
                    {
                        "gpu_id": int(gpu_id),
                        "busy_time_sec": busy,
                        "idle_time_sec": max(window_sec - busy, 0.0),
                        "utilization": max(0.0, min(1.0, busy / window_sec)),
                        "task_count": int(gpu_count.get(gpu_id, 0)),
                    }
                )
            return {
                "series_start_time": float(series_start),
                "series_end_time": float(series_end),
                "series_window_sec": float(window_sec),
                "gpu_summary": gpu_summary,
                "task_runtime_records": task_runtime_records,
                "gpu_task_records": gpu_task_records,
                "reconstructed_from": str(source_path),
            }

        def persist_reconstructed_runtime(tasks, source_path):
            if len(tasks) < required_real_runtime_tasks:
                return
            existing_count = 0
            if os.path.exists(run_real_stats_path):
                try:
                    with open(run_real_stats_path, "r", encoding="utf-8") as f:
                        existing_payload = json.load(f)
                    existing_records = existing_payload.get("task_runtime_records", [])
                    if isinstance(existing_records, list):
                        existing_count = len(existing_records)
                except Exception:
                    existing_count = 0
            if existing_count >= len(tasks):
                return
            payload = build_reconstructed_runtime_payload(tasks, source_path)
            if payload is None:
                return
            with open(run_real_stats_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        def load_real_runtime_tasks():
            candidate_results = []
            flattened_scheduler_tasks = []
            expected_candidate_sequence = []

            def normalize_real_tasks(tasks):
                if not isinstance(tasks, list) or len(tasks) == 0:
                    return []
                normalized = []
                last_end_by_gpu = {}
                for task in sorted(tasks, key=lambda x: (float(x["start"]), int(x["gpu"]))):
                    gpu_idx = int(task["gpu"])
                    start_ts = float(task["start"])
                    end_ts = float(task["end"])
                    duration = max(float(task["duration"]), 0.0)
                    if duration <= 0.0 or end_ts <= start_ts:
                        continue
                    prev_end = last_end_by_gpu.get(gpu_idx, None)
                    if prev_end is not None and start_ts < prev_end:
                        shift = prev_end - start_ts
                        start_ts = prev_end
                        end_ts = max(end_ts + shift, start_ts)
                        duration = max(end_ts - start_ts, 1e-6)
                    normalized.append(
                        {
                            "gpu": gpu_idx,
                            "type": str(task.get("type", "full")),
                            "start": start_ts,
                            "duration": duration,
                            "end": end_ts,
                            "task_id": task.get("task_id", None),
                            "model_id": task.get("model_id", None),
                            "candidate_index": task.get("candidate_index", None),
                            "estimated_tokens": float(task.get("estimated_tokens", 0.0)),
                        }
                    )
                    last_end_by_gpu[gpu_idx] = end_ts
                if scheduler_task_count > 0 and len(normalized) > scheduler_task_count:
                    normalized = normalized[-scheduler_task_count:]
                return normalized

            def add_candidate_result(tasks, source_path, source_rank):
                normalized_tasks = normalize_real_tasks(tasks)
                if len(normalized_tasks) == 0:
                    return
                overlap_count = 0
                last_end_by_gpu = {}
                for task in normalized_tasks:
                    gpu_idx = int(task["gpu"])
                    start_ts = float(task["start"])
                    prev_end = last_end_by_gpu.get(gpu_idx, None)
                    if prev_end is not None and start_ts < prev_end - 1e-6:
                        overlap_count += 1
                    last_end_by_gpu[gpu_idx] = max(float(task["end"]), float(prev_end)) if prev_end is not None else float(task["end"])
                count_gap = abs(len(normalized_tasks) - scheduler_task_count) if scheduler_task_count > 0 else 0
                enough_tasks = 0 if len(normalized_tasks) >= required_real_runtime_tasks else 1
                candidate_results.append(
                    (
                        (enough_tasks, overlap_count, count_gap, source_rank, -len(normalized_tasks)),
                        normalized_tasks,
                        source_path,
                    )
                )

            for rec in plot_scheduler_history:
                for task in rec.get("tasks", []):
                    if not isinstance(task, dict):
                        continue
                    candidate_index = task.get("candidate_index", None)
                    if not isinstance(candidate_index, int):
                        continue
                    flattened_scheduler_tasks.append(
                        {
                            "gpu": int(task.get("gpu", 0)),
                            "candidate_index": int(candidate_index),
                        }
                    )
                    expected_candidate_sequence.append(int(candidate_index))
            for real_stats_path in real_stats_candidates:
                if not os.path.exists(real_stats_path):
                    continue
                try:
                    with open(real_stats_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    continue
                records = payload.get("task_runtime_records", [])
                if not isinstance(records, list):
                    continue
                tasks = []
                for task_order, rec in enumerate(records):
                    if not isinstance(rec, dict):
                        continue
                    start_ts = rec.get("start_time", None)
                    end_ts = rec.get("end_time", None)
                    gpu_ids = rec.get("gpu_ids", [])
                    if not isinstance(start_ts, (int, float)) or not isinstance(end_ts, (int, float)):
                        continue
                    if not isinstance(gpu_ids, list) or len(gpu_ids) == 0:
                        continue
                    duration = float(end_ts) - float(start_ts)
                    if duration <= 0:
                        continue
                    tasks.append(
                        {
                            "gpu": int(gpu_ids[0]),
                            "type": "full",
                            "start": float(start_ts),
                            "duration": float(duration),
                            "end": float(end_ts),
                            "task_id": rec.get("task_id", None),
                            "estimated_tokens": float(rec.get("estimated_tokens", 0.0)),
                            "task_key": int(task_order),
                        }
                    )
                if len(tasks) == 0:
                    continue
                min_start = min(task["start"] for task in tasks)
                max_end = max(task["end"] for task in tasks)
                epoch_based = float(min_start) > 1e9 and float(max_end) > 1e9
                payload_window = payload.get("series_window_sec", None)
                if not isinstance(payload_window, (int, float)) or not np.isfinite(float(payload_window)) or float(payload_window) <= 1e-6:
                    payload_window = float(max_end - min_start)
                checkpoint_scaled = False
                if (
                    (not epoch_based)
                    and checkpoint_window is not None
                    and len(tasks) >= (max(8, int(np.ceil(float(scheduler_task_count) * 0.25))) if scheduler_task_count > 0 else 8)
                ):
                    ratio = float(checkpoint_window["span"] / float(payload_window)) if float(payload_window) > 0 else None
                    if (
                        ratio is not None
                        and np.isfinite(ratio)
                        and ratio > 5.0
                        and checkpoint_window["span"] > 600.0
                        and float(payload_window) < 600.0
                    ):
                        base = float(checkpoint_window["start"])
                        scaled_tasks = []
                        for task in tasks:
                            start_rel = float(task["start"]) - float(min_start)
                            end_rel = float(task["end"]) - float(min_start)
                            start_abs = base + start_rel * float(ratio)
                            end_abs = base + end_rel * float(ratio)
                            scaled_tasks.append(
                                {
                                    **task,
                                    "start": float(start_abs),
                                    "end": float(end_abs),
                                    "duration": float(max(end_abs - start_abs, 1e-6)),
                                }
                            )
                        tasks = scaled_tasks
                        epoch_based = True
                        checkpoint_scaled = True
                source_rank = 0 if epoch_based else 90
                add_candidate_result(tasks, real_stats_path, source_rank)

            outputs_root = os.path.join(project_root, "outputs")
            if os.path.isdir(outputs_root):
                launched_tasks = []
                ordered_log_model_ids = []
                launch_pattern = re.compile(
                    r"为任务 (?P<task_id>task_[^\s]+) (?:冷启动常驻|启动).*?端口: (?P<port>\d+).*?"
                    r"(?:.*/)?(?P<model_id>merged_model_[^，\s/]+)，GPU: \[(?P<gpu_ids>[^\]]+)\]"
                )
                log_model_pattern = re.compile(rf"/{re.escape(run_label)}/[^，\s]*/(merged_model_[^，\s/]+)")
                for log_path in log_path_candidates:
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                            launch_lines = f.readlines()
                    except Exception:
                        continue
                    for line in launch_lines:
                        match = launch_pattern.search(line)
                        if match is not None:
                            gpu_tokens = [item.strip() for item in match.group("gpu_ids").split(",") if item.strip()]
                            if len(gpu_tokens) > 0:
                                try:
                                    gpu_idx = int(gpu_tokens[0])
                                except Exception:
                                    gpu_idx = None
                                if gpu_idx is not None:
                                    launched_tasks.append(
                                        {
                                            "task_id": str(match.group("task_id")),
                                            "model_id": str(match.group("model_id")),
                                            "gpu": gpu_idx,
                                        }
                                    )
                        if run_label and (("冷启动常驻vLLM服务器" in line) or ("热加载模型:" in line)):
                            model_match = log_model_pattern.search(line)
                            if model_match is not None:
                                ordered_log_model_ids.append(str(model_match.group(1)))

                if scheduler_task_count > 0 and len(launched_tasks) > scheduler_task_count:
                    launched_tasks = launched_tasks[-scheduler_task_count:]
                if total_task_count > 0 and len(ordered_log_model_ids) >= total_task_count and init_task_count > 0:
                    ordered_log_model_ids = ordered_log_model_ids[init_task_count : init_task_count + scheduler_task_count]
                elif scheduler_task_count > 0 and len(ordered_log_model_ids) > scheduler_task_count:
                    ordered_log_model_ids = ordered_log_model_ids[-scheduler_task_count:]

                output_model_windows = {}
                ordered_output_windows = []
                for output_name in sorted(os.listdir(outputs_root)):
                    output_dir = os.path.join(outputs_root, output_name)
                    if not os.path.isdir(output_dir):
                        continue
                    reports_dir = os.path.join(output_dir, "reports")
                    predictions_dir = os.path.join(output_dir, "predictions")
                    if not os.path.isdir(reports_dir) and not os.path.isdir(predictions_dir):
                        continue
                    try:
                        start_ts = datetime.datetime.strptime(output_name, "%Y%m%d_%H%M%S").timestamp()
                    except Exception:
                        start_ts = float(os.path.getmtime(output_dir))
                    end_ts = float(start_ts)
                    for root_dir, _, file_names in os.walk(output_dir):
                        for file_name in file_names:
                            file_path = os.path.join(root_dir, file_name)
                            try:
                                end_ts = max(end_ts, float(os.path.getmtime(file_path)))
                            except Exception:
                                continue
                    seen_output_models = set()
                    for model_root in [reports_dir, predictions_dir]:
                        if not os.path.isdir(model_root):
                            continue
                        for model_name in os.listdir(model_root):
                            model_path = os.path.join(model_root, model_name)
                            if not os.path.isdir(model_path) or model_name in seen_output_models:
                                continue
                            seen_output_models.add(model_name)
                            model_window = {
                                "start": float(start_ts),
                                "end": float(max(end_ts, start_ts)),
                            }
                            output_model_windows[model_name] = model_window
                            candidate_match = re.match(r"merged_model_(\d+)_", str(model_name))
                            if candidate_match is not None:
                                ordered_output_windows.append(
                                    {
                                        "model_id": str(model_name),
                                        "candidate_index": int(candidate_match.group(1)),
                                        "start": float(start_ts),
                                        "end": float(max(end_ts, start_ts)),
                                    }
                                )

                reconstructed_tasks = []
                for launched_task in launched_tasks:
                    model_window = output_model_windows.get(launched_task["model_id"])
                    if model_window is None:
                        continue
                    start_ts = float(model_window["start"])
                    end_ts = float(model_window["end"])
                    duration = end_ts - start_ts
                    if duration <= 0:
                        continue
                    reconstructed_tasks.append(
                        {
                            "gpu": int(launched_task["gpu"]),
                            "type": "full",
                            "start": start_ts,
                            "duration": float(duration),
                            "end": end_ts,
                        }
                    )
                reconstructed_tasks.sort(key=lambda x: (x["start"], x["gpu"]))
                if len(log_path_candidates) > 0:
                    add_candidate_result(
                        reconstructed_tasks,
                        f"{log_path_candidates[0]} + {outputs_root}",
                        2,
                    )

                ordered_output_windows.sort(key=lambda x: (x["start"], x["candidate_index"], x["model_id"]))
                if len(expected_candidate_sequence) > 0 and len(ordered_output_windows) >= len(expected_candidate_sequence):
                    matched_slice_start = None
                    output_candidate_sequence = [item["candidate_index"] for item in ordered_output_windows]
                    target_length = len(expected_candidate_sequence)
                    for start_idx in range(len(output_candidate_sequence) - target_length, -1, -1):
                        end_idx = start_idx + target_length
                        if output_candidate_sequence[start_idx:end_idx] == expected_candidate_sequence:
                            matched_slice_start = start_idx
                            break
                    if matched_slice_start is not None:
                        matched_windows = ordered_output_windows[
                            matched_slice_start : matched_slice_start + target_length
                        ]
                        scheduled_output_tasks = []
                        for scheduler_task, model_window in zip(flattened_scheduler_tasks, matched_windows):
                            duration = float(model_window["end"] - model_window["start"])
                            if duration <= 0:
                                continue
                            scheduled_output_tasks.append(
                                {
                                    "gpu": int(scheduler_task["gpu"]),
                                    "type": "full",
                                    "start": float(model_window["start"]),
                                    "duration": float(duration),
                                    "end": float(model_window["end"]),
                                }
                            )
                        scheduled_output_tasks.sort(key=lambda x: (x["start"], x["gpu"]))
                        add_candidate_result(
                            scheduled_output_tasks,
                            f"{outputs_root} (scheduler-aligned real output windows)",
                            1,
                        )

                if len(ordered_log_model_ids) > 0:
                    scheduled_output_tasks = []
                    for scheduler_task, model_id in zip(flattened_scheduler_tasks, ordered_log_model_ids):
                        model_window = output_model_windows.get(model_id, None)
                        if model_window is None:
                            continue
                        duration = float(model_window["end"] - model_window["start"])
                        if duration <= 0:
                            continue
                        scheduled_output_tasks.append(
                            {
                                "gpu": int(scheduler_task["gpu"]),
                                "type": "full",
                                "start": float(model_window["start"]),
                                "duration": duration,
                                "end": float(model_window["end"]),
                                "candidate_index": int(scheduler_task["candidate_index"]),
                                "model_id": str(model_id),
                                "task_id": f"task_{model_id}",
                            }
                        )
                    scheduled_output_tasks.sort(key=lambda x: (x["start"], x["gpu"]))
                    add_candidate_result(
                        scheduled_output_tasks,
                        f"{outputs_root} (log-aligned real output windows)",
                        1,
                    )

            if len(candidate_results) == 0:
                return [], None
            candidate_results.sort(key=lambda item: item[0])
            _, selected_tasks, selected_path = candidate_results[0]
            return selected_tasks, selected_path

        has_absolute_task = any(
            isinstance(task, dict) and ("absolute_start" in task)
            for rec in plot_scheduler_history
            for task in rec.get("tasks", [])
        )
        debug_gpu_intervals = {}
        max_gpu = 0
        real_runtime_tasks, real_runtime_path = load_real_runtime_tasks()
        persist_reconstructed_runtime(real_runtime_tasks, real_runtime_path)
        use_real_runtime = len(real_runtime_tasks) >= required_real_runtime_tasks
        if len(real_runtime_tasks) > 0 and not use_real_runtime:
            print(
                "检测到真实运行记录数量不足，改用调度导出数据绘图: "
                f"real_tasks={len(real_runtime_tasks)}, scheduler_tasks={scheduler_task_count}, "
                f"source={real_runtime_path}"
            )
        if use_real_runtime:
            min_start = min(item["start"] for item in real_runtime_tasks)
            all_end = max(item["end"] for item in real_runtime_tasks) - min_start
            for item in real_runtime_tasks:
                start_rel = float(item["start"] - min_start)
                duration = float(item["duration"])
                gpu_idx = int(item["gpu"])
                task_type = item["type"]
                task_key = item.get("candidate_index", item.get("task_key", None))
                label = None
                if task_type == "full" and not full_labeled:
                    label = "full"
                    full_labeled = True
                ax.barh(
                    gpu_idx,
                    duration,
                    left=start_rel,
                    color=_task_color(task_type, task_key),
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.3,
                    label=label,
                )
                if gpu_idx not in debug_gpu_intervals:
                    debug_gpu_intervals[gpu_idx] = []
                debug_gpu_intervals[gpu_idx].append(
                    {"start": float(start_rel), "end": float(start_rel + duration), "duration": float(duration)}
                )
                max_gpu = max(max_gpu, gpu_idx + 1)
            for gpu_idx in range(max_gpu):
                busy_spans = sorted(debug_gpu_intervals.get(gpu_idx, []), key=lambda x: x["start"])
                cursor = 0.0
                for span in busy_spans:
                    idle = max(span["start"] - cursor, 0.0)
                    if idle > 1e-9:
                        idle_label = None
                        if not idle_labeled:
                            idle_label = "idle"
                            idle_labeled = True
                        ax.barh(
                            gpu_idx,
                            idle,
                            left=cursor,
                            color="#d9d9d9",
                            alpha=0.8,
                            edgecolor="none",
                            label=idle_label,
                        )
                    cursor = max(cursor, span["end"])
                tail_idle = max(all_end - cursor, 0.0)
                if tail_idle > 1e-9:
                    idle_label = None
                    if not idle_labeled:
                        idle_label = "idle"
                        idle_labeled = True
                    ax.barh(
                        gpu_idx,
                        tail_idle,
                        left=cursor,
                        color="#d9d9d9",
                        alpha=0.8,
                        edgecolor="none",
                        label=idle_label,
                    )
            ax.set_xlabel("Wall Time (s)")
            ax.set_title("GPU Schedule Gantt (real runtime)")
        elif has_absolute_task:
            all_tasks = []
            for rec in plot_scheduler_history:
                for task in rec.get("tasks", []):
                    if not isinstance(task, dict):
                        continue
                    if "absolute_start" not in task:
                        continue
                    gpu_idx = int(task.get("gpu", 0))
                    start_abs = float(task["absolute_start"])
                    duration = float(task.get("duration", 0.0))
                    end_abs = start_abs + max(duration, 0.0)
                    all_tasks.append(
                        {
                            "gpu": gpu_idx,
                            "type": str(task.get("type", "full")),
                            "start": start_abs,
                            "duration": max(duration, 0.0),
                            "end": end_abs,
                            "task_key": task.get("candidate_index", len(all_tasks)),
                        }
                    )
                    max_gpu = max(max_gpu, gpu_idx + 1)
            if len(all_tasks) > 0:
                min_start = min(item["start"] for item in all_tasks)
                all_end = max(item["end"] for item in all_tasks) - min_start
                for item in sorted(all_tasks, key=lambda x: (x["gpu"], x["start"])):
                    task_type = item["type"]
                    task_key = item.get("candidate_index", item.get("task_key", None))
                    start_rel = float(item["start"] - min_start)
                    label = None
                    if task_type == "full" and not full_labeled:
                        label = "full"
                        full_labeled = True
                    ax.barh(
                        int(item["gpu"]),
                        float(item["duration"]),
                        left=start_rel,
                        color=_task_color(task_type, task_key),
                        alpha=0.85,
                        edgecolor="black",
                        linewidth=0.3,
                        label=label,
                    )
                    if item["gpu"] not in debug_gpu_intervals:
                        debug_gpu_intervals[item["gpu"]] = []
                    debug_gpu_intervals[item["gpu"]].append(
                        {
                            "start": start_rel,
                            "end": float(start_rel + float(item["duration"])),
                            "duration": float(item["duration"]),
                        }
                    )
                for gpu_idx in range(max_gpu):
                    busy_spans = sorted(debug_gpu_intervals.get(gpu_idx, []), key=lambda x: x["start"])
                    cursor = 0.0
                    for span in busy_spans:
                        idle = max(span["start"] - cursor, 0.0)
                        if idle > 1e-9:
                            idle_label = None
                            if not idle_labeled:
                                idle_label = "idle"
                                idle_labeled = True
                            ax.barh(
                                gpu_idx,
                                idle,
                                left=cursor,
                                color="#d9d9d9",
                                alpha=0.8,
                                edgecolor="none",
                                label=idle_label,
                            )
                        cursor = max(cursor, span["end"])
                    tail_idle = max(all_end - cursor, 0.0)
                    if tail_idle > 1e-9:
                        idle_label = None
                        if not idle_labeled:
                            idle_label = "idle"
                            idle_labeled = True
                        ax.barh(
                            gpu_idx,
                            tail_idle,
                            left=cursor,
                            color="#d9d9d9",
                            alpha=0.8,
                            edgecolor="none",
                            label=idle_label,
                        )
        else:
            offset = 0.0
            scheduler_uses_wall_time = True
            for rec in plot_scheduler_history:
                iter_span = float(rec["estimated_makespan"])
                iter_wall = rec.get("wall_time_sec", None)
                if isinstance(iter_wall, (int, float)) and float(iter_wall) > 0.0:
                    iter_wall_span = float(iter_wall)
                else:
                    iter_wall_span = iter_span
                    scheduler_uses_wall_time = False
                scale = float(iter_wall_span / max(iter_span, 1e-9)) if iter_span > 0 else 1.0
                gpu_loads = rec["gpu_loads"]
                max_gpu = max(max_gpu, len(gpu_loads))
                for task in rec["tasks"]:
                    task_type = task["type"]
                    task_key = task.get("candidate_index", None)
                    label = None
                    if task_type == "full" and not full_labeled:
                        label = "full"
                        full_labeled = True
                    ax.barh(
                        int(task["gpu"]),
                        float(task["duration"]) * scale,
                        left=offset + float(task["start"]) * scale,
                        color=_task_color(task_type, task_key),
                        alpha=0.85,
                        edgecolor="black",
                        linewidth=0.3,
                        label=label,
                    )
                    if int(task["gpu"]) not in debug_gpu_intervals:
                        debug_gpu_intervals[int(task["gpu"])] = []
                    task_start = offset + float(task["start"]) * scale
                    task_end = task_start + float(task["duration"]) * scale
                    debug_gpu_intervals[int(task["gpu"])].append(
                        {
                            "start": float(task_start),
                            "end": float(task_end),
                            "duration": float(float(task["duration"]) * scale),
                        }
                    )
                for gpu_idx, load in enumerate(gpu_loads):
                    idle = max(iter_wall_span - float(load) * scale, 0.0)
                    if idle > 0:
                        idle_label = None
                        if not idle_labeled:
                            idle_label = "idle"
                            idle_labeled = True
                        ax.barh(
                            gpu_idx,
                            idle,
                            left=offset + float(load) * scale,
                            color="#d9d9d9",
                            alpha=0.8,
                            edgecolor="none",
                            label=idle_label,
                        )
                offset += iter_wall_span
        debug_intervals_path = os.path.join(run_dir, "gpu_debug_intervals.json")
        with open(debug_intervals_path, "w") as f:
            json.dump({str(k): v for k, v in sorted(debug_gpu_intervals.items())}, f, indent=2)
        print("GPU使用时间段调试输出:")
        for gpu_idx in sorted(debug_gpu_intervals.keys()):
            spans = sorted(debug_gpu_intervals[gpu_idx], key=lambda x: x["start"])
            if len(spans) == 0:
                print(f"  GPU {gpu_idx}: 无任务")
                continue
            span_str = ", ".join(
                [f"[{item['start']:.2f}, {item['end']:.2f}]({item['duration']:.2f}s)" for item in spans]
            )
            print(f"  GPU {gpu_idx}: {span_str}")
        use_wall_time_unit = use_real_runtime or has_absolute_task or (
            "scheduler_uses_wall_time" in locals() and scheduler_uses_wall_time
        )
        if use_wall_time_unit:
            ax.set_xlabel("Wall Time (s)")
            if not use_real_runtime:
                ax.set_title("GPU Schedule Gantt (wall time)")
        else:
            ax.set_xlabel("Estimated Time Units")
        ax.set_ylabel("GPU")
        if not use_wall_time_unit:
            ax.set_title("GPU Schedule Gantt (full/idle)")
        ax.set_yticks(list(range(max_gpu)))
        if full_labeled or idle_labeled:
            ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "gpu_schedule_gantt.png"), dpi=160)
        fig.savefig(os.path.join(run_dir, "gpu_schedule_gantt_real_runtime.png"), dpi=160)
        plt.close(fig)

    def sample_initial_z(n_samples: int) -> torch.Tensor:
        z = draw_sobol_samples(bounds=z_bounds_eff, n=n_samples, q=1).squeeze(1)
        return z.to(**tkwargs)

    def generate_initial_data(n=initial_samples, custom_solutions=None):
        if isinstance(shared_initial_dataset, dict):
            shared_x = torch.as_tensor(shared_initial_dataset.get("decision_variables", []), **tkwargs)
            shared_obj_true = torch.as_tensor(shared_initial_dataset.get("objectives", []), **tkwargs)
            shared_info = shared_initial_dataset.get("metrics", [{} for _ in range(shared_x.shape[0])])
            if shared_x.ndim == 2 and shared_x.shape == (n, dim) and shared_obj_true.shape == (n, num_objectives):
                train_z = encode_x_to_z(shared_x)
                train_obj = shared_obj_true + torch.randn_like(shared_obj_true) * NOISE_SE
                return train_z, shared_x, train_obj, shared_obj_true, shared_info, []
        if custom_solutions is not None and len(custom_solutions) > 0:
            num_custom = min(len(custom_solutions), n)
            custom_x = []
            for val in custom_solutions[:num_custom]:
                custom_sol = torch.full((1, dim), float(val), **tkwargs).clamp(lower.unsqueeze(0), upper.unsqueeze(0))
                custom_x.append(custom_sol)
            remaining = n - num_custom
            if remaining > 0:
                random_z = sample_initial_z(remaining)
                random_x = decode_z_to_x(random_z)
                train_x = torch.cat(custom_x + [random_x], dim=0)
            else:
                train_x = torch.cat(custom_x, dim=0)
        else:
            random_z = sample_initial_z(n)
            train_x = decode_z_to_x(random_z)
        train_z = encode_x_to_z(train_x)
        train_x, train_obj, train_obj_true, train_info, initial_cleanup_paths = evaluate_from_z(
            train_z,
            eval_limit=full_eval_limits,
            eval_mode="full",
        )
        return train_z, train_x, train_obj, train_obj_true, train_info, initial_cleanup_paths

    def initialize_model(train_z, train_obj):
        train_z_normalized = normalize(train_z, z_bounds_eff)
        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]
            noise = NOISE_SE[i].to(device=train_y.device, dtype=train_y.dtype)
            train_yvar = torch.full_like(train_y, noise ** 2)
            train_z_normalized = train_z_normalized.to(dtype=train_y.dtype, device=train_y.device)
            models.append(SingleTaskGP(train_z_normalized, train_y, train_yvar))
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def fit_gpytorch_mll(mll):
        botorch_fit_gpytorch_mll(mll)

    def propose_candidates(model, train_z, train_obj, train_obj_true, train_info, sampler, q, pending_z=None):
        _ = train_obj
        def extract_avg_tokens(metric_item):
            if not isinstance(metric_item, dict):
                return None
            aime_val = metric_item.get("aime25", {}).get("mean_tokens_num", None)
            gpqa_val = metric_item.get("gpqa_diamond", {}).get("mean_tokens_num", None)
            vals = []
            if isinstance(aime_val, (int, float)) and np.isfinite(aime_val) and aime_val > 0:
                vals.append(float(aime_val))
            if isinstance(gpqa_val, (int, float)) and np.isfinite(gpqa_val) and gpqa_val > 0:
                vals.append(float(gpqa_val))
            if len(vals) == 0:
                return None
            return float(np.mean(vals))

        train_z_normalized = normalize(train_z, z_bounds_eff)
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=problem_ref_point.to(device=train_z.device, dtype=train_z.dtype),
            X_baseline=train_z_normalized,
            prune_baseline=True,
            sampler=sampler,
        )
        if pending_z is not None and pending_z.numel() > 0:
            pending_z = pending_z.to(device=train_z.device, dtype=train_z.dtype)
            pending_z_normalized = normalize(pending_z, z_bounds_eff)
            acq.set_X_pending(pending_z_normalized)
        candidate_pool_size = int(max(1, q))
        if enable_gap_aware_postprocess:
            candidate_pool_size = max(
                candidate_pool_size,
                int(max(2, q) * max(1, int(gap_candidate_pool_multiplier))),
            )
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=standard_bounds_eff.to(device=train_z.device, dtype=train_z.dtype),
            q=candidate_pool_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 100},
            sequential=True,
        )
        candidate_values = []
        for cand_idx in range(candidates.shape[0]):
            xi = candidates[cand_idx : cand_idx + 1].unsqueeze(0)
            val = acq(xi).detach().reshape(-1)[0]
            candidate_values.append(val)
        candidate_values = torch.stack(candidate_values).to(**tkwargs)
        f2_posterior = model.models[1].posterior(candidates).mean.detach().reshape(-1).to(**tkwargs)
        observed_f2 = train_obj_true[:, 1].detach().cpu().numpy()
        observed_tokens = []
        observed_f2_valid = []
        for idx, metric_item in enumerate(train_info):
            token_val = extract_avg_tokens(metric_item)
            if token_val is None:
                continue
            if idx >= observed_f2.shape[0]:
                continue
            observed_tokens.append(token_val)
            observed_f2_valid.append(float(observed_f2[idx]))
        if len(observed_tokens) >= 2 and np.std(observed_f2_valid) > 0:
            slope, intercept = np.polyfit(np.array(observed_f2_valid, dtype=float), np.array(observed_tokens, dtype=float), 1)
            estimated_tokens = f2_posterior * float(slope) + float(intercept)
        elif len(observed_tokens) == 1:
            estimated_tokens = torch.full_like(f2_posterior, float(observed_tokens[0]))
        else:
            estimated_tokens = torch.abs(f2_posterior) + 1.0
        estimated_tokens = torch.clamp(estimated_tokens, min=1.0)
        candidate_z = unnormalize(candidates.detach(), bounds=z_bounds_eff)
        candidate_obj_mean = model.posterior(candidates).mean.detach().reshape(candidates.shape[0], -1).to(**tkwargs)
        if enable_gap_aware_postprocess:
            observed_obj = train_obj_true.detach().to(**tkwargs)
            pending_z_normalized = None
            if pending_z is not None and pending_z.numel() > 0:
                pending_z_normalized = normalize(pending_z, z_bounds_eff).to(**tkwargs)
            selected_idx, selected_score_details = _select_gap_aware_indices(
                candidate_repr=candidates.detach().to(**tkwargs),
                candidate_values=candidate_values,
                candidate_obj_mean=candidate_obj_mean,
                observed_obj=observed_obj,
                q=q,
                pending_repr=pending_z_normalized,
                gap_reward_weight=gap_reward_weight,
                pending_penalty_weight=gap_pending_penalty_weight,
            )
        else:
            sorted_idx = torch.argsort(candidate_values, descending=True)
            selected_idx = sorted_idx[: min(int(max(1, q)), sorted_idx.numel())]
            selected_score_details = []
        selected_z = candidate_z[selected_idx]
        selected_costs = estimated_tokens[selected_idx]
        selected_values = candidate_values[selected_idx]
        if enable_gap_aware_postprocess and len(selected_score_details) > 0:
            round_record = {
                "proposal_round": int(proposal_score_state["round"]),
                "iteration": int(current_iteration),
                "requested_q": int(q),
                "selected_count": int(selected_idx.numel()),
                "candidate_pool_size": int(candidates.shape[0]),
                "pending_count": int(0 if pending_z is None else pending_z.shape[0]),
                "scores": [],
            }
            for detail in selected_score_details:
                candidate_index = int(detail["candidate_index"])
                round_record["scores"].append(
                    {
                        **detail,
                        "candidate_z": candidate_z[candidate_index].detach().cpu().tolist(),
                        "predicted_objectives": candidate_obj_mean[candidate_index].detach().cpu().tolist(),
                        "acq_value": float(candidate_values[candidate_index].item()),
                    }
                )
            candidate_score_history.append(round_record)
            proposal_score_state["round"] += 1
        return selected_z, selected_values, selected_costs

    def optimize_acq_and_observe(model, train_z, train_obj, sampler):
        selected_z, _, selected_costs = propose_candidates(
            model=model,
            train_z=train_z,
            train_obj=train_obj,
            train_obj_true=train_obj_true,
            train_info=train_info,
            sampler=sampler,
            q=BATCH_SIZE,
        )
        gpu_slots = max(1, int(scheduler_gpu_count))
        gpu_loads = torch.zeros(gpu_slots, **tkwargs)
        task_segments = []
        for idx in range(selected_z.shape[0]):
            min_gpu = int(torch.argmin(gpu_loads).item())
            start_load = float(gpu_loads[min_gpu].item())
            duration = float(selected_costs[idx].item())
            task_segments.append(
                {
                    "gpu": min_gpu,
                    "start": start_load,
                    "duration": duration,
                    "type": "full",
                    "candidate_index": int(idx),
                }
            )
            gpu_loads[min_gpu] += selected_costs[idx]
        new_z = selected_z
        new_x, new_obj, new_obj_true, new_info, cleanup_paths = evaluate_from_z(
            new_z,
            eval_limit=full_eval_limits,
            eval_mode="full",
            estimated_tokens=selected_costs,
        )
        schedule_record = {
            "iteration": int(current_iteration),
            "mode": "sync",
            "gpu_count": gpu_slots,
            "estimated_makespan": float(torch.max(gpu_loads).item()) if gpu_loads.numel() > 0 else 0.0,
            "gpu_loads": [float(v) for v in gpu_loads.detach().cpu().tolist()],
            "tasks": task_segments,
            "counts": {
                "full": int(selected_z.shape[0]),
            },
            "wall_time_sec": 0.0,
        }
        schedule_history_by_iter[int(current_iteration)] = schedule_record
        return new_z, new_x, new_obj, new_obj_true, new_info, cleanup_paths

    def compute_hv(objectives, ref_point_t):
        if not isinstance(objectives, torch.Tensor):
            objectives = torch.as_tensor(objectives, dtype=dtype)
        if not isinstance(ref_point_t, torch.Tensor):
            ref_point_t = torch.as_tensor(ref_point_t, dtype=objectives.dtype)

        objectives_cpu = objectives.detach().to(device="cpu", dtype=torch.float64)
        ref_point_cpu = ref_point_t.detach().to(device="cpu", dtype=torch.float64)

        if objectives_cpu.ndim == 1:
            objectives_cpu = objectives_cpu.unsqueeze(0)
        finite_mask = torch.isfinite(objectives_cpu).all(dim=-1)
        valid_objectives = objectives_cpu[finite_mask]

        if valid_objectives.numel() == 0:
            return 0.0

        try:
            bd = FastNondominatedPartitioning(ref_point=ref_point_cpu, Y=valid_objectives)
            hv_value = bd.compute_hypervolume()
            if isinstance(hv_value, torch.Tensor):
                hv_scalar = hv_value.item()
            else:
                hv_scalar = float(hv_value)
            if not np.isfinite(hv_scalar):
                return 0.0
            return float(hv_scalar)
        except RuntimeError as exc:
            print(f"警告: hypervolume 计算失败，回退为 0.0: {exc}")
            return 0.0

    def to_json_safe(value):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, dict):
            return {str(key): to_json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_json_safe(item) for item in value]
        return value

    def save_checkpoint(iteration, train_z, train_x, train_obj, train_obj_true, train_info, hvs, hv_curve):
        checkpoint = {
            "iteration": iteration,
            "hvs": to_json_safe(hvs),
            "hv_curve": to_json_safe(hv_curve),
            "candidate_score_history": to_json_safe(candidate_score_history),
            "evaluated_solutions": {
                "decision_variables": train_x.cpu().tolist(),
                "objectives": train_obj_true.cpu().tolist(),
                "metrics": to_json_safe(train_info),
            },
        }
        if isinstance(eval_metadata, dict) and len(eval_metadata) > 0:
            checkpoint["evaluation_settings"] = to_json_safe(eval_metadata)
        with open(os.path.join(run_dir, f"checkpoint_iter_{iteration}.json"), "w") as f:
            json.dump(checkpoint, f)
        state = {
            "train_z": train_z.cpu(),
            "train_x": train_x.cpu(),
            "train_obj": train_obj.cpu(),
            "train_obj_true": train_obj_true.cpu(),
            "train_info": train_info,
            "iteration": iteration,
            "hvs": hvs,
            "hv_curve": hv_curve,
            "evaluated_solutions": checkpoint["evaluated_solutions"],
            "evaluated_count": int(train_z.shape[0]),
            "top_layers": top_layers,
            "groups": groups,
            "m_prior": m_prior_t.cpu(),
            "u_prior": u_prior_t.cpu(),
            "rho": rho,
            "scheduler_history": scheduler_history,
            "candidate_score_history": candidate_score_history,
        }
        if isinstance(eval_metadata, dict) and len(eval_metadata) > 0:
            state["evaluation_settings"] = eval_metadata
        torch.save(state, os.path.join(run_dir, f"checkpoint_iter_{iteration}.pt"))
        torch.save(state, os.path.join(run_dir, "checkpoint_latest.pt"))

    def load_checkpoint():
        p = os.path.join(run_dir, "checkpoint_latest.pt")
        if not os.path.exists(p):
            return None
        state = torch.load(p, map_location="cpu")
        train_z = state["train_z"].to(**tkwargs)
        train_x = state.get("train_x", decode_z_to_x(train_z)).to(**tkwargs)
        train_obj = state["train_obj"].to(**tkwargs)
        train_obj_true = state["train_obj_true"].to(**tkwargs)
        train_info = state.get("train_info", [{} for _ in range(train_z.shape[0])])
        iteration = state["iteration"]
        hvs = state["hvs"]
        hv_curve = state.get("hv_curve", [])
        if len(hv_curve) == 0 and len(hvs) > 0:
            hv_curve = [{"evaluations": int(train_z.shape[0]), "hypervolume": float(hvs[-1])}]
        evaluated_count = int(state.get("evaluated_count", train_z.shape[0]))
        loaded_history = state.get("scheduler_history", [])
        scheduler_history.clear()
        scheduler_history.extend(loaded_history)
        loaded_candidate_score_history = state.get("candidate_score_history", [])
        candidate_score_history.clear()
        if isinstance(loaded_candidate_score_history, list):
            candidate_score_history.extend(loaded_candidate_score_history)
        proposal_score_state["round"] = len(candidate_score_history)
        return train_z, train_x, train_obj, train_obj_true, train_info, iteration, hvs, hv_curve, evaluated_count

    if verbose:
        print(f"使用Prior-BO+qNEHVI优化多目标问题 (原始维度: {dim}, 优化维度: {z_dim}, 目标数: {num_objectives})")
        print(f"设备: {tkwargs['device']}, 数据类型: {tkwargs['dtype']}")
        print(f"运行ID: {run_id}, 检查点目录: {run_dir}")
        print(f"top_layers: {top_layers}")
        print(f"groups: {groups}")
        print(f"每轮候选数: {BATCH_SIZE}")
        print(f"调度GPU数: {max(1, int(scheduler_gpu_count))}")
        print(f"最大评估次数: {int(max_evaluations)}, 异步模式: {bool(async_mode)}")

    scheduler_history = []
    schedule_history_by_iter = {}
    candidate_score_history = []
    proposal_score_state = {"round": 0}
    current_iteration = 0
    timeline_origin_ts = time.time()

    def append_initial_scheduler_records(info_list):
        if not isinstance(info_list, list) or len(info_list) == 0:
            return
        parsed_items = []
        for info_item in info_list:
            if not isinstance(info_item, dict):
                continue
            start_ts = info_item.get("mock_start_ts", None)
            end_ts = info_item.get("mock_end_ts", None)
            gpu_id = info_item.get("mock_gpu_id", None)
            if not isinstance(start_ts, (int, float)) or not isinstance(end_ts, (int, float)):
                continue
            if not isinstance(gpu_id, (int, float)):
                continue
            parsed_items.append({"start_ts": float(start_ts), "end_ts": float(end_ts), "gpu_id": int(gpu_id)})
        if len(parsed_items) == 0:
            return
        parsed_items.sort(key=lambda x: x["start_ts"])
        gpu_count_for_record = max(1, int(scheduler_gpu_count))
        for idx, item in enumerate(parsed_items):
            duration = max(float(item["end_ts"]) - float(item["start_ts"]), 1e-6)
            gpu_slot = int(item["gpu_id"]) % gpu_count_for_record
            start_rel = max(float(item["start_ts"]) - float(timeline_origin_ts), 0.0)
            gpu_loads = [0.0 for _ in range(gpu_count_for_record)]
            gpu_loads[gpu_slot] = float(duration)
            scheduler_history.append(
                {
                    "iteration": int(idx + 1),
                    "mode": "init",
                    "gpu_count": gpu_count_for_record,
                    "estimated_makespan": float(duration),
                    "gpu_loads": gpu_loads,
                    "tasks": [
                        {
                            "type": "full",
                            "gpu": int(gpu_slot),
                            "start": 0.0,
                            "duration": float(duration),
                            "absolute_start": float(start_rel),
                        }
                    ],
                    "counts": {"full": 1},
                    "wall_time_sec": float(duration),
                    "candidate_tokens_est": 0.0,
                    "remaining_eta_sec": 0.0,
                    "estimated_finish_time": "",
                }
            )

    checkpoint = load_checkpoint()
    if checkpoint is not None:
        train_z, train_x, train_obj, train_obj_true, train_info, start_iteration, hvs, hv_curve, start_evaluated_count = checkpoint
        if verbose:
            print(f"成功加载检查点，从迭代 {start_iteration} 继续")
            print(f"当前超体积: {hvs[-1]:.4f}")
            print(f"已评估数量: {start_evaluated_count}")
    else:
        train_z, train_x, train_obj, train_obj_true, train_info, initial_cleanup_paths = generate_initial_data(
            custom_solutions=custom_initial_solutions
        )
        hvs = [compute_hv(train_obj_true, problem_ref_point)]
        hv_curve = [{"evaluations": int(train_z.shape[0]), "hypervolume": float(hvs[-1])}]
        start_iteration = train_z.shape[0]
        start_evaluated_count = int(train_z.shape[0])
        if verbose:
            print(f"初始超体积: {hvs[-1]:.4f}")
        append_initial_scheduler_records(train_info)
        save_checkpoint(0, train_z, train_x, train_obj, train_obj_true, train_info, hvs, hv_curve)
        if iteration_callback is not None and isinstance(initial_cleanup_paths, list) and len(initial_cleanup_paths) > 0:
            if hasattr(iteration_callback, "cleanup_paths"):
                iteration_callback.cleanup_paths = list(initial_cleanup_paths)
            if hasattr(iteration_callback, "async_mode"):
                iteration_callback.async_mode = False
            iteration_callback(0, train_x, train_obj_true, hvs)

    def append_observation(new_z, new_x, new_obj, new_obj_true, new_info):
        nonlocal train_z, train_x, train_obj, train_obj_true, train_info
        train_z = torch.cat([train_z, new_z.to(train_z.device)], dim=0)
        train_x = torch.cat([train_x, new_x.to(train_x.device)], dim=0)
        train_obj = torch.cat([train_obj, new_obj.to(train_obj.device)], dim=0)
        train_obj_true = torch.cat([train_obj_true, new_obj_true.to(train_obj_true.device)], dim=0)
        train_info.extend(new_info)

    def calc_remaining_eta(avg_runtime_sec, inflight_records, done_count, total_count):
        remain_count = max(total_count - done_count, 0)
        inflight_remaining = 0.0
        now_ts = time.time()
        for item in inflight_records:
            elapsed = now_ts - float(item["start_ts"])
            inflight_remaining += max(float(item["est_runtime_sec"]) - elapsed, 0.0)
        if scheduler_gpu_count <= 0:
            queue_eta = 0.0
        else:
            queue_eta = max(remain_count - len(inflight_records), 0) * float(avg_runtime_sec) / max(1, int(scheduler_gpu_count))
        return inflight_remaining + queue_eta

    try:
        if not async_mode:
            max_sync_steps = int(np.ceil(max(0, int(max_evaluations) - train_z.shape[0]) / max(1, int(BATCH_SIZE))))
            for step_idx in range(1, max_sync_steps + 1):
                iter_start_ts = time.time()
                current_iteration = step_idx
                mll, model = initialize_model(train_z, train_obj)
                fit_gpytorch_mll(mll)
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
                new_z, new_x, new_obj, new_obj_true, new_info, new_cleanup_paths = optimize_acq_and_observe(model, train_z, train_obj, sampler)
                remain_cap = int(max_evaluations) - train_z.shape[0]
                if remain_cap < new_z.shape[0]:
                    keep = max(0, remain_cap)
                    new_z = new_z[:keep]
                    new_x = new_x[:keep]
                    new_obj = new_obj[:keep]
                    new_obj_true = new_obj_true[:keep]
                    new_info = new_info[:keep]
                    if isinstance(new_cleanup_paths, list):
                        new_cleanup_paths = new_cleanup_paths[:keep]
                iter_wall = time.time() - iter_start_ts
                rec = schedule_history_by_iter.get(step_idx, None)
                if rec is None:
                    rec = {
                        "iteration": int(step_idx),
                        "mode": "sync",
                        "gpu_count": max(1, int(scheduler_gpu_count)),
                        "estimated_makespan": 0.0,
                        "gpu_loads": [],
                        "tasks": [],
                        "counts": {"full": int(new_z.shape[0])},
                        "wall_time_sec": float(iter_wall),
                    }
                else:
                    rec["wall_time_sec"] = float(iter_wall)
                    rec["counts"] = {"full": int(new_z.shape[0])}
                scheduler_history.append(rec)
                append_observation(new_z, new_x, new_obj, new_obj_true, new_info)
                new_hv = compute_hv(train_obj_true, problem_ref_point)
                hvs.append(new_hv)
                hv_curve.append({"evaluations": int(train_z.shape[0]), "hypervolume": float(new_hv)})
                if verbose:
                    print(f"同步步 {step_idx:>2}: 超体积 = {new_hv:.4f}")
                if iteration_callback is not None:
                    if hasattr(iteration_callback, "cleanup_paths"):
                        iteration_callback.cleanup_paths = list(new_cleanup_paths) if isinstance(new_cleanup_paths, list) else []
                    if hasattr(iteration_callback, "async_mode"):
                        iteration_callback.async_mode = False
                    iteration_callback(step_idx, train_x, train_obj_true, hvs)
                save_checkpoint(step_idx, train_z, train_x, train_obj, train_obj_true, train_info, hvs, hv_curve)
                del mll, model, sampler, new_z, new_x, new_obj, new_obj_true
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                if train_z.shape[0] >= int(max_evaluations):
                    break
        else:
            total_target = int(max_evaluations)
            completed_counter = int(start_evaluated_count)
            submit_counter = 0
            completion_walltimes = []
            inflight = []
            model_dirty = True
            mll = None
            model = None
            sampler = None
            candidate_generation_lock = threading.Lock()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(scheduler_gpu_count))) as executor:
                while completed_counter < total_target or len(inflight) > 0:
                    done_items = [item for item in inflight if item["future"].done()]
                    for item in done_items:
                        inflight.remove(item)
                        new_x, new_obj, new_obj_true, new_info, new_cleanup_paths = item["future"].result()
                        new_z = item["z"].to(**tkwargs)
                        append_observation(new_z, new_x, new_obj, new_obj_true, new_info)
                        completed_counter += int(new_z.shape[0])
                        completion_walltimes.append(time.time() - float(item["start_ts"]))
                        model_dirty = True
                        hv_val = compute_hv(train_obj_true, problem_ref_point)
                        hvs.append(hv_val)
                        hv_curve.append({"evaluations": int(completed_counter), "hypervolume": float(hv_val)})
                        current_iteration = completed_counter
                        item_runtime = max(float(time.time() - float(item["start_ts"])), 1e-6)
                        gpu_count_for_record = max(1, int(scheduler_gpu_count))
                        info_gpu_idx = None
                        if isinstance(new_info, list) and len(new_info) > 0 and isinstance(new_info[0], dict):
                            raw_gpu = new_info[0].get("mock_gpu_id", None)
                            if isinstance(raw_gpu, (int, float)):
                                info_gpu_idx = int(raw_gpu)
                        task_gpu_slot = (
                            int(info_gpu_idx) % gpu_count_for_record
                            if info_gpu_idx is not None
                            else int(item.get("gpu_slot", 0)) % gpu_count_for_record
                        )
                        task_start_rel = max(float(item["start_ts"]) - float(timeline_origin_ts), 0.0)
                        gpu_loads = [0.0 for _ in range(gpu_count_for_record)]
                        gpu_loads[task_gpu_slot] = float(item_runtime)
                        rec = {
                            "iteration": int(current_iteration),
                            "mode": "async",
                            "gpu_count": gpu_count_for_record,
                            "estimated_makespan": float(item_runtime),
                            "gpu_loads": gpu_loads,
                            "tasks": [
                                {
                                    "type": "full",
                                    "gpu": int(task_gpu_slot),
                                    "start": 0.0,
                                    "duration": float(item_runtime),
                                    "absolute_start": float(task_start_rel),
                                }
                            ],
                            "counts": {"full": int(new_z.shape[0])},
                            "wall_time_sec": float(item_runtime),
                            "candidate_tokens_est": float(item["candidate_tokens_est"]),
                            "remaining_eta_sec": calc_remaining_eta(
                                np.mean(completion_walltimes) if len(completion_walltimes) > 0 else item["est_runtime_sec"],
                                inflight,
                                completed_counter,
                                total_target,
                            ),
                            "estimated_finish_time": time.strftime(
                                "%Y-%m-%d %H:%M:%S",
                                time.localtime(
                                    time.time()
                                    + calc_remaining_eta(
                                        np.mean(completion_walltimes) if len(completion_walltimes) > 0 else item["est_runtime_sec"],
                                        inflight,
                                        completed_counter,
                                        total_target,
                                    )
                                ),
                            ),
                        }
                        scheduler_history.append(rec)
                        if verbose:
                            print(
                                f"异步完成 {completed_counter}/{total_target}: 超体积={hv_val:.4f}, "
                                f"剩余约 {rec['remaining_eta_sec']:.1f}s, 预计完成 {rec['estimated_finish_time']}"
                            )
                        if iteration_callback is not None:
                            if hasattr(iteration_callback, "cleanup_paths"):
                                iteration_callback.cleanup_paths = list(new_cleanup_paths) if isinstance(new_cleanup_paths, list) else []
                            if hasattr(iteration_callback, "async_mode"):
                                iteration_callback.async_mode = True
                            iteration_callback(current_iteration, train_x, train_obj_true, hvs)
                        save_checkpoint(current_iteration, train_z, train_x, train_obj, train_obj_true, train_info, hvs, hv_curve)
                        gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                    q_new = 0
                    if callable(idle_gpu_count_fn):
                        try:
                            slots = max(int(idle_gpu_count_fn()), 0)
                        except Exception:
                            slots = max(1, int(scheduler_gpu_count)) - len(inflight)
                    else:
                        slots = max(1, int(scheduler_gpu_count)) - len(inflight)
                    remaining_budget = total_target - (completed_counter + len(inflight))
                    if slots > 0 and remaining_budget > 0:
                        with candidate_generation_lock:
                            if model_dirty and completed_counter < total_target:
                                mll, model = initialize_model(train_z, train_obj)
                                fit_gpytorch_mll(mll)
                                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
                                model_dirty = False
                                if verbose:
                                    print(
                                        f"代理模型已更新: train_samples={int(train_z.shape[0])}, "
                                        f"completed={int(completed_counter)}, inflight={len(inflight)}"
                                    )
                            if not model_dirty:
                                if len(inflight) == 0:
                                    q_new = min(slots, remaining_budget)
                                elif len(done_items) > 0:
                                    q_new = min(len(done_items), slots, remaining_budget)
                                else:
                                    q_new = 0
                                    if not callable(idle_gpu_count_fn):
                                        q_new = min(slots, remaining_budget)
                                        soonest_finish = min(
                                            max(float(item["est_runtime_sec"]) - (time.time() - float(item["start_ts"])), 0.0)
                                            for item in inflight
                                        )
                                        mean_runtime = np.mean(completion_walltimes) if len(completion_walltimes) > 0 else 1.0
                                        if soonest_finish <= float(wait_for_completion_threshold) * max(mean_runtime, 1.0):
                                            q_new = 0
                                if q_new > 0:
                                    if verbose:
                                        print(
                                            f"生成候选: q={int(q_new)}, surrogate_train_samples={int(train_z.shape[0])}, "
                                            f"completed={int(completed_counter)}, inflight={len(inflight)}, slots={int(slots)}"
                                        )
                                    pending_z = None
                                    if len(inflight) > 0:
                                        pending_batches = [
                                            item["z"].detach().clone().to(**tkwargs)
                                            for item in inflight
                                            if isinstance(item.get("z", None), torch.Tensor) and item["z"].numel() > 0
                                        ]
                                        if len(pending_batches) > 0:
                                            pending_z = torch.cat(pending_batches, dim=0)
                                    candidate_z, _, candidate_cost = propose_candidates(
                                        model=model,
                                        train_z=train_z,
                                        train_obj=train_obj,
                                        train_obj_true=train_obj_true,
                                        train_info=train_info,
                                        sampler=sampler,
                                        q=q_new,
                                        pending_z=pending_z,
                                    )
                                    for cand_idx in range(int(q_new)):
                                        cand_z = candidate_z[cand_idx : cand_idx + 1]
                                        est_runtime = float(candidate_cost[cand_idx].item())
                                        gpu_slot = int(submit_counter % max(1, int(scheduler_gpu_count)))
                                        fut = executor.submit(
                                            evaluate_from_z,
                                            cand_z,
                                            full_eval_limits,
                                            "full",
                                            float(candidate_cost[cand_idx].item()),
                                        )
                                        submit_counter += 1
                                        inflight.append(
                                            {
                                                "submit_id": submit_counter,
                                                "future": fut,
                                                "z": cand_z.detach().clone(),
                                                "start_ts": time.time(),
                                                "est_runtime_sec": est_runtime,
                                                "candidate_tokens_est": est_runtime,
                                                "gpu_slot": int(gpu_slot),
                                            }
                                        )
                    if len(done_items) == 0 and q_new == 0:
                        time.sleep(0.2)
    except Exception as e:
        print(f"错误: 优化过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        save_scheduler_reports()
        return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id

    save_scheduler_reports()
    return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
