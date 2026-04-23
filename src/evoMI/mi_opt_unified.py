#!/usr/bin/env python3

import argparse
import ast
import hashlib
import inspect
import json
import os
import random
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evoMI.checkpoint_runtime import (
    build_eval_metadata,
    build_hv_curve,
    build_sync_scheduler_history,
    ensure_latest_checkpoint_metadata,
    rebuild_runtime_reports_from_checkpoint,
    save_runtime_reports,
)
from src.evoMI.model_reproduction import (
    build_eval_cache_config,
    build_eval_cache_namespace,
    build_eval_setting_id,
    load_cached_results,
    normalize_eval_limit,
    save_results_to_cache,
)
from src.evoMI.mm_mo_optimizer import mm_mo_optimizer
from src.evoMI.moead_cmaes_prior_optimizer import moead_cmaes_prior_optimizer
from src.evoMI.optimizer import prior_bo_optimizer
from src.evoMI.qehvi_optimizer import qehvi_optimizer


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _default_reasoning_specs(limit=None):
    aime_limit = 5 if limit is None else limit
    gpqa_limit = 60 if limit is None else limit
    return [
        {
            "dataset_id": "opencompass/AIME2025",
            "split": "test",
            "text_field": "question",
            "subset_field": "subset",
            "subset_values": ["AIME2025-I"],
            "limit": aime_limit,
        },
        {
            "dataset_id": "opencompass/AIME2025",
            "split": "test",
            "text_field": "question",
            "subset_field": "subset",
            "subset_values": ["AIME2025-II"],
            "limit": aime_limit,
        },
        {
            "dataset_id": "AI-ModelScope/gpqa_diamond",
            "split": "train",
            "text_field": "Question",
            "limit": gpqa_limit,
        },
    ]


def _default_general_specs(limit=None):
    gpqa_limit = 60 if limit is None else limit
    return [
        {
            "dataset_id": "AI-ModelScope/gpqa_diamond",
            "split": "train",
            "text_field": "Question",
            "limit": gpqa_limit,
        }
    ]


def _resolve_eval_limits(eval_profile, full_eval_limits=None, eval_aime_limit=None, eval_gpqa_limit=None, eval_gsm8k_limit=None, eval_math_limit=None):
    if full_eval_limits is not None:
        return normalize_eval_limit(eval_profile, full_eval_limits)
    profile = str(eval_profile or "aime_gpqa").strip().lower()
    if profile == "gsm8k_gpqa":
        candidate = {
            "gsm8k": eval_gsm8k_limit if eval_gsm8k_limit is not None else 100,
            "gpqa_diamond": eval_gpqa_limit if eval_gpqa_limit is not None else 100,
        }
    elif profile == "math500_level5_gpqa":
        candidate = {
            "math_500": eval_math_limit,
            "gpqa_diamond": eval_gpqa_limit,
        }
    else:
        candidate = {
            "aime25": eval_aime_limit,
            "gpqa_diamond": eval_gpqa_limit,
        }
    return normalize_eval_limit(profile, candidate)


def _align_prior_length(values, dim):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] == dim:
        return arr
    if arr.shape[0] < dim:
        repeat_n = int(np.ceil(dim / arr.shape[0]))
        return np.tile(arr, repeat_n)[:dim]
    return arr[:dim]


def _resolve_runtime_device(device, available_gpus):
    if isinstance(device, torch.device):
        device = str(device)
    if isinstance(device, int):
        device = f"cuda:{int(device)}"
    device_str = str(device or "cpu").strip().lower()
    if device_str == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    if device_str == "cuda":
        if available_gpus:
            return f"cuda:{int(available_gpus[0])}"
        return "cuda:0"
    if device_str.startswith("cuda:"):
        try:
            int(device_str.split(":", 1)[1])
            return device_str
        except ValueError:
            if available_gpus:
                return f"cuda:{int(available_gpus[0])}"
            return "cuda:0"
    if available_gpus:
        return f"cuda:{int(available_gpus[0])}"
    return "cuda:0"


def _build_blueprint_priors(**kwargs):
    from src.evoMI.mi_opt_optimizer import _build_blueprint_priors as _impl

    return _impl(**kwargs)


def _save_settings(params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "settings.json"), "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, ensure_ascii=False)


def _save_optimization_results(result_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pareto_x = result_dict.get("pareto_x", np.array([]))
    pareto_y = result_dict.get("pareto_y", np.array([]))
    np.save(os.path.join(output_dir, "pareto_decision_variables.npy"), pareto_x)
    np.save(os.path.join(output_dir, "pareto_objectives.npy"), pareto_y)
    np.save(os.path.join(output_dir, "all_evaluated_variables.npy"), result_dict.get("all_x", np.array([])))
    np.save(os.path.join(output_dir, "all_evaluated_objectives.npy"), result_dict.get("all_y", np.array([])))
    results = {
        "pareto_solutions": [
            {"decision_variables": x.tolist(), "objectives": y.tolist()}
            for x, y in zip(pareto_x, pareto_y)
        ],
        "hypervolume_history": result_dict.get("hypervolume_history", []),
        "total_evaluations": len(result_dict.get("all_y", [])),
        "best_hypervolume": max(result_dict.get("hypervolume_history", [0])) if result_dict.get("hypervolume_history") else 0,
    }
    with open(os.path.join(output_dir, "optimization_results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


def _visualize_optimization_results(result_dict, output_dir):
    from src.evoMI.optimization_reporting import reporter

    reporter.visualize_optimization_results(result_dict, output_dir)


def _create_optimizer_config(
    custom_initial_solutions=None,
    num_blocks=8,
    num_objectives=2,
    BATCH_SIZE=4,
    NUM_RESTARTS=10,
    RAW_SAMPLES=512,
    MC_SAMPLES=128,
    N_BATCH=50,
    verbose=True,
    device="cpu",
    dtype=torch.double,
    initial_samples=8,
    noise_level=0.0001,
    run_id="block_test0",
    checkpoint_dir="./checkpoints",
    optimize_density=1,
):
    if optimize_density == 1:
        dim = num_blocks + 1
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim]).to(dtype)
    elif optimize_density == 2:
        dim = (num_blocks + 1) * 2
        bounds = torch.tensor(
            [[0.0] * (num_blocks + 1) + [0.6] * (num_blocks + 1), [1.0] * (num_blocks + 1) + [1.0] * (num_blocks + 1)]
        ).to(dtype)
    elif optimize_density == 3:
        dim = (num_blocks + 1) * 3
        bounds = torch.tensor(
            [
                [0.0] * (num_blocks + 1) + [0.6] * (num_blocks + 1) + [0.00] * (num_blocks + 1),
                [1.0] * (num_blocks + 1) + [1.0] * (num_blocks + 1) + [0.05] * (num_blocks + 1),
            ]
        ).to(dtype)
    else:
        dim = num_blocks + 1
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim]).to(dtype)

    return {
        "dim": dim,
        "num_objectives": num_objectives,
        "bounds": bounds,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_RESTARTS": NUM_RESTARTS,
        "RAW_SAMPLES": RAW_SAMPLES,
        "MC_SAMPLES": MC_SAMPLES,
        "N_BATCH": N_BATCH,
        "verbose": verbose,
        "device": device,
        "dtype": dtype,
        "initial_samples": initial_samples,
        "noise_level": noise_level,
        "ref_point": torch.tensor([-0.2, -0.2]).to(dtype),
        "run_id": run_id,
        "checkpoint_dir": checkpoint_dir,
        "custom_initial_solutions": custom_initial_solutions,
    }


def _clean_generated_model_dirs(cache_dir, model_dirs):
    cache_dir_abs = os.path.abspath(cache_dir)
    for model_dir in model_dirs:
        model_dir_abs = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir_abs):
            continue
        if os.path.commonpath([cache_dir_abs, model_dir_abs]) != cache_dir_abs:
            continue
        shutil.rmtree(model_dir_abs)
        parent_dir = os.path.dirname(model_dir_abs)
        while os.path.commonpath([cache_dir_abs, parent_dir]) == cache_dir_abs and parent_dir != cache_dir_abs:
            if os.path.isdir(parent_dir) and len(os.listdir(parent_dir)) == 0:
                os.rmdir(parent_dir)
                parent_dir = os.path.dirname(parent_dir)
                continue
            break


def _clean_eval_task_dirs(cache_dir):
    cache_dir_abs = os.path.abspath(cache_dir)
    if not os.path.isdir(cache_dir_abs):
        return
    for root, dir_names, _ in os.walk(cache_dir_abs, topdown=False):
        for dir_name in dir_names:
            child_path = os.path.join(root, dir_name)
            if os.path.isdir(child_path) and dir_name.startswith("merged_model_"):
                shutil.rmtree(child_path)
        if root != cache_dir_abs and os.path.isdir(root) and len(os.listdir(root)) == 0:
            os.rmdir(root)


def _read_process_table():
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,args="],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []
    process_rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        process_rows.append({
            "pid": pid,
            "ppid": ppid,
            "command": parts[2],
        })
    return process_rows


def _select_stale_vllm_pool_processes(process_rows, pool_root, current_pid=None):
    pool_root_abs = os.path.abspath(pool_root)
    current_pid = os.getpid() if current_pid is None else int(current_pid)
    process_map = {row["pid"]: row for row in process_rows if isinstance(row, dict)}
    vllm_pids = set()
    launcher_pids = set()
    for row in process_map.values():
        pid = int(row["pid"])
        if pid == current_pid:
            continue
        command = str(row.get("command", ""))
        if "vllm.entrypoints.openai.api_server" in command and pool_root_abs in command:
            vllm_pids.add(pid)
    if len(vllm_pids) == 0:
        return []
    children_by_parent = {}
    for row in process_map.values():
        children_by_parent.setdefault(int(row["ppid"]), []).append(int(row["pid"]))
    for pid in list(vllm_pids):
        parent_pid = int(process_map[pid]["ppid"])
        parent_row = process_map.get(parent_pid)
        if parent_row is None or parent_pid == current_pid:
            continue
        parent_command = str(parent_row.get("command", ""))
        if "python" not in os.path.basename(parent_command.split(" ", 1)[0]):
            continue
        child_pids = [
            child_pid
            for child_pid in children_by_parent.get(parent_pid, [])
            if child_pid != current_pid
        ]
        if child_pids and all(child_pid in vllm_pids for child_pid in child_pids):
            launcher_pids.add(parent_pid)
    ordered_pids = sorted(vllm_pids) + sorted(launcher_pids)
    return ordered_pids


def _terminate_stale_vllm_pool_processes(pool_root):
    target_pids = _select_stale_vllm_pool_processes(_read_process_table(), pool_root)
    if len(target_pids) == 0:
        return []
    terminated_pids = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        remaining_pids = []
        for pid in target_pids:
            try:
                os.kill(pid, 0)
            except OSError:
                if pid not in terminated_pids:
                    terminated_pids.append(pid)
                continue
            try:
                os.kill(pid, sig)
            except OSError:
                if pid not in terminated_pids:
                    terminated_pids.append(pid)
                continue
            remaining_pids.append(pid)
        if len(remaining_pids) == 0:
            break
        time.sleep(2 if sig == signal.SIGTERM else 1)
        next_remaining = []
        for pid in remaining_pids:
            try:
                os.kill(pid, 0)
                next_remaining.append(pid)
            except OSError:
                if pid not in terminated_pids:
                    terminated_pids.append(pid)
        target_pids = next_remaining
        if len(target_pids) == 0:
            break
    return terminated_pids


def _create_iteration_callback(cache_dir, cleanup_interval=1):
    def callback(iteration, x, y, hypervolume):
        cleanup_paths = getattr(callback, "cleanup_paths", [])
        async_mode = bool(getattr(callback, "async_mode", False))
        callback.cleanup_paths = []
        if cleanup_paths:
            _clean_generated_model_dirs(cache_dir, cleanup_paths)
            return
        if (not async_mode) and (iteration + 1) % cleanup_interval == 0:
            _clean_eval_task_dirs(cache_dir)

    callback.cleanup_paths = []
    callback.async_mode = False
    return callback


def _normalize_algorithm_name(algorithm):
    value = str(algorithm).strip().lower().replace("-", "_")
    alias_map = {
        "priorbo": "priorbo",
        "prior_bo": "priorbo",
        "prior_bo_qnehvi": "priorbo",
        "qnehvi": "qnehvi",
        "qehvi": "qnehvi",
        "momm": "momm",
        "mm_mo": "momm",
        "mmmo": "momm",
        "moead_cmaes": "moead_cmaes",
        "moead": "moead_cmaes",
        "moead_prior": "moead_cmaes",
    }
    if value not in alias_map:
        raise ValueError(f"不支持的算法: {algorithm}")
    return alias_map[value]


def _get_algorithm_preset(algorithm):
    requested = str(algorithm).strip().lower().replace("-", "_")
    normalized = _normalize_algorithm_name(requested)
    preset = {
        "requested_algorithm": requested,
        "normalized_algorithm": normalized,
        "run_prefix": requested,
        "async_mode": None,
        "enable_importance_update": None,
        "enable_importance_guidance": None,
        "enable_importance_weighted_acq": None,
        "enable_importance_prior_cutoff": None,
        "enable_gap_aware_postprocess": None,
    }
    return preset


def _safe_gpu_count(available_gpus):
    if available_gpus is None:
        return max(1, torch.cuda.device_count())
    return max(1, len(available_gpus))


def _filter_callable_kwargs(func, kwargs):
    signature = inspect.signature(func)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _build_result_dict(result):
    return {
        "pareto_x": result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0],
        "pareto_y": result[1].cpu().numpy() if isinstance(result[1], torch.Tensor) else result[1],
        "all_x": result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0],
        "all_y": result[1].cpu().numpy() if isinstance(result[1], torch.Tensor) else result[1],
        "all_metrics": result[2],
        "hypervolume_history": result[3] if len(result) > 3 else [],
        "problem_ref_point": result[4].tolist() if isinstance(result[4], torch.Tensor) else result[4],
        "run_id": result[5] if len(result) > 5 else None,
    }


def _to_json_compatible(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    return value


def _sanitize_metric_items(metric_items, expected_count):
    if not isinstance(metric_items, list):
        return [{} for _ in range(expected_count)]
    sanitized_items = []
    for item in metric_items[:expected_count]:
        if not isinstance(item, dict):
            sanitized_items.append({})
            continue
        sanitized_items.append(
            {
                str(key): _to_json_compatible(value)
                for key, value in item.items()
                if key != "_cleanup_model_dirs"
            }
        )
    while len(sanitized_items) < expected_count:
        sanitized_items.append({})
    return sanitized_items


def _collect_cleanup_paths(metric_items):
    cleanup_paths = []
    if not isinstance(metric_items, list):
        return cleanup_paths
    for item in metric_items:
        if not isinstance(item, dict):
            continue
        item_cleanup_paths = item.get("_cleanup_model_dirs", [])
        if not isinstance(item_cleanup_paths, list):
            continue
        cleanup_paths.extend(
            path for path in item_cleanup_paths if isinstance(path, str) and len(path) > 0
        )
    if len(cleanup_paths) == 0:
        return cleanup_paths
    return list(dict.fromkeys(cleanup_paths))


def _compute_initial_hypervolume(objectives, ref_point, dtype, device):
    objective_tensor = torch.as_tensor(objectives, dtype=dtype, device=device)
    if objective_tensor.ndim != 2 or objective_tensor.shape[0] == 0:
        return None
    ref_tensor = torch.as_tensor(ref_point, dtype=dtype, device=device).reshape(-1)
    if ref_tensor.shape[0] != objective_tensor.shape[1]:
        return None
    partitioning = FastNondominatedPartitioning(ref_point=ref_tensor, Y=objective_tensor)
    return float(partitioning.compute_hypervolume().item())


def _expand_custom_initial_solutions(custom_initial_solutions, dim, bounds, dtype, device):
    if custom_initial_solutions is None:
        return torch.empty((0, dim), dtype=dtype, device=device)
    initial_tensor = torch.as_tensor(custom_initial_solutions, dtype=dtype, device=device)
    if initial_tensor.numel() == 0:
        return torch.empty((0, dim), dtype=dtype, device=device)
    if initial_tensor.ndim == 0:
        initial_tensor = initial_tensor.unsqueeze(0)
    if initial_tensor.ndim == 1:
        if initial_tensor.numel() == dim:
            initial_tensor = initial_tensor.unsqueeze(0)
        else:
            initial_tensor = initial_tensor.unsqueeze(1).repeat(1, dim)
    else:
        initial_tensor = initial_tensor.reshape(initial_tensor.shape[0], -1)
    if initial_tensor.shape[-1] != dim:
        raise ValueError(f"自定义初始样本维度应为 {dim}，当前为 {initial_tensor.shape[-1]}")
    lower = bounds[0].to(dtype=dtype, device=device)
    upper = bounds[1].to(dtype=dtype, device=device)
    return initial_tensor.clamp(lower, upper)


def _generate_shared_initial_points(bounds, dim, initial_samples, custom_initial_solutions, seed, dtype, device):
    if initial_samples <= 0:
        return torch.empty((0, dim), dtype=dtype, device=device)
    lower = bounds[0].to(dtype=dtype, device=device)
    upper = bounds[1].to(dtype=dtype, device=device)
    custom_points = _expand_custom_initial_solutions(
        custom_initial_solutions=custom_initial_solutions,
        dim=dim,
        bounds=bounds,
        dtype=dtype,
        device=device,
    )
    if custom_points.shape[0] >= initial_samples:
        return custom_points[:initial_samples].clone()
    remaining = int(initial_samples - custom_points.shape[0])
    sobol_engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=int(seed))
    sobol_points = sobol_engine.draw(remaining).to(dtype=dtype, device=device)
    sobol_points = lower.unsqueeze(0) + (upper - lower).unsqueeze(0) * sobol_points
    if custom_points.shape[0] == 0:
        return sobol_points
    return torch.cat([custom_points, sobol_points], dim=0)


def _parse_initial_objective_result(result, batch_size, dtype, device):
    if isinstance(result, tuple) and len(result) == 3:
        obj_true, metric_items, cleanup_paths = result
    elif isinstance(result, tuple) and len(result) == 2:
        obj_true, metric_items = result
        cleanup_paths = _collect_cleanup_paths(metric_items)
    else:
        obj_true = result
        metric_items = [{} for _ in range(batch_size)]
        cleanup_paths = []
    obj_true = torch.as_tensor(obj_true, dtype=dtype, device=device)
    metric_items = _sanitize_metric_items(metric_items, batch_size)
    cleanup_paths = [
        path for path in cleanup_paths
        if isinstance(path, str) and len(path) > 0
    ]
    return obj_true, metric_items, list(dict.fromkeys(cleanup_paths))


def _build_shared_initial_cache_payload(
    dim,
    num_objectives,
    bounds,
    initial_samples,
    custom_initial_solutions,
    seed,
    num_blocks,
    alpha,
    beta,
    base_model_path,
    task_model_paths,
    fusion_method,
    optimize_density,
):
    return {
        "cache_type": "shared_initial_dataset_v1",
        "dim": int(dim),
        "num_objectives": int(num_objectives),
        "bounds": _to_json_compatible(bounds),
        "initial_samples": int(initial_samples),
        "custom_initial_solutions": _to_json_compatible(custom_initial_solutions),
        "seed": int(seed),
        "num_blocks": int(num_blocks),
        "alpha": float(alpha),
        "beta": float(beta),
        "base_model_path": os.path.abspath(base_model_path) if base_model_path is not None else None,
        "task_model_paths": [os.path.abspath(path) for path in (task_model_paths or [])],
        "fusion_method": fusion_method,
        "optimize_density": optimize_density,
    }


def _load_or_create_shared_initial_dataset(
    objective_function,
    checkpoint_root,
    cache_dir,
    eval_config,
    bounds,
    dim,
    num_objectives,
    initial_samples,
    custom_initial_solutions,
    seed,
    num_blocks,
    alpha,
    beta,
    base_model_path,
    task_model_paths,
    fusion_method,
    optimize_density,
    ref_point,
    device,
    dtype,
    verbose,
):
    if initial_samples is None or int(initial_samples) <= 0:
        return None
    model_cache_dir = os.path.abspath(cache_dir)
    cache_payload = _build_shared_initial_cache_payload(
        dim=dim,
        num_objectives=num_objectives,
        bounds=bounds,
        initial_samples=initial_samples,
        custom_initial_solutions=custom_initial_solutions,
        seed=seed,
        num_blocks=num_blocks,
        alpha=alpha,
        beta=beta,
        base_model_path=base_model_path,
        task_model_paths=task_model_paths,
        fusion_method=fusion_method,
        optimize_density=optimize_density,
    )
    cache_key = hashlib.md5(
        json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    shared_initial_cache_dir = os.path.join(
        checkpoint_root,
        "evaluation_cache",
        "shared_initial",
        build_eval_cache_namespace(eval_config),
    )
    os.makedirs(shared_initial_cache_dir, exist_ok=True)
    cache_path = os.path.join(shared_initial_cache_dir, f"{cache_key}.json")
    cached = load_cached_results(cache_path, expected_eval_config=eval_config)
    if cached is not None and cached.get("problem_config") == cache_payload:
        initial_dataset = cached.get("initial_dataset", {})
        train_x = torch.as_tensor(initial_dataset.get("decision_variables", []), dtype=dtype, device=device)
        train_obj_true = torch.as_tensor(initial_dataset.get("objectives", []), dtype=dtype, device=device)
        cached_hv = initial_dataset.get("hypervolume")
        if train_x.ndim == 2 and train_x.shape == (int(initial_samples), int(dim)) and train_obj_true.shape == (int(initial_samples), int(num_objectives)):
            if cached_hv is None:
                cached_hv = _compute_initial_hypervolume(
                    objectives=train_obj_true,
                    ref_point=ref_point,
                    dtype=dtype,
                    device=device,
                )
            if verbose:
                print(f"复用统一初始样本缓存: {cache_path}")
            return {
                "decision_variables": train_x.detach().cpu(),
                "objectives": train_obj_true.detach().cpu(),
                "metrics": [{} for _ in range(train_x.shape[0])],
                "hypervolume": cached_hv,
                "cache_path": cache_path,
                "source": "cache",
            }
    train_x = _generate_shared_initial_points(
        bounds=bounds,
        dim=dim,
        initial_samples=int(initial_samples),
        custom_initial_solutions=custom_initial_solutions,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    objective_signature = inspect.signature(objective_function)
    objective_kwargs = {}
    if "eval_limit" in objective_signature.parameters:
        objective_kwargs["eval_limit"] = eval_config.get("limit")
    if "eval_mode" in objective_signature.parameters:
        objective_kwargs["eval_mode"] = "full"
    if "estimated_tokens" in objective_signature.parameters:
        objective_kwargs["estimated_tokens"] = None
    objective_result = objective_function(train_x, **objective_kwargs)
    obj_true, train_info, cleanup_paths = _parse_initial_objective_result(
        objective_result,
        batch_size=train_x.shape[0],
        dtype=dtype,
        device=device,
    )
    _clean_generated_model_dirs(
        model_cache_dir,
        cleanup_paths,
    )
    _clean_eval_task_dirs(model_cache_dir)
    initial_hv = _compute_initial_hypervolume(
        objectives=obj_true,
        ref_point=ref_point,
        dtype=dtype,
        device=device,
    )
    initial_dataset = {
        "decision_variables": train_x.detach().cpu().tolist(),
        "objectives": obj_true.detach().cpu().tolist(),
        "hypervolume": initial_hv,
    }
    save_results_to_cache(
        results={
            "problem_config": cache_payload,
            "initial_dataset": initial_dataset,
        },
        cache_path=cache_path,
        eval_config=eval_config,
    )
    if verbose:
        print(f"生成并缓存统一初始样本: {cache_path}")
    return {
        "decision_variables": train_x.detach().cpu(),
        "objectives": obj_true.detach().cpu(),
        "metrics": train_info,
        "hypervolume": initial_hv,
        "cache_path": cache_path,
        "source": "new",
    }


def _ensure_checkpoint_visualization_compatibility(
    run_dir,
    train_info,
    hvs,
    initial_samples,
    batch_size,
    scheduler_gpu_count,
    async_mode=False,
):
    scheduler_path = os.path.join(run_dir, "scheduler_usage.json")
    hv_path = os.path.join(run_dir, "hypervolume_curve.json")
    expected_scheduler_history = build_sync_scheduler_history(
        train_info=train_info,
        initial_samples=initial_samples,
        batch_size=batch_size,
        scheduler_gpu_count=scheduler_gpu_count,
    )
    expected_hv_curve = build_hv_curve(
        hvs=hvs,
        initial_samples=initial_samples,
        batch_size=batch_size,
        total_evaluations=len(train_info),
    )

    scheduler_history = None
    if os.path.exists(scheduler_path):
        with open(scheduler_path, "r", encoding="utf-8") as handle:
            loaded_scheduler_history = json.load(handle)
        if isinstance(loaded_scheduler_history, list):
            scheduler_history = loaded_scheduler_history
    if scheduler_history is None:
        scheduler_history, hv_curve = rebuild_runtime_reports_from_checkpoint(
            run_dir=run_dir,
            initial_samples=initial_samples,
            batch_size=batch_size,
            scheduler_gpu_count=scheduler_gpu_count,
            async_mode=async_mode,
        )
        return
    else:
        is_sync_history = all(
            isinstance(record, dict) and str(record.get("mode", "sync")).lower() == "sync"
            for record in scheduler_history
        )
        has_absolute_task = any(
            isinstance(task, dict) and ("absolute_start" in task)
            for record in scheduler_history
            if isinstance(record, dict)
            for task in record.get("tasks", [])
        )
        if is_sync_history and bool(async_mode) and not has_absolute_task:
            scheduler_history, hv_curve = rebuild_runtime_reports_from_checkpoint(
                run_dir=run_dir,
                initial_samples=initial_samples,
                batch_size=batch_size,
                scheduler_gpu_count=scheduler_gpu_count,
                async_mode=True,
            )
            return
        if is_sync_history and len(scheduler_history) != len(expected_scheduler_history):
            scheduler_history = expected_scheduler_history

    hv_curve = None
    if os.path.exists(hv_path):
        with open(hv_path, "r", encoding="utf-8") as handle:
            loaded_hv_curve = json.load(handle)
        if isinstance(loaded_hv_curve, list):
            hv_curve = loaded_hv_curve
    if hv_curve is None:
        hv_curve = expected_hv_curve
    else:
        current_last_eval = int(hv_curve[-1]["evaluations"]) if len(hv_curve) > 0 else 0
        expected_last_eval = int(expected_hv_curve[-1]["evaluations"]) if len(expected_hv_curve) > 0 else 0
        if len(hv_curve) != len(expected_hv_curve) or current_last_eval != expected_last_eval:
            hv_curve = expected_hv_curve

    save_runtime_reports(
        run_dir=run_dir,
        scheduler_history=scheduler_history,
        hv_curve=hv_curve,
    )
    ensure_latest_checkpoint_metadata(
        run_dir=run_dir,
        scheduler_history=scheduler_history,
        hv_curve=hv_curve,
    )


def _build_objective_bundle(
    algorithm,
    objective_function,
    custom_initial_solutions,
    num_blocks,
    checkpoint_root,
    cache_dir,
    alpha,
    beta,
    base_model,
    expert_model,
    base_model_path,
    task_model_paths,
    fusion_method,
    optimize_density,
    max_tokens,
    max_model_len,
    available_gpus,
    eval_profile,
    eval_limits,
    eval_repeats,
    eval_seed,
):
    if objective_function is not None:
        if hasattr(objective_function, "get_idle_gpu_count"):
            objective_callable = objective_function
        else:
            objective_callable = objective_function
        return {
            "objective_function": objective_callable,
            "merged_blocks": None,
            "base_model_results": None,
            "expert_model_results": None,
            "requires_shutdown": False,
            "shutdown": None,
        }

    from src.evoMI.mi_block_fusion import calculate_merged_blocks
    from src.evoMI.mi_opt_saasbo2 import (
        collect_newly_completed_tasks,
        get_idle_gpu_count,
        get_shared_vllm_manager,
        initialize_model_evaluations,
        model_merge_optimization_function,
        set_available_gpus,
        shutdown_shared_vllm_manager,
        start_async_model_merge_evaluation_session,
    )

    set_available_gpus(available_gpus)
    get_shared_vllm_manager(max_model_len=max_model_len if max_model_len is not None else (max_tokens + 3000))

    merged_blocks = calculate_merged_blocks(
        task_model_paths=task_model_paths,
        num_blocks=num_blocks,
        alpha=alpha,
        beta=beta,
        checkpoint_dir=cache_dir,
        metric="L2-norm",
        partition_method="hybrid",
    )
    base_model_results, expert_model_results = initialize_model_evaluations(
        base_model,
        expert_model,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        eval_limit=eval_limits,
        eval_profile=eval_profile,
        eval_repeats=eval_repeats,
        eval_seed=eval_seed,
        cache_root=checkpoint_root,
    )

    def wrapped_optimization_function(x, eval_limit=None, eval_mode="full", estimated_tokens=None):
        return model_merge_optimization_function(
            x,
            merged_blocks=merged_blocks,
            num_blocks=num_blocks,
            cache_dir=cache_dir,
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            fusion_method=fusion_method,
            base_model_results=base_model_results,
            expert_model_results=expert_model_results,
            optimize_density=optimize_density,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            eval_limit=eval_limit if eval_limit is not None else eval_limits,
            eval_mode=eval_mode,
            estimated_tokens=estimated_tokens,
            eval_profile=eval_profile,
            eval_repeats=eval_repeats,
            eval_seed=eval_seed,
        )

    def start_async_session(x, eval_limit=None, estimated_tokens=None):
        return start_async_model_merge_evaluation_session(
            decision_matrix=x,
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            base_output_dir=cache_dir,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            merged_blocks=merged_blocks,
            num_blocks=num_blocks,
            fusion_method=fusion_method,
            base_model_results=base_model_results,
            expert_model_results=expert_model_results,
            optimize_density=optimize_density,
            eval_limit=eval_limit if eval_limit is not None else eval_limits,
            estimated_tokens=estimated_tokens,
            eval_profile=eval_profile,
            eval_repeats=eval_repeats,
            eval_seed=eval_seed,
            finalize_series=True,
        )

    wrapped_optimization_function.start_async_session = start_async_session

    if algorithm == "priorbo":
        wrapped_optimization_function.get_idle_gpu_count = get_idle_gpu_count
        wrapped_optimization_function.collect_newly_completed_tasks = collect_newly_completed_tasks

    return {
        "objective_function": wrapped_optimization_function,
        "merged_blocks": merged_blocks,
        "base_model_results": base_model_results,
        "expert_model_results": expert_model_results,
        "requires_shutdown": True,
        "shutdown": shutdown_shared_vllm_manager,
    }


def _prepare_algorithm_kwargs(
    algorithm,
    config,
    merged_blocks,
    optimize_density,
    num_blocks,
    initial_importance,
    enable_blueprint_priors,
    m_prior,
    u_prior,
    base_model_path,
    task_model_paths,
    device,
    output_dir,
    reasoning_specs,
    general_specs,
    proxy_batch_size,
    proxy_max_length,
    proxy_pooling,
    proxy_dtype,
    patch_topk,
    beta_shared,
    lambda_spec,
    w_param,
    w_shared,
    w_patch_abs,
    v_spec,
    v_ratio,
    v_patch,
    use_saas,
    enable_importance_prior,
    enable_importance_update,
    enable_importance_guidance,
    enable_importance_weighted_acq,
    enable_importance_prior_cutoff,
    importance_prior_cutoff_evals,
    learning_rate,
    async_mode,
    enable_pending_aware_acq,
):
    optimizer_map = {
        "priorbo": prior_bo_optimizer,
        "qnehvi": qehvi_optimizer,
        "momm": mm_mo_optimizer,
        "moead_cmaes": moead_cmaes_prior_optimizer,
    }
    optimizer_fn = optimizer_map[algorithm]
    optimizer_kwargs = dict(config)

    effective_proxy_metrics = None
    if algorithm in {"priorbo", "moead_cmaes"}:
        effective_m_prior = m_prior
        effective_u_prior = u_prior
        needs_blueprint_priors = (
            enable_blueprint_priors
            and merged_blocks is not None
            and (effective_m_prior is None or effective_u_prior is None)
        )
        if needs_blueprint_priors:
            reason_model_path = task_model_paths[0] if len(task_model_paths) > 0 else base_model_path
            effective_m_prior, effective_u_prior, effective_proxy_metrics = _build_blueprint_priors(
                base_model_path=base_model_path,
                reason_model_path=reason_model_path,
                device=device,
                output_dir=output_dir,
                reasoning_specs=reasoning_specs,
                general_specs=general_specs,
                proxy_batch_size=proxy_batch_size,
                proxy_max_length=proxy_max_length,
                proxy_pooling=proxy_pooling,
                proxy_dtype=proxy_dtype,
                patch_topk=patch_topk,
                beta_shared=beta_shared,
                lambda_spec=lambda_spec,
                w_param=w_param,
                w_shared=w_shared,
                w_patch_abs=w_patch_abs,
                v_spec=v_spec,
                v_ratio=v_ratio,
                v_patch=v_patch,
            )
        if effective_m_prior is not None:
            optimizer_kwargs["m_prior"] = _align_prior_length(effective_m_prior, config["dim"])
        if effective_u_prior is not None:
            optimizer_kwargs["u_prior"] = _align_prior_length(effective_u_prior, config["dim"])

    return optimizer_fn, _filter_callable_kwargs(optimizer_fn, optimizer_kwargs)


def main_optimization(
    algorithm="priorbo",
    objective_function=None,
    custom_initial_solutions=None,
    num_blocks=36,
    num_objectives=2,
    BATCH_SIZE=4,
    NUM_RESTARTS=10,
    RAW_SAMPLES=512,
    MC_SAMPLES=128,
    N_BATCH=20,
    verbose=True,
    device="cuda",
    dtype=torch.double,
    initial_samples=4,
    noise_level=0.0001,
    run_id=None,
    checkpoint_root="./checkpoints",
    cache_dir="output/mi_optimization_temp",
    alpha=1.0,
    beta=0.0,
    base_model=None,
    expert_model=None,
    base_model_path="models/Qwen3-4B-Instruct-2507",
    task_model_paths=None,
    fusion_method="task_arithmetic",
    optimize_density=1,
    seed=42,
    rho=0.5,
    topk=6,
    n_groups=4,
    enable_grouping=False,
    max_tokens=35000,
    max_model_len=48000,
    available_gpus=None,
    reasoning_specs=None,
    general_specs=None,
    reasoning_limit=None,
    general_limit=None,
    proxy_batch_size=2,
    proxy_max_length=256,
    proxy_pooling="last_token",
    proxy_dtype="float16",
    patch_topk=8,
    beta_shared=1.0,
    lambda_spec=1.0,
    w_param=0.30,
    w_shared=0.40,
    w_patch_abs=0.30,
    v_spec=0.40,
    v_ratio=0.20,
    v_patch=0.40,
    max_evaluations=88,
    eval_profile="aime_gpqa",
    eval_aime_limit=4,
    eval_gsm8k_limit=None,
    eval_math_limit=None,
    eval_gpqa_limit=20,
    eval_repeats=None,
    async_mode=False,
    wait_for_completion_threshold=0.15,
    full_eval_limits=None,
    enable_blueprint_priors=True,
    m_prior=None,
    u_prior=None,
    initial_importance=None,
    use_saas=True,
    enable_importance_prior=True,
    enable_importance_update=None,
    enable_importance_guidance=None,
    enable_importance_weighted_acq=False,
    enable_importance_prior_cutoff=None,
    importance_prior_cutoff_evals=24,
    learning_rate=0.1,
    enable_gap_aware_postprocess=False,
    gap_reward_weight=0.25,
    gap_pending_penalty_weight=0.15,
    gap_candidate_pool_multiplier=3,
    enable_pending_aware_acq=True,
):
    algorithm_preset = _get_algorithm_preset(algorithm)
    algorithm = algorithm_preset["normalized_algorithm"]
    if algorithm_preset["async_mode"] is not None:
        async_mode = bool(algorithm_preset["async_mode"])
    if algorithm_preset["enable_importance_weighted_acq"] is not None:
        enable_importance_weighted_acq = bool(algorithm_preset["enable_importance_weighted_acq"])
    if algorithm_preset["enable_gap_aware_postprocess"] is not None:
        enable_gap_aware_postprocess = bool(algorithm_preset["enable_gap_aware_postprocess"])
    if enable_importance_update is None:
        preset_update = algorithm_preset["enable_importance_update"]
        enable_importance_update = True if preset_update is None else bool(preset_update)
    if enable_importance_guidance is None:
        preset_guidance = algorithm_preset["enable_importance_guidance"]
        enable_importance_guidance = True if preset_guidance is None else bool(preset_guidance)
    if enable_importance_prior_cutoff is None:
        preset_cutoff = algorithm_preset["enable_importance_prior_cutoff"]
        enable_importance_prior_cutoff = True if preset_cutoff is None else bool(preset_cutoff)
    _set_global_seed(seed)

    if run_id is None:
        run_id = f"{algorithm_preset['run_prefix']}_{time.strftime('%Y%m%d_%H%M%S')}"
    if base_model is None:
        base_model = ["models/Qwen3-4B-Instruct-2507", "models/Qwen3-4B-thinking-2507"]
    if expert_model is None:
        expert_model = ["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"]
    if task_model_paths is None:
        task_model_paths = ["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"]
    if reasoning_specs is None:
        reasoning_specs = _default_reasoning_specs(reasoning_limit)
    if general_specs is None:
        general_specs = _default_general_specs(general_limit)
    if available_gpus is None:
        available_gpus = list(range(torch.cuda.device_count())) if torch.cuda.device_count() > 0 else [0]
    runtime_device = _resolve_runtime_device(device, available_gpus)
    if runtime_device.startswith("cuda:"):
        torch.cuda.set_device(torch.device(runtime_device))
    eval_profile = str(eval_profile or "aime_gpqa").strip().lower()
    full_eval_limits = _resolve_eval_limits(
        eval_profile=eval_profile,
        full_eval_limits=full_eval_limits,
        eval_aime_limit=eval_aime_limit,
        eval_gpqa_limit=eval_gpqa_limit,
        eval_gsm8k_limit=eval_gsm8k_limit,
        eval_math_limit=eval_math_limit,
    )
    eval_config = build_eval_cache_config(
        eval_profile=eval_profile,
        eval_limit=full_eval_limits,
        repeats=eval_repeats,
        max_tokens=max_tokens,
        seed=seed,
    )
    eval_setting_id = build_eval_setting_id(eval_config)
    eval_metadata = build_eval_metadata(
        eval_profile=eval_profile,
        eval_limits=full_eval_limits,
        eval_repeats=eval_repeats,
        eval_setting_id=eval_setting_id,
    )

    start_time = time.time()
    checkpoint_root = os.path.abspath(checkpoint_root)
    cache_dir = os.path.abspath(cache_dir)
    run_dir = os.path.join(checkpoint_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    output_dir = os.path.join(run_dir, "output", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    _clean_eval_task_dirs(cache_dir)
    vllm_pool_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "output",
        "vllm_server_pool",
    )
    cleaned_vllm_pids = _terminate_stale_vllm_pool_processes(vllm_pool_root)
    if cleaned_vllm_pids:
        print(f"启动前已清理 {len(cleaned_vllm_pids)} 个历史残留 vLLM Python 进程")

    params = {
        "algorithm": algorithm,
        "custom_initial_solutions": custom_initial_solutions,
        "num_blocks": num_blocks,
        "num_objectives": num_objectives,
        "batch_size": BATCH_SIZE,
        "num_restarts": NUM_RESTARTS,
        "raw_samples": RAW_SAMPLES,
        "mc_samples": MC_SAMPLES,
        "n_batch": N_BATCH,
        "verbose": verbose,
        "device": runtime_device,
        "dtype": str(dtype),
        "initial_samples": initial_samples,
        "noise_level": noise_level,
        "run_id": run_id,
        "checkpoint_root": checkpoint_root,
        "cache_dir": cache_dir,
        "alpha": alpha,
        "beta": beta,
        "base_model": base_model,
        "expert_model": expert_model,
        "base_model_path": base_model_path,
        "task_model_paths": task_model_paths,
        "fusion_method": fusion_method,
        "optimize_density": optimize_density,
        "seed": seed,
        "rho": rho,
        "topk": topk,
        "n_groups": n_groups,
        "enable_grouping": enable_grouping,
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "available_gpus": available_gpus,
        "max_evaluations": max_evaluations,
        "eval_profile": eval_profile,
        "eval_aime_limit": eval_aime_limit,
        "eval_gsm8k_limit": eval_gsm8k_limit,
        "eval_math_limit": eval_math_limit,
        "eval_gpqa_limit": eval_gpqa_limit,
        "eval_repeats": eval_repeats,
        "full_eval_limits": full_eval_limits,
        "eval_setting_id": eval_setting_id,
        "async_mode": async_mode,
        "wait_for_completion_threshold": wait_for_completion_threshold,
        "enable_blueprint_priors": enable_blueprint_priors,
        "enable_importance_prior": enable_importance_prior,
        "enable_importance_update": enable_importance_update,
        "enable_importance_guidance": enable_importance_guidance,
        "enable_importance_weighted_acq": enable_importance_weighted_acq,
        "enable_importance_prior_cutoff": enable_importance_prior_cutoff,
        "importance_prior_cutoff_evals": importance_prior_cutoff_evals,
        "use_saas": use_saas,
        "enable_gap_aware_postprocess": enable_gap_aware_postprocess,
        "gap_reward_weight": gap_reward_weight,
        "gap_pending_penalty_weight": gap_pending_penalty_weight,
        "gap_candidate_pool_multiplier": gap_candidate_pool_multiplier,
        "enable_pending_aware_acq": enable_pending_aware_acq,
    }
    _save_settings(params, output_dir)

    config = _create_optimizer_config(
        custom_initial_solutions=custom_initial_solutions,
        num_blocks=num_blocks,
        num_objectives=num_objectives,
        BATCH_SIZE=BATCH_SIZE,
        NUM_RESTARTS=NUM_RESTARTS,
        RAW_SAMPLES=RAW_SAMPLES,
        MC_SAMPLES=MC_SAMPLES,
        N_BATCH=N_BATCH,
        verbose=verbose,
        device=runtime_device,
        dtype=dtype,
        initial_samples=initial_samples,
        noise_level=noise_level,
        run_id=run_id,
        checkpoint_dir=checkpoint_root,
        optimize_density=optimize_density,
    )
    config["seed"] = seed
    config["scheduler_gpu_count"] = _safe_gpu_count(available_gpus)
    config["max_evaluations"] = max_evaluations
    config["async_mode"] = async_mode
    config["wait_for_completion_threshold"] = wait_for_completion_threshold
    config["full_eval_limits"] = full_eval_limits
    config["eval_profile"] = eval_profile
    config["eval_repeats"] = eval_repeats
    config["eval_setting_id"] = eval_setting_id
    config["eval_metadata"] = eval_metadata
    config["rho"] = rho
    config["topk"] = topk
    config["n_groups"] = n_groups
    config["enable_grouping"] = enable_grouping
    config["enable_gap_aware_postprocess"] = enable_gap_aware_postprocess
    config["gap_reward_weight"] = gap_reward_weight
    config["gap_pending_penalty_weight"] = gap_pending_penalty_weight
    config["gap_candidate_pool_multiplier"] = gap_candidate_pool_multiplier
    config["enable_pending_aware_acq"] = enable_pending_aware_acq

    bundle = None
    try:
        bundle = _build_objective_bundle(
            algorithm=algorithm,
            objective_function=objective_function,
            custom_initial_solutions=custom_initial_solutions,
            num_blocks=num_blocks,
            checkpoint_root=checkpoint_root,
            cache_dir=cache_dir,
            alpha=alpha,
            beta=beta,
            base_model=base_model,
            expert_model=expert_model,
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            fusion_method=fusion_method,
            optimize_density=optimize_density,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            available_gpus=available_gpus,
            eval_profile=eval_profile,
            eval_limits=full_eval_limits,
            eval_repeats=eval_repeats,
            eval_seed=seed,
        )
        config["shared_initial_dataset"] = _load_or_create_shared_initial_dataset(
            objective_function=bundle["objective_function"],
            checkpoint_root=checkpoint_root,
            cache_dir=cache_dir,
            eval_config=eval_config,
            bounds=config["bounds"],
            dim=config["dim"],
            num_objectives=num_objectives,
            initial_samples=initial_samples,
            custom_initial_solutions=custom_initial_solutions,
            seed=seed,
            num_blocks=num_blocks,
            alpha=alpha,
            beta=beta,
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            fusion_method=fusion_method,
            optimize_density=optimize_density,
            ref_point=config["ref_point"],
            device=torch.device(runtime_device),
            dtype=dtype,
            verbose=verbose,
        )
        iteration_callback = _create_iteration_callback(cache_dir=cache_dir, cleanup_interval=1)
        optimizer_fn, optimizer_kwargs = _prepare_algorithm_kwargs(
            algorithm=algorithm,
            config=config,
            merged_blocks=bundle["merged_blocks"],
            optimize_density=optimize_density,
            num_blocks=num_blocks,
            initial_importance=initial_importance,
            enable_blueprint_priors=enable_blueprint_priors,
            m_prior=m_prior,
            u_prior=u_prior,
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            device=runtime_device,
            output_dir=output_dir,
            reasoning_specs=reasoning_specs,
            general_specs=general_specs,
            proxy_batch_size=proxy_batch_size,
            proxy_max_length=proxy_max_length,
            proxy_pooling=proxy_pooling,
            proxy_dtype=proxy_dtype,
            patch_topk=patch_topk,
            beta_shared=beta_shared,
            lambda_spec=lambda_spec,
            w_param=w_param,
            w_shared=w_shared,
            w_patch_abs=w_patch_abs,
            v_spec=v_spec,
            v_ratio=v_ratio,
            v_patch=v_patch,
            use_saas=use_saas,
            enable_importance_prior=enable_importance_prior,
            enable_importance_update=enable_importance_update,
            enable_importance_guidance=enable_importance_guidance,
            enable_importance_weighted_acq=enable_importance_weighted_acq,
            enable_importance_prior_cutoff=enable_importance_prior_cutoff,
            importance_prior_cutoff_evals=importance_prior_cutoff_evals,
            learning_rate=learning_rate,
            async_mode=async_mode,
            enable_pending_aware_acq=enable_pending_aware_acq,
        )
        result = optimizer_fn(
            bundle["objective_function"],
            iteration_callback=iteration_callback,
            **optimizer_kwargs,
        )
    finally:
        if bundle is not None and bundle.get("requires_shutdown", False):
            shutdown_fn = bundle.get("shutdown")
            if shutdown_fn is not None:
                shutdown_fn()

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\n优化完成！总耗时: {elapsed_time:.2f} 秒 ({elapsed_time / 3600:.2f} 小时)")

    _ensure_checkpoint_visualization_compatibility(
        run_dir=run_dir,
        train_info=result[2],
        hvs=result[3] if len(result) > 3 else [],
        initial_samples=initial_samples,
        batch_size=BATCH_SIZE,
        scheduler_gpu_count=config["scheduler_gpu_count"],
        async_mode=async_mode,
    )

    result_dict = _build_result_dict(result)
    _save_optimization_results(result_dict, output_dir)
    _visualize_optimization_results(result_dict, output_dir)

    if verbose:
        pareto_x = result_dict.get("pareto_x", np.array([]))
        hypervolume_history = result_dict.get("hypervolume_history", [])
        best_hypervolume = max(hypervolume_history) if len(hypervolume_history) > 0 else 0
        print(f"\n找到 {len(pareto_x)} 个帕累托最优解")
        print("\n=== 优化统计信息 ===")
        print(f"总评估次数: {len(result_dict.get('all_x', []))}")
        print(f"初始样本数: {config['initial_samples']}")
        print(f"最大评估次数: {config['max_evaluations']}")
        print(f"每次候选数: {config['BATCH_SIZE']}")
        print(f"最佳超体积: {best_hypervolume}")
        print(f"\n所有结果已保存到: {output_dir}")

    return result_dict


def _parse_custom_initial_solutions(raw_value):
    if raw_value is None:
        return None
    if os.path.exists(raw_value):
        data = np.load(raw_value)
        return data.tolist() if isinstance(data, np.ndarray) else data
    return np.asarray(ast.literal_eval(raw_value), dtype=np.float64).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一模型合并优化入口")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="priorbo",
        choices=[
            "priorbo",
            "qnehvi",
            "momm",
            "moead_cmaes",
        ],
    )
    parser.add_argument("--custom-initial-solutions", type=str, default=None)
    parser.add_argument("--num-blocks", type=int, default=36)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--num-objectives", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--raw-samples", type=int, default=512)
    parser.add_argument("--mc-samples", type=int, default=128)
    parser.add_argument("--n-batch", type=int, default=20)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--initial-samples", type=int, default=4)
    parser.add_argument("--noise-level", type=float, default=0.0001)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default="./checkpoints")
    parser.add_argument("--cache-dir", type=str, default="output/mi_optimization_temp")
    parser.add_argument("--base-model-path", type=str, default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--task-model-paths", nargs="+", default=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"])
    parser.add_argument("--base-model", nargs="+", default=["models/Qwen3-4B-Instruct-2507", "models/Qwen3-4B-thinking-2507"])
    parser.add_argument("--expert-model", nargs="+", default=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"])
    parser.add_argument("--fusion-method", type=str, default="task_arithmetic")
    parser.add_argument("--optimize-density", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--n-groups", type=int, default=4)
    parser.add_argument("--enable-grouping", action="store_true", default=False)
    parser.add_argument("--max-tokens", type=int, default=35000)
    parser.add_argument("--max-model-len", type=int, default=48000)
    parser.add_argument("--available-gpus", nargs="+", type=int, default=None)
    parser.add_argument("--max-evaluations", type=int, default=88)
    parser.add_argument("--eval-profile", type=str, default="aime_gpqa", choices=["aime_gpqa", "gsm8k_gpqa", "math500_level5_gpqa"])
    parser.add_argument("--eval-aime-limit", type=int, default=4)
    parser.add_argument("--eval-gsm8k-limit", type=int, default=None)
    parser.add_argument("--eval-math-limit", type=int, default=None)
    parser.add_argument("--eval-gpqa-limit", type=int, default=20)
    parser.add_argument("--async-mode", action="store_true", default=False)
    parser.add_argument("--wait-for-completion-threshold", type=float, default=0.15)
    parser.add_argument("--disable-blueprint-priors", action="store_true", default=False)
    parser.add_argument("--disable-importance-prior", action="store_true", default=False)
    parser.add_argument("--disable-importance-update", action="store_true", default=False)
    parser.add_argument("--enable-importance-update", action="store_true", default=False)
    parser.add_argument("--disable-importance-guidance", action="store_true", default=False)
    parser.add_argument("--enable-importance-guidance", action="store_true", default=False)
    parser.add_argument("--enable-importance-weighted-acq", action="store_true", default=False)
    parser.add_argument("--disable-saas", action="store_true", default=False)
    parser.add_argument("--disable-importance-prior-cutoff", action="store_true", default=False)
    parser.add_argument("--enable-importance-prior-cutoff", action="store_true", default=False)
    parser.add_argument("--importance-prior-cutoff-evals", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--enable-gap-aware-postprocess", action="store_true", default=False)
    parser.add_argument("--gap-reward-weight", type=float, default=0.25)
    parser.add_argument("--gap-pending-penalty-weight", type=float, default=0.15)
    parser.add_argument("--gap-candidate-pool-multiplier", type=int, default=3)
    parser.add_argument("--disable-pending-aware-acq", action="store_true", default=False)

    cli_args = parser.parse_args()
    algorithm_preset = _get_algorithm_preset(cli_args.algorithm)
    algorithm_name = algorithm_preset["normalized_algorithm"]
    enable_importance_update = True
    if cli_args.enable_importance_update:
        enable_importance_update = True
    if cli_args.disable_importance_update:
        enable_importance_update = False
    if algorithm_preset["enable_importance_update"] is not None:
        enable_importance_update = bool(algorithm_preset["enable_importance_update"])
    enable_importance_guidance = True
    if cli_args.enable_importance_guidance:
        enable_importance_guidance = True
    if cli_args.disable_importance_guidance:
        enable_importance_guidance = False
    if algorithm_preset["enable_importance_guidance"] is not None:
        enable_importance_guidance = bool(algorithm_preset["enable_importance_guidance"])
    enable_importance_prior_cutoff = True
    if cli_args.enable_importance_prior_cutoff:
        enable_importance_prior_cutoff = True
    if cli_args.disable_importance_prior_cutoff:
        enable_importance_prior_cutoff = False
    custom_initial_solutions = _parse_custom_initial_solutions(cli_args.custom_initial_solutions)
    main_optimization(
        algorithm=cli_args.algorithm,
        custom_initial_solutions=custom_initial_solutions,
        num_blocks=cli_args.num_blocks,
        num_objectives=cli_args.num_objectives,
        BATCH_SIZE=cli_args.batch_size,
        NUM_RESTARTS=cli_args.num_restarts,
        RAW_SAMPLES=cli_args.raw_samples,
        MC_SAMPLES=cli_args.mc_samples,
        N_BATCH=cli_args.n_batch,
        verbose=cli_args.verbose,
        device=cli_args.device,
        initial_samples=cli_args.initial_samples,
        noise_level=cli_args.noise_level,
        run_id=cli_args.run_id,
        checkpoint_root=cli_args.checkpoint_root,
        cache_dir=cli_args.cache_dir,
        alpha=cli_args.alpha,
        beta=cli_args.beta,
        base_model=cli_args.base_model,
        expert_model=cli_args.expert_model,
        base_model_path=cli_args.base_model_path,
        task_model_paths=cli_args.task_model_paths,
        fusion_method=cli_args.fusion_method,
        optimize_density=cli_args.optimize_density,
        seed=cli_args.seed,
        rho=cli_args.rho,
        topk=cli_args.topk,
        n_groups=cli_args.n_groups,
        enable_grouping=cli_args.enable_grouping,
        max_tokens=cli_args.max_tokens,
        max_model_len=cli_args.max_model_len,
        available_gpus=cli_args.available_gpus,
        max_evaluations=cli_args.max_evaluations,
        eval_profile=cli_args.eval_profile,
        eval_aime_limit=cli_args.eval_aime_limit,
        eval_gsm8k_limit=cli_args.eval_gsm8k_limit,
        eval_math_limit=cli_args.eval_math_limit,
        eval_gpqa_limit=cli_args.eval_gpqa_limit,
        async_mode=cli_args.async_mode,
        wait_for_completion_threshold=cli_args.wait_for_completion_threshold,
        enable_blueprint_priors=not cli_args.disable_blueprint_priors,
        enable_importance_prior=not cli_args.disable_importance_prior,
        enable_importance_update=enable_importance_update,
        enable_importance_guidance=enable_importance_guidance,
        use_saas=not cli_args.disable_saas,
        enable_importance_weighted_acq=cli_args.enable_importance_weighted_acq,
        enable_importance_prior_cutoff=enable_importance_prior_cutoff,
        importance_prior_cutoff_evals=cli_args.importance_prior_cutoff_evals,
        learning_rate=cli_args.learning_rate,
        enable_gap_aware_postprocess=cli_args.enable_gap_aware_postprocess,
        gap_reward_weight=cli_args.gap_reward_weight,
        gap_pending_penalty_weight=cli_args.gap_pending_penalty_weight,
        gap_candidate_pool_multiplier=cli_args.gap_candidate_pool_multiplier,
        enable_pending_aware_acq=not cli_args.disable_pending_aware_acq,
    )
