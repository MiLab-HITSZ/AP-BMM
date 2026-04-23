from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


RuntimeHistory = List[Dict[str, Any]]
HypervolumeCurve = List[Dict[str, float]]


def build_evaluation_metadata(
    eval_profile: Optional[str] = None,
    eval_limits: Optional[Dict[str, Any]] = None,
    eval_repeats: Optional[Dict[str, Any]] = None,
    eval_setting_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if eval_profile is None and eval_limits is None and eval_repeats is None and eval_setting_id is None:
        return None
    metadata = {
        "eval_profile": None if eval_profile is None else str(eval_profile),
        "eval_limits": eval_limits,
        "eval_repeats": eval_repeats,
        "eval_setting_id": eval_setting_id,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def extract_average_tokens(metric_item: Any) -> Optional[float]:
    if not isinstance(metric_item, dict):
        return None
    values: List[float] = []
    for dataset_metrics in metric_item.values():
        if not isinstance(dataset_metrics, dict):
            continue
        token_value = dataset_metrics.get("mean_tokens_num", None)
        if isinstance(token_value, (int, float)) and np.isfinite(token_value) and token_value > 0:
            values.append(float(token_value))
    if not values:
        return None
    return float(np.mean(values))


def estimate_metric_runtime(metric_item: Any, fallback: float = 1.0) -> float:
    avg_tokens = extract_average_tokens(metric_item)
    if avg_tokens is not None:
        return max(float(avg_tokens), 1.0)
    return max(float(fallback), 1e-6)


def build_synchronous_iteration_record(
    iteration: int,
    metric_items: Sequence[Any],
    gpu_count: int = 1,
    candidate_index_offset: int = 0,
) -> Dict[str, Any]:
    effective_gpu_count = max(1, int(gpu_count))
    gpu_loads = [0.0 for _ in range(effective_gpu_count)]
    tasks = []
    for local_index, metric_item in enumerate(metric_items):
        gpu_index = int(np.argmin(gpu_loads))
        duration = estimate_metric_runtime(metric_item)
        start_time = float(gpu_loads[gpu_index])
        tasks.append(
            {
                "gpu": int(gpu_index),
                "start": start_time,
                "duration": float(duration),
                "type": "full",
                "candidate_index": int(candidate_index_offset + local_index),
            }
        )
        gpu_loads[gpu_index] += duration
    wall_time = float(max(gpu_loads) if gpu_loads else 0.0)
    return {
        "iteration": int(iteration),
        "mode": "sync",
        "gpu_count": effective_gpu_count,
        "estimated_makespan": wall_time,
        "gpu_loads": [float(value) for value in gpu_loads],
        "tasks": tasks,
        "counts": {"full": int(len(tasks))},
        "wall_time_sec": wall_time,
    }


def build_synchronous_runtime_history(
    train_info: Sequence[Any],
    initial_samples: int,
    batch_size: int,
    gpu_count: int = 1,
) -> RuntimeHistory:
    history: RuntimeHistory = []
    total = len(train_info)
    if total <= 0:
        return history
    head = min(max(int(initial_samples), 0), total)
    if head > 0:
        history.append(
            build_synchronous_iteration_record(
                iteration=0,
                metric_items=train_info[:head],
                gpu_count=gpu_count,
                candidate_index_offset=0,
            )
        )
    offset = head
    iteration = 1
    effective_batch_size = max(1, int(batch_size))
    while offset < total:
        batch_metrics = train_info[offset : min(offset + effective_batch_size, total)]
        history.append(
            build_synchronous_iteration_record(
                iteration=iteration,
                metric_items=batch_metrics,
                gpu_count=gpu_count,
                candidate_index_offset=offset,
            )
        )
        offset += len(batch_metrics)
        iteration += 1
    return history


def build_asynchronous_runtime_history(
    train_info: Sequence[Any],
    initial_samples: int,
    gpu_count: int = 1,
) -> RuntimeHistory:
    history: RuntimeHistory = []
    total = len(train_info)
    if total <= 0:
        return history
    head = min(max(int(initial_samples), 0), total)
    if head > 0:
        init_record = build_synchronous_iteration_record(
            iteration=0,
            metric_items=train_info[:head],
            gpu_count=gpu_count,
            candidate_index_offset=0,
        )
        init_record["mode"] = "init"
        history.append(init_record)
    effective_gpu_count = max(1, int(gpu_count))
    gpu_available = [0.0 for _ in range(effective_gpu_count)]
    offset = head
    while offset < total:
        metric_item = train_info[offset]
        gpu_index = int(np.argmin(gpu_available))
        duration = estimate_metric_runtime(metric_item)
        absolute_start = float(gpu_available[gpu_index])
        gpu_available[gpu_index] += duration
        gpu_loads = [0.0 for _ in range(effective_gpu_count)]
        gpu_loads[gpu_index] = float(duration)
        history.append(
            {
                "iteration": int(offset + 1),
                "mode": "async",
                "gpu_count": effective_gpu_count,
                "estimated_makespan": float(duration),
                "gpu_loads": gpu_loads,
                "tasks": [
                    {
                        "gpu": int(gpu_index),
                        "start": 0.0,
                        "absolute_start": float(absolute_start),
                        "duration": float(duration),
                        "type": "full",
                        "candidate_index": int(offset),
                    }
                ],
                "counts": {"full": 1},
                "wall_time_sec": float(duration),
            }
        )
        offset += 1
    return history


def build_hypervolume_curve(
    hvs: Sequence[float],
    initial_samples: int,
    batch_size: int,
    total_evaluations: int,
) -> HypervolumeCurve:
    curve: HypervolumeCurve = []
    total = max(int(total_evaluations), 0)
    if len(hvs) == 0:
        return curve
    initial_count = min(max(int(initial_samples), 0), total) if total > 0 else max(int(initial_samples), 0)
    effective_batch_size = max(1, int(batch_size))
    for idx, hv_value in enumerate(hvs):
        if idx == 0:
            evaluations = initial_count
        else:
            evaluations = initial_count + idx * effective_batch_size
            if total > 0:
                evaluations = min(evaluations, total)
        curve.append({"evaluations": int(evaluations), "hypervolume": float(hv_value)})
    return curve


def save_optimizer_checkpoint(
    run_dir: str,
    iteration: int,
    train_x: torch.Tensor,
    train_obj_true: torch.Tensor,
    train_info: Sequence[Any],
    hvs: Sequence[float],
    train_obj: Optional[torch.Tensor] = None,
    extra_json: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    scheduler_history: Optional[RuntimeHistory] = None,
    hv_curve: Optional[HypervolumeCurve] = None,
    eval_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_json = {
        "iteration": int(iteration),
        "hvs": [float(value) for value in hvs],
        "evaluated_solutions": {
            "decision_variables": train_x.detach().cpu().tolist(),
            "objectives": train_obj_true.detach().cpu().tolist(),
            "metrics": list(train_info),
        },
    }
    if isinstance(eval_metadata, dict) and eval_metadata:
        checkpoint_json["evaluation_settings"] = eval_metadata
    if isinstance(extra_json, dict):
        checkpoint_json.update(extra_json)
    with open(os.path.join(run_dir, f"checkpoint_iter_{int(iteration)}.json"), "w", encoding="utf-8") as handle:
        json.dump(checkpoint_json, handle, ensure_ascii=False, indent=2)

    state: Dict[str, Any] = {
        "train_x": train_x.detach().cpu(),
        "train_obj_true": train_obj_true.detach().cpu(),
        "train_info": list(train_info),
        "iteration": int(iteration),
        "hvs": [float(value) for value in hvs],
        "evaluated_solutions": checkpoint_json["evaluated_solutions"],
    }
    if train_obj is not None:
        state["train_obj"] = train_obj.detach().cpu()
    if scheduler_history is not None:
        state["scheduler_history"] = scheduler_history
    if hv_curve is not None:
        state["hv_curve"] = hv_curve
    if isinstance(eval_metadata, dict) and eval_metadata:
        state["evaluation_settings"] = eval_metadata
    if isinstance(extra_state, dict):
        state.update(extra_state)

    torch.save(state, os.path.join(run_dir, f"checkpoint_iter_{int(iteration)}.pt"))
    torch.save(state, os.path.join(run_dir, "checkpoint_latest.pt"))


def load_optimizer_checkpoint(run_dir: str, tkwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    latest_path = os.path.join(run_dir, "checkpoint_latest.pt")
    if not os.path.exists(latest_path):
        return None
    state = torch.load(latest_path, map_location="cpu")
    if "train_x" in state and isinstance(state["train_x"], torch.Tensor):
        state["train_x"] = state["train_x"].to(**tkwargs)
    if "train_obj" in state and isinstance(state["train_obj"], torch.Tensor):
        state["train_obj"] = state["train_obj"].to(**tkwargs)
    if "train_obj_true" in state and isinstance(state["train_obj_true"], torch.Tensor):
        state["train_obj_true"] = state["train_obj_true"].to(**tkwargs)
    return state


def update_latest_checkpoint_runtime_artifacts(
    run_dir: str,
    scheduler_history: Optional[RuntimeHistory] = None,
    hv_curve: Optional[HypervolumeCurve] = None,
) -> None:
    latest_path = os.path.join(run_dir, "checkpoint_latest.pt")
    if not os.path.exists(latest_path):
        return

    def _patch_checkpoint(checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if scheduler_history is not None:
            state["scheduler_history"] = scheduler_history
        if hv_curve is not None:
            state["hv_curve"] = hv_curve
        torch.save(state, checkpoint_path)

    _patch_checkpoint(latest_path)
    iter_paths = sorted(
        path for path in os.listdir(run_dir)
        if path.startswith("checkpoint_iter_") and path.endswith(".pt")
    )
    if iter_paths:
        _patch_checkpoint(os.path.join(run_dir, iter_paths[-1]))


def save_runtime_artifacts(
    run_dir: str,
    scheduler_history: Optional[RuntimeHistory] = None,
    hv_curve: Optional[HypervolumeCurve] = None,
) -> None:
    hv_curve = [] if hv_curve is None else list(hv_curve)
    scheduler_history = [] if scheduler_history is None else list(scheduler_history)

    if hv_curve:
        with open(os.path.join(run_dir, "hypervolume_curve.json"), "w", encoding="utf-8") as handle:
            json.dump(hv_curve, handle, ensure_ascii=False, indent=2)
        hv_x = [int(item["evaluations"]) for item in hv_curve]
        hv_y = [float(item["hypervolume"]) for item in hv_curve]
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.plot(hv_x, hv_y, marker="o")
        axis.set_xlabel("Evaluations")
        axis.set_ylabel("Hypervolume")
        axis.set_title("Hypervolume Curve")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "hypervolume_curve.png"), dpi=160)
        plt.close(fig)

    if not scheduler_history:
        return

    with open(os.path.join(run_dir, "scheduler_usage.json"), "w", encoding="utf-8") as handle:
        json.dump(scheduler_history, handle, ensure_ascii=False, indent=2)

    iterations = [int(item["iteration"]) for item in scheduler_history]
    wall_times = [float(item.get("wall_time_sec", item.get("estimated_makespan", 0.0))) for item in scheduler_history]
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(iterations, wall_times, marker="o")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Wall Time (s)")
    axis.set_title("Iteration Wall Time")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "iteration_wall_time.png"), dpi=160)
    plt.close(fig)


def ensure_runtime_artifacts(
    run_dir: str,
    initial_samples: int,
    batch_size: int,
    gpu_count: int = 1,
    async_mode: bool = False,
) -> Tuple[RuntimeHistory, HypervolumeCurve]:
    latest_path = os.path.join(run_dir, "checkpoint_latest.pt")
    if not os.path.exists(latest_path):
        return [], []
    try:
        state = torch.load(latest_path, map_location="cpu")
    except Exception:
        return [], []

    train_info = state.get("train_info", []) if isinstance(state, dict) else []
    hvs = state.get("hvs", []) if isinstance(state, dict) else []
    scheduler_history = state.get("scheduler_history", []) if isinstance(state, dict) else []
    hv_curve = state.get("hv_curve", []) if isinstance(state, dict) else []

    if not isinstance(hv_curve, list) or not hv_curve:
        hv_curve = build_hypervolume_curve(
            hvs=hvs if isinstance(hvs, list) else [],
            initial_samples=initial_samples,
            batch_size=batch_size,
            total_evaluations=len(train_info) if isinstance(train_info, list) else 0,
        )

    if not isinstance(scheduler_history, list) or not scheduler_history:
        if async_mode:
            scheduler_history = build_asynchronous_runtime_history(
                train_info=train_info if isinstance(train_info, list) else [],
                initial_samples=initial_samples,
                gpu_count=gpu_count,
            )
        else:
            scheduler_history = build_synchronous_runtime_history(
                train_info=train_info if isinstance(train_info, list) else [],
                initial_samples=initial_samples,
                batch_size=batch_size,
                gpu_count=gpu_count,
            )

    save_runtime_artifacts(run_dir=run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
    update_latest_checkpoint_runtime_artifacts(run_dir=run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
    return scheduler_history, hv_curve


# Backward-compatible wrappers kept while refactoring internal callers.
def build_eval_metadata(*args, **kwargs):
    return build_evaluation_metadata(*args, **kwargs)


def build_sync_schedule_record(
    iteration,
    metric_items,
    scheduler_gpu_count=1,
    candidate_index_offset=0,
):
    return build_synchronous_iteration_record(
        iteration=iteration,
        metric_items=metric_items,
        gpu_count=scheduler_gpu_count,
        candidate_index_offset=candidate_index_offset,
    )


def build_sync_scheduler_history(
    train_info,
    initial_samples,
    batch_size,
    scheduler_gpu_count=1,
):
    return build_synchronous_runtime_history(
        train_info=train_info,
        initial_samples=initial_samples,
        batch_size=batch_size,
        gpu_count=scheduler_gpu_count,
    )


def build_async_scheduler_history(
    train_info,
    initial_samples,
    scheduler_gpu_count=1,
):
    return build_asynchronous_runtime_history(
        train_info=train_info,
        initial_samples=initial_samples,
        gpu_count=scheduler_gpu_count,
    )


def build_hv_curve(*args, **kwargs):
    return build_hypervolume_curve(*args, **kwargs)


def save_standard_checkpoint(*args, **kwargs):
    return save_optimizer_checkpoint(*args, **kwargs)


def load_standard_checkpoint(*args, **kwargs):
    return load_optimizer_checkpoint(*args, **kwargs)


def ensure_latest_checkpoint_metadata(*args, **kwargs):
    return update_latest_checkpoint_runtime_artifacts(*args, **kwargs)


def rebuild_runtime_reports_from_checkpoint(
    run_dir,
    initial_samples,
    batch_size,
    scheduler_gpu_count=1,
    async_mode=False,
):
    return ensure_runtime_artifacts(
        run_dir=run_dir,
        initial_samples=initial_samples,
        batch_size=batch_size,
        gpu_count=scheduler_gpu_count,
        async_mode=async_mode,
    )


def save_runtime_reports(*args, **kwargs):
    return save_runtime_artifacts(*args, **kwargs)
