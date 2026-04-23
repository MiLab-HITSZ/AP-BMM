import colorsys
import glob
import hashlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def build_eval_metadata(eval_profile=None, eval_limits=None, eval_repeats=None, eval_setting_id=None):
    if eval_profile is None and eval_limits is None and eval_repeats is None and eval_setting_id is None:
        return None
    metadata = {
        "eval_profile": None if eval_profile is None else str(eval_profile),
        "eval_limits": eval_limits,
        "eval_repeats": eval_repeats,
        "eval_setting_id": eval_setting_id,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def extract_avg_tokens(metric_item):
    if not isinstance(metric_item, dict):
        return None
    values = []
    for dataset_metrics in metric_item.values():
        if not isinstance(dataset_metrics, dict):
            continue
        token_value = dataset_metrics.get("mean_tokens_num", None)
        if isinstance(token_value, (int, float)) and np.isfinite(token_value) and token_value > 0:
            values.append(float(token_value))
    if len(values) == 0:
        return None
    return float(np.mean(values))


def estimate_task_duration(metric_item, fallback=1.0):
    avg_tokens = extract_avg_tokens(metric_item)
    if avg_tokens is not None:
        return max(float(avg_tokens), 1.0)
    return max(float(fallback), 1e-6)


def _normalize_task_key(task_key):
    if task_key is None:
        return None
    if isinstance(task_key, (int, np.integer)):
        return int(task_key)
    if isinstance(task_key, float) and np.isfinite(task_key):
        return int(task_key)
    text = str(task_key).strip()
    if len(text) == 0:
        return None
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _task_identity(task):
    if not isinstance(task, dict):
        return None
    for key in ["task_key", "candidate_index", "task_id", "model_id"]:
        value = task.get(key, None)
        normalized = _normalize_task_key(value)
        if normalized is not None:
            return normalized
    return None


def _task_color(task_type, task_key=None):
    if str(task_type) != "full":
        return "#1f77b4"
    key = _normalize_task_key(task_key)
    if key is None:
        return "#1f77b4"
    hue = (key * 0.618033988749895) % 1.0
    sat = 0.55 + 0.10 * (key % 3)
    val = 0.80 + 0.08 * (key % 2)
    return colorsys.hsv_to_rgb(min(max(hue, 0.0), 1.0), min(sat, 0.95), min(val, 0.98))


def build_sync_schedule_record(iteration, metric_items, scheduler_gpu_count=1, candidate_index_offset=0):
    gpu_count = max(1, int(scheduler_gpu_count))
    gpu_loads = [0.0 for _ in range(gpu_count)]
    tasks = []
    for local_idx, metric_item in enumerate(metric_items):
        gpu_idx = int(np.argmin(gpu_loads))
        duration = estimate_task_duration(metric_item)
        start_time = float(gpu_loads[gpu_idx])
        tasks.append(
            {
                "gpu": int(gpu_idx),
                "start": start_time,
                "duration": float(duration),
                "type": "full",
                "candidate_index": int(candidate_index_offset + local_idx),
            }
        )
        gpu_loads[gpu_idx] += duration
    return {
        "iteration": int(iteration),
        "mode": "sync",
        "gpu_count": int(gpu_count),
        "estimated_makespan": float(max(gpu_loads) if len(gpu_loads) > 0 else 0.0),
        "gpu_loads": [float(v) for v in gpu_loads],
        "tasks": tasks,
        "counts": {"full": int(len(metric_items))},
        "wall_time_sec": float(max(gpu_loads) if len(gpu_loads) > 0 else 0.0),
    }


def build_sync_scheduler_history(train_info, initial_samples, batch_size, scheduler_gpu_count=1):
    history = []
    total = len(train_info)
    if total <= 0:
        return history
    head = min(max(int(initial_samples), 0), total)
    if head > 0:
        history.append(
            build_sync_schedule_record(
                iteration=0,
                metric_items=train_info[:head],
                scheduler_gpu_count=scheduler_gpu_count,
                candidate_index_offset=0,
            )
        )
    offset = head
    iteration = 1
    effective_batch = max(1, int(batch_size))
    while offset < total:
        batch_metrics = train_info[offset : min(offset + effective_batch, total)]
        history.append(
            build_sync_schedule_record(
                iteration=iteration,
                metric_items=batch_metrics,
                scheduler_gpu_count=scheduler_gpu_count,
                candidate_index_offset=offset,
            )
        )
        offset += len(batch_metrics)
        iteration += 1
    return history


def build_async_scheduler_history(train_info, initial_samples, scheduler_gpu_count=1):
    history = []
    total = len(train_info)
    if total <= 0:
        return history
    head = min(max(int(initial_samples), 0), total)
    if head > 0:
        init_record = build_sync_schedule_record(
            iteration=0,
            metric_items=train_info[:head],
            scheduler_gpu_count=scheduler_gpu_count,
            candidate_index_offset=0,
        )
        init_record["mode"] = "init"
        history.append(init_record)
    offset = head
    gpu_count = max(1, int(scheduler_gpu_count))
    gpu_available = [0.0 for _ in range(gpu_count)]
    while offset < total:
        metric_item = train_info[offset]
        gpu_idx = int(np.argmin(gpu_available))
        duration = estimate_task_duration(metric_item)
        absolute_start = float(gpu_available[gpu_idx])
        gpu_available[gpu_idx] += duration
        gpu_loads = [0.0 for _ in range(gpu_count)]
        gpu_loads[gpu_idx] = float(duration)
        history.append(
            {
                "iteration": int(offset + 1),
                "mode": "async",
                "gpu_count": int(gpu_count),
                "estimated_makespan": float(duration),
                "gpu_loads": gpu_loads,
                "tasks": [
                    {
                        "gpu": int(gpu_idx),
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


def build_hv_curve(hvs, initial_samples, batch_size, total_evaluations):
    curve = []
    total = max(int(total_evaluations), 0)
    if len(hvs) == 0:
        return curve
    initial_count = min(max(int(initial_samples), 0), total) if total > 0 else max(int(initial_samples), 0)
    effective_batch = max(1, int(batch_size))
    for idx, hv_value in enumerate(hvs):
        if idx == 0:
            evaluations = initial_count if total > 0 else initial_count
        else:
            evaluations = initial_count + idx * effective_batch
            if total > 0:
                evaluations = min(evaluations, total)
        curve.append({"evaluations": int(evaluations), "hypervolume": float(hv_value)})
    return curve


def save_standard_checkpoint(
    run_dir,
    iteration,
    train_x,
    train_obj_true,
    train_info,
    hvs,
    train_obj=None,
    extra_json=None,
    extra_state=None,
    scheduler_history=None,
    hv_curve=None,
    eval_metadata=None,
):
    checkpoint_json = {
        "iteration": int(iteration),
        "hvs": [float(v) for v in hvs],
        "evaluated_solutions": {
            "decision_variables": train_x.detach().cpu().tolist(),
            "objectives": train_obj_true.detach().cpu().tolist(),
            "metrics": train_info,
        },
    }
    if isinstance(eval_metadata, dict) and len(eval_metadata) > 0:
        checkpoint_json["evaluation_settings"] = eval_metadata
    if isinstance(extra_json, dict):
        checkpoint_json.update(extra_json)
    with open(os.path.join(run_dir, f"checkpoint_iter_{int(iteration)}.json"), "w", encoding="utf-8") as handle:
        json.dump(checkpoint_json, handle, ensure_ascii=False, indent=2)

    state = {
        "train_x": train_x.detach().cpu(),
        "train_obj_true": train_obj_true.detach().cpu(),
        "train_info": train_info,
        "iteration": int(iteration),
        "hvs": [float(v) for v in hvs],
        "evaluated_solutions": checkpoint_json["evaluated_solutions"],
    }
    if train_obj is not None:
        state["train_obj"] = train_obj.detach().cpu()
    if scheduler_history is not None:
        state["scheduler_history"] = scheduler_history
    if hv_curve is not None:
        state["hv_curve"] = hv_curve
    if isinstance(eval_metadata, dict) and len(eval_metadata) > 0:
        state["evaluation_settings"] = eval_metadata
    if isinstance(extra_state, dict):
        state.update(extra_state)

    torch.save(state, os.path.join(run_dir, f"checkpoint_iter_{int(iteration)}.pt"))
    torch.save(state, os.path.join(run_dir, "checkpoint_latest.pt"))


def load_standard_checkpoint(run_dir, tkwargs):
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


def ensure_latest_checkpoint_metadata(run_dir, scheduler_history=None, hv_curve=None):
    latest_path = os.path.join(run_dir, "checkpoint_latest.pt")
    if not os.path.exists(latest_path):
        return
    state = torch.load(latest_path, map_location="cpu")
    if scheduler_history is not None:
        state["scheduler_history"] = scheduler_history
    if hv_curve is not None:
        state["hv_curve"] = hv_curve
    torch.save(state, latest_path)

    iter_paths = sorted(glob.glob(os.path.join(run_dir, "checkpoint_iter_*.pt")))
    if len(iter_paths) == 0:
        return
    last_iter_path = iter_paths[-1]
    state = torch.load(last_iter_path, map_location="cpu")
    if scheduler_history is not None:
        state["scheduler_history"] = scheduler_history
    if hv_curve is not None:
        state["hv_curve"] = hv_curve
    torch.save(state, last_iter_path)


def load_runtime_checkpoint_state(run_dir):
    latest_path = os.path.join(run_dir, "checkpoint_latest.pt")
    if not os.path.exists(latest_path):
        return None
    try:
        return torch.load(latest_path, map_location="cpu")
    except Exception:
        return None


def rebuild_runtime_reports_from_checkpoint(
    run_dir,
    initial_samples,
    batch_size,
    scheduler_gpu_count=1,
    async_mode=False,
):
    def has_absolute_task(history):
        return any(
            isinstance(task, dict) and ("absolute_start" in task)
            for record in history
            if isinstance(record, dict)
            for task in record.get("tasks", [])
        )

    def is_sync_only(history):
        if len(history) == 0:
            return False
        return all(
            isinstance(record, dict) and str(record.get("mode", "sync")).lower() in {"sync", "init"}
            for record in history
        )

    state = load_runtime_checkpoint_state(run_dir)
    train_info = []
    hvs = []
    checkpoint_scheduler_history = None
    checkpoint_hv_curve = None
    if isinstance(state, dict):
        train_info = state.get("train_info", [])
        hvs = state.get("hvs", [])
        checkpoint_scheduler_history = state.get("scheduler_history", None)
        checkpoint_hv_curve = state.get("hv_curve", None)

    scheduler_history = checkpoint_scheduler_history if isinstance(checkpoint_scheduler_history, list) else None
    hv_curve = checkpoint_hv_curve if isinstance(checkpoint_hv_curve, list) else None

    scheduler_path = os.path.join(run_dir, "scheduler_usage.json")
    if scheduler_history is None and os.path.exists(scheduler_path):
        try:
            with open(scheduler_path, "r", encoding="utf-8") as handle:
                loaded_scheduler_history = json.load(handle)
            if isinstance(loaded_scheduler_history, list):
                scheduler_history = loaded_scheduler_history
        except Exception:
            scheduler_history = None

    if (
        isinstance(scheduler_history, list)
        and bool(async_mode)
        and is_sync_only(scheduler_history)
        and not has_absolute_task(scheduler_history)
    ):
        scheduler_history = None

    hv_path = os.path.join(run_dir, "hypervolume_curve.json")
    if hv_curve is None and os.path.exists(hv_path):
        try:
            with open(hv_path, "r", encoding="utf-8") as handle:
                loaded_hv_curve = json.load(handle)
            if isinstance(loaded_hv_curve, list):
                hv_curve = loaded_hv_curve
        except Exception:
            hv_curve = None

    total_evaluations = len(train_info) if isinstance(train_info, list) else 0
    if hv_curve is None:
        hv_curve = build_hv_curve(
            hvs=hvs if isinstance(hvs, list) else [],
            initial_samples=initial_samples,
            batch_size=batch_size,
            total_evaluations=total_evaluations,
        )

    if scheduler_history is None or len(scheduler_history) == 0:
        if bool(async_mode):
            scheduler_history = build_async_scheduler_history(
                train_info=train_info if isinstance(train_info, list) else [],
                initial_samples=initial_samples,
                scheduler_gpu_count=scheduler_gpu_count,
            )
        else:
            scheduler_history = build_sync_scheduler_history(
                train_info=train_info if isinstance(train_info, list) else [],
                initial_samples=initial_samples,
                batch_size=batch_size,
                scheduler_gpu_count=scheduler_gpu_count,
            )

    save_runtime_reports(run_dir=run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
    ensure_latest_checkpoint_metadata(run_dir=run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
    return scheduler_history, hv_curve


def save_runtime_reports(run_dir, scheduler_history=None, hv_curve=None):
    if hv_curve is None:
        hv_curve = []
    if scheduler_history is None:
        scheduler_history = []

    if len(hv_curve) > 0:
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

    if len(scheduler_history) == 0:
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

    def is_initialization_record(record):
        if not isinstance(record, dict):
            return False
        mode = str(record.get("mode", "")).lower()
        if mode == "init":
            return True
        iteration = record.get("iteration", None)
        return isinstance(iteration, int) and int(iteration) == 0

    comparison_eval_budget = 40
    init_task_count = sum(
        len(record.get("tasks", []))
        for record in scheduler_history
        if is_initialization_record(record) and isinstance(record, dict)
    )
    target_post_init_tasks = max(0, int(comparison_eval_budget) - int(init_task_count))

    def trim_history_to_budget(history_records):
        if not isinstance(history_records, list) or len(history_records) == 0:
            return []
        async_like = any(
            isinstance(record, dict) and str(record.get("mode", "")).lower() == "async"
            for record in history_records
        )
        trimmed = []
        if async_like:
            for record in history_records:
                if not isinstance(record, dict):
                    continue
                tasks = record.get("tasks", [])
                if not isinstance(tasks, list):
                    continue
                kept_tasks = []
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    candidate_index = task.get("candidate_index", None)
                    if isinstance(candidate_index, int):
                        if int(candidate_index) < int(comparison_eval_budget):
                            kept_tasks.append(task)
                    elif len(kept_tasks) < target_post_init_tasks:
                        kept_tasks.append(task)
                if len(kept_tasks) == 0:
                    continue
                record_new = dict(record)
                record_new["tasks"] = kept_tasks
                record_new["counts"] = {"full": int(len(kept_tasks))}
                trimmed.append(record_new)
            return trimmed

        remaining = int(target_post_init_tasks)
        for record in history_records:
            if remaining <= 0:
                break
            if not isinstance(record, dict):
                continue
            tasks = record.get("tasks", [])
            if not isinstance(tasks, list) or len(tasks) == 0:
                continue
            take_n = min(int(len(tasks)), int(remaining))
            kept_tasks = tasks[:take_n]
            if len(kept_tasks) == 0:
                continue
            record_new = dict(record)
            record_new["tasks"] = kept_tasks
            record_new["counts"] = {"full": int(len(kept_tasks))}
            trimmed.append(record_new)
            remaining -= int(len(kept_tasks))
        return trimmed

    plot_scheduler_history = [record for record in scheduler_history if not is_initialization_record(record)]
    plot_scheduler_history = trim_history_to_budget(plot_scheduler_history)
    if len(plot_scheduler_history) == 0:
        return
    plot_task_count = sum(len(record.get("tasks", [])) for record in plot_scheduler_history)
    required_real_runtime_tasks = max(1, int(np.ceil(float(plot_task_count) * 0.8))) if plot_task_count > 0 else 1
    run_real_stats_path = os.path.join(run_dir, "vllm_gpu_usage_stats.json")
    run_root = os.path.abspath(os.path.dirname(run_dir))
    run_id = os.path.basename(run_root)
    run_label = os.path.basename(run_dir)
    run_label_prefix = f"{run_id}_"
    if run_label.startswith(run_label_prefix):
        run_label = run_label[len(run_label_prefix) :]
    def persist_reconstructed_runtime(_tasks, _source_path):
        # Keep runtime statistics strictly from real vLLM traces.
        return

    def load_real_runtime_tasks():
        candidate_results = []

        def normalize_real_tasks(tasks):
            if not isinstance(tasks, list) or len(tasks) == 0:
                return []
            normalized = []
            last_end_by_gpu = {}
            for task in sorted(tasks, key=lambda item: (float(item["start"]), int(item["gpu"]))):
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
                        "task_key": _task_identity(task),
                    }
                )
                last_end_by_gpu[gpu_idx] = end_ts
            if plot_task_count > 0 and len(normalized) > plot_task_count:
                normalized = normalized[-plot_task_count:]
            return normalized

        def add_candidate_result(tasks, source_path, source_rank, meta=None):
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
            count_gap = abs(len(normalized_tasks) - plot_task_count) if plot_task_count > 0 else 0
            enough_tasks = 0 if len(normalized_tasks) >= required_real_runtime_tasks else 1
            candidate_results.append(
                (
                    (enough_tasks, overlap_count, count_gap, source_rank, -len(normalized_tasks)),
                    normalized_tasks,
                    source_path,
                    {} if meta is None else dict(meta),
                )
            )

        if os.path.exists(run_real_stats_path):
            try:
                with open(run_real_stats_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                payload = {}
            if isinstance(payload, dict) and "reconstructed_from" in payload:
                return [], None, {}
            records = payload.get("task_runtime_records", [])
            if isinstance(records, list):
                tasks = []
                series_start = payload.get("series_start_time", None)
                series_end = payload.get("series_end_time", None)
                epoch_based = (
                    isinstance(series_start, (int, float))
                    and isinstance(series_end, (int, float))
                    and float(series_start) > 1e9
                    and float(series_end) > 1e9
                )
                if not epoch_based:
                    return [], None, {}
                for task_order, record in enumerate(records):
                    if not isinstance(record, dict):
                        continue
                    start_ts = record.get("start_time", None)
                    end_ts = record.get("end_time", None)
                    gpu_ids = record.get("gpu_ids", [])
                    if not isinstance(start_ts, (int, float)) or not isinstance(end_ts, (int, float)):
                        continue
                    if not isinstance(gpu_ids, list) or len(gpu_ids) == 0:
                        continue
                    duration = float(end_ts) - float(start_ts)
                    if duration <= 0.0:
                        continue
                    tasks.append(
                        {
                            "gpu": int(gpu_ids[0]),
                            "type": "full",
                            "start": float(start_ts),
                            "duration": float(duration),
                            "end": float(end_ts),
                            "task_id": record.get("task_id", None),
                            "estimated_tokens": float(record.get("estimated_tokens", 0.0)),
                            "task_key": _task_identity(record) or int(task_order),
                        }
                    )
                expected_count = int(plot_task_count)
                if len(tasks) > 0:
                    durations = [max(float(item["duration"]), 0.0) for item in tasks if float(item["duration"]) > 0.0]
                    filtered_tasks = list(tasks)
                    if len(durations) >= 8:
                        q1 = float(np.percentile(durations, 25))
                        short_threshold = max(1.0, q1 * 0.2)
                        filtered_tasks = [item for item in tasks if float(item["duration"]) >= short_threshold]
                        if len(filtered_tasks) == 0:
                            filtered_tasks = list(tasks)
                    if expected_count > 0 and len(filtered_tasks) > expected_count:
                        ordered_tasks = sorted(filtered_tasks, key=lambda item: float(item["start"]))
                        gpu_count_hint = max(
                            [
                                int(task.get("gpu", 0)) + 1
                                for record in plot_scheduler_history
                                for task in record.get("tasks", [])
                                if isinstance(task, dict)
                            ]
                            + [1]
                        )
                        best_window = None
                        best_score = None
                        window_size = int(expected_count)
                        for start_idx in range(0, len(ordered_tasks) - window_size + 1):
                            window = ordered_tasks[start_idx : start_idx + window_size]
                            if len(window) == 0:
                                continue
                            min_start = float(window[0]["start"])
                            first_start_by_gpu = {}
                            for item in window:
                                gpu_idx = int(item.get("gpu", 0))
                                if gpu_idx not in first_start_by_gpu:
                                    first_start_by_gpu[gpu_idx] = float(item["start"])
                            if len(first_start_by_gpu) == 0:
                                continue
                            startup_spread = float(max(first_start_by_gpu.values()) - min(first_start_by_gpu.values()))
                            startup_missing_gpu = max(0, int(gpu_count_hint - len(first_start_by_gpu)))
                            window_span = float(window[-1]["end"] - window[0]["start"])
                            # 优先选择启动最紧凑、覆盖GPU更完整的窗口；次级约束为窗口整体跨度更短。
                            score = (
                                startup_spread
                                + startup_missing_gpu * 1e6
                                + max(window_span, 0.0) * 1e-3
                                + max(min_start, 0.0) * 1e-6
                            )
                            if best_score is None or score < best_score:
                                best_score = score
                                best_window = window
                        if best_window is None:
                            filtered_tasks = ordered_tasks[-expected_count:]
                        else:
                            filtered_tasks = best_window
                    tasks = filtered_tasks
                source_rank = 0
                time_reference = "epoch"
                add_candidate_result(
                    tasks,
                    run_real_stats_path,
                    source_rank,
                    meta={"time_reference": time_reference},
                )

        # Real-runtime-only mode: do not mix scheduler/log/output reconstructions.
        if len(candidate_results) == 0:
            return [], None, {}
        candidate_results.sort(key=lambda item: item[0])
        _, selected_tasks, selected_path, selected_meta = candidate_results[0]
        if os.path.abspath(str(selected_path)) != os.path.abspath(run_real_stats_path):
            return [], None, {}
        if str(selected_meta.get("time_reference", "")).strip().lower() != "epoch":
            return [], None, {}
        return selected_tasks, selected_path, selected_meta

    max_gpu = 0
    debug_gpu_intervals = {}
    use_wall_time_unit = False
    title = "GPU Schedule Gantt"
    full_labeled = False
    idle_labeled = False

    real_runtime_tasks, real_runtime_path, real_runtime_meta = load_real_runtime_tasks()
    persist_reconstructed_runtime(real_runtime_tasks, real_runtime_path)
    use_real_runtime = len(real_runtime_tasks) >= required_real_runtime_tasks
    if not use_real_runtime:
        summary_path = os.path.join(run_dir, "gpu_runtime_summary.json")
        if os.path.exists(summary_path):
            os.remove(summary_path)
        return
    is_sync_history = all(
        isinstance(record, dict) and str(record.get("mode", "")).lower() in {"sync", "init"}
        for record in scheduler_history
    ) if len(scheduler_history) > 0 else False

    def compress_sync_real_runtime_tasks(tasks):
        if (not is_sync_history) or (not isinstance(tasks, list)) or len(tasks) == 0:
            return tasks
        sync_records = [
            record
            for record in plot_scheduler_history
            if isinstance(record, dict) and str(record.get("mode", "sync")).lower() == "sync"
        ]
        batch_specs = []
        for record in sync_records:
            tasks_in_record = record.get("tasks", [])
            if not isinstance(tasks_in_record, list) or len(tasks_in_record) == 0:
                continue
            batch_specs.append(
                {
                    "size": int(len(tasks_in_record)),
                    "gpu_count": max(1, int(record.get("gpu_count", 1))),
                }
            )
        if len(batch_specs) == 0:
            return tasks

        ordered_tasks = sorted(tasks, key=lambda item: (float(item["start"]), int(item["gpu"])))
        needed_tasks = sum(spec["size"] for spec in batch_specs)
        if needed_tasks <= 0:
            return ordered_tasks
        if len(ordered_tasks) > needed_tasks:
            ordered_tasks = ordered_tasks[-needed_tasks:]
        elif len(ordered_tasks) < needed_tasks:
            # 如果真实任务数更少，截断批次规格到可用任务数
            remaining = len(ordered_tasks)
            trimmed_specs = []
            for spec in reversed(batch_specs):
                if remaining <= 0:
                    break
                take = min(int(spec["size"]), int(remaining))
                if take > 0:
                    trimmed_specs.append({"size": int(take), "gpu_count": int(spec["gpu_count"])})
                    remaining -= int(take)
            batch_specs = list(reversed(trimmed_specs))

        compressed = []
        cursor = 0.0
        pointer = 0
        for spec in batch_specs:
            size = int(spec["size"])
            gpu_count = max(1, int(spec["gpu_count"]))
            chunk = ordered_tasks[pointer : pointer + size]
            pointer += size
            if len(chunk) == 0:
                continue
            gpu_available = [0.0 for _ in range(gpu_count)]
            chunk_span = 0.0
            for item in chunk:
                duration = max(float(item["duration"]), 0.0)
                gpu_idx = int(np.argmin(gpu_available))
                start_rel = float(gpu_available[gpu_idx])
                gpu_available[gpu_idx] += duration
                start_new = float(cursor + start_rel)
                end_new = float(start_new + duration)
                chunk_span = max(chunk_span, float(start_rel + duration))
                compressed.append(
                    {
                        **item,
                        "gpu": int(gpu_idx),
                        "start": start_new,
                        "end": end_new,
                        "duration": float(duration),
                    }
                )
            cursor += float(chunk_span)
        return compressed if len(compressed) > 0 else ordered_tasks

    if use_real_runtime and is_sync_history:
        real_runtime_tasks = compress_sync_real_runtime_tasks(real_runtime_tasks)

    fig, axis = plt.subplots(figsize=(18, 6))

    use_wall_time_unit = True
    title = (
        "GPU Schedule Gantt (real runtime, gap-compressed)"
        if is_sync_history
        else "GPU Schedule Gantt (real runtime)"
    )
    min_start = min(item["start"] for item in real_runtime_tasks)
    all_end = max(item["end"] for item in real_runtime_tasks) - min_start
    durations_all = []
    for item in real_runtime_tasks:
        start_rel = float(item["start"] - min_start)
        duration = float(item["duration"])
        durations_all.append(duration)
        gpu_idx = int(item["gpu"])
        label = None
        if not full_labeled:
            label = "full"
            full_labeled = True
        task_key = _task_identity(item)
        axis.barh(
            gpu_idx,
            duration,
            left=start_rel,
            height=0.6,
            color=_task_color(item.get("type", "full"), task_key),
            alpha=0.9,
            edgecolor="black",
            linewidth=0.3,
            label=label,
        )
        debug_gpu_intervals.setdefault(gpu_idx, []).append(
            {"start": float(start_rel), "end": float(start_rel + duration), "duration": float(duration)}
        )
        max_gpu = max(max_gpu, gpu_idx + 1)
    for gpu_idx in range(max_gpu):
        busy_spans = sorted(debug_gpu_intervals.get(gpu_idx, []), key=lambda item: item["start"])
        cursor = 0.0
        for span in busy_spans:
            idle = max(float(span["start"]) - cursor, 0.0)
            if idle > 1e-9:
                label = None
                if not idle_labeled:
                    label = "idle"
                    idle_labeled = True
                axis.barh(
                    gpu_idx,
                    idle,
                    left=cursor,
                    height=0.6,
                    color="#d9d9d9",
                    alpha=0.8,
                    edgecolor="none",
                    label=label,
                )
            cursor = max(cursor, float(span["end"]))
        tail_idle = max(all_end - cursor, 0.0)
        if tail_idle > 1e-9:
            label = None
            if not idle_labeled:
                label = "idle"
                idle_labeled = True
            axis.barh(
                gpu_idx,
                tail_idle,
                left=cursor,
                height=0.6,
                color="#d9d9d9",
                alpha=0.8,
                edgecolor="none",
                label=label,
            )

    gpu_busy_time = {
        str(gpu_idx): float(sum(span["duration"] for span in debug_gpu_intervals.get(gpu_idx, [])))
        for gpu_idx in range(max_gpu)
    }
    makespan = float(max(all_end, 1e-9))
    mean_duration = float(np.mean(durations_all)) if len(durations_all) > 0 else 0.0
    median_duration = float(np.median(durations_all)) if len(durations_all) > 0 else 0.0
    p90_duration = float(np.percentile(durations_all, 90)) if len(durations_all) > 0 else 0.0
    gpu_utilization = {
        str(gpu_idx): float(gpu_busy_time[str(gpu_idx)] / makespan) if makespan > 0 else 0.0
        for gpu_idx in range(max_gpu)
    }
    summary_payload = {
        "source": "real_runtime",
        "time_reference": str(real_runtime_meta.get("time_reference", "unknown")),
        "task_count": int(len(real_runtime_tasks)),
        "gpu_count": int(max_gpu),
        "makespan_sec": makespan,
        "task_duration_sec": {
            "mean": mean_duration,
            "median": median_duration,
            "p90": p90_duration,
            "min": float(min(durations_all)) if len(durations_all) > 0 else 0.0,
            "max": float(max(durations_all)) if len(durations_all) > 0 else 0.0,
        },
        "gpu_busy_time_sec": gpu_busy_time,
        "gpu_utilization": gpu_utilization,
        "cluster_utilization": float(sum(gpu_busy_time.values()) / (max_gpu * makespan)) if max_gpu > 0 and makespan > 0 else 0.0,
        "is_sync_history": bool(is_sync_history),
        "budget_task_count": int(plot_task_count),
    }
    with open(os.path.join(run_dir, "gpu_runtime_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "gpu_debug_intervals.json"), "w", encoding="utf-8") as handle:
        json.dump({str(key): value for key, value in sorted(debug_gpu_intervals.items())}, handle, ensure_ascii=False, indent=2)

    axis.set_xlabel("Wall Time (s)" if use_wall_time_unit else "Estimated Time")
    axis.set_ylabel("GPU")
    axis.set_title(title)
    axis.set_yticks(list(range(max_gpu)))
    axis.set_yticklabels([f"GPU {idx}" for idx in range(max_gpu)])
    if full_labeled or idle_labeled:
        axis.legend(loc="upper right")
    axis.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "gpu_schedule_gantt.png"), dpi=160)
    real_runtime_path = os.path.join(run_dir, "gpu_schedule_gantt_real_runtime.png")
    if use_real_runtime:
        fig.savefig(real_runtime_path, dpi=160)
    elif os.path.exists(real_runtime_path):
        os.remove(real_runtime_path)
    plt.close(fig)
