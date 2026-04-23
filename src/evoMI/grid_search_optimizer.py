import os
import gc
import inspect
import datetime
import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from src.evoMI.checkpoint_runtime import (
    build_eval_metadata,
    build_hv_curve,
    build_sync_schedule_record,
    load_standard_checkpoint,
    save_runtime_reports,
    save_standard_checkpoint,
)


def grid_search_optimizer(
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

    if isinstance(ref_point, torch.Tensor):
        problem_ref_point = ref_point.to(**tkwargs)
    else:
        problem_ref_point = torch.full((num_objectives,), ref_point, **tkwargs)

    if max_evaluations is None:
        max_evaluations = int(initial_samples + N_BATCH * BATCH_SIZE)
    max_evaluations = max(1, int(max_evaluations))

    if full_eval_limits is None:
        full_eval_limits = {"aime25": 5, "gpqa_diamond": 60}
    if eval_metadata is None:
        eval_metadata = build_eval_metadata(
            eval_profile=eval_profile,
            eval_limits=full_eval_limits,
            eval_repeats=eval_repeats,
            eval_setting_id=eval_setting_id,
        )

    lower = bounds[0]
    upper = bounds[1]

    objective_signature = inspect.signature(objective_function)
    objective_supports_eval_limit = "eval_limit" in objective_signature.parameters
    objective_supports_eval_mode = "eval_mode" in objective_signature.parameters
    objective_supports_estimated_tokens = "estimated_tokens" in objective_signature.parameters
    objective_supports_async_session = hasattr(objective_function, "start_async_session")

    def collect_cleanup_paths(info_items):
        cleanup_paths = []
        if not isinstance(info_items, list):
            return cleanup_paths
        for item in info_items:
            if not isinstance(item, dict):
                continue
            item_cleanup_paths = item.get("_cleanup_model_dirs", [])
            if not isinstance(item_cleanup_paths, list):
                continue
            cleanup_paths.extend(
                path for path in item_cleanup_paths
                if isinstance(path, str) and len(path) > 0
            )
        return list(dict.fromkeys(cleanup_paths))

    def parse_objective_result(result, batch_size):
        if isinstance(result, tuple) and len(result) == 3:
            obj_true, info, cleanup_paths = result
        elif isinstance(result, tuple) and len(result) == 2:
            obj_true, info = result
            cleanup_paths = collect_cleanup_paths(info)
        else:
            obj_true = result
            info = [{} for _ in range(batch_size)]
            cleanup_paths = []
        if not isinstance(info, list):
            info = [{} for _ in range(batch_size)]
        return obj_true.to(**tkwargs), info, cleanup_paths if isinstance(cleanup_paths, list) else []

    def evaluate_batch(x_batch):
        kwargs = {}
        if objective_supports_eval_limit:
            kwargs["eval_limit"] = full_eval_limits
        if objective_supports_eval_mode:
            kwargs["eval_mode"] = "full"
        if objective_supports_estimated_tokens:
            kwargs["estimated_tokens"] = None
        result = objective_function(x_batch, **kwargs)
        obj_true, info, cleanup_paths = parse_objective_result(result, x_batch.shape[0])
        return x_batch, obj_true, info, cleanup_paths

    def start_async_session(x_batch):
        kwargs = {}
        if objective_supports_eval_limit:
            kwargs["eval_limit"] = full_eval_limits
        if objective_supports_estimated_tokens:
            kwargs["estimated_tokens"] = None
        return objective_function.start_async_session(x_batch, **kwargs)

    def normalize_async_objective(objective_row):
        if isinstance(objective_row, torch.Tensor):
            tensor = objective_row.detach().to(**tkwargs).reshape(-1)
        else:
            tensor = torch.as_tensor(np.asarray(objective_row, dtype=float), **tkwargs).reshape(-1)
        if tensor.numel() < num_objectives:
            padded = torch.zeros((num_objectives,), **tkwargs)
            if tensor.numel() > 0:
                padded[:tensor.numel()] = tensor
            tensor = padded
        return tensor[:num_objectives].reshape(1, -1)

    def compute_hypervolume(objectives):
        partitioning = FastNondominatedPartitioning(ref_point=problem_ref_point, Y=objectives)
        return float(partitioning.compute_hypervolume().item())

    scheduler_history = []
    hv_curve = []

    def save_checkpoint(iteration, train_x, train_obj_true, train_info, hvs, grid_metadata):
        save_standard_checkpoint(
            run_dir=run_dir,
            iteration=iteration,
            train_x=train_x,
            train_obj_true=train_obj_true,
            train_info=train_info,
            hvs=hvs,
            extra_json={"grid_metadata": grid_metadata},
            extra_state={"grid_metadata": grid_metadata},
            scheduler_history=scheduler_history,
            hv_curve=hv_curve,
            eval_metadata=eval_metadata,
        )

    def tensor_to_key(x_row):
        rounded = torch.round(x_row.detach().cpu() * 1e8) / 1e8
        return tuple(float(v) for v in rounded.tolist())

    def build_candidate_grid(total_budget):
        candidate_count = max(1, int(total_budget))
        if candidate_count == 1:
            weight_values = [0.5]
        else:
            weight_values = np.linspace(0.05, 0.95, candidate_count).tolist()
        candidates = [
            torch.full((dim,), float(weight), **tkwargs).clamp(lower, upper)
            for weight in weight_values
        ]
        candidate_x = torch.stack(candidates, dim=0) if len(candidates) > 0 else torch.empty((0, dim), **tkwargs)
        return candidate_x, {
            "weight_values": weight_values,
            "uniform_weight": True,
        }

    candidate_x, candidate_spec = build_candidate_grid(max_evaluations)
    total_target = int(candidate_x.shape[0])
    grid_metadata = {
        "grid_counts": [int(total_target)],
        "total_candidates": int(total_target),
        "budget": int(total_target),
        "async_mode": True,
        **candidate_spec,
    }

    if verbose:
        print(f"使用统一权重网格搜索优化多目标问题 (维度: {dim}, 目标数: {num_objectives})")
        print(f"运行ID: {run_id}, 检查点目录: {run_dir}")
        print(
            f"统一权重候选数: {grid_metadata['total_candidates']}, "
            f"权重范围: {grid_metadata['weight_values'][0]:.4f}-{grid_metadata['weight_values'][-1]:.4f}"
        )

    checkpoint = load_standard_checkpoint(run_dir, tkwargs)
    if checkpoint is not None:
        train_x = checkpoint["train_x"]
        train_obj_true = checkpoint["train_obj_true"]
        train_info = checkpoint.get("train_info", [{} for _ in range(train_x.shape[0])])
        hvs = [float(v) for v in checkpoint.get("hvs", [])]
        start_iteration = int(checkpoint.get("iteration", -1)) + 1
        scheduler_history = checkpoint.get("scheduler_history", [])
        hv_curve = checkpoint.get(
            "hv_curve",
            build_hv_curve(
                hvs=hvs,
                initial_samples=0,
                batch_size=BATCH_SIZE,
                total_evaluations=train_x.shape[0],
            ),
        )
        if verbose:
            print(f"成功加载检查点，从迭代 {start_iteration} 继续")
            if len(hvs) > 0:
                print(f"当前超体积: {hvs[-1]:.4f}")
    else:
        train_x = torch.empty((0, dim), **tkwargs)
        train_obj_true = torch.empty((0, num_objectives), **tkwargs)
        train_info = []
        hvs = []
        start_iteration = 0
    if train_x.shape[0] > 0:
        evaluated_keys = {tensor_to_key(row) for row in train_x}
        remaining_candidates = [
            row for row in candidate_x
            if tensor_to_key(row) not in evaluated_keys
        ]
        candidate_x = torch.stack(remaining_candidates, dim=0) if len(remaining_candidates) > 0 else torch.empty((0, dim), **tkwargs)
    total_target = int(train_x.shape[0] + candidate_x.shape[0])
    grid_metadata = {
        "grid_counts": [int(candidate_x.shape[0])],
        "total_candidates": int(total_target),
        "remaining_candidates": int(candidate_x.shape[0]),
        "budget": int(total_target),
        "async_mode": True,
        **candidate_spec,
    }
    if verbose and train_x.shape[0] > 0:
        print(f"剩余统一权重候选数: {candidate_x.shape[0]}")

    try:
        if candidate_x.shape[0] > 0:
            iteration = int(start_iteration)
            if objective_supports_async_session:
                if verbose:
                    print(
                        f"网格搜索将使用全局异步补位评测 {candidate_x.shape[0]} 个候选, "
                        f"scheduler_gpu_count={scheduler_gpu_count}"
                    )
                session = start_async_session(candidate_x)
                completed_in_run = 0
                processed_candidate_indices = set()
                while not session.is_finished():
                    next_item = session.get_next_result(timeout=1.0)
                    if next_item is None:
                        continue
                    relative_index = int(next_item.get("candidate_index", -1))
                    if relative_index < 0 or relative_index >= candidate_x.shape[0] or relative_index in processed_candidate_indices:
                        continue
                    processed_candidate_indices.add(relative_index)
                    single_x = candidate_x[relative_index: relative_index + 1]
                    single_obj_true = normalize_async_objective(next_item.get("objective", np.zeros((num_objectives,), dtype=float)))
                    single_metric = next_item.get("metric", {})
                    single_info = [single_metric if isinstance(single_metric, dict) else {}]
                    cleanup_paths = collect_cleanup_paths(single_info)
                    train_x = torch.cat([train_x, single_x], dim=0)
                    train_obj_true = torch.cat([train_obj_true, single_obj_true], dim=0)
                    train_info.extend(single_info)
                    hv_value = compute_hypervolume(train_obj_true)
                    hvs.append(hv_value)
                    hv_curve = build_hv_curve(
                        hvs=hvs,
                        initial_samples=0,
                        batch_size=1,
                        total_evaluations=train_x.shape[0],
                    )
                    iteration = int(start_iteration + completed_in_run)
                    completed_in_run += 1
                    scheduler_history.append(
                        build_sync_schedule_record(
                            iteration=iteration,
                            metric_items=single_info,
                            scheduler_gpu_count=scheduler_gpu_count,
                            candidate_index_offset=train_x.shape[0] - len(single_info),
                        )
                    )
                    remaining_count = int(candidate_x.shape[0] - completed_in_run)
                    current_grid_metadata = {
                        **grid_metadata,
                        "remaining_candidates": remaining_count,
                    }
                    save_checkpoint(iteration, train_x, train_obj_true, train_info, hvs, current_grid_metadata)
                    if verbose:
                        print(f"迭代 {iteration:>2}: 超体积 = {hv_value:.4f}, 已评估 = {train_x.shape[0]}/{total_target}")
                    if iteration_callback is not None:
                        try:
                            if len(cleanup_paths) > 0:
                                existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
                                iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + cleanup_paths))
                            iteration_callback(iteration, train_x, train_obj_true, hvs)
                        except Exception as callback_exc:
                            print(f"警告: 迭代回调函数执行失败: {callback_exc}")
                    gc.collect()
                    if torch.device(device).type == "cuda":
                        torch.cuda.empty_cache()
                session.finalize()
            else:
                evaluation_batch_size = min(
                    max(1, int(BATCH_SIZE)),
                    max(1, int(scheduler_gpu_count)),
                    max(1, int(candidate_x.shape[0])),
                )
                if verbose:
                    print(
                        f"网格搜索将按批次评测候选: batch_size={evaluation_batch_size}, "
                        f"scheduler_gpu_count={scheduler_gpu_count}"
                    )
                for batch_start in range(0, candidate_x.shape[0], evaluation_batch_size):
                    batch_end = min(batch_start + evaluation_batch_size, candidate_x.shape[0])
                    current_x = candidate_x[batch_start:batch_end]
                    new_x, new_obj_true, new_info, cleanup_paths = evaluate_batch(current_x)
                    batch_count = int(new_x.shape[0])
                    if batch_count == 0:
                        continue
                    if len(new_info) < batch_count:
                        new_info = list(new_info) + ([{}] * (batch_count - len(new_info)))
                    elif len(new_info) > batch_count:
                        new_info = list(new_info[:batch_count])
                    for batch_offset in range(batch_count):
                        iteration = int(start_iteration + batch_start + batch_offset)
                        single_x = new_x[batch_offset: batch_offset + 1]
                        single_obj_true = new_obj_true[batch_offset: batch_offset + 1]
                        single_info = [new_info[batch_offset]]
                        train_x = torch.cat([train_x, single_x], dim=0)
                        train_obj_true = torch.cat([train_obj_true, single_obj_true], dim=0)
                        train_info.extend(single_info)
                        hv_value = compute_hypervolume(train_obj_true)
                        hvs.append(hv_value)
                        hv_curve = build_hv_curve(
                            hvs=hvs,
                            initial_samples=0,
                            batch_size=1,
                            total_evaluations=train_x.shape[0],
                        )
                        scheduler_history.append(
                            build_sync_schedule_record(
                                iteration=iteration,
                                metric_items=single_info,
                                scheduler_gpu_count=scheduler_gpu_count,
                                candidate_index_offset=train_x.shape[0] - len(single_info),
                            )
                        )
                        remaining_count = int(candidate_x.shape[0] - batch_start - batch_offset - 1)
                        current_grid_metadata = {
                            **grid_metadata,
                            "remaining_candidates": remaining_count,
                        }
                        save_checkpoint(iteration, train_x, train_obj_true, train_info, hvs, current_grid_metadata)
                        if verbose:
                            print(f"迭代 {iteration:>2}: 超体积 = {hv_value:.4f}, 已评估 = {train_x.shape[0]}/{total_target}")
                        if iteration_callback is not None:
                            try:
                                if len(cleanup_paths) > 0:
                                    existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
                                    iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + cleanup_paths))
                                iteration_callback(iteration, train_x, train_obj_true, hvs)
                            except Exception as callback_exc:
                                print(f"警告: 迭代回调函数执行失败: {callback_exc}")
                        gc.collect()
                        if torch.device(device).type == "cuda":
                            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"错误: 网格搜索优化过程中发生异常: {exc}")
        import traceback
        traceback.print_exc()
        save_runtime_reports(run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
        return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id

    save_runtime_reports(run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
    return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
