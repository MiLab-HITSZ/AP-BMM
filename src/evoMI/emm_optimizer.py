import os
import gc
import json
import inspect
import datetime
import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from src.evoMI.checkpoint_runtime import (
    build_eval_metadata,
    build_hv_curve,
    build_sync_schedule_record,
    load_standard_checkpoint,
    save_runtime_reports,
    save_standard_checkpoint,
)


def emm_optimizer(
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

    lower = bounds[0]
    upper = bounds[1]
    span = (upper - lower).clamp_min(1e-8)
    if m_prior is None:
        mean = ((lower + upper) * 0.5).clone()
    else:
        mean = torch.tensor(m_prior, **tkwargs).flatten().clamp(lower, upper)
        if mean.numel() != dim:
            raise ValueError(f"m_prior长度应为{dim}，当前为{mean.numel()}")
    if u_prior is None:
        sigma = float(span.mean().item() * 0.3)
    else:
        prior_radius = torch.tensor(u_prior, **tkwargs).flatten()
        if prior_radius.numel() != dim:
            raise ValueError(f"u_prior长度应为{dim}，当前为{prior_radius.numel()}")
        if torch.min(prior_radius) >= 0 and torch.max(prior_radius) <= 1.0:
            prior_radius = prior_radius * span
        sigma = float(prior_radius.mean().item() * max(rho, 1e-3))
    sigma = max(sigma, 1e-3)

    if max_evaluations is None:
        max_evaluations = int(initial_samples + N_BATCH * BATCH_SIZE)
    if full_eval_limits is None:
        full_eval_limits = {"aime25": 5, "gpqa_diamond": 60}
    if eval_metadata is None:
        eval_metadata = build_eval_metadata(
            eval_profile=eval_profile,
            eval_limits=full_eval_limits,
            eval_repeats=eval_repeats,
            eval_setting_id=eval_setting_id,
        )

    objective_signature = inspect.signature(objective_function)
    objective_supports_eval_limit = "eval_limit" in objective_signature.parameters
    objective_supports_eval_mode = "eval_mode" in objective_signature.parameters
    objective_supports_estimated_tokens = "estimated_tokens" in objective_signature.parameters

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

    lambda_ = max(int(BATCH_SIZE), 4)
    mu = max(lambda_ // 2, 1)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / weights.sum()
    weights_t = torch.tensor(weights, **tkwargs)
    mu_eff = float(1.0 / np.sum(weights ** 2))
    c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0, 0.0) + c_sigma
    c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
    c_mu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))
    expected_norm = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    p_sigma = torch.zeros(dim, **tkwargs)
    p_c = torch.zeros(dim, **tkwargs)
    cov = torch.eye(dim, **tkwargs)

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
        return obj_true, info, cleanup_paths if isinstance(cleanup_paths, list) else []

    def evaluate_batch(x_batch, estimated_tokens=None):
        kwargs = {}
        if objective_supports_eval_limit:
            kwargs["eval_limit"] = full_eval_limits
        if objective_supports_eval_mode:
            kwargs["eval_mode"] = "full"
        if objective_supports_estimated_tokens and estimated_tokens is not None:
            kwargs["estimated_tokens"] = estimated_tokens
        result = objective_function(x_batch, **kwargs)
        obj_true, info, cleanup_paths = parse_objective_result(result, x_batch.shape[0])
        return x_batch, obj_true.to(**tkwargs), info, cleanup_paths

    def compute_hypervolume(objectives):
        partitioning = FastNondominatedPartitioning(ref_point=problem_ref_point, Y=objectives)
        return float(partitioning.compute_hypervolume().item())

    def scalarize(objectives):
        if objectives.shape[1] != num_objectives:
            raise ValueError(f"目标维度不匹配: 期望{num_objectives}，实际{objectives.shape[1]}")
        obj_min = objectives.min(dim=0).values
        obj_span = (objectives.max(dim=0).values - obj_min).clamp_min(1e-8)
        normalized = (objectives - obj_min) / obj_span
        equal_weights = torch.full((num_objectives,), 1.0 / num_objectives, **tkwargs)
        return (normalized * equal_weights.unsqueeze(0)).sum(dim=-1)

    def sample_initial_points(n=initial_samples):
        if isinstance(shared_initial_dataset, dict):
            shared_x = torch.as_tensor(shared_initial_dataset.get("decision_variables", []), **tkwargs)
            shared_obj_true = torch.as_tensor(shared_initial_dataset.get("objectives", []), **tkwargs)
            shared_info = shared_initial_dataset.get("metrics", [{} for _ in range(shared_x.shape[0])])
            if shared_x.ndim == 2 and shared_x.shape == (n, dim) and shared_obj_true.shape == (n, num_objectives):
                return shared_x, shared_obj_true, shared_info
        if custom_initial_solutions is not None and len(custom_initial_solutions) > 0:
            custom_x = []
            for value in custom_initial_solutions[:n]:
                custom_x.append(torch.full((1, dim), value, **tkwargs))
            if len(custom_x) > 0:
                custom_x = torch.cat(custom_x, dim=0)
            else:
                custom_x = torch.empty((0, dim), **tkwargs)
            remaining = max(n - custom_x.shape[0], 0)
            if remaining > 0:
                sobol_x = draw_sobol_samples(bounds=bounds, n=remaining, q=1).squeeze(1).to(**tkwargs)
                train_x = torch.cat([custom_x, sobol_x], dim=0)
            else:
                train_x = custom_x[:n]
        else:
            train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1).to(**tkwargs)
        train_x, train_obj_true, train_info, initial_cleanup_paths = evaluate_batch(train_x)
        if iteration_callback is not None and len(initial_cleanup_paths) > 0:
            existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
            iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + initial_cleanup_paths))
        return train_x, train_obj_true, train_info

    scheduler_history = []
    hv_curve = []

    def save_checkpoint(iteration, train_x, train_obj_true, train_info, hvs):
        save_standard_checkpoint(
            run_dir=run_dir,
            iteration=iteration,
            train_x=train_x,
            train_obj_true=train_obj_true,
            train_info=train_info,
            hvs=hvs,
            extra_state={
                "mean": mean.detach().cpu(),
                "cov": cov.detach().cpu(),
                "sigma": float(sigma),
                "p_sigma": p_sigma.detach().cpu(),
                "p_c": p_c.detach().cpu(),
            },
            scheduler_history=scheduler_history,
            hv_curve=hv_curve,
            eval_metadata=eval_metadata,
        )

    if verbose:
        print(f"使用EMM同步优化多目标问题 (维度: {dim}, 目标数: {num_objectives})")
        print(f"运行ID: {run_id}, 检查点目录: {run_dir}")
        print(f"等权重标量化搜索，批大小: {lambda_}, 初始sigma: {sigma:.4f}")

    checkpoint = load_standard_checkpoint(run_dir, tkwargs)
    if checkpoint is not None:
        train_x = checkpoint["train_x"]
        train_obj_true = checkpoint["train_obj_true"]
        train_info = checkpoint.get("train_info", [{} for _ in range(train_x.shape[0])])
        start_iteration = int(checkpoint.get("iteration", 0))
        hvs = [float(v) for v in checkpoint.get("hvs", [])]
        scheduler_history = checkpoint.get("scheduler_history", [])
        hv_curve = checkpoint.get(
            "hv_curve",
            build_hv_curve(
                hvs=hvs,
                initial_samples=initial_samples,
                batch_size=BATCH_SIZE,
                total_evaluations=train_x.shape[0],
            ),
        )
        mean = checkpoint.get("mean", mean.detach().cpu()).to(**tkwargs)
        cov = checkpoint.get("cov", cov.detach().cpu()).to(**tkwargs)
        sigma = float(checkpoint.get("sigma", sigma))
        p_sigma = checkpoint.get("p_sigma", p_sigma.detach().cpu()).to(**tkwargs)
        p_c = checkpoint.get("p_c", p_c.detach().cpu()).to(**tkwargs)
        if verbose:
            print(f"成功加载检查点，从迭代 {start_iteration} 继续")
            if len(hvs) > 0:
                print(f"当前超体积: {hvs[-1]:.4f}")
    else:
        train_x, train_obj_true, train_info = sample_initial_points()
        hvs = [compute_hypervolume(train_obj_true)]
        hv_curve = build_hv_curve(
            hvs=hvs,
            initial_samples=initial_samples,
            batch_size=BATCH_SIZE,
            total_evaluations=train_x.shape[0],
        )
        scheduler_history = [
            build_sync_schedule_record(
                iteration=0,
                metric_items=train_info,
                scheduler_gpu_count=scheduler_gpu_count,
                candidate_index_offset=0,
            )
        ]
        save_checkpoint(0, train_x, train_obj_true, train_info, hvs)
        start_iteration = 0

    try:
        total_target = min(int(max_evaluations), int(initial_samples + N_BATCH * BATCH_SIZE))
        iteration = int(start_iteration)
        current_evaluations = train_x.shape[0]
        while current_evaluations < total_target:
            iteration += 1
            remaining = total_target - current_evaluations
            population_size = min(lambda_, remaining)

            eigvals, eigvecs = torch.linalg.eigh(cov + 1e-8 * torch.eye(dim, **tkwargs))
            eigvals = eigvals.clamp_min(1e-10)
            sqrt_cov = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

            z = torch.randn(population_size, dim, **tkwargs)
            candidates = mean.unsqueeze(0) + sigma * (z @ sqrt_cov.T)
            candidates = candidates.clamp(lower.unsqueeze(0), upper.unsqueeze(0))

            new_x, new_obj_true, new_info, cleanup_paths = evaluate_batch(candidates)
            train_x = torch.cat([train_x, new_x], dim=0)
            train_obj_true = torch.cat([train_obj_true, new_obj_true], dim=0)
            train_info.extend(new_info)
            current_evaluations = train_x.shape[0]

            scores = scalarize(new_obj_true)
            ranked = torch.argsort(scores, descending=True)
            elites = new_x[ranked[:mu]]
            elite_z = z[ranked[:mu]]

            old_mean = mean.clone()
            mean = torch.sum(elites * weights_t[: elites.shape[0]].unsqueeze(1), dim=0)
            y = (mean - old_mean) / sigma
            inv_sqrt_cov = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
            p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (inv_sqrt_cov @ y)
            norm_p_sigma = float(torch.linalg.norm(p_sigma).item())
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 2.0 / (dim + 1.0)) * expected_norm else 0.0
            p_c = (1.0 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * y

            rank_mu = torch.zeros_like(cov)
            elite_count = elites.shape[0]
            effective_weights = weights_t[:elite_count]
            if effective_weights.sum() <= 0:
                effective_weights = torch.full((elite_count,), 1.0 / elite_count, **tkwargs)
            else:
                effective_weights = effective_weights / effective_weights.sum()
            for idx in range(elite_count):
                diff = (elites[idx] - old_mean) / sigma
                rank_mu = rank_mu + effective_weights[idx] * torch.outer(diff, diff)

            cov = (
                (1.0 - c1 - c_mu) * cov
                + c1 * (torch.outer(p_c, p_c) + (1.0 - h_sigma) * c_c * (2.0 - c_c) * cov)
                + c_mu * rank_mu
            )
            cov = 0.5 * (cov + cov.T) + 1e-8 * torch.eye(dim, **tkwargs)
            sigma = float(sigma * np.exp((c_sigma / d_sigma) * (norm_p_sigma / expected_norm - 1.0)))
            sigma = max(min(sigma, float(span.max().item() * 2.0)), 1e-4)

            hv_value = compute_hypervolume(train_obj_true)
            hvs.append(hv_value)
            hv_curve = build_hv_curve(
                hvs=hvs,
                initial_samples=initial_samples,
                batch_size=BATCH_SIZE,
                total_evaluations=current_evaluations,
            )
            scheduler_history.append(
                build_sync_schedule_record(
                    iteration=iteration,
                    metric_items=new_info,
                    scheduler_gpu_count=scheduler_gpu_count,
                    candidate_index_offset=current_evaluations - len(new_info),
                )
            )
            if verbose:
                best_scalar = float(scores.max().item()) if scores.numel() > 0 else float("nan")
                print(f"迭代 {iteration:>2}: 超体积 = {hv_value:.4f}, 标量最优 = {best_scalar:.4f}, 已评估 = {current_evaluations}")
            if iteration_callback is not None:
                if len(cleanup_paths) > 0:
                    existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
                    iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + cleanup_paths))
                iteration_callback(iteration, train_x, train_obj_true, hvs)
            save_checkpoint(iteration, train_x, train_obj_true, train_info, hvs)
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    except Exception as exc:
        print(f"错误: EMM优化过程中发生异常: {exc}")
        import traceback
        traceback.print_exc()
        save_runtime_reports(run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
        return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id

    save_runtime_reports(run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
    return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
