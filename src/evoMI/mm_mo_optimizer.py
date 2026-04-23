import os
import gc
import json
import time
import inspect
import datetime
import numpy as np
import torch
from src.evoMI.runtime_artifacts import (
    build_eval_metadata,
    build_hv_curve,
    build_sync_schedule_record,
    load_standard_checkpoint,
    save_runtime_reports,
    save_standard_checkpoint,
)
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.fit import fit_gpytorch_mll as botorch_fit_gpytorch_mll


def mm_mo_optimizer(
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

    standard_bounds = torch.zeros(2, dim, **tkwargs)
    standard_bounds[1] = 1
    noise_se = torch.full((num_objectives,), noise_level, **tkwargs)
    if isinstance(ref_point, torch.Tensor):
        problem_ref_point = ref_point.to(**tkwargs)
    else:
        problem_ref_point = torch.full((num_objectives,), ref_point, **tkwargs)

    lower = bounds[0]
    upper = bounds[1]
    span = (upper - lower).clamp_min(1e-8)
    if m_prior is None:
        prior_center = (lower + upper) * 0.5
    else:
        prior_center = torch.tensor(m_prior, **tkwargs).flatten().clamp(lower, upper)
    if u_prior is None:
        prior_radius = span * 0.5
    else:
        prior_radius = torch.tensor(u_prior, **tkwargs).flatten()
        if prior_radius.numel() != dim:
            raise ValueError(f"u_prior长度应为{dim}，当前为{prior_radius.numel()}")
        if torch.min(prior_radius) >= 0 and torch.max(prior_radius) <= 1.0:
            prior_radius = prior_radius * span
        prior_radius = prior_radius.clamp_min(1e-8)

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

    def parse_objective_result(result, batch_size):
        cleanup_paths = []
        if isinstance(result, tuple) and len(result) == 3:
            obj_true, info, cleanup_paths = result
        elif isinstance(result, tuple) and len(result) == 2:
            obj_true, info = result
            cleanup_paths = collect_cleanup_paths(info)
        else:
            obj_true = result
            info = [{} for _ in range(batch_size)]
        if not isinstance(info, list):
            info = [{} for _ in range(batch_size)]
        return obj_true, info, cleanup_paths if isinstance(cleanup_paths, list) else []

    def evaluate_batch(x_batch, eval_limit=None, eval_mode="full", estimated_tokens=None):
        kwargs = {}
        if objective_supports_eval_limit:
            kwargs["eval_limit"] = eval_limit
        if objective_supports_eval_mode:
            kwargs["eval_mode"] = eval_mode
        if objective_supports_estimated_tokens and estimated_tokens is not None:
            kwargs["estimated_tokens"] = estimated_tokens
        result = objective_function(x_batch, **kwargs)
        obj_true, info, cleanup_paths = parse_objective_result(result, x_batch.shape[0])
        obj_true = obj_true.to(**tkwargs)
        obj = obj_true + torch.randn_like(obj_true) * noise_se
        return x_batch, obj, obj_true, info, cleanup_paths

    def generate_initial_data(n=initial_samples):
        if isinstance(shared_initial_dataset, dict):
            shared_x = torch.as_tensor(shared_initial_dataset.get("decision_variables", []), **tkwargs)
            shared_obj_true = torch.as_tensor(shared_initial_dataset.get("objectives", []), **tkwargs)
            shared_info = shared_initial_dataset.get("metrics", [{} for _ in range(shared_x.shape[0])])
            if shared_x.ndim == 2 and shared_x.shape == (n, dim) and shared_obj_true.shape == (n, num_objectives):
                shared_obj = shared_obj_true + torch.randn_like(shared_obj_true) * noise_se
                return shared_x, shared_obj, shared_obj_true, shared_info
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
        train_x, train_obj, train_obj_true, train_info, initial_cleanup_paths = evaluate_batch(
            train_x,
            eval_limit=full_eval_limits,
            eval_mode="full",
        )
        if iteration_callback is not None and len(initial_cleanup_paths) > 0:
            existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
            iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + initial_cleanup_paths))
        return train_x, train_obj, train_obj_true, train_info

    def initialize_model(train_x, train_obj):
        normalized_x = normalize(train_x, bounds)
        models = []
        for index in range(train_obj.shape[-1]):
            train_y = train_obj[..., index:index + 1]
            train_yvar = torch.full_like(train_y, noise_se[index] ** 2)
            models.append(SingleTaskGP(normalized_x, train_y, train_yvar))
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def compute_hypervolume(objectives):
        partitioning = FastNondominatedPartitioning(ref_point=problem_ref_point, Y=objectives)
        return float(partitioning.compute_hypervolume().item())

    def is_non_dominated(points):
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

    def hypervolume_contributions(objectives):
        base_hv = compute_hypervolume(objectives)
        contributions = torch.zeros(objectives.shape[0], **tkwargs)
        for idx in range(objectives.shape[0]):
            if objectives.shape[0] == 1:
                contributions[idx] = base_hv
            else:
                reduced = torch.cat([objectives[:idx], objectives[idx + 1:]], dim=0)
                contributions[idx] = base_hv - compute_hypervolume(reduced)
        return contributions

    def select_elites(train_x, train_obj_true):
        pareto_mask = is_non_dominated(train_obj_true)
        pareto_x = train_x[pareto_mask]
        pareto_obj = train_obj_true[pareto_mask]
        if pareto_x.shape[0] == 0:
            return train_x[: min(topk, train_x.shape[0])]
        contributions = hypervolume_contributions(pareto_obj)
        rank = torch.argsort(contributions, descending=True)
        elite_count = min(max(2, topk), pareto_x.shape[0])
        return pareto_x[rank[:elite_count]]

    def generate_w2s_candidates(train_x, train_obj_true, candidate_count):
        elites = select_elites(train_x, train_obj_true)
        if elites.shape[0] == 0:
            return torch.empty((0, dim), **tkwargs)
        generated = []
        total = max(candidate_count, BATCH_SIZE)
        for _ in range(total):
            if elites.shape[0] >= 3:
                chosen = torch.randperm(elites.shape[0], device=elites.device)[:3]
                r1, r2, r3 = elites[chosen[0]], elites[chosen[1]], elites[chosen[2]]
            elif elites.shape[0] == 2:
                r1, r2 = elites[0], elites[1]
                r3 = train_x[torch.randint(train_x.shape[0], (1,), device=train_x.device).item()]
            else:
                r1 = elites[0]
                r2 = train_x[torch.randint(train_x.shape[0], (1,), device=train_x.device).item()]
                r3 = train_x[torch.randint(train_x.shape[0], (1,), device=train_x.device).item()]
            scale = 0.5 + 0.3 * torch.rand(1, **tkwargs)
            candidate = r1 + scale * (r2 - r3)
            if torch.rand(1, **tkwargs).item() < 0.5:
                perturb = torch.randn(dim, **tkwargs) * (prior_radius * 0.15 * max(rho, 1e-3))
                candidate = candidate + perturb
            candidate = 0.7 * candidate + 0.3 * prior_center
            generated.append(candidate.clamp(lower, upper))
        return torch.stack(generated, dim=0)

    def deduplicate_candidates(candidates):
        if candidates.shape[0] == 0:
            return candidates
        clipped = candidates.clamp(lower.unsqueeze(0), upper.unsqueeze(0))
        rounded = torch.round(clipped * 1e6) / 1e6
        kept_rows = []
        seen = set()
        for index in range(rounded.shape[0]):
            key = tuple(float(v) for v in rounded[index].detach().cpu().tolist())
            if key in seen:
                continue
            seen.add(key)
            kept_rows.append(index)
        return clipped[kept_rows]

    def select_candidates(model, train_x, train_obj, train_obj_true, sampler):
        baseline_x = normalize(train_x, bounds)
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=problem_ref_point,
            X_baseline=baseline_x,
            prune_baseline=True,
            sampler=sampler,
        )

        acq_pool_size = max(BATCH_SIZE * 3, BATCH_SIZE + 2)
        qehvi_candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=acq_pool_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 100},
            sequential=True,
        )
        qehvi_candidates = unnormalize(qehvi_candidates.detach(), bounds)

        w2s_candidates = generate_w2s_candidates(train_x, train_obj_true, acq_pool_size)
        candidate_pool = torch.cat([qehvi_candidates, w2s_candidates], dim=0)
        candidate_pool = deduplicate_candidates(candidate_pool)
        if candidate_pool.shape[0] == 0:
            candidate_pool = qehvi_candidates[:BATCH_SIZE]

        normalized_pool = normalize(candidate_pool, bounds)
        acq_values = acq_func(normalized_pool.unsqueeze(1)).flatten()
        posterior = model.posterior(normalized_pool)
        variance = posterior.variance
        if variance.dim() == 3:
            variance = variance.squeeze(1)
        uncertainty = variance.mean(dim=-1)
        fi_score = 1.0 / (uncertainty + 1e-8)

        shortlist_size = min(candidate_pool.shape[0], max(BATCH_SIZE * 3, BATCH_SIZE))
        acquisition_rank = torch.argsort(acq_values, descending=True)[:shortlist_size]
        shortlisted_pool = candidate_pool[acquisition_rank]
        shortlisted_uncertainty = uncertainty[acquisition_rank]
        shortlisted_fi = fi_score[acquisition_rank]
        final_score = shortlisted_uncertainty - 0.05 * shortlisted_fi
        selected_rank = torch.argsort(final_score, descending=True)[:BATCH_SIZE]
        return shortlisted_pool[selected_rank]

    scheduler_history = []
    hv_curve = []

    def save_checkpoint(iteration, train_x, train_obj, train_obj_true, train_info, hvs):
        save_standard_checkpoint(
            run_dir=run_dir,
            iteration=iteration,
            train_x=train_x,
            train_obj=train_obj,
            train_obj_true=train_obj_true,
            train_info=train_info,
            hvs=hvs,
            scheduler_history=scheduler_history,
            hv_curve=hv_curve,
            eval_metadata=eval_metadata,
        )

    if verbose:
        print(f"使用MM-MO同步优化多目标问题 (维度: {dim}, 目标数: {num_objectives})")
        print(f"运行ID: {run_id}, 检查点目录: {run_dir}")
        print(f"每轮候选数: {BATCH_SIZE}, 初始样本数: {initial_samples}, 最大评估数: {max_evaluations}")

    checkpoint = load_standard_checkpoint(run_dir, tkwargs)
    if checkpoint is not None:
        train_x = checkpoint["train_x"]
        train_obj = checkpoint["train_obj"]
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
        if verbose:
            print(f"成功加载检查点，从迭代 {start_iteration} 继续")
            if len(hvs) > 0:
                print(f"当前超体积: {hvs[-1]:.4f}")
    else:
        train_x, train_obj, train_obj_true, train_info = generate_initial_data()
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
        save_checkpoint(0, train_x, train_obj, train_obj_true, train_info, hvs)
        start_iteration = 0

    try:
        total_target = min(int(max_evaluations), int(initial_samples + N_BATCH * BATCH_SIZE))
        current_evaluations = train_x.shape[0]
        iteration = int(start_iteration)
        while current_evaluations < total_target:
            iteration += 1
            mll, model = initialize_model(train_x, train_obj)
            botorch_fit_gpytorch_mll(mll)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            batch_q = min(BATCH_SIZE, total_target - current_evaluations)
            selected_candidates = select_candidates(model, train_x, train_obj, train_obj_true, sampler)[:batch_q]
            new_x, new_obj, new_obj_true, new_info, cleanup_paths = evaluate_batch(
                selected_candidates,
                eval_limit=full_eval_limits,
                eval_mode="full",
            )
            train_x = torch.cat([train_x, new_x], dim=0)
            train_obj = torch.cat([train_obj, new_obj], dim=0)
            train_obj_true = torch.cat([train_obj_true, new_obj_true], dim=0)
            train_info.extend(new_info)
            current_evaluations = train_x.shape[0]
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
                print(f"迭代 {iteration:>2}: 超体积 = {hv_value:.4f}, 已评估 = {current_evaluations}")
            if iteration_callback is not None:
                if len(cleanup_paths) > 0:
                    existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
                    iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + cleanup_paths))
                iteration_callback(iteration, train_x, train_obj_true, hvs)
            save_checkpoint(iteration, train_x, train_obj, train_obj_true, train_info, hvs)
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    except Exception as exc:
        print(f"错误: MM-MO优化过程中发生异常: {exc}")
        import traceback
        traceback.print_exc()
        save_runtime_reports(run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
        return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id

    save_runtime_reports(run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
    return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
