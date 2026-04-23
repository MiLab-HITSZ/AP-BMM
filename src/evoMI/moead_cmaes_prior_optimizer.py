import gc
import datetime
import inspect
import os

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.sampling import draw_sobol_samples

from src.evoMI.checkpoint_runtime import (
    build_eval_metadata,
    build_hv_curve,
    build_sync_schedule_record,
    load_standard_checkpoint,
    save_runtime_reports,
    save_standard_checkpoint,
)


def moead_cmaes_prior_optimizer(
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

    def decode_z_to_x(z_batch):
        n = z_batch.shape[0]
        x = m_prior_t.unsqueeze(0).repeat(n, 1)
        ptr = 0
        for layer_idx in top_layers:
            x[:, layer_idx] = m_prior_t[layer_idx] + rho * u_prior_t[layer_idx] * z_batch[:, ptr]
            ptr += 1
        for group in groups:
            zg = z_batch[:, ptr].unsqueeze(1)
            idx = torch.tensor(group, device=z_batch.device, dtype=torch.long)
            x[:, idx] = m_prior_t[idx].unsqueeze(0) + rho * u_prior_t[idx].unsqueeze(0) * zg
            ptr += 1
        return x.clamp(lower.unsqueeze(0), upper.unsqueeze(0))

    def encode_x_to_z(x_batch):
        z = torch.zeros((x_batch.shape[0], z_dim), **tkwargs)
        ptr = 0
        denom = (rho * u_prior_t).clamp_min(1e-8)
        for layer_idx in top_layers:
            z[:, ptr] = ((x_batch[:, layer_idx] - m_prior_t[layer_idx]) / denom[layer_idx]).clamp(-1, 1)
            ptr += 1
        for group in groups:
            idx = torch.tensor(group, device=x_batch.device, dtype=torch.long)
            z_group = ((x_batch[:, idx] - m_prior_t[idx].unsqueeze(0)) / denom[idx].unsqueeze(0)).mean(dim=1)
            z[:, ptr] = z_group.clamp(-1, 1)
            ptr += 1
        return z

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
        if not isinstance(cleanup_paths, list):
            cleanup_paths = []
        return obj_true, info, cleanup_paths

    def evaluate_from_z(z_batch, estimated_tokens=None):
        kwargs = {}
        if objective_supports_eval_limit:
            kwargs["eval_limit"] = full_eval_limits
        if objective_supports_eval_mode:
            kwargs["eval_mode"] = "full"
        if objective_supports_estimated_tokens and estimated_tokens is not None:
            kwargs["estimated_tokens"] = estimated_tokens
        x_batch = decode_z_to_x(z_batch)
        result = objective_function(x_batch, **kwargs)
        obj_true, info, cleanup_paths = parse_objective_result(result, x_batch.shape[0])
        return x_batch, obj_true.to(**tkwargs), info, cleanup_paths

    def compute_hypervolume(objectives):
        partitioning = FastNondominatedPartitioning(ref_point=problem_ref_point, Y=objectives)
        return float(partitioning.compute_hypervolume().item())

    def compute_normalized_objectives(objectives):
        if objectives.numel() == 0:
            return objectives
        obj_min = train_obj_true.min(dim=0).values if train_obj_true.shape[0] > 0 else objectives.min(dim=0).values
        obj_max = train_obj_true.max(dim=0).values if train_obj_true.shape[0] > 0 else objectives.max(dim=0).values
        obj_min = torch.minimum(obj_min, objectives.min(dim=0).values)
        obj_max = torch.maximum(obj_max, objectives.max(dim=0).values)
        obj_span = (obj_max - obj_min).clamp_min(1e-8)
        return (objectives - obj_min.unsqueeze(0)) / obj_span.unsqueeze(0)

    def compute_scores_for_weights(objectives, weight_subset):
        normalized = compute_normalized_objectives(objectives).clamp(0.0, 1.5)
        safe_weights = torch.where(weight_subset > 0, weight_subset, torch.full_like(weight_subset, 1e-3))
        penalties = (1.0 - normalized.unsqueeze(1)) * safe_weights.unsqueeze(0)
        tchebycheff = penalties.max(dim=-1).values
        aggregate = penalties.sum(dim=-1)
        return -(tchebycheff + 0.05 * aggregate)

    def build_weight_vectors():
        subproblem_count = max(8, min(24, int(BATCH_SIZE) * 4))
        if num_objectives <= 1:
            return torch.ones((1, 1), **tkwargs)
        if num_objectives == 2:
            grid = torch.linspace(0.0, 1.0, steps=subproblem_count, **tkwargs)
            return torch.stack([grid, 1.0 - grid], dim=1)
        samples = np.random.dirichlet(np.ones(num_objectives), size=subproblem_count)
        return torch.tensor(samples, **tkwargs)

    def build_neighbor_indices(weight_vectors):
        pairwise = torch.cdist(weight_vectors, weight_vectors, p=2)
        neighbor_count = max(2, min(weight_vectors.shape[0], max(3, int(np.ceil(weight_vectors.shape[0] / 4)))))
        return torch.topk(pairwise, k=neighbor_count, dim=1, largest=False).indices

    def build_recombination_weights(elite_count):
        elite_count = max(int(elite_count), 1)
        weight_values = np.log(elite_count + 0.5) - np.log(np.arange(1, elite_count + 1))
        weight_values = np.maximum(weight_values, 1e-8)
        weight_values = weight_values / weight_values.sum()
        return torch.tensor(weight_values, **tkwargs)

    lambda_state = max(2, min(4, int(BATCH_SIZE)))
    mu_state = max(lambda_state // 2, 1)
    base_weights = np.log(mu_state + 0.5) - np.log(np.arange(1, mu_state + 1))
    base_weights = base_weights / base_weights.sum()
    mu_eff = float(1.0 / np.sum(base_weights ** 2))
    c_sigma = (mu_eff + 2.0) / (z_dim + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(np.sqrt((mu_eff - 1.0) / (z_dim + 1.0)) - 1.0, 0.0) + c_sigma
    c_c = (4.0 + mu_eff / z_dim) / (z_dim + 4.0 + 2.0 * mu_eff / z_dim)
    c1 = 2.0 / ((z_dim + 1.3) ** 2 + mu_eff)
    c_mu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((z_dim + 2.0) ** 2 + mu_eff))
    expected_norm = np.sqrt(z_dim) * (1.0 - 1.0 / (4.0 * z_dim) + 1.0 / (21.0 * z_dim * z_dim))

    weight_vectors = build_weight_vectors()
    neighbor_indices = build_neighbor_indices(weight_vectors)
    num_subproblems = int(weight_vectors.shape[0])
    prior_center_z = torch.zeros(z_dim, **tkwargs)
    initial_sigma = float(min(0.45, max(0.12, 0.35 * max(float(rho), 0.35))))

    def initialize_state_tensors():
        means = prior_center_z.unsqueeze(0).repeat(num_subproblems, 1).clone()
        sigmas = torch.full((num_subproblems,), initial_sigma, **tkwargs)
        covs = torch.eye(z_dim, **tkwargs).unsqueeze(0).repeat(num_subproblems, 1, 1)
        p_sigma_all = torch.zeros((num_subproblems, z_dim), **tkwargs)
        p_c_all = torch.zeros((num_subproblems, z_dim), **tkwargs)
        best_scores = torch.full((num_subproblems,), float("-inf"), **tkwargs)
        best_z = torch.zeros((num_subproblems, z_dim), **tkwargs)
        best_x = decode_z_to_x(best_z)
        best_obj = torch.full((num_subproblems, num_objectives), float("-inf"), **tkwargs)
        has_best = torch.zeros((num_subproblems,), dtype=torch.bool, device=tkwargs["device"])
        return means, sigmas, covs, p_sigma_all, p_c_all, best_scores, best_z, best_x, best_obj, has_best

    def sample_initial_points(n=initial_samples):
        initial_cleanup_paths = []
        if isinstance(shared_initial_dataset, dict):
            shared_x = torch.as_tensor(shared_initial_dataset.get("decision_variables", []), **tkwargs)
            shared_obj_true = torch.as_tensor(shared_initial_dataset.get("objectives", []), **tkwargs)
            shared_info = shared_initial_dataset.get("metrics", [{} for _ in range(shared_x.shape[0])])
            if shared_x.ndim == 2 and shared_x.shape[1] == dim and shared_obj_true.ndim == 2 and shared_obj_true.shape[1] == num_objectives:
                if shared_x.shape[0] >= n:
                    return shared_x[:n], encode_x_to_z(shared_x[:n]), shared_obj_true[:n], shared_info[:n], initial_cleanup_paths
        if custom_initial_solutions is not None and len(custom_initial_solutions) > 0:
            custom_x = []
            for value in custom_initial_solutions[:n]:
                custom_x.append(torch.full((1, dim), value, **tkwargs))
            custom_x = torch.cat(custom_x, dim=0) if len(custom_x) > 0 else torch.empty((0, dim), **tkwargs)
            remaining = max(n - custom_x.shape[0], 0)
            if remaining > 0:
                sobol_x = draw_sobol_samples(bounds=bounds, n=remaining, q=1).squeeze(1).to(**tkwargs)
                train_x_local = torch.cat([custom_x, sobol_x], dim=0)
            else:
                train_x_local = custom_x[:n]
        else:
            train_x_local = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1).to(**tkwargs)
        train_z_local = encode_x_to_z(train_x_local)
        train_x_local, train_obj_true_local, train_info_local, initial_cleanup_paths = evaluate_from_z(train_z_local)
        return train_x_local, train_z_local, train_obj_true_local, train_info_local, initial_cleanup_paths

    scheduler_history = []
    hv_curve = []
    train_x = torch.empty((0, dim), **tkwargs)
    train_z = torch.empty((0, z_dim), **tkwargs)
    train_obj_true = torch.empty((0, num_objectives), **tkwargs)
    train_info = []
    (
        means,
        sigmas,
        covs,
        p_sigma_all,
        p_c_all,
        subproblem_best_scores,
        subproblem_best_z,
        subproblem_best_x,
        subproblem_best_obj,
        subproblem_has_best,
    ) = initialize_state_tensors()

    def refresh_subproblem_incumbents(candidate_z, candidate_x, candidate_obj):
        nonlocal subproblem_best_scores, subproblem_best_z, subproblem_best_x, subproblem_best_obj, subproblem_has_best, means
        if candidate_obj.shape[0] == 0:
            return
        score_matrix = compute_scores_for_weights(candidate_obj, weight_vectors)
        best_values, best_indices = score_matrix.max(dim=0)
        improvement_mask = (~subproblem_has_best) | (best_values > subproblem_best_scores)
        if improvement_mask.any():
            improved_indices = torch.nonzero(improvement_mask, as_tuple=False).flatten()
            chosen_candidates = best_indices[improved_indices]
            subproblem_best_scores[improved_indices] = best_values[improved_indices]
            subproblem_best_z[improved_indices] = candidate_z[chosen_candidates]
            subproblem_best_x[improved_indices] = candidate_x[chosen_candidates]
            subproblem_best_obj[improved_indices] = candidate_obj[chosen_candidates]
            subproblem_has_best[improved_indices] = True
            means[improved_indices] = 0.6 * means[improved_indices] + 0.4 * subproblem_best_z[improved_indices]

    def save_checkpoint(iteration, hvs):
        save_standard_checkpoint(
            run_dir=run_dir,
            iteration=iteration,
            train_x=train_x,
            train_obj_true=train_obj_true,
            train_info=train_info,
            hvs=hvs,
            extra_json={
                "optimizer_state": {
                    "algorithm": "moead_cmaes",
                    "num_subproblems": int(num_subproblems),
                    "z_dim": int(z_dim),
                    "top_layers": top_layers,
                    "groups": groups,
                    "rho": float(rho),
                    "weight_vectors": weight_vectors.detach().cpu().tolist(),
                }
            },
            extra_state={
                "train_z": train_z.detach().cpu(),
                "means": means.detach().cpu(),
                "sigmas": sigmas.detach().cpu(),
                "covs": covs.detach().cpu(),
                "p_sigma_all": p_sigma_all.detach().cpu(),
                "p_c_all": p_c_all.detach().cpu(),
                "subproblem_best_scores": subproblem_best_scores.detach().cpu(),
                "subproblem_best_z": subproblem_best_z.detach().cpu(),
                "subproblem_best_x": subproblem_best_x.detach().cpu(),
                "subproblem_best_obj": subproblem_best_obj.detach().cpu(),
                "subproblem_has_best": subproblem_has_best.detach().cpu(),
                "weight_vectors": weight_vectors.detach().cpu(),
                "neighbor_indices": neighbor_indices.detach().cpu(),
                "top_layers": top_layers,
                "groups": groups,
                "m_prior": m_prior_t.detach().cpu(),
                "u_prior": u_prior_t.detach().cpu(),
                "rho": float(rho),
            },
            scheduler_history=scheduler_history,
            hv_curve=hv_curve,
            eval_metadata=eval_metadata,
        )

    if verbose:
        print(f"使用MOEA-D/CMA-ES+先验同步优化多目标问题 (原始维度: {dim}, 优化维度: {z_dim}, 目标数: {num_objectives})")
        print(f"运行ID: {run_id}, 检查点目录: {run_dir}")
        print(f"子问题数: {num_subproblems}, 初始sigma: {initial_sigma:.4f}, 批大小: {BATCH_SIZE}")
        if async_mode:
            print("当前算法为同步实现，将忽略async_mode配置")

    checkpoint = load_standard_checkpoint(run_dir, tkwargs)
    if checkpoint is not None:
        train_x = checkpoint["train_x"]
        train_obj_true = checkpoint["train_obj_true"]
        train_info = checkpoint.get("train_info", [{} for _ in range(train_x.shape[0])])
        train_z = checkpoint.get("train_z", encode_x_to_z(train_x).detach().cpu()).to(**tkwargs)
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
        means = checkpoint.get("means", means.detach().cpu()).to(**tkwargs)
        sigmas = checkpoint.get("sigmas", sigmas.detach().cpu()).to(**tkwargs)
        covs = checkpoint.get("covs", covs.detach().cpu()).to(**tkwargs)
        p_sigma_all = checkpoint.get("p_sigma_all", p_sigma_all.detach().cpu()).to(**tkwargs)
        p_c_all = checkpoint.get("p_c_all", p_c_all.detach().cpu()).to(**tkwargs)
        subproblem_best_scores = checkpoint.get("subproblem_best_scores", subproblem_best_scores.detach().cpu()).to(**tkwargs)
        subproblem_best_z = checkpoint.get("subproblem_best_z", subproblem_best_z.detach().cpu()).to(**tkwargs)
        subproblem_best_x = checkpoint.get("subproblem_best_x", subproblem_best_x.detach().cpu()).to(**tkwargs)
        subproblem_best_obj = checkpoint.get("subproblem_best_obj", subproblem_best_obj.detach().cpu()).to(**tkwargs)
        subproblem_has_best = checkpoint.get("subproblem_has_best", subproblem_has_best.detach().cpu()).to(device=tkwargs["device"])
        if verbose:
            print(f"成功加载检查点，从迭代 {start_iteration} 继续")
            if len(hvs) > 0:
                print(f"当前超体积: {hvs[-1]:.4f}")
    else:
        train_x, train_z, train_obj_true, train_info, initial_cleanup_paths = sample_initial_points()
        refresh_subproblem_incumbents(train_z, train_x, train_obj_true)
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
        save_checkpoint(0, hvs)
        if iteration_callback is not None and len(initial_cleanup_paths) > 0:
            existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
            iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + initial_cleanup_paths))
            iteration_callback(0, train_x, train_obj_true, hvs)
        start_iteration = 0

    try:
        total_target = min(int(max_evaluations), int(initial_samples + N_BATCH * BATCH_SIZE))
        iteration = int(start_iteration)
        current_evaluations = train_x.shape[0]
        active_pointer = int(current_evaluations % max(num_subproblems, 1))

        while current_evaluations < total_target:
            iteration += 1
            remaining = total_target - current_evaluations
            population_size = min(int(BATCH_SIZE), remaining)
            offspring_per_subproblem = 2 if population_size >= 4 else 1
            active_count = int(np.ceil(population_size / max(offspring_per_subproblem, 1)))

            active_subproblems = []
            for offset in range(active_count):
                active_subproblems.append(int((active_pointer + offset) % num_subproblems))
            active_pointer = int((active_pointer + active_count) % num_subproblems)

            candidate_batches = []
            candidate_owner_slices = []
            owner_sampling_cache = {}
            produced = 0

            for owner_idx in active_subproblems:
                local_count = min(offspring_per_subproblem, population_size - produced)
                if local_count <= 0:
                    break
                neighborhood = neighbor_indices[owner_idx]
                valid_neighbors = neighborhood[subproblem_has_best[neighborhood]]
                if valid_neighbors.numel() > 0:
                    neighbor_center = subproblem_best_z[valid_neighbors].mean(dim=0)
                    parent_mean = 0.65 * means[owner_idx] + 0.35 * neighbor_center
                else:
                    parent_mean = means[owner_idx]
                cov_k = covs[owner_idx] + 1e-8 * torch.eye(z_dim, **tkwargs)
                eigvals, eigvecs = torch.linalg.eigh(cov_k)
                eigvals = eigvals.clamp_min(1e-10)
                sqrt_cov = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
                standard_noise = torch.randn(local_count, z_dim, **tkwargs)
                local_z = parent_mean.unsqueeze(0) + sigmas[owner_idx] * (standard_noise @ sqrt_cov.T)
                if valid_neighbors.numel() >= 2:
                    perm = torch.randperm(valid_neighbors.numel(), device=valid_neighbors.device)
                    diff = subproblem_best_z[valid_neighbors[perm[0]]] - subproblem_best_z[valid_neighbors[perm[1]]]
                    local_z = local_z + 0.15 * diff.unsqueeze(0)
                local_z = local_z.clamp(z_bounds_eff[0].unsqueeze(0), z_bounds_eff[1].unsqueeze(0))
                owner_sampling_cache[owner_idx] = {
                    "eigvals": eigvals,
                    "eigvecs": eigvecs,
                }
                start_idx = produced
                end_idx = produced + local_count
                candidate_owner_slices.append((owner_idx, start_idx, end_idx))
                candidate_batches.append(local_z)
                produced += local_count

            candidate_z = torch.cat(candidate_batches, dim=0) if len(candidate_batches) > 0 else torch.empty((0, z_dim), **tkwargs)
            candidate_x, candidate_obj_true, candidate_info, cleanup_paths = evaluate_from_z(candidate_z)

            train_x = torch.cat([train_x, candidate_x], dim=0)
            train_z = torch.cat([train_z, candidate_z], dim=0)
            train_obj_true = torch.cat([train_obj_true, candidate_obj_true], dim=0)
            train_info.extend(candidate_info)
            current_evaluations = train_x.shape[0]

            previous_best_scores = subproblem_best_scores.clone()
            refresh_subproblem_incumbents(candidate_z, candidate_x, candidate_obj_true)

            for owner_idx, start_idx, end_idx in candidate_owner_slices:
                local_z = candidate_z[start_idx:end_idx]
                local_obj = candidate_obj_true[start_idx:end_idx]
                if local_z.shape[0] == 0:
                    continue
                owner_weights = weight_vectors[owner_idx : owner_idx + 1]
                local_scores = compute_scores_for_weights(local_obj, owner_weights).squeeze(1)
                ranked = torch.argsort(local_scores, descending=True)
                elite_count = min(max(1, mu_state), local_z.shape[0])
                elite_z = local_z[ranked[:elite_count]]
                effective_weights = build_recombination_weights(elite_count)
                old_mean = means[owner_idx].clone()
                recombined = torch.sum(elite_z * effective_weights.unsqueeze(1), dim=0)
                neighborhood = neighbor_indices[owner_idx]
                valid_neighbors = neighborhood[subproblem_has_best[neighborhood]]
                if valid_neighbors.numel() > 0:
                    neighborhood_center = subproblem_best_z[valid_neighbors].mean(dim=0)
                    new_mean = 0.75 * recombined + 0.25 * neighborhood_center
                else:
                    new_mean = recombined
                cached = owner_sampling_cache[owner_idx]
                eigvals = cached["eigvals"]
                eigvecs = cached["eigvecs"]
                inv_sqrt_cov = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
                sigma_k = float(sigmas[owner_idx].item())
                y = (new_mean - old_mean) / max(sigma_k, 1e-8)
                p_sigma_all[owner_idx] = (
                    (1.0 - c_sigma) * p_sigma_all[owner_idx]
                    + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (inv_sqrt_cov @ y)
                )
                norm_p_sigma = float(torch.linalg.norm(p_sigma_all[owner_idx]).item())
                h_sigma = 1.0 if norm_p_sigma < (1.4 + 2.0 / (z_dim + 1.0)) * expected_norm else 0.0
                p_c_all[owner_idx] = (
                    (1.0 - c_c) * p_c_all[owner_idx]
                    + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * y
                )
                rank_mu = torch.zeros((z_dim, z_dim), **tkwargs)
                for elite_idx in range(elite_count):
                    diff = (elite_z[elite_idx] - old_mean) / max(sigma_k, 1e-8)
                    rank_mu = rank_mu + effective_weights[elite_idx] * torch.outer(diff, diff)
                updated_cov = (
                    (1.0 - c1 - c_mu) * covs[owner_idx]
                    + c1 * (
                        torch.outer(p_c_all[owner_idx], p_c_all[owner_idx])
                        + (1.0 - h_sigma) * c_c * (2.0 - c_c) * covs[owner_idx]
                    )
                    + c_mu * rank_mu
                )
                covs[owner_idx] = 0.5 * (updated_cov + updated_cov.T) + 1e-8 * torch.eye(z_dim, **tkwargs)
                updated_sigma = sigma_k * np.exp((c_sigma / d_sigma) * (norm_p_sigma / expected_norm - 1.0))
                best_local_score = float(local_scores.max().item())
                if best_local_score <= float(previous_best_scores[owner_idx].item()):
                    updated_sigma *= 1.03
                sigmas[owner_idx] = torch.tensor(min(max(updated_sigma, 0.03), 1.25), **tkwargs)
                means[owner_idx] = new_mean.clamp(z_bounds_eff[0], z_bounds_eff[1])

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
                    metric_items=candidate_info,
                    scheduler_gpu_count=scheduler_gpu_count,
                    candidate_index_offset=current_evaluations - len(candidate_info),
                )
            )

            if verbose:
                current_best = float(subproblem_best_scores.max().item()) if subproblem_best_scores.numel() > 0 else float("nan")
                sigma_mean = float(sigmas.mean().item()) if sigmas.numel() > 0 else float("nan")
                print(
                    f"迭代 {iteration:>2}: 超体积 = {hv_value:.4f}, "
                    f"最佳分解分数 = {current_best:.4f}, 平均sigma = {sigma_mean:.4f}, 已评估 = {current_evaluations}"
                )

            if iteration_callback is not None:
                if len(cleanup_paths) > 0:
                    existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
                    iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + cleanup_paths))
                iteration_callback(iteration, train_x, train_obj_true, hvs)

            save_checkpoint(iteration, hvs)
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    except Exception as exc:
        print(f"错误: MOEA-D/CMA-ES+先验优化过程中发生异常: {exc}")
        import traceback
        traceback.print_exc()
        save_runtime_reports(run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
        return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id

    save_runtime_reports(run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
    return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
