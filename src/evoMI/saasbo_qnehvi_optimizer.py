import os
import gc
import time
import inspect
import threading
import concurrent.futures
import torch
import numpy as np
import datetime
from src.evoMI.optimization_reporting import reporter
from src.evoMI.runtime_artifacts import (
    build_hv_curve,
    build_sync_schedule_record,
    load_standard_checkpoint,
    save_runtime_reports,
    save_standard_checkpoint,
)
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.settings import fast_computations
from botorch.fit import fit_gpytorch_mll as botorch_fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval
from gpytorch.priors import GammaPrior, HalfCauchyPrior
from botorch.models.transforms import Normalize as BoTorchNormalize, Standardize

def _to_numpy_vector(values):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _robust_normalize(values):
    arr = _to_numpy_vector(values)
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.full_like(arr, 0.5)
    finite_vals = arr[finite_mask]
    median = np.median(finite_vals)
    mad = np.median(np.abs(finite_vals - median))
    if mad < 1e-8:
        min_v = np.min(finite_vals)
        max_v = np.max(finite_vals)
        if max_v - min_v < 1e-8:
            normalized = np.full_like(arr, 0.5)
            normalized[~finite_mask] = 0.5
            return normalized
        normalized = (arr - min_v) / (max_v - min_v)
        normalized[~finite_mask] = 0.5
        return np.clip(normalized, 0.0, 1.0)
    z_score = (arr - median) / (1.4826 * mad + 1e-8)
    normalized = 1.0 / (1.0 + np.exp(-z_score))
    normalized[~finite_mask] = 0.5
    return np.clip(normalized, 0.0, 1.0)


def _smooth_scores(values):
    arr = _to_numpy_vector(values)
    if arr.size <= 2:
        return arr
    smoothed = arr.copy()
    smoothed[1:-1] = 0.25 * arr[:-2] + 0.5 * arr[1:-1] + 0.25 * arr[2:]
    return smoothed


def _normalize_importance_scores(values, fill_value=0.5):
    arr = _to_numpy_vector(values)
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.full_like(arr, fill_value)
    finite_vals = arr[finite_mask]
    min_v = np.min(finite_vals)
    max_v = np.max(finite_vals)
    if max_v - min_v > 1e-8:
        normalized = (arr - min_v) / (max_v - min_v)
    else:
        scale = max(abs(max_v), 1e-8)
        normalized = arr / scale
    normalized[~finite_mask] = fill_value
    return np.clip(normalized, 0.0, 1.0)


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


def _calibrate_importance_scores(values, fill_value=0.5, floor=0.05, rank_blend=0.35):
    normalized = _normalize_importance_scores(values, fill_value=fill_value)
    if normalized.size == 0:
        return normalized
    rank_info = _rank_importance(normalized)
    rank_score = 1.0 - rank_info["rank_ratio"]
    blended = (1.0 - rank_blend) * normalized + rank_blend * rank_score
    return np.clip(floor + (1.0 - floor) * blended, floor, 1.0)


def _rank_importance(values):
    arr = _to_numpy_vector(values)
    n = arr.size
    if n == 0:
        return {
            "values": arr,
            "sorted_indices": np.asarray([], dtype=np.int64),
            "rank_position": np.asarray([], dtype=np.int64),
            "rank_ratio": arr,
        }
    safe_arr = np.where(np.isfinite(arr), arr, -np.inf)
    sorted_indices = np.argsort(-safe_arr, kind="stable")
    rank_position = np.empty(n, dtype=np.int64)
    rank_position[sorted_indices] = np.arange(n, dtype=np.int64)
    denominator = max(n - 1, 1)
    rank_ratio = rank_position.astype(np.float64) / denominator
    return {
        "values": arr,
        "sorted_indices": sorted_indices,
        "rank_position": rank_position,
        "rank_ratio": rank_ratio,
    }


def _importance_category_by_rank(values):
    rank_info = _rank_importance(values)
    rank_ratio = rank_info["rank_ratio"]
    categories = np.empty(rank_ratio.shape[0], dtype=object)
    if rank_ratio.size == 0:
        return categories, rank_info
    categories[rank_ratio <= 0.10] = "关键"
    categories[(rank_ratio > 0.10) & (rank_ratio <= 0.30)] = "重要"
    categories[(rank_ratio > 0.30) & (rank_ratio <= 0.60)] = "中等"
    categories[(rank_ratio > 0.60) & (rank_ratio <= 0.85)] = "次要"
    categories[rank_ratio > 0.85] = "可忽略"
    return categories, rank_info


def _sampling_scale_by_rank(values):
    rank_info = _rank_importance(values)
    rank_ratio = rank_info["rank_ratio"]
    scales = np.empty(rank_ratio.shape[0], dtype=np.float64)
    if rank_ratio.size == 0:
        return scales
    scales[rank_ratio <= 0.15] = 0.5
    scales[(rank_ratio > 0.15) & (rank_ratio <= 0.40)] = 0.3
    scales[(rank_ratio > 0.40) & (rank_ratio <= 0.70)] = 0.15
    scales[rank_ratio > 0.70] = 0.05
    return scales


def _lengthscale_by_rank(values):
    rank_info = _rank_importance(values)
    rank_ratio = rank_info["rank_ratio"]
    lengthscales = np.empty(rank_ratio.shape[0], dtype=np.float64)
    if rank_ratio.size == 0:
        return lengthscales
    lengthscales[rank_ratio <= 0.15] = 0.5
    lengthscales[(rank_ratio > 0.15) & (rank_ratio <= 0.40)] = 1.0
    lengthscales[(rank_ratio > 0.40) & (rank_ratio <= 0.70)] = 2.0
    lengthscales[rank_ratio > 0.70] = 5.0
    return lengthscales


def _align_prior_vector(values, dim, fill_value=0.5):
    if values is None:
        return np.full(dim, fill_value, dtype=np.float64)
    arr = _to_numpy_vector(values)
    if arr.size == dim:
        return arr
    if arr.size == dim - 1 and arr.size > 0:
        return np.concatenate([arr, [float(np.mean(arr))]])
    if arr.size == 0:
        return np.full(dim, fill_value, dtype=np.float64)
    if arr.size > dim:
        return arr[:dim]
    repeat = int(np.ceil(dim / max(arr.size, 1)))
    return np.tile(arr, repeat)[:dim]


def _build_prior_saas_importance(dim, prior_proxy_metrics=None, fallback_importance=None):
    if isinstance(prior_proxy_metrics, dict):
        param_score = _align_prior_vector(prior_proxy_metrics.get("param_score"), dim, fill_value=0.5)
        act_reason = _align_prior_vector(prior_proxy_metrics.get("act_reason"), dim, fill_value=0.5)
        param_norm = _robust_normalize(param_score)
        act_norm = _robust_normalize(act_reason)
        combined = 0.5 * param_norm + 0.5 * act_norm
        combined = _smooth_scores(combined)
        return np.clip(combined, 0.05, 1.0)
    if fallback_importance is None:
        return None
    fallback = _align_prior_vector(fallback_importance, dim, fill_value=0.5)
    return np.clip(fallback, 0.05, 1.0)


def saasbo_qnehvi_optimizer(
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
    dtype=torch.double,  # 强制使用double精度
    initial_samples=10,
    noise_level=0.01,
    iteration_callback=None,
    ref_point=-1.1,
    run_id=None,
    checkpoint_dir="./checkpoints",
    custom_initial_solutions=None,
    custom_initial_points=None,
    # SAASBO and importance parameters
    use_saas=True,
    initial_importance=None,
    enable_importance_prior=True,
    enable_importance_update=True,
    enable_importance_guidance=True,
    enable_importance_weighted_acq=False,
    learning_rate=0.1,
    # 随机种子参数，用于结果复现
    seed=42,
    shared_initial_dataset=None,
    scheduler_gpu_count=4,
    max_evaluations=None,
    async_mode=False,
    wait_for_completion_threshold=0.15,
    enable_pending_aware_acq=True,
    enable_gap_aware_postprocess=False,
    gap_reward_weight=0.25,
    gap_pending_penalty_weight=0.15,
    gap_candidate_pool_multiplier=3,
    enable_importance_prior_cutoff=True,
    importance_prior_cutoff_evals=24,
):
    """
    使用SAASBO+qNEHVI算法优化多目标问题的封装函数
    
    参数:
    ----------
    objective_function : callable
        目标函数，输入为形状为(batch_size, dim)的张量，输出为形状为(batch_size, num_objectives)的张量
    dim : int, optional
        决策变量的维度，默认为3
    num_objectives : int, optional
        目标函数的数量，默认为2
    bounds : torch.Tensor, optional
        决策变量的边界，形状为(2, dim)，如果为None则默认[0, 1]边界
    BATCH_SIZE : int, optional
        每轮迭代选择的候选点数量，默认为5
    NUM_RESTARTS : int, optional
        优化获取函数时的重启次数，默认为20
    RAW_SAMPLES : int, optional
        用于初始化优化的原始样本数量，默认为512
    MC_SAMPLES : int, optional
        蒙特卡洛采样数量，默认为128
    N_BATCH : int, optional
        优化的总轮数，默认为40
    verbose : bool, optional
        是否打印优化进度，默认为True
    device : str, optional
        计算设备，默认为"cpu"
    dtype : torch.dtype, optional
        数据类型，默认为torch.double
    initial_samples : int, optional
        初始采样点数量，默认为10
    noise_level : float, optional
        观察噪声标准差，默认为0.01
    iteration_callback : callable, optional
        每轮迭代完成后调用的回调函数，函数签名为callback(iteration, train_x, train_obj_true, hvs)，默认为None
    ref_point : float or torch.Tensor, optional
        超体积计算的参考点，默认为-1.1
    run_id : str, optional
        运行的唯一标识符，如果为None则自动生成时间戳格式的ID
    checkpoint_dir : str, optional
        检查点保存的根目录，默认为"./checkpoints"
    custom_initial_solutions : list, optional
        用户自定义的初始解列表
    use_saas : bool, optional
        是否使用SAAS先验，默认为True
    initial_importance : torch.Tensor, optional
        初始重要性权重，形状为(dim,)，默认为None（均匀分布）
    enable_importance_prior : bool, optional
        是否在模型中集成重要性先验，默认为True
    enable_importance_update : bool, optional
        是否在优化过程中更新重要性估计，默认为True
    enable_importance_guidance : bool, optional
        是否在搜索策略中使用重要性指导，默认为True
    enable_importance_weighted_acq : bool, optional
        是否在获取函数中加入重要性权重，默认为False
        注意：该参数使用基于探索质量的加权逻辑，鼓励在重要维度上探索不足的点，而不是简单地远离已知点
    learning_rate : float, optional
        重要性更新学习率，默认为0.1
    
    返回:
    -------
    train_x : torch.Tensor
        所有评估点的决策变量值
    train_obj_true : torch.Tensor
        所有评估点的真实目标函数值
    hvs : list
        每轮迭代的超体积值
    problem_ref_point : torch.Tensor
        问题的参考点
    run_id : str
        本次运行的唯一标识符
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 设置tkwargs
    tkwargs = {
        "dtype": dtype,
        "device": torch.device(device),
    }
    
    # 生成或使用提供的run_id
    if run_id is None:
        # 使用当前时间生成唯一ID
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_dir = os.path.join(checkpoint_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # 设置边界
    if bounds is None:
        bounds = torch.zeros(2, dim, **tkwargs)
        bounds[1] = 1
    else:
        # 确保bounds在正确的设备和数据类型上
        bounds = bounds.to(**tkwargs)
    
    # 标准化边界
    standard_bounds = torch.zeros(2, dim, **tkwargs)
    standard_bounds[1] = 1
    
    # 设置噪声
    NOISE_SE = torch.full((num_objectives,), noise_level, **tkwargs)
    if max_evaluations is None:
        max_evaluations = int(initial_samples + N_BATCH * BATCH_SIZE)
    
    # 确定参考点（根据botorch的要求，参考点应该是目标空间中的一个点，所有真实目标都应该大于这个点）
    # 这里我们假设目标是最大化的，所以设置参考点为一个较小的值
    if type(ref_point) == torch.Tensor:
        problem_ref_point = ref_point
    else:
        problem_ref_point = torch.full((num_objectives,), ref_point, **tkwargs)
    
    # 初始化重要性
    if initial_importance is None:
        importance = torch.ones(dim, **tkwargs) * 0.5
    else:
        importance = initial_importance.clone().to(**tkwargs)

    objective_signature = inspect.signature(objective_function)
    objective_supports_estimated_tokens = "estimated_tokens" in objective_signature.parameters
    idle_gpu_count_fn = getattr(objective_function, "get_idle_gpu_count", None)
    collect_completed_tasks_fn = getattr(objective_function, "collect_newly_completed_tasks", None)

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

    completed_task_runtime_by_id = {}

    def extract_task_id(info_items):
        if not isinstance(info_items, list):
            return None
        for item in info_items:
            if not isinstance(item, dict):
                continue
            task_id = item.get("_task_id", item.get("task_id", None))
            if isinstance(task_id, str) and len(task_id) > 0:
                return task_id
        return None

    def refresh_completed_task_runtime_cache():
        if not callable(collect_completed_tasks_fn):
            return
        try:
            completed_records = collect_completed_tasks_fn()
        except Exception:
            return
        if not isinstance(completed_records, list):
            return
        for record in completed_records:
            if not isinstance(record, dict):
                continue
            task_id = record.get("task_id", None)
            if isinstance(task_id, str) and len(task_id) > 0:
                completed_task_runtime_by_id[task_id] = record

    def pop_completed_task_runtime(task_id):
        if not isinstance(task_id, str) or len(task_id) == 0:
            return None
        refresh_completed_task_runtime_cache()
        return completed_task_runtime_by_id.pop(task_id, None)
    
    def get_importance_from_saas_model(model):
        """
        从SAAS模型中直接提取维度重要性信息
        
        参数:
            model: ModelListGP或SingleTaskGP对象
        
        返回:
            importance: 维度重要性向量，已归一化到[0,1]区间
        """
        if isinstance(model, ModelListGP):
            # 多目标情况：每个目标有自己的模型
            num_models = len(model.models)
            all_lengthscales = []
            
            for model_idx, model in enumerate(model.models):
                if hasattr(model.covar_module, 'base_kernel'):
                    # ScaleKernel + MaternKernel 结构
                    base_kernel = model.covar_module.base_kernel
                    if hasattr(base_kernel, 'lengthscale'):
                        lengthscale = base_kernel.lengthscale.detach().cpu()
                        # 移除batch维度 (1, dim) -> (dim,)
                        if lengthscale.dim() > 1:
                            lengthscale = lengthscale.squeeze(0)
                        all_lengthscales.append(lengthscale)
            
            if all_lengthscales:
                # 将所有模型的长度尺度堆叠起来
                lengthscales_stack = torch.stack(all_lengthscales, dim=0)  # (num_models, dim)
                
                # 方法1: 取几何平均（对长度尺度这种乘法量更合适）
                # 长度尺度越小，该维度越敏感（变化范围小但对目标影响大）
                mean_log_lengthscale = torch.log(lengthscales_stack).mean(dim=0)
                importance = 1.0 / torch.exp(mean_log_lengthscale)
                
                # 方法2: 取中位数，对异常值更鲁棒
                # median_lengthscale = lengthscales_stack.median(dim=0).values
                # importance = 1.0 / median_lengthscale
                
                # 方法3: 考虑每个目标的不同权重（基于目标函数的尺度）
                # 可以在这里添加目标权重的计算
                
        elif isinstance(model, SingleTaskGP):
            # 单目标情况
            if hasattr(model.covar_module, 'base_kernel'):
                base_kernel = model.covar_module.base_kernel
                if hasattr(base_kernel, 'lengthscale'):
                    lengthscale = base_kernel.lengthscale.detach().cpu()
                    if lengthscale.dim() > 1:
                        lengthscale = lengthscale.squeeze(0)
                    importance = 1.0 / lengthscale
        
        else:
            # 默认情况，返回均匀分布
            importance = torch.ones(dim, **tkwargs)
        
        importance = _calibrate_importance_scores(importance, floor=0.05)
        return torch.as_tensor(importance, **tkwargs)
    
    def update_importance_with_saas_info(model, current_importance=None):
        """
        结合SAAS模型信息和当前重要性进行更新
        
        参数:
            model: 拟合好的SAAS模型
            current_importance: 当前的重要性估计
        
        返回:
            更新后的重要性
        """
        # 从SAAS模型中获取基础重要性
        importance_floor = 0.05
        saas_importance = get_importance_from_saas_model(model)
        saas_importance = saas_importance.to(**tkwargs)
        
        if current_importance is None:
            return saas_importance
        
        calibrated_current = _calibrate_importance_scores(current_importance, floor=importance_floor)
        current_importance = torch.as_tensor(calibrated_current, **tkwargs)
        
        current_prob = torch.clamp(
            (current_importance - importance_floor) / (1.0 - importance_floor),
            min=1e-4,
            max=1.0 - 1e-4,
        )
        saas_prob = torch.clamp(
            (saas_importance - importance_floor) / (1.0 - importance_floor),
            min=1e-4,
            max=1.0 - 1e-4,
        )
        current_logit = torch.logit(current_prob)
        saas_logit = torch.logit(saas_prob)
        
        current_max_idx = current_importance.argmax()
        saas_max_idx = saas_importance.argmax()
        
        if current_max_idx == saas_max_idx:
            lr = 0.5
        else:
            lr = 0.2
        
        updated_logit = (1 - lr) * current_logit + lr * saas_logit
        updated_prob = torch.sigmoid(updated_logit)
        updated_importance = importance_floor + (1.0 - importance_floor) * updated_prob
        return torch.as_tensor(updated_importance, **tkwargs)
    
    def update_importance_from_data(train_x, train_obj):
        """从数据中更新重要性估计（保持函数签名兼容，实际使用update_importance_with_saas_info）"""
        return importance
    
    def importance_based_lhs(n_samples: int) -> torch.Tensor:
        """自适应稀疏采样：针对n_samples < dim但不算极端的情况"""
        # 生成基础均匀分布样本
        samples = draw_sobol_samples(bounds, n_samples, 1).squeeze(1)
        samples = samples.to(**tkwargs)
        perturbation_scales = torch.as_tensor(_sampling_scale_by_rank(importance), **tkwargs)
        
        # 基准值，默认为0.5
        baseline_value = 0.5
        
        # 对每个样本进行调整
        for i in range(n_samples):
            # 创建以baseline_value为基准的调整
            adjusted_sample = torch.ones(dim, **tkwargs) * baseline_value
            
            for dim_idx in range(dim):
                perturbation = perturbation_scales[dim_idx] * torch.randn(1, **tkwargs)
                adjusted_sample[dim_idx] = baseline_value + perturbation
                adjusted_sample[dim_idx] = torch.clamp(
                    adjusted_sample[dim_idx],
                    bounds[0, dim_idx],
                    bounds[1, dim_idx],
                )
            
            samples[i] = adjusted_sample
        
        return samples
    
    def uniform_lhs(n_samples: int) -> torch.Tensor:
        """均匀拉丁超立方采样"""
        samples = draw_sobol_samples(bounds, n_samples, 1).squeeze(1)
        return samples.to(**tkwargs)
    
    def generate_initial_data(n=initial_samples, custom_solutions=None):
        """生成初始训练数据"""
        if isinstance(shared_initial_dataset, dict):
            shared_x = torch.as_tensor(shared_initial_dataset.get("decision_variables", []), **tkwargs)
            shared_obj_true = torch.as_tensor(shared_initial_dataset.get("objectives", []), **tkwargs)
            shared_info = shared_initial_dataset.get("metrics", [{} for _ in range(shared_x.shape[0])])
            if shared_x.ndim == 2 and shared_x.shape == (n, dim) and shared_obj_true.shape == (n, num_objectives):
                train_obj = shared_obj_true + torch.randn_like(shared_obj_true, **tkwargs) * NOISE_SE
                return shared_x, train_obj, shared_obj_true, shared_info
        if custom_initial_points is not None:
            seed_x = torch.as_tensor(custom_initial_points, **tkwargs)
            if seed_x.ndim == 1:
                seed_x = seed_x.unsqueeze(0)
            if seed_x.ndim != 2 or seed_x.shape[1] != dim:
                raise ValueError(f"custom_initial_points形状应为(n, {dim})，当前为{tuple(seed_x.shape)}")
            seed_x = seed_x.clamp(bounds[0].unsqueeze(0), bounds[1].unsqueeze(0))
            num_seed = min(seed_x.shape[0], n)
            remaining = n - num_seed
            if remaining > 0:
                lhs_func = importance_based_lhs if enable_importance_guidance else uniform_lhs
                random_x = lhs_func(remaining)
                train_x = torch.cat([seed_x[:num_seed], random_x], dim=0)
            else:
                train_x = seed_x[:n]
        elif custom_solutions is not None and len(custom_solutions) > 0:
            # 计算需要生成的自定义解数量
            num_custom = len(custom_solutions)
            # 确保不超过总样本数
            num_custom = min(num_custom, n)
            
            # 生成自定义解
            custom_x = []
            for val in custom_solutions:
                # 生成全是val的解，长度为dim
                custom_sol = torch.full((1, dim), val, **tkwargs)
                custom_x.append(custom_sol)
            
            # 合并自定义解
            custom_x = torch.cat(custom_x, dim=0)
            
            # 计算还需要生成的样本数
            remaining = n - num_custom
            
            if remaining > 0:
                # 使用基于重要性的LHS生成剩余样本
                lhs_func = importance_based_lhs if enable_importance_guidance else uniform_lhs
                random_x = lhs_func(remaining)
                # 合并自定义解和随机解
                train_x = torch.cat([custom_x, random_x], dim=0)
            else:
                # 如果自定义解数量大于等于n，只取前n个
                train_x = custom_x[:n]
        else:
            # 没有自定义解，使用基于重要性的LHS或均匀LHS生成
            lhs_func = importance_based_lhs if enable_importance_guidance else uniform_lhs
            train_x = lhs_func(n)
        
        # 调用目标函数，获取目标函数值和评测结果
        result = objective_function(train_x)
        
        # 处理返回结果，支持两种格式：仅返回目标函数值，或返回目标函数值和评测结果
        initial_cleanup_paths = []
        if isinstance(result, tuple) and len(result) == 3:
            train_obj_true, train_info, initial_cleanup_paths = result
        elif isinstance(result, tuple) and len(result) == 2:
            train_obj_true, train_info = result
            initial_cleanup_paths = collect_cleanup_paths(train_info)
        else:
            train_obj_true = result
            train_info = [{} for _ in range(train_x.shape[0])]  # 为空的评测结果
        
        # 确保train_obj_true在正确的设备和数据类型上
        train_obj_true = train_obj_true.to(**tkwargs)
        # 生成相同设备和数据类型上的随机噪声
        train_obj = train_obj_true + torch.randn_like(train_obj_true, **tkwargs) * NOISE_SE
        if iteration_callback is not None and len(initial_cleanup_paths) > 0:
            existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
            iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + initial_cleanup_paths))
        
        return train_x, train_obj, train_obj_true, train_info
    
    def create_saas_model(train_x, train_obj):
        """创建SAAS模型"""
        # 确保train_x与train_obj使用double精度
        train_x = train_x.to(dtype=torch.double, device=train_obj.device)
        train_y = train_obj.to(dtype=torch.double, device=train_obj.device)
        models = []
        current_eval_count = int(train_x.shape[0])
        importance_prior_active = bool(enable_importance_prior)
        if importance_prior_active and bool(enable_importance_prior_cutoff):
            importance_prior_active = current_eval_count < int(importance_prior_cutoff_evals)
        
        for i in range(train_obj.shape[-1]):
            train_y_i = train_y[..., i : i + 1]
            
            # 强制使用double精度创建所有模型组件
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=dim,
                    lengthscale_constraint=Interval(0.005, 10.0)
                )
            )
            
            if use_saas and importance_prior_active:
                # 设置基于重要性排序的长度尺度初始值
                with torch.no_grad():
                    initial_lengthscales = torch.as_tensor(
                        _lengthscale_by_rank(importance),
                        **tkwargs,
                    )
                    initial_lengthscales = initial_lengthscales.view(1, dim).to(dtype=torch.double)
                    covar_module.base_kernel.lengthscale = initial_lengthscales
                
            # 创建模型
            model = SingleTaskGP(
                train_x,
                train_y_i,
                covar_module=covar_module,
                input_transform=BoTorchNormalize(d=dim),
                outcome_transform=Standardize(m=1)
            )
            
            if use_saas:
                # 设置SAAS先验
                model.likelihood.noise_covar.register_prior(
                    "noise_prior",
                    GammaPrior(1.1, 0.05),
                    "raw_noise"
                )
                
                model.covar_module.outputscale_prior = GammaPrior(2.0, 0.15)
            
            # 确保模型参数使用double精度
            model = model.double()
            models.append(model)
        
        model = ModelListGP(*models)
        # 确保所有模型组件使用double精度
        model = model.double()
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model
    
    def initialize_model(train_x, train_obj):
        """初始化模型"""
        return create_saas_model(train_x, train_obj)
    
    def fit_gpytorch_mll(mll):
        """训练模型，添加重试机制"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 尝试使用不同的优化器参数
                optimizer_kwargs = {
                    'options': {
                        'maxiter': 1000,
                        'maxfun': 1000
                    },
                    'method': 'L-BFGS-B'
                }
                botorch_fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)
                return  # 成功拟合，退出函数
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise  # 超过最大重试次数，抛出异常
                # 降低学习率并重试
                print(f"模型拟合失败，重试 {retry_count}/{max_retries}: {str(e)}")
    
    def extract_avg_tokens(metric_item):
        if not isinstance(metric_item, dict):
            return None
        values = []
        stack = [metric_item]
        while stack:
            current = stack.pop()
            if not isinstance(current, dict):
                continue
            for key, value in current.items():
                if isinstance(value, dict):
                    stack.append(value)
                elif key == "mean_tokens_num" and isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
                    values.append(float(value))
        if len(values) == 0:
            return None
        return float(np.mean(values))

    def estimate_candidate_cost(candidates, train_x, train_info):
        if not isinstance(train_info, list) or len(train_info) == 0 or train_x.numel() == 0:
            return torch.ones(candidates.shape[0], device=candidates.device, dtype=candidates.dtype)
        hist_x = []
        hist_cost = []
        for x_item, info_item in zip(train_x, train_info):
            token_est = extract_avg_tokens(info_item)
            if token_est is None:
                continue
            hist_x.append(x_item.detach())
            hist_cost.append(token_est)
        if len(hist_cost) == 0:
            return torch.ones(candidates.shape[0], device=candidates.device, dtype=candidates.dtype)
        hist_x = torch.stack(hist_x).to(device=candidates.device, dtype=candidates.dtype)
        hist_cost_t = torch.as_tensor(hist_cost, device=candidates.device, dtype=candidates.dtype)
        candidate_costs = []
        for cand in candidates:
            nearest_idx = torch.argmin(torch.norm(hist_x - cand.unsqueeze(0), dim=-1))
            candidate_costs.append(hist_cost_t[nearest_idx])
        return torch.stack(candidate_costs)

    def propose_candidates(model, train_x, train_obj, train_obj_true, sampler, q, pending_x=None):
        """优化qLogNEHVI获取函数，并返回新的候选点"""
        _ = train_obj
        # 确保所有数据在同一设备和数据类型上
        ref_point_device = train_x.device
        ref_point_dtype = train_x.dtype
        
        # 确保bounds在正确的设备和数据类型上
        device_bounds = bounds.to(device=ref_point_device, dtype=ref_point_dtype)
        
        # 归一化训练数据
        train_x_normalized = normalize(train_x, device_bounds)
        train_x_normalized = train_x_normalized.to(dtype=ref_point_dtype)
        
        # 创建获取函数
        ref_point = problem_ref_point.to(device=ref_point_device, dtype=ref_point_dtype)
        
        # 创建获取函数
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_x_normalized,
            prune_baseline=True,
            sampler=sampler,
        )
        
        if enable_pending_aware_acq and pending_x is not None and pending_x.numel() > 0:
            pending_x = pending_x.to(device=ref_point_device, dtype=ref_point_dtype)
            pending_x_normalized = normalize(pending_x, device_bounds)
            acq_func.set_X_pending(pending_x_normalized)

        # 如果启用了重要性加权获取函数，使用保持符号的重要性加权
        if enable_importance_weighted_acq:
            # 确保importance在正确的设备和数据类型上
            device_importance = importance.to(device=ref_point_device, dtype=ref_point_dtype)
            
            class SignPreservingImportanceWeighting(torch.nn.Module):
                """保持符号的重要性加权"""
                
                def __init__(self, base_acq_func, importance_scores, X_baseline):
                    super().__init__()
                    self.base_acq_func = base_acq_func
                    self.importance = importance_scores
                    self.X_baseline = X_baseline
                    
                def forward(self, X):
                    # 1. 计算基础获取函数值
                    base_value = self.base_acq_func(X)
                    
                    # 2. 计算探索质量因子
                    exploration_factor = self.compute_exploration_factor(X)
                    
                    # 3. 关键：保持基础值的符号
                    # 如果基础值为正，乘以>1的因子增强
                    # 如果基础值为负，乘以<1的因子减弱（不改变符号）
                    weighted_value = self.apply_sign_preserving_weight(base_value, exploration_factor)
                    
                    return weighted_value
                
                def compute_exploration_factor(self, X):
                    """计算探索因子（1.0附近）"""
                    if self.X_baseline is None or len(self.X_baseline) == 0:
                        return 1.0
                    
                    # 对于初始条件生成阶段，X的形状可能不同
                    # 直接返回一个标量因子，避免维度不匹配
                    if X.dim() == 2:
                        # 初始条件生成阶段，X的形状是 (num_restarts, dim)
                        # 计算所有初始点的平均探索因子
                        min_weighted_distances = []
                        for i in range(X.shape[0]):
                            candidate = X[i, :]
                            min_distance = float('inf')
                            for baseline_point in self.X_baseline:
                                diff = torch.abs(candidate - baseline_point)
                                weighted_diff = diff * self.importance
                                weighted_distance = weighted_diff.sum().item()
                                min_distance = min(min_distance, weighted_distance)
                            min_weighted_distances.append(min_distance)
                        
                        # 计算平均距离
                        avg_distance = torch.tensor(min_weighted_distances, device=X.device).mean().item()
                        # 映射到因子，确保张量在X的设备上
                        return 0.95 + 0.1 * torch.sigmoid(torch.tensor(avg_distance * 3.0, device=X.device))
                    
                    if X.dim() == 3:
                        # 正常优化阶段，X的形状是 (1, q, dim)
                        q, d = X.shape[1], X.shape[2]
                        factors = []
                        
                        for i in range(q):
                            candidate = X[0, i, :]
                            
                            # 计算在重要维度上的探索程度
                            min_weighted_distance = float('inf')
                            for baseline_point in self.X_baseline:
                                diff = torch.abs(candidate - baseline_point)
                                weighted_diff = diff * self.importance
                                weighted_distance = weighted_diff.sum()
                                min_weighted_distance = min(min_weighted_distance, weighted_distance.item())
                            
                            # 将距离映射到[0.9, 1.1]的因子
                        # 距离越大，因子越接近1.1
                        # 确保factor张量与X在同一设备上
                        factor = 0.95 + 0.1 * torch.sigmoid(torch.tensor(min_weighted_distance * 3.0, device=X.device))
                        factors.append(factor)
                        
                        return torch.stack(factors)
                    
                    # 默认返回标量因子
                    return 1.0
                
                def apply_sign_preserving_weight(self, base_values, factors):
                    """应用保持符号的权重"""
                    # 处理因子形状不匹配的情况
                    # 如果factors是标量，直接应用
                    if isinstance(factors, float):
                        factors = torch.tensor(factors, device=base_values.device, dtype=base_values.dtype)
                    
                    # 如果因子维度与base_values不匹配，直接返回base_values
                    # 这种情况在初始条件生成阶段经常发生
                    if factors.dim() > 0 and factors.shape != base_values.shape:
                        return base_values
                    
                    # 分离符号和幅度
                    signs = torch.sign(base_values)
                    magnitudes = torch.abs(base_values)
                    
                    # 对正值的点应用增强，对负值的点应用减弱
                    # 对于正值：乘以因子（>=1）
                    # 对于负值：乘以(2-因子)（<=1）
                    weighted_magnitudes = torch.where(
                        signs > 0,
                        magnitudes * factors,  # 正值增强
                        magnitudes * (2.0 - factors)  # 负值减弱
                    )
                    
                    # 重新组合
                    weighted_values = signs * weighted_magnitudes
                    
                    return weighted_values
            
            # 使用保持符号的重要性加权
            acq_func = SignPreservingImportanceWeighting(acq_func, device_importance, train_x_normalized)
        
        # 确保standard_bounds在正确的设备和数据类型上
        device_standard_bounds = standard_bounds.to(device=ref_point_device, dtype=ref_point_dtype)
        
        # 优化获取函数
        candidate_pool_size = int(max(1, q))
        if enable_gap_aware_postprocess:
            candidate_pool_size = max(
                candidate_pool_size,
                int(max(2, q) * max(1, int(gap_candidate_pool_multiplier))),
            )
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=device_standard_bounds,
            q=candidate_pool_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 100},
            sequential=True,
        )

        with torch.no_grad():
            candidate_values = []
            for cand_idx in range(candidates.shape[0]):
                xi = candidates[cand_idx : cand_idx + 1].unsqueeze(0)
                candidate_values.append(acq_func(xi).detach().reshape(-1)[0])
            candidate_values = torch.stack(candidate_values).to(device=ref_point_device, dtype=ref_point_dtype)
            candidate_obj_mean = model.posterior(candidates).mean.detach().reshape(candidates.shape[0], -1)
            candidate_x = unnormalize(candidates.detach(), bounds=device_bounds).to(
                device=ref_point_device, dtype=ref_point_dtype
            )
            if enable_gap_aware_postprocess:
                pending_x_normalized = None
                if pending_x is not None and pending_x.numel() > 0:
                    pending_x_normalized = normalize(pending_x, device_bounds).to(
                        device=ref_point_device,
                        dtype=ref_point_dtype,
                    )
                selected_idx, selected_score_details = _select_gap_aware_indices(
                    candidate_repr=candidates.detach().to(device=ref_point_device, dtype=ref_point_dtype),
                    candidate_values=candidate_values,
                    candidate_obj_mean=candidate_obj_mean.to(device=ref_point_device, dtype=ref_point_dtype),
                    observed_obj=train_obj_true.detach().to(device=ref_point_device, dtype=ref_point_dtype),
                    q=q,
                    pending_repr=pending_x_normalized,
                    gap_reward_weight=gap_reward_weight,
                    pending_penalty_weight=gap_pending_penalty_weight,
                )
            else:
                sorted_idx = torch.argsort(candidate_values, descending=True)
                selected_idx = sorted_idx[: min(int(max(1, q)), sorted_idx.numel())]
                selected_score_details = []
            new_x = candidate_x[selected_idx]
            candidate_cost = estimate_candidate_cost(new_x, train_x, train_info)
            if enable_gap_aware_postprocess and len(selected_score_details) > 0:
                round_record = {
                    "proposal_round": int(proposal_score_state["round"]),
                    "iteration": int(train_x.shape[0]),
                    "requested_q": int(q),
                    "selected_count": int(selected_idx.numel()),
                    "candidate_pool_size": int(candidates.shape[0]),
                    "pending_count": int(0 if pending_x is None else pending_x.shape[0]),
                    "scores": [],
                }
                for detail in selected_score_details:
                    candidate_index = int(detail["candidate_index"])
                    round_record["scores"].append(
                        {
                            **detail,
                            "candidate_x": candidate_x[candidate_index].detach().cpu().tolist(),
                            "predicted_objectives": candidate_obj_mean[candidate_index].detach().cpu().tolist(),
                            "acq_value": float(candidate_values[candidate_index].item()),
                        }
                    )
                candidate_score_history.append(round_record)
                proposal_score_state["round"] += 1

        del train_x_normalized, acq_func, candidates
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return new_x, candidate_cost

    def evaluate_candidates(x_batch, estimated_tokens=None):
        result_kwargs = {}
        if objective_supports_estimated_tokens:
            result_kwargs["estimated_tokens"] = estimated_tokens
        result = objective_function(x_batch, **result_kwargs)
        cleanup_paths = []
        if isinstance(result, tuple) and len(result) == 3:
            new_obj_true, new_info, cleanup_paths = result
        elif isinstance(result, tuple) and len(result) == 2:
            new_obj_true, new_info = result
            cleanup_paths = collect_cleanup_paths(new_info)
        else:
            new_obj_true = result
            new_info = [{} for _ in range(x_batch.shape[0])]
        new_obj_true = new_obj_true.to(device=x_batch.device, dtype=x_batch.dtype)
        device_noise_se = NOISE_SE.to(device=x_batch.device, dtype=x_batch.dtype)
        new_obj = new_obj_true + torch.randn_like(new_obj_true, device=x_batch.device, dtype=x_batch.dtype) * device_noise_se
        return x_batch, new_obj, new_obj_true, new_info, cleanup_paths
    
    def compute_pareto_hypervolume(objectives, ref_point):
        """计算帕累托前沿的超体积"""
        # 确保ref_point和objectives在同一个设备上
        ref_point = ref_point.to(device=objectives.device, dtype=objectives.dtype)
        bd = FastNondominatedPartitioning(ref_point=ref_point, Y=objectives)
        return bd.compute_hypervolume().item()

    scheduler_history = []
    hv_curve = []
    candidate_score_history = []
    proposal_score_state = {"round": 0}

    def save_checkpoint(iteration, train_x, train_obj, train_obj_true, train_info, hvs, run_dir):
        """
        保存优化过程的检查点
        """
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
            extra_json={
                "importance": importance.detach().cpu().tolist(),
                "candidate_score_history": candidate_score_history,
            },
            extra_state={
                "importance": importance.detach().cpu(),
                "candidate_score_history": candidate_score_history,
            },
        )

    def load_checkpoint(run_dir, tkwargs):
        """
        加载最新的检查点
        """
        checkpoint = load_standard_checkpoint(run_dir, tkwargs)
        if checkpoint is None:
            return None

        train_x = checkpoint['train_x'].to(**tkwargs)
        train_obj = checkpoint['train_obj'].to(**tkwargs)
        train_obj_true = checkpoint['train_obj_true'].to(**tkwargs)
        iteration = checkpoint['iteration']
        hvs = checkpoint['hvs']
        importance = checkpoint['importance'].to(**tkwargs)
        train_info = checkpoint.get('train_info', [{} for _ in range(train_x.shape[0])])
        loaded_hv_curve = checkpoint.get("hv_curve", [])
        hv_curve.clear()
        if isinstance(loaded_hv_curve, list):
            hv_curve.extend(loaded_hv_curve)
        loaded_scheduler_history = checkpoint.get("scheduler_history", [])
        scheduler_history.clear()
        if isinstance(loaded_scheduler_history, list):
            scheduler_history.extend(loaded_scheduler_history)
        loaded_candidate_score_history = checkpoint.get("candidate_score_history", [])
        candidate_score_history.clear()
        if isinstance(loaded_candidate_score_history, list):
            candidate_score_history.extend(loaded_candidate_score_history)
        proposal_score_state["round"] = len(candidate_score_history)

        return train_x, train_obj, train_obj_true, train_info, iteration, hvs, importance
    
    def get_importance_report():
        """
        获取重要性报告
        
        返回:
        -------
        dict: 包含重要性报告的字典
        """
        import pandas as pd
        
        categories, rank_info = _importance_category_by_rank(importance)
        data = []
        for i, imp in enumerate(importance):
            data.append({
                'variable': i,
                'importance': imp.item(),
                'category': str(categories[i]),
                'rank': int(rank_info['rank_position'][i]) + 1,
            })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values('importance', ascending=False)
        
        # 转换为字典格式返回
        return {
            'importance_values': importance.tolist(),
            'importance_report': df.to_dict(orient='records'),
            'average_importance': importance.mean().item(),
            'critical_variables': len(df[df['category'] == '关键']),
            'important_variables': len(df[df['category'] == '重要']),
            'medium_variables': len(df[df['category'] == '中等']),
            'minor_variables': len(df[df['category'] == '次要']),
            'negligible_variables': len(df[df['category'] == '可忽略'])
        }
    
    def get_performance_summary(train_x, train_obj_true, hvs):
        """
        获取性能摘要
        
        参数:
        ----------
        train_x : torch.Tensor
            所有评估点的决策变量值
        train_obj_true : torch.Tensor
            真实目标函数值
        hvs : list
            超体积历史
        
        返回:
        -------
        dict: 包含性能摘要的字典
        """
        summary = {
            'total_evaluations': len(train_x),
            'best_hypervolume': hvs[-1] if hvs else 0.0,
            'average_importance': importance.mean().item(),
            'total_iterations': len(hvs) - 1 if hvs else 0,
            'dimensions': dim,
            'num_objectives': num_objectives
        }
        
        # 计算目标函数的统计信息
        if train_obj_true.numel() > 0:
            summary['objective_min'] = train_obj_true.min().item()
            summary['objective_max'] = train_obj_true.max().item()
            summary['objective_mean'] = train_obj_true.mean().item()
        
        return summary
    
    # 开始优化过程
    if verbose:
        print(f"使用SAASBO+qNEHVI算法优化多目标问题 (维度: {dim}, 目标数: {num_objectives})")
        print(f"设备: {tkwargs['device']}, 数据类型: {tkwargs['dtype']}")
        print(f"运行ID: {run_id}, 检查点目录: {run_dir}")
        print(f"SAAS配置: 使用SAAS先验={use_saas}")
        print(f"重要性配置: 先验={enable_importance_prior}, 更新={enable_importance_update}, 指导={enable_importance_guidance}, 加权获取={enable_importance_weighted_acq}")
        print(f"重要性先验截断: 启用={enable_importance_prior_cutoff}, 阈值评估数={int(importance_prior_cutoff_evals)}")
        print(f"Pending-aware获取: {bool(enable_pending_aware_acq)}")
        print(f"最大评估次数: {int(max_evaluations)}, 异步模式: {bool(async_mode)}")
    
    # 尝试加载检查点
    checkpoint = load_checkpoint(run_dir, tkwargs)
    if checkpoint is not None:
        train_x, train_obj, train_obj_true, train_info, start_iteration, hvs, importance = checkpoint
        if len(hv_curve) == 0 and len(hvs) > 0:
            hv_curve.extend(
                build_hv_curve(
                    hvs=hvs,
                    initial_samples=initial_samples,
                    batch_size=BATCH_SIZE,
                    total_evaluations=int(train_x.shape[0]),
                )
            )
        if verbose:
            print(f"成功加载检查点，从迭代 {start_iteration} 继续")
            print(f"当前超体积: {hvs[-1]:.4f}")
            if enable_importance_prior and enable_importance_prior_cutoff:
                print(
                    f"当前评估数: {int(train_x.shape[0])}，重要性先验注入={'开启' if int(train_x.shape[0]) < int(importance_prior_cutoff_evals) else '关闭'}"
                )
    else:
        # 生成初始数据
        train_x, train_obj, train_obj_true, train_info = generate_initial_data(custom_solutions=custom_initial_solutions)
        
        # 记录超体积
        hvs = []
        initial_hv = compute_pareto_hypervolume(train_obj_true, problem_ref_point)
        hvs.append(initial_hv)
        hv_curve.append({"evaluations": int(train_x.shape[0]), "hypervolume": float(initial_hv)})
        init_record = build_sync_schedule_record(
            iteration=0,
            metric_items=train_info,
            scheduler_gpu_count=scheduler_gpu_count,
            candidate_index_offset=0,
        )
        init_record["mode"] = "init"
        scheduler_history.append(init_record)
        if verbose:
            print(f"初始超体积: {initial_hv:.4f}")
        
        start_iteration = 0
        # 保存初始状态
        save_checkpoint(0, train_x, train_obj, train_obj_true, train_info, hvs, run_dir)
    
    try:
        if not async_mode:
            max_sync_steps = int(np.ceil(max(0, int(max_evaluations) - train_x.shape[0]) / max(1, int(BATCH_SIZE))))
            for step_idx in range(1, max_sync_steps + 1):
                iteration = start_iteration + step_idx
                mll, model = initialize_model(train_x, train_obj)
                fit_gpytorch_mll(mll)

                if enable_importance_update and train_x.shape[0] > 2:
                    importance = update_importance_with_saas_info(model, importance)
                    if verbose:
                        print(f"迭代 {iteration}: 从SAAS模型更新重要性")
                        print(f"  重要性值: {importance.detach().cpu().numpy()[:5]}...")
                        print(f"  最重要维度: {importance.argmax().item()}")

                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
                remain_cap = int(max_evaluations) - train_x.shape[0]
                q_new = min(max(1, int(BATCH_SIZE)), max(remain_cap, 0))
                if q_new <= 0:
                    break
                new_x, candidate_cost = propose_candidates(model, train_x, train_obj, train_obj_true, sampler, q=q_new)
                new_x, new_obj, new_obj_true, new_info, cleanup_paths = evaluate_candidates(
                    new_x,
                    estimated_tokens=candidate_cost,
                )

                train_x = torch.cat([train_x, new_x.to(train_x.device)])
                train_obj = torch.cat([train_obj, new_obj.to(train_obj.device)])
                train_obj_true = torch.cat([train_obj_true, new_obj_true.to(train_obj_true.device)])
                train_info.extend(new_info)

                with torch.no_grad():
                    new_hv = compute_pareto_hypervolume(train_obj_true, problem_ref_point)
                hvs.append(new_hv)
                hv_curve.append({"evaluations": int(train_x.shape[0]), "hypervolume": float(new_hv)})
                scheduler_history.append(
                    build_sync_schedule_record(
                        iteration=iteration,
                        metric_items=new_info,
                        scheduler_gpu_count=scheduler_gpu_count,
                        candidate_index_offset=int(train_x.shape[0] - len(new_info)),
                    )
                )

                if verbose:
                    print(f"迭代 {iteration:>2}: 超体积 = {new_hv:.4f}")
                    if tkwargs['device'].type == 'cuda':
                        print(f"    GPU内存: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

                if iteration_callback is not None:
                    try:
                        if len(cleanup_paths) > 0:
                            existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
                            iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + cleanup_paths))
                        if hasattr(iteration_callback, "async_mode"):
                            iteration_callback.async_mode = False
                        iteration_callback(iteration, train_x, train_obj_true, hvs)
                    except Exception as e:
                        print(f"警告: 迭代回调函数执行失败: {e}")

                if enable_importance_prior and (iteration + 1) % 5 == 0:
                    if verbose:
                        print(f"\n=== 迭代 {iteration+1} - 重要性报告 ===")
                    importance_report = get_importance_report()
                    if verbose:
                        print(f"平均重要性: {importance_report['average_importance']:.4f}")
                        print(f"关键变量数: {importance_report['critical_variables']}")
                        print(f"重要变量数: {importance_report['important_variables']}")
                        print(f"中等变量数: {importance_report['medium_variables']}")
                        print(f"次要变量数: {importance_report['minor_variables']}")
                        print(f"可忽略变量数: {importance_report['negligible_variables']}")
                        print("前5个重要变量:")
                        for i, item in enumerate(importance_report['importance_report'][:5]):
                            print(f"  变量 {item['variable']}: 重要性 = {item['importance']:.4f} ({item['category']})")

                save_checkpoint(iteration, train_x, train_obj, train_obj_true, train_info, hvs, run_dir)
                del mll, model, sampler
                gc.collect()
                if tkwargs['device'].type == 'cuda':
                    torch.cuda.empty_cache()
        else:
            total_target = int(max_evaluations)
            completed_counter = int(train_x.shape[0])
            completion_walltimes = []
            inflight = []
            model_dirty = True
            mll = None
            model = None
            sampler = None
            submit_counter = 0
            candidate_generation_lock = threading.Lock()
            timeline_origin_ts = time.time()

            def append_observation(new_x, new_obj, new_obj_true, new_info):
                nonlocal train_x, train_obj, train_obj_true, train_info
                train_x = torch.cat([train_x, new_x.to(train_x.device)], dim=0)
                train_obj = torch.cat([train_obj, new_obj.to(train_obj.device)], dim=0)
                train_obj_true = torch.cat([train_obj_true, new_obj_true.to(train_obj_true.device)], dim=0)
                train_info.extend(new_info)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(scheduler_gpu_count))) as executor:
                while completed_counter < total_target or len(inflight) > 0:
                    done_items = [item for item in inflight if item["future"].done()]
                    for item in done_items:
                        inflight.remove(item)
                        new_x, new_obj, new_obj_true, new_info, cleanup_paths = item["future"].result()
                        append_observation(new_x, new_obj, new_obj_true, new_info)
                        completed_counter += int(new_x.shape[0])
                        completion_walltimes.append(time.time() - float(item["start_ts"]))
                        model_dirty = True
                        with torch.no_grad():
                            hv_val = compute_pareto_hypervolume(train_obj_true, problem_ref_point)
                        hvs.append(hv_val)
                        hv_curve.append({"evaluations": int(completed_counter), "hypervolume": float(hv_val)})
                        iteration_id = int(completed_counter)
                        task_id = extract_task_id(new_info)
                        completed_runtime = pop_completed_task_runtime(task_id)
                        fallback_runtime = max(float(time.time() - float(item["start_ts"])), 1e-6)
                        item_runtime = float(fallback_runtime)
                        task_gpu_slot = int(item.get("gpu_slot", 0)) % max(1, int(scheduler_gpu_count))
                        task_start_rel = max(float(item["start_ts"]) - float(timeline_origin_ts), 0.0)
                        runtime_source = "scheduler_fallback"
                        runtime_trusted = False
                        if isinstance(completed_runtime, dict):
                            gpu_ids = completed_runtime.get("gpu_ids", [])
                            if isinstance(gpu_ids, list) and len(gpu_ids) > 0:
                                try:
                                    task_gpu_slot = int(gpu_ids[0])
                                except Exception:
                                    pass
                            start_time = completed_runtime.get("start_time", None)
                            end_time = completed_runtime.get("end_time", None)
                            runtime_sec = completed_runtime.get("runtime_sec", None)
                            if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)) and float(end_time) > float(start_time):
                                item_runtime = max(float(end_time) - float(start_time), 1e-6)
                                task_start_rel = max(float(start_time) - float(timeline_origin_ts), 0.0)
                                runtime_source = "real_runtime"
                                runtime_trusted = True
                            elif isinstance(runtime_sec, (int, float)) and float(runtime_sec) > 0.0:
                                item_runtime = max(float(runtime_sec), 1e-6)
                                runtime_source = "real_gpu_mapping"
                                runtime_trusted = True
                            else:
                                runtime_source = "real_gpu_mapping"
                        if not runtime_trusted and verbose:
                            if isinstance(task_id, str) and len(task_id) > 0:
                                print(
                                    f"警告: 候选 {int(completed_counter - 1)} 未能从真实任务记录恢复 runtime，"
                                    f"将仅保留 fallback 时长 {fallback_runtime:.2f}s 供调试使用。"
                                )
                            else:
                                print(
                                    f"警告: 候选 {int(completed_counter - 1)} 缺少 task_id，"
                                    f"无法回连真实 runtime；当前 fallback 时长 {fallback_runtime:.2f}s 不应用于单解时间分析。"
                                )
                        gpu_axis_count = max(max(1, int(scheduler_gpu_count)), int(task_gpu_slot) + 1)
                        gpu_loads = [0.0 for _ in range(gpu_axis_count)]
                        gpu_loads[task_gpu_slot] = float(item_runtime)
                        scheduler_history.append(
                            {
                                "iteration": int(iteration_id),
                                "mode": "async",
                                "gpu_count": int(gpu_axis_count),
                                "estimated_makespan": float(item_runtime),
                                "gpu_loads": gpu_loads,
                                "tasks": [
                                    {
                                        "type": "full",
                                        "gpu": int(task_gpu_slot),
                                        "start": 0.0,
                                        "duration": float(item_runtime),
                                        "absolute_start": float(task_start_rel),
                                        "candidate_index": int(completed_counter - 1),
                                        "task_id": task_id,
                                        "runtime_source": runtime_source,
                                        "runtime_trusted": bool(runtime_trusted),
                                        "fallback_runtime_sec": float(fallback_runtime),
                                    }
                                ],
                                "counts": {"full": int(new_x.shape[0])},
                                "wall_time_sec": float(item_runtime),
                                "candidate_tokens_est": float(item["est_runtime_sec"]),
                                "runtime_trusted": bool(runtime_trusted),
                                "fallback_runtime_sec": float(fallback_runtime),
                            }
                        )
                        if verbose:
                            print(f"异步完成 {completed_counter}/{total_target}: 超体积 = {hv_val:.4f}")
                        if iteration_callback is not None:
                            try:
                                if len(cleanup_paths) > 0:
                                    existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
                                    iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + cleanup_paths))
                                if hasattr(iteration_callback, "async_mode"):
                                    iteration_callback.async_mode = True
                                iteration_callback(iteration_id, train_x, train_obj_true, hvs)
                            except Exception as e:
                                print(f"警告: 迭代回调函数执行失败: {e}")
                        save_checkpoint(iteration_id, train_x, train_obj, train_obj_true, train_info, hvs, run_dir)
                        gc.collect()
                        if tkwargs['device'].type == 'cuda':
                            torch.cuda.empty_cache()

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
                                mll, model = initialize_model(train_x, train_obj)
                                fit_gpytorch_mll(mll)
                                if enable_importance_update and train_x.shape[0] > 2:
                                    importance = update_importance_with_saas_info(model, importance)
                                    if verbose:
                                        print(f"异步更新重要性: {importance.detach().cpu().numpy()[:5]}...")
                                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
                                model_dirty = False
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
                                    pending_x = None
                                    if len(inflight) > 0:
                                        pending_batches = [
                                            item["x"].detach().clone().to(**tkwargs)
                                            for item in inflight
                                            if isinstance(item.get("x", None), torch.Tensor) and item["x"].numel() > 0
                                        ]
                                        if len(pending_batches) > 0:
                                            pending_x = torch.cat(pending_batches, dim=0)
                                    candidate_x, candidate_cost = propose_candidates(
                                        model=model,
                                        train_x=train_x,
                                        train_obj=train_obj,
                                        train_obj_true=train_obj_true,
                                        sampler=sampler,
                                        q=q_new,
                                        pending_x=pending_x,
                                    )
                                    for cand_idx in range(int(q_new)):
                                        cand_x = candidate_x[cand_idx : cand_idx + 1]
                                        est_runtime = float(candidate_cost[cand_idx].item()) if candidate_cost.numel() > cand_idx else 1.0
                                        fut = executor.submit(
                                            evaluate_candidates,
                                            cand_x,
                                            est_runtime,
                                        )
                                        submit_counter += 1
                                        gpu_slot = int((submit_counter - 1) % max(1, int(scheduler_gpu_count)))
                                        inflight.append(
                                            {
                                                "submit_id": submit_counter,
                                                "future": fut,
                                                "x": cand_x.detach().clone(),
                                                "start_ts": time.time(),
                                                "est_runtime_sec": max(est_runtime, 1e-6),
                                                "gpu_slot": gpu_slot,
                                            }
                                        )
                    if len(done_items) == 0 and completed_counter < total_target and (len(inflight) > 0 or slots <= 0):
                        time.sleep(0.2)
    except Exception as e:
        print(f"错误: 优化过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        save_runtime_reports(run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
        # 即使出错也返回已有的结果
        return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
    
    # 打印最终性能摘要
    if verbose:
        print(f"\n=== 优化完成 - 性能摘要 ===")
        performance_summary = get_performance_summary(train_x, train_obj_true, hvs)
        print(f"总评估次数: {performance_summary['total_evaluations']}")
        print(f"总迭代次数: {performance_summary['total_iterations']}")
        print(f"最佳超体积: {performance_summary['best_hypervolume']:.6f}")
        print(f"平均重要性: {performance_summary['average_importance']:.4f}")
        
        # 打印重要性分布
        importance_report = get_importance_report()
        print(f"\n重要性分布:")
        print(f"  关键变量: {importance_report['critical_variables']}")
        print(f"  重要变量: {importance_report['important_variables']}")
        print(f"  中等变量: {importance_report['medium_variables']}")
        print(f"  次要变量: {importance_report['minor_variables']}")
        print(f"  可忽略变量: {importance_report['negligible_variables']}")

    save_runtime_reports(run_dir, scheduler_history=scheduler_history, hv_curve=hv_curve)
    return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id


def prior_saas_bo_optimizer(
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
    use_saas=True,
    initial_importance=None,
    enable_importance_prior=True,
    enable_importance_update=False,
    enable_importance_guidance=False,
    enable_importance_weighted_acq=False,
    learning_rate=0.1,
    seed=42,
    shared_initial_dataset=None,
    scheduler_gpu_count=4,
    max_evaluations=None,
    async_mode=False,
    wait_for_completion_threshold=0.15,
    enable_pending_aware_acq=True,
    enable_gap_aware_postprocess=False,
    gap_reward_weight=0.25,
    gap_pending_penalty_weight=0.15,
    gap_candidate_pool_multiplier=3,
    prior_proxy_metrics=None,
    enable_importance_prior_cutoff=False,
    importance_prior_cutoff_evals=24,
):
    tkwargs = {"dtype": dtype, "device": torch.device(device)}
    if bounds is None:
        bounds_tensor = torch.zeros(2, dim, **tkwargs)
        bounds_tensor[1] = 1
    else:
        bounds_tensor = bounds.to(**tkwargs)
    prior_importance = _build_prior_saas_importance(
        dim=dim,
        prior_proxy_metrics=prior_proxy_metrics,
        fallback_importance=initial_importance,
    )
    if prior_importance is not None:
        prior_importance = torch.as_tensor(prior_importance, **tkwargs)
    return saasbo_qnehvi_optimizer(
        objective_function=objective_function,
        dim=dim,
        num_objectives=num_objectives,
        bounds=bounds_tensor,
        BATCH_SIZE=BATCH_SIZE,
        NUM_RESTARTS=NUM_RESTARTS,
        RAW_SAMPLES=RAW_SAMPLES,
        MC_SAMPLES=MC_SAMPLES,
        N_BATCH=N_BATCH,
        verbose=verbose,
        device=device,
        dtype=dtype,
        initial_samples=initial_samples,
        noise_level=noise_level,
        iteration_callback=iteration_callback,
        ref_point=ref_point,
        run_id=run_id,
        checkpoint_dir=checkpoint_dir,
        custom_initial_solutions=custom_initial_solutions,
        use_saas=use_saas,
        initial_importance=prior_importance,
        enable_importance_prior=enable_importance_prior,
        enable_importance_update=enable_importance_update,
        enable_importance_guidance=enable_importance_guidance,
        enable_importance_weighted_acq=enable_importance_weighted_acq,
        learning_rate=learning_rate,
        seed=seed,
        shared_initial_dataset=shared_initial_dataset,
        scheduler_gpu_count=scheduler_gpu_count,
        max_evaluations=max_evaluations,
        async_mode=async_mode,
        wait_for_completion_threshold=wait_for_completion_threshold,
        enable_pending_aware_acq=enable_pending_aware_acq,
        enable_gap_aware_postprocess=enable_gap_aware_postprocess,
        gap_reward_weight=gap_reward_weight,
        gap_pending_penalty_weight=gap_pending_penalty_weight,
        gap_candidate_pool_multiplier=gap_candidate_pool_multiplier,
        enable_importance_prior_cutoff=enable_importance_prior_cutoff,
        importance_prior_cutoff_evals=importance_prior_cutoff_evals,
    )

def get_pareto_optimal_points(points):
    """从点集中筛选出帕累托最优解"""
    # 转换为numpy数组
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    # 标记是否为帕累托最优
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        # 检查每个点是否被其他点支配
        for j in range(points.shape[0]):
            if i != j and is_pareto[j]:
                # 如果点j支配点i
                if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                    is_pareto[i] = False
                    break
    
    return points[is_pareto]

def plot_pareto_results(train_obj_true, problem_name="Multi-objective Problem", 
                       true_pareto_front=None, ref_point=None, save_path=None):
    """
    绘制帕累托前沿结果（使用VisualizationTool）
    """
    # 使用可视化工具类进行绘图
    reporter.plot_pareto_results(train_obj_true, problem_name, true_pareto_front, ref_point, save_path)

def plot_hypervolume_history(hvs, problem_name="Multi-objective Problem", save_path=None):
    """
    绘制超体积随迭代的变化（使用VisualizationTool）
    """
    # 使用可视化工具类进行绘图
    reporter.plot_hypervolume_history(hvs, problem_name, save_path)
