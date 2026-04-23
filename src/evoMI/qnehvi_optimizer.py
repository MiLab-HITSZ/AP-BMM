import os
import gc
import torch
import numpy as np
import datetime
from src.evoMI.runtime_artifacts import build_eval_metadata, load_standard_checkpoint, save_standard_checkpoint
from src.evoMI.optimization_reporting import reporter
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.settings import fast_computations
from botorch.fit import fit_gpytorch_mll as botorch_fit_gpytorch_mll


def qnehvi_optimizer(
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
    # 随机种子参数，用于结果复现
    seed=42,
    full_eval_limits=None,
    eval_profile="aime_gpqa",
    eval_repeats=None,
    eval_setting_id=None,
    eval_metadata=None,
    shared_initial_dataset=None,
):
    """
    使用qEHVI算法优化多目标问题的封装函数
    
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
    if eval_metadata is None:
        eval_metadata = build_eval_metadata(
            eval_profile=eval_profile,
            eval_limits=full_eval_limits,
            eval_repeats=eval_repeats,
            eval_setting_id=eval_setting_id,
        )

    def collect_cleanup_paths(info_items):
        cleanup_paths = []
        if not isinstance(info_items, list):
            return cleanup_paths
        for item in info_items:
            if not isinstance(item, dict):
                continue
            item_cleanup_paths = item.pop("_cleanup_model_dirs", [])
            if isinstance(item_cleanup_paths, list):
                cleanup_paths.extend(
                    path for path in item_cleanup_paths if isinstance(path, str) and len(path) > 0
                )
        if len(cleanup_paths) == 0:
            return cleanup_paths
        return list(dict.fromkeys(cleanup_paths))
    
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
    
    # 确定参考点（根据botorch的要求，参考点应该是目标空间中的一个点，所有真实目标都应该大于这个点）
    # 这里我们假设目标是最大化的，所以设置参考点为一个较小的值
    if type(ref_point) == torch.Tensor:
        problem_ref_point = ref_point
    else:
        problem_ref_point = torch.full((num_objectives,), ref_point, **tkwargs)
    
    def generate_initial_data(n=initial_samples, custom_solutions=None):
        """生成初始训练数据"""
        if isinstance(shared_initial_dataset, dict):
            shared_x = torch.as_tensor(shared_initial_dataset.get("decision_variables", []), **tkwargs)
            shared_obj_true = torch.as_tensor(shared_initial_dataset.get("objectives", []), **tkwargs)
            shared_info = shared_initial_dataset.get("metrics", [{} for _ in range(shared_x.shape[0])])
            if shared_x.ndim == 2 and shared_x.shape == (n, dim) and shared_obj_true.shape == (n, num_objectives):
                train_obj = shared_obj_true + torch.randn_like(shared_obj_true, **tkwargs) * NOISE_SE
                return shared_x, train_obj, shared_obj_true, shared_info
        # 处理自定义初始解
        if custom_solutions is not None and len(custom_solutions) > 0:
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
                # 生成剩余的样本
                random_x = draw_sobol_samples(bounds=bounds, n=remaining, q=1).squeeze(1)
                random_x = random_x.to(**tkwargs)
                # 合并自定义解和随机解
                train_x = torch.cat([custom_x, random_x], dim=0)
            else:
                # 如果自定义解数量大于等于n，只取前n个
                train_x = custom_x[:n]
        else:
            # 没有自定义解，按原算法生成
            train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)
            train_x = train_x.to(**tkwargs)
        
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
    
    def initialize_model(train_x, train_obj):
        """初始化模型"""
        # 确保train_x与train_obj具有相同的数据类型和设备
        train_x = train_x.to(dtype=train_obj.dtype, device=train_obj.device)
        # 确保bounds在与train_x相同的设备上
        device_bounds = bounds.to(device=train_x.device)
        train_x_normalized = normalize(train_x, device_bounds)
        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]
            # 确保NOISE_SE在与train_y相同的设备和数据类型上
            device_noise_se = NOISE_SE[i].to(device=train_y.device, dtype=train_y.dtype)
            train_yvar = torch.full_like(train_y, device_noise_se ** 2)
            # 确保train_x_normalized与train_y具有相同的数据类型和设备
            train_x_normalized = train_x_normalized.to(dtype=train_y.dtype, device=train_y.device)
            models.append(
                SingleTaskGP(train_x_normalized, train_y, train_yvar)
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model
    
    def fit_gpytorch_mll(mll):
        """训练模型"""
        botorch_fit_gpytorch_mll(mll)
    
    def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
        """优化qLogEHVI获取函数，并返回新的候选点和观察值"""
        # 确保所有数据在同一设备和数据类型上
        ref_point_device = train_x.device
        ref_point_dtype = train_x.dtype
        
        # 确保bounds在正确的设备和数据类型上
        device_bounds = bounds.to(device=ref_point_device, dtype=ref_point_dtype)
        
        # 将训练数据归一化，并确保数据类型一致
        train_x_normalized = normalize(train_x, device_bounds)
        train_x_normalized = train_x_normalized.to(dtype=ref_point_dtype)
        
        # 准备qLogNoisyExpectedHypervolumeImprovement所需的参数
        device_ref_point = problem_ref_point.to(device=ref_point_device, dtype=ref_point_dtype)
        
        # 使用qLogNoisyExpectedHypervolumeImprovement获取函数
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=device_ref_point,
            X_baseline=train_x_normalized,
            prune_baseline=True,
            sampler=sampler,
        )
        
        # 确保standard_bounds在正确的设备和数据类型上
        device_standard_bounds = standard_bounds.to(device=ref_point_device, dtype=ref_point_dtype)
        
        # 优化获取函数
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=device_standard_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 100},
            sequential=True,
        )
        
        # 观察新值
        with torch.no_grad():
            new_x = unnormalize(candidates.detach(), bounds=device_bounds)
            # 确保new_x在正确的设备和数据类型上
            new_x = new_x.to(device=ref_point_device, dtype=ref_point_dtype)
            
            # 调用目标函数，获取目标函数值和评测结果
            result = objective_function(new_x)
            
            # 处理返回结果，支持两种格式：仅返回目标函数值，或返回目标函数值和评测结果
            cleanup_paths = []
            if isinstance(result, tuple) and len(result) == 3:
                new_obj_true, new_info, cleanup_paths = result
            elif isinstance(result, tuple) and len(result) == 2:
                new_obj_true, new_info = result
                cleanup_paths = collect_cleanup_paths(new_info)
            else:
                new_obj_true = result
                new_info = [{} for _ in range(new_x.shape[0])]  # 为空的评测结果
            
            # 确保new_obj_true在正确的设备和数据类型上
            new_obj_true = new_obj_true.to(device=ref_point_device, dtype=ref_point_dtype)
            # 确保NOISE_SE在正确的设备和数据类型上
            device_noise_se = NOISE_SE.to(device=ref_point_device, dtype=ref_point_dtype)
            # 生成相同设备和数据类型上的随机噪声
            new_obj = new_obj_true + torch.randn_like(new_obj_true, device=ref_point_device, dtype=ref_point_dtype) * device_noise_se
        
        # 显式释放临时变量
        del train_x_normalized, acq_func, candidates
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return new_x, new_obj, new_obj_true, new_info, cleanup_paths
    
    def compute_pareto_hypervolume(objectives, ref_point):
        """计算帕累托前沿的超体积"""
        # 确保ref_point和objectives在同一个设备上
        ref_point = ref_point.to(device=objectives.device, dtype=objectives.dtype)
        bd = FastNondominatedPartitioning(ref_point=ref_point, Y=objectives)
        return bd.compute_hypervolume().item()

    def save_checkpoint(iteration, train_x, train_obj, train_obj_true, train_info, hvs, run_dir, tkwargs):
        """
        保存优化过程的检查点
        
        参数:
        ----------
        iteration : int
            当前迭代次数
        train_x : torch.Tensor
            所有评估点的决策变量值
        train_obj : torch.Tensor
            带噪声的目标函数值
        train_obj_true : torch.Tensor
            真实目标函数值
        train_info : list
            每个评估点的评测结果信息
        hvs : list
            超体积历史
        run_dir : str
            保存检查点的目录
        tkwargs : dict
            设备和数据类型参数
        """
        save_standard_checkpoint(
            run_dir=run_dir,
            iteration=iteration,
            train_x=train_x,
            train_obj=train_obj,
            train_obj_true=train_obj_true,
            train_info=train_info,
            hvs=hvs,
            eval_metadata=eval_metadata,
        )

    def load_checkpoint(run_dir, tkwargs):
        """
        加载最新的检查点
        
        参数:
        ----------
        run_dir : str
            检查点所在的目录
        tkwargs : dict
            设备和数据类型参数
        
        返回:
        -------
        tuple or None
            (train_x, train_obj, train_obj_true, train_info, iteration, hvs) 或 None（如果没有找到检查点）
        """
        checkpoint = load_standard_checkpoint(run_dir, tkwargs)
        if checkpoint is None:
            return None

        train_x = checkpoint['train_x']
        train_obj = checkpoint['train_obj']
        train_obj_true = checkpoint['train_obj_true']
        iteration = checkpoint['iteration']
        hvs = checkpoint['hvs']
        train_info = checkpoint.get('train_info', [{} for _ in range(train_x.shape[0])])

        return train_x, train_obj, train_obj_true, train_info, iteration, hvs
    
    # 开始优化过程
    if verbose:
        print(f"使用qEHVI算法优化多目标问题 (维度: {dim}, 目标数: {num_objectives})")
        print(f"设备: {tkwargs['device']}, 数据类型: {tkwargs['dtype']}")
        print(f"运行ID: {run_id}, 检查点目录: {run_dir}")
    
    # 尝试加载检查点
    checkpoint = load_checkpoint(run_dir, tkwargs)
    if checkpoint is not None:
        train_x, train_obj, train_obj_true, train_info, start_iteration, hvs = checkpoint
        if verbose:
            print(f"成功加载检查点，从迭代 {start_iteration} 继续")
            print(f"当前超体积: {hvs[-1]:.4f}")
    else:
        # 生成初始数据
        train_x, train_obj, train_obj_true, train_info = generate_initial_data(custom_solutions=custom_initial_solutions)
        initial_cleanup_paths = collect_cleanup_paths(train_info)
        if iteration_callback is not None and len(initial_cleanup_paths) > 0:
            iteration_callback.cleanup_paths = initial_cleanup_paths
        
        # 记录超体积
        hvs = []
        initial_hv = compute_pareto_hypervolume(train_obj_true, problem_ref_point)
        hvs.append(initial_hv)
        if verbose:
            print(f"初始超体积: {initial_hv:.4f}")
        
        start_iteration = 0
        # 保存初始状态
        save_checkpoint(0, train_x, train_obj, train_obj_true, train_info, hvs, run_dir, tkwargs)
    
    try:
        # 运行剩余轮数的贝叶斯优化
        for iteration in range(start_iteration + 1, N_BATCH + 1):
            # 初始化并训练模型
            mll, model = initialize_model(train_x, train_obj)
            fit_gpytorch_mll(mll)
            
            # 定义QMC采样器
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            
            # 优化获取函数并获取新的观察值
            new_x, new_obj, new_obj_true, new_info, cleanup_paths = optimize_qnehvi_and_get_observation(
                model, train_x, train_obj, sampler
            )
            
            # 更新训练点，确保所有张量在同一设备上
            train_x = torch.cat([train_x, new_x.to(train_x.device)])
            train_obj = torch.cat([train_obj, new_obj.to(train_obj.device)])
            train_obj_true = torch.cat([train_obj_true, new_obj_true.to(train_obj_true.device)])
            train_info.extend(new_info)  # 更新评测信息列表
            
            # 计算新的超体积
            with torch.no_grad():
                new_hv = compute_pareto_hypervolume(train_obj_true, problem_ref_point)
            hvs.append(new_hv)
            
            # 打印进度
            if verbose:
                print(f"迭代 {iteration:>2}: 超体积 = {new_hv:.4f}")
                # 打印当前GPU内存使用情况
                if tkwargs['device'].type == 'cuda':
                    print(f"    GPU内存: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
            
            # 执行迭代回调函数（如果提供）
            if iteration_callback is not None:
                try:
                    if len(cleanup_paths) > 0:
                        existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
                        iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + cleanup_paths))
                    iteration_callback(iteration, train_x, train_obj_true, hvs)
                except Exception as e:
                    print(f"警告: 迭代回调函数执行失败: {e}")
            
            # 保存当前迭代状态
            save_checkpoint(iteration, train_x, train_obj, train_obj_true, train_info, hvs, run_dir, tkwargs)
    except Exception as e:
        print(f"错误: 优化过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        # 即使出错也返回已有的结果
        return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
    
    return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id


def get_pareto_optimal_points(points):
    """从点集中筛选出帕累托最优解"""
    # 转换为numpy数组
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
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
    
    参数:
    ----------
    train_obj_true : torch.Tensor
        所有评估点的真实目标函数值
    problem_name : str, optional
        问题名称，默认为"Multi-objective Problem"
    true_pareto_front : tuple of np.ndarray, optional
        真实帕累托前沿的目标值，格式为(f1, f2, ..., fn)
    ref_point : torch.Tensor, optional
        参考点
    save_path : str, optional
        保存图像的路径
    """
    # 使用可视化工具类进行绘图
    reporter.plot_pareto_results(train_obj_true, problem_name, true_pareto_front, ref_point, save_path)


def plot_hypervolume_history(hvs, problem_name="Multi-objective Problem", save_path=None):
    """
    绘制超体积随迭代的变化（使用VisualizationTool）
    
    参数:
    ----------
    hvs : list
        每轮迭代的超体积值
    problem_name : str, optional
        问题名称
    save_path : str, optional
        保存图像的路径
    """
    # 使用可视化工具类进行绘图
    reporter.plot_hypervolume_history(hvs, problem_name, save_path)


# Backward-compatible alias kept during the naming cleanup.
qehvi_optimizer = qnehvi_optimizer
