import os
import gc
import torch
import numpy as np
import datetime
import json
from src.evoMI.optimization_reporting import reporter
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
from gpytorch.kernels import ScaleKernel, MaternKernel, ProductKernel
from gpytorch.constraints import Interval
from gpytorch.priors import GammaPrior, HalfCauchyPrior
from botorch.models.transforms import Normalize as BoTorchNormalize, Standardize

def saasbo_qnehvi_two_stage(
    objective_function,
    dim=36,
    num_objectives=2,
    bounds=None,
    BATCH_SIZE=5,
    NUM_RESTARTS=20,
    RAW_SAMPLES=512,
    MC_SAMPLES=128,
    N_BATCH=20,  # 总迭代次数
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
    # SAASBO和重要性参数
    use_saas=True,
    initial_importance=None,
    initial_importance_dict=None,  # 不同块数对应的重要性初始化值字典
    enable_importance_prior=True,
    enable_importance_update=True,
    enable_importance_guidance=True,
    enable_importance_weighted_acq=False,
    learning_rate=0.1,
    # 分块配置
    block_configs=None,  # 块划分配置列表，每个元素包含start_iter, end_iter, n_blocks
    block_mapping='repeat',  # 映射方式
    merged_blocks_dict=None,  # 不同块数对应的合并块字典
    seed=42,
    # 真实层数配置
    real_dim=36,  # 真实的层数，用于第二阶段精细搜索
):
    """
    块优化版本的saasbo_qnehvi优化器
    根据block_configs配置在不同迭代阶段使用不同的分块数
    
    参数:
    objective_function: 目标函数，输入为形状为(batch_size, dim)的张量，输出为形状为(batch_size, num_objectives)的张量
    dim: 决策变量维度，默认为36
    num_objectives: 目标函数数量，默认为2
    bounds: 变量边界，形状为(2, dim)，如果为None则默认[0, 1]边界
    BATCH_SIZE: 每轮迭代选择的候选点数量，默认为5
    NUM_RESTARTS: 优化获取函数时的重启次数，默认为20
    RAW_SAMPLES: 用于初始化优化的原始样本数量，默认为512
    MC_SAMPLES: 蒙特卡洛采样数量，默认为128
    N_BATCH: 优化的总轮数，默认为20
    verbose: 是否打印优化进度，默认为True
    device: 计算设备，默认为"cpu"
    dtype: 数据类型，默认为torch.double
    initial_samples: 初始采样点数量，默认为10
    noise_level: 观察噪声标准差，默认为0.01
    iteration_callback: 每轮迭代完成后调用的回调函数
    ref_point: 超体积计算的参考点，默认为-1.1
    run_id: 运行的唯一标识符，如果为None则自动生成时间戳格式的ID
    checkpoint_dir: 检查点保存的根目录，默认为"./checkpoints"
    custom_initial_solutions: 用户自定义的初始解列表
    use_saas: 是否使用SAAS先验，默认为True
    initial_importance: 初始重要性权重，形状为(dim,)，默认为None（均匀分布）
    initial_importance_dict: 不同块数对应的重要性初始化值字典，键为块数，值为对应的重要性权重
    enable_importance_prior: 是否在模型中集成重要性先验，默认为True
    enable_importance_update: 是否在优化过程中更新重要性估计，默认为True
    enable_importance_guidance: 是否在搜索策略中使用重要性指导，默认为True
    enable_importance_weighted_acq: 是否在获取函数中加入重要性权重，默认为False
    learning_rate: 重要性更新学习率，默认为0.1
    block_configs: 块划分配置列表，每个元素是一个字典，包含：
        - start_iter: 开始迭代次数（包含）
        - end_iter: 结束迭代次数（包含）
        - n_blocks: 该阶段使用的分块数
        例如：[{"start_iter": 0, "end_iter": 9, "n_blocks": 6}, {"start_iter": 10, "end_iter": 19, "n_blocks": 36}]
        如果为None，则使用默认配置：前10次迭代使用6块，之后使用精细搜索
    block_mapping: 映射方式，默认为'repeat'
    seed: 随机种子，默认为42
    real_dim: 真实的层数，用于精细搜索，默认为36
    
    返回:
    train_x: 所有评估点的决策变量值
    train_obj_true: 所有评估点的真实目标函数值
    train_info: 所有评估点的评测信息
    hvs: 每轮迭代的超体积值
    problem_ref_point: 超体积计算的参考点
    run_id: 本次运行的唯一标识符
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
        bounds = bounds.to(**tkwargs)
    
    # 标准化边界
    standard_bounds = torch.zeros(2, dim, **tkwargs)
    standard_bounds[1] = 1
    
    # 设置噪声
    NOISE_SE = torch.full((num_objectives,), noise_level, **tkwargs)
    
    # 确定参考点
    if type(ref_point) == torch.Tensor:
        problem_ref_point = ref_point
    else:
        problem_ref_point = torch.full((num_objectives,), ref_point, **tkwargs)
    
    # 处理块配置
    if block_configs is None:
        # 默认配置：前10次迭代使用6块，之后使用精细搜索
        block_configs = [
            {"start_iter": 0, "end_iter": 9, "n_blocks": 6},
            {"start_iter": 10, "end_iter": N_BATCH - 1, "n_blocks": real_dim}
        ]
    
    def get_current_block_config(iteration):
        """根据当前迭代次数获取对应的块配置"""
        for config in block_configs:
            if config["start_iter"] <= iteration <= config["end_iter"]:
                return config
        # 默认返回最后一个配置
        return block_configs[-1]
    
    def create_block_mapping(current_n_blocks):
        """创建分块映射函数，根据实际的分块信息而不是均匀划分"""
        # 使用real_dim作为真实维度
        actual_dim = real_dim
        
        if block_mapping == 'repeat':
            def block_to_full(x_block):
                """块变量 -> 高维变量（重复）"""
                n_samples = x_block.shape[0]
                x_full = torch.zeros(n_samples, actual_dim+1, **tkwargs)
                
                # 确保x_block的维度是current_n_blocks + 1
                expected_dim = current_n_blocks + 1
                if x_block.shape[1] != expected_dim:
                    print(f"警告：x_block的维度 {x_block.shape[1]} 与预期维度 {expected_dim} 不匹配")
                    x_block = torch.concat([x_block, x_block[:, -1:]], dim=1)
                
                # 如果有merged_blocks_dict，使用实际的分块信息
                if merged_blocks_dict is not None and current_n_blocks in merged_blocks_dict:
                    merged_blocks = merged_blocks_dict[current_n_blocks]
                    # 遍历每个合并块，将块变量值分配给对应的层
                    # 注意：x_block的最后一个维度是额外的决策变量，不参与分块映射
                    for i, (block_layers, _) in enumerate(merged_blocks):
                        # 遍历块内的所有层
                        for layer_idx in block_layers:
                            if layer_idx < actual_dim:  # 确保层索引不越界
                                x_full[:, layer_idx] = x_block[:, i]

                x_full[:, -1] = x_block[:, -1]  
                return x_full
            
            def full_to_block(x_full):
                """高维变量 -> 块变量（平均）"""
                n_samples = x_full.shape[0]
                # 当块数为N时，返回N+1个决策变量
                x_block = torch.zeros(n_samples, current_n_blocks + 1, **tkwargs)
                
                # 如果有merged_blocks_dict，使用实际的分块信息
                if merged_blocks_dict is not None and current_n_blocks in merged_blocks_dict:
                    merged_blocks = merged_blocks_dict[current_n_blocks]
                    # 遍历每个合并块，计算块内所有层的平均值
                    for i, (block_layers, _) in enumerate(merged_blocks):
                        # 收集块内所有层的值
                        layer_values = []
                        for layer_idx in block_layers:
                            if layer_idx < actual_dim:  # 确保层索引不越界
                                layer_values.append(x_full[:, layer_idx])
                        # 计算平均值
                        if layer_values:
                            x_block[:, i] = torch.mean(torch.stack(layer_values, dim=1), dim=1)
                else:
                    # 如果没有merged_blocks_dict，使用均匀划分
                    current_block_size = actual_dim // current_n_blocks
                    for i in range(current_n_blocks):
                        start = i * current_block_size
                        end = min((i + 1) * current_block_size, actual_dim)
                        x_block[:, i] = torch.mean(x_full[:, start:end], dim=1)
                
                # 最后一个维度使用平均值
                x_block[:, -1] = x_full[:, -1]
                
                return x_block
        else:
            raise ValueError(f"不支持的映射方式: {block_mapping}")
        
        return block_to_full, full_to_block
    
    def smart_checkpoint_conversion(checkpoint_x, checkpoint_n_blocks, current_n_blocks):
        """智能转换checkpoint中的解，处理决策变量与块数不匹配的情况"""
        # 如果块数相同，直接返回
        if checkpoint_n_blocks == current_n_blocks:
            return checkpoint_x
        
        # 确定当前模式：如果块数小于real_dim，就是分块模式，否则是精细模式
        checkpoint_is_block_mode = checkpoint_n_blocks < real_dim
        current_is_block_mode = current_n_blocks < real_dim
        
        # 从分块模式转换到另一种分块模式
        if checkpoint_is_block_mode and current_is_block_mode:
            # 先将分块解转换为原始解
            block_to_full, _ = create_block_mapping(checkpoint_n_blocks)
            full_x = block_to_full(checkpoint_x)
            # 再将原始解转换为新的分块解
            _, full_to_block = create_block_mapping(current_n_blocks)
            new_block_x = full_to_block(full_x)
            print(f"分块模式转换：{checkpoint_n_blocks}块 -> {current_n_blocks}块，转换后维度: {new_block_x.shape[1]}")
            return new_block_x
        
        # 如果是从分块模式转换到精细模式
        elif checkpoint_is_block_mode and not current_is_block_mode:
            # 先将分块解转换为原始解
            block_to_full, _ = create_block_mapping(checkpoint_n_blocks)
            full_x = block_to_full(checkpoint_x)
            print(f"分块模式转换到精细模式：{checkpoint_n_blocks}块 -> {current_n_blocks}块，转换后维度: {full_x.shape[1]}")
            return full_x
        
        # 如果是从精细模式转换到分块模式
        elif not checkpoint_is_block_mode and current_is_block_mode:
            # 将原始解转换为分块解
            _, full_to_block = create_block_mapping(current_n_blocks)
            block_x = full_to_block(checkpoint_x)
            print(f"精细模式转换到分块模式：{checkpoint_n_blocks}块 -> {current_n_blocks}块，转换后维度: {block_x.shape[1]}")
            return block_x
        
        # 精细模式到精细模式的转换（直接返回）
        else:
            print(f"精细模式转换到精细模式：{checkpoint_n_blocks}块 -> {current_n_blocks}块，直接返回，维度: {checkpoint_x.shape[1]}")
            return checkpoint_x
    
    def create_gp_model(train_x, train_obj):
        """创建GP模型"""
        # 确保train_x与train_obj使用double精度
        train_x = train_x.to(dtype=torch.double, device=train_obj.device)
        train_y = train_obj.to(dtype=torch.double, device=train_obj.device)
        models = []
        
        # 获取当前train_x的实际维度
        current_dim = train_x.shape[1]
        
        for i in range(train_obj.shape[-1]):
            train_y_i = train_y[..., i : i + 1]
            
            # 创建协方差模块，使用当前维度
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=current_dim,
                    lengthscale_constraint=Interval(0.005, 10.0)
                )
            )
            
            # 设置基于重要性的长度尺度初始值
            if enable_importance_prior:
                with torch.no_grad():
                    if current_dim < dim:
                        # 分块模式：使用重要性的前current_dim个值
                        initial_lengthscales = 0.05 + 4.95 * (1.0 - importance[:current_dim])
                    else:
                        # 精细模式：使用完整的重要性值
                        initial_lengthscales = 0.05 + 4.95 * (1.0 - importance)
                    initial_lengthscales = initial_lengthscales.view(1, current_dim).to(dtype=torch.double)
                    covar_module.base_kernel.lengthscale = initial_lengthscales
            
            # 创建模型，使用当前维度创建输入变换
            model = SingleTaskGP(
                train_x,
                train_y_i,
                covar_module=covar_module,
                input_transform=BoTorchNormalize(d=current_dim),
                outcome_transform=Standardize(m=1)
            )
            
            # 设置SAAS先验
            if use_saas:
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
        return create_gp_model(train_x, train_obj)

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
    
    def generate_initial_data(n=initial_samples, custom_solutions=None, current_n_blocks=None):
        """生成初始训练数据"""
        # 处理自定义初始解
        if custom_solutions is not None and len(custom_solutions) > 0:
            # 计算需要生成的自定义解数量
            num_custom = len(custom_solutions)
            # 确保不超过总样本数
            num_custom = min(num_custom, n)
            
            # 生成自定义解
            custom_x = []
            for val in custom_solutions:
                # 如果是分块模式，生成块维度+1的解（块数+1个决策变量）
                if current_n_blocks is not None and current_n_blocks < dim:
                    # 当块数为N时，决策变量数量为N+1
                    custom_sol = torch.full((1, current_n_blocks + 1), val, **tkwargs)
                else:
                    # 精细模式：生成原始维度的解
                    custom_sol = torch.full((1, dim), val, **tkwargs)
                custom_x.append(custom_sol)
            
            # 合并自定义解
            custom_x = torch.cat(custom_x, dim=0)
            
            # 计算还需要生成的样本数
            remaining = n - num_custom
            
            if remaining > 0:
                # 生成随机样本
                if current_n_blocks is not None and current_n_blocks < dim:
                    # 分块模式：生成块维度+1的随机样本（块数+1个决策变量）
                    train_x = torch.rand(remaining, current_n_blocks + 1, **tkwargs)
                else:
                    # 精细模式：生成原始维度的随机样本
                    train_x = torch.rand(remaining, dim, **tkwargs)
                # 合并自定义解和随机解
                train_x = torch.cat([custom_x, train_x], dim=0)
            else:
                # 如果自定义解数量大于等于n，只取前n个
                train_x = custom_x[:n]
        else:
            # 没有自定义解，生成随机样本
            if current_n_blocks is not None and current_n_blocks < dim:
                # 分块模式：生成块维度+1的随机样本（块数+1个决策变量）
                train_x = torch.rand(n, current_n_blocks + 1, **tkwargs)
            else:
                # 精细模式：生成原始维度的随机样本
                train_x = torch.rand(n, dim, **tkwargs)
        
        # 调用目标函数，获取目标函数值和评测结果
        result = objective_function(train_x)
        
        # 处理返回结果
        initial_cleanup_paths = []
        if isinstance(result, tuple) and len(result) == 3:
            train_obj_true, train_info, initial_cleanup_paths = result
        elif isinstance(result, tuple) and len(result) == 2:
            train_obj_true, train_info = result
            initial_cleanup_paths = collect_cleanup_paths(train_info)
        else:
            train_obj_true = result
            train_info = [{} for _ in range(train_x.shape[0])]
        
        # 确保train_obj_true在正确的设备和数据类型上
        train_obj_true = train_obj_true.to(**tkwargs)
        # 生成相同设备和数据类型上的随机噪声
        train_obj = train_obj_true + torch.randn_like(train_obj_true, **tkwargs) * NOISE_SE
        if iteration_callback is not None and len(initial_cleanup_paths) > 0:
            existing_cleanup_paths = getattr(iteration_callback, "cleanup_paths", [])
            iteration_callback.cleanup_paths = list(dict.fromkeys(existing_cleanup_paths + initial_cleanup_paths))
        
        return train_x, train_obj, train_obj_true, train_info
    
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
    
    def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler, current_dim, current_bounds):
        """优化qNEHVI获取函数，并返回新的候选点和观察值"""
        # 确保所有数据在同一设备和数据类型上（强制使用double精度）
        ref_point_device = train_x.device
        ref_point_dtype = torch.double
        
        # 确保训练数据使用double精度
        train_x = train_x.to(dtype=torch.double, device=ref_point_device)
        train_obj = train_obj.to(dtype=torch.double, device=ref_point_device)
        
        # 获取当前训练数据的实际维度
        actual_current_dim = train_x.shape[1]
        print(f"optimize_qnehvi_and_get_observation: 实际训练数据维度 {actual_current_dim}, current_dim {current_dim}")
        
        # 确保边界维度与训练数据维度匹配
        if current_bounds.shape[1] != actual_current_dim:
            print(f"警告：边界维度 {current_bounds.shape[1]} 与训练数据维度 {actual_current_dim} 不匹配，重新创建边界")
            # 重新创建边界，维度为训练数据的实际维度
            current_bounds = torch.zeros(2, actual_current_dim, **tkwargs)
            current_bounds[1] = 1
            print(f"重新创建边界，维度: {current_bounds.shape[1]}")
        
        # 归一化训练数据
        train_x_normalized = normalize(train_x, current_bounds)
        train_x_normalized = train_x_normalized.to(dtype=ref_point_dtype)
        print(f"归一化后训练数据维度: {train_x_normalized.shape[1]}")
        
        # 创建获取函数（确保模型使用double精度）
        model = model.double()
        ref_point = problem_ref_point.to(device=ref_point_device, dtype=ref_point_dtype)
        
        # 重新创建模型，确保输入变换使用当前维度
        print(f"创建获取函数，使用的模型维度: {actual_current_dim}")
        
        # 重新创建获取函数，确保使用当前维度的X_baseline
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_x_normalized,
            prune_baseline=True,
            sampler=sampler,
        )
        print(f"获取函数创建成功，X_baseline维度: {acq_func.X_baseline.shape[1]}")
        
        # 优化获取函数（强制使用double精度）
        # 使用训练数据的实际维度作为边界维度，确保生成的初始条件维度匹配
        standard_bounds = torch.zeros(2, actual_current_dim, dtype=torch.double, device=ref_point_device)
        standard_bounds[1] = 1
        print(f"优化获取函数，使用的边界维度: {actual_current_dim}")
        
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 100, "dtype": torch.double},
        )
        print(f"获取函数优化成功，生成的候选点维度: {candidates.shape[1]}")
        
        # 确保候选点使用double精度
        candidates = candidates.to(dtype=torch.double)
        
        # 观察新值
        with torch.no_grad():
            new_x = unnormalize(candidates.detach(), bounds=current_bounds)
            # 确保new_x在正确的设备和数据类型上
            new_x = new_x.to(device=ref_point_device, dtype=ref_point_dtype)
            
            # 调用目标函数，获取目标函数值和评测结果
            result = objective_function(new_x)
            
            # 处理返回结果
            cleanup_paths = []
            if isinstance(result, tuple) and len(result) == 3:
                new_obj_true, new_info, cleanup_paths = result
            elif isinstance(result, tuple) and len(result) == 2:
                new_obj_true, new_info = result
                cleanup_paths = collect_cleanup_paths(new_info)
            else:
                new_obj_true = result
                new_info = [{} for _ in range(new_x.shape[0])]
            
            # 确保new_obj_true在正确的设备和数据类型上
            new_obj_true = new_obj_true.to(device=ref_point_device, dtype=ref_point_dtype)
            # 生成相同设备和数据类型上的随机噪声
            device_noise_se = NOISE_SE.to(device=ref_point_device, dtype=ref_point_dtype)
            new_obj = new_obj_true + torch.randn_like(new_obj_true, device=ref_point_device, dtype=ref_point_dtype) * device_noise_se
        
        # 显式释放临时变量
        del train_x_normalized, acq_func, candidates
        torch.cuda.empty_cache()
        
        return new_x, new_obj, new_obj_true, new_info, cleanup_paths
    
    def compute_pareto_hypervolume(objectives, ref_point):
        """计算帕累托前沿的超体积"""
        # 确保ref_point和objectives在同一个设备上
        ref_point = ref_point.to(device=objectives.device, dtype=objectives.dtype)
        bd = FastNondominatedPartitioning(ref_point=ref_point, Y=objectives)
        return bd.compute_hypervolume().item()
    
    def save_checkpoint(iteration, train_x, train_obj, train_obj_true, train_info, hvs, run_dir, tkwargs, current_n_blocks):
        """保存优化过程的检查点"""
        # 创建检查点字典
        checkpoint = {
            'iteration': iteration,
            'hvs': hvs,
            'current_n_blocks': current_n_blocks,
            'evaluated_solutions': {
                'decision_variables': train_x.cpu().tolist(),
                'objectives': train_obj_true.cpu().tolist(),
                'metrics': train_info
            }
        }
        
        # 保存非张量数据到JSON文件
        json_path = os.path.join(run_dir, f'checkpoint_iter_{iteration}.json')
        with open(json_path, 'w') as f:
            json.dump(checkpoint, f)
        
        # 保存张量数据
        torch.save({
            'iteration': iteration,
            'hvs': hvs,
            'current_n_blocks': current_n_blocks,
            'train_x': train_x.cpu(),
            'train_obj': train_obj.cpu(),
            'train_obj_true': train_obj_true.cpu(),
            'train_info': train_info
        }, os.path.join(run_dir, f'checkpoint_iter_{iteration}.pt'))
        
        # 同时保存最新的检查点，方便后续重启
        torch.save({
            'iteration': iteration,
            'hvs': hvs,
            'current_n_blocks': current_n_blocks,
            'train_x': train_x.cpu(),
            'train_obj': train_obj.cpu(),
            'train_obj_true': train_obj_true.cpu(),
            'train_info': train_info
        }, os.path.join(run_dir, 'checkpoint_latest.pt'))
    
    def load_checkpoint(run_dir, tkwargs):
        """加载最新的检查点"""
        latest_checkpoint_path = os.path.join(run_dir, 'checkpoint_latest.pt')
        if not os.path.exists(latest_checkpoint_path):
            return None
        
        # 加载检查点
        checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
        
        # 转换到指定的设备和数据类型
        train_x = checkpoint['train_x'].to(**tkwargs)
        train_obj = checkpoint['train_obj'].to(**tkwargs)
        train_obj_true = checkpoint['train_obj_true'].to(**tkwargs)
        iteration = checkpoint['iteration']
        hvs = checkpoint['hvs']
        # 使用默认块配置的第一个块数作为默认值
        default_n_blocks = block_configs[0]["n_blocks"]
        current_n_blocks = checkpoint.get('current_n_blocks', default_n_blocks)
        
        # 加载评测信息，如果不存在则创建空列表
        train_info = checkpoint.get('train_info', [{} for _ in range(train_x.shape[0])])
        
        return train_x, train_obj, train_obj_true, train_info, iteration, hvs, current_n_blocks
    
    # 开始优化过程
    if verbose:
        print("="*60)
        print(f"使用SAASBO+qNEHVI块优化算法")
        print(f"维度: {dim}, 目标数: {num_objectives}")
        print(f"设备: {tkwargs['device']}, 数据类型: {tkwargs['dtype']}")
        print(f"运行ID: {run_id}, 检查点目录: {run_dir}")
        print(f"块配置: {block_configs}")
        print(f"SAAS配置: 使用SAAS先验={use_saas}")
        print("="*60)
    
    # 尝试加载检查点
    checkpoint = load_checkpoint(run_dir, tkwargs)
    if checkpoint is not None:
        train_x, train_obj, train_obj_true, train_info, start_iteration, hvs, checkpoint_n_blocks = checkpoint
        if verbose:
            print(f"成功加载检查点，从迭代 {start_iteration} 继续")
            print(f"当前超体积: {hvs[-1]:.4f}")
    else:
        # 生成初始数据，使用第一个块配置的n_blocks
        initial_block_config = block_configs[0]
        initial_n_blocks = initial_block_config["n_blocks"]
        train_x, train_obj, train_obj_true, train_info = generate_initial_data(custom_solutions=custom_initial_solutions, current_n_blocks=initial_n_blocks)
        
        # 记录超体积
        hvs = []
        initial_hv = compute_pareto_hypervolume(train_obj_true, problem_ref_point)
        hvs.append(initial_hv)
        if verbose:
            print(f"初始超体积: {initial_hv:.4f}")
        
        start_iteration = 0
        # 初始化checkpoint_n_blocks，使用初始块数
        checkpoint_n_blocks = initial_n_blocks
        # 保存初始状态
        save_checkpoint(0, train_x, train_obj, train_obj_true, train_info, hvs, run_dir, tkwargs, initial_n_blocks)
    
    try:
        # 运行优化
        for iteration in range(start_iteration + 1, N_BATCH + 1):
            # 确定当前块配置
            current_config = get_current_block_config(iteration - 1)  # 使用iteration-1作为索引，因为初始状态是iteration 0
            current_n_blocks = current_config["n_blocks"]
            
            # 确定当前阶段
            if current_n_blocks < dim:
                current_stage = "block"
            else:
                current_stage = "fine"
            
            if verbose:
                print(f"\n迭代 {iteration}/{N_BATCH}: {current_stage} 优化 (块数: {current_n_blocks})")
            
            # 智能转换checkpoint中的解，确保训练数据维度与当前块配置匹配
            if checkpoint_n_blocks != current_n_blocks:
                print(f"切换块配置: 从 {checkpoint_n_blocks} 块到 {current_n_blocks} 块")
                # 转换训练数据，确保维度匹配
                train_x = smart_checkpoint_conversion(train_x, checkpoint_n_blocks, current_n_blocks)
                checkpoint_n_blocks = current_n_blocks
                print(f"转换后训练数据维度: {train_x.shape[1]}")
                
                # 重新计算初始重要性，确保与当前块配置匹配
                if initial_importance_dict is not None and current_n_blocks in initial_importance_dict:
                    initial_importance = initial_importance_dict[current_n_blocks]
                    print(f"使用块数 {current_n_blocks} 对应的重要性初始化值")
                
                # 初始化重要性，确保与当前块配置匹配
                if initial_importance_dict is not None and current_n_blocks in initial_importance_dict:
                    importance = initial_importance_dict[current_n_blocks].clone().to(**tkwargs)
                else:
                    importance = torch.ones(dim, **tkwargs) * 0.5
            else:
                # 根据当前块数获取对应的重要性初始化值
                if initial_importance_dict is not None and current_n_blocks in initial_importance_dict:
                    importance = initial_importance_dict[current_n_blocks].clone().to(**tkwargs)
                    if verbose:
                        print(f"  使用块数 {current_n_blocks} 对应的重要性初始化值")
                else:
                    importance = torch.ones(dim, **tkwargs) * 0.5
            
            # 创建映射函数
            block_to_full, full_to_block = create_block_mapping(current_n_blocks)
            
            # 确定当前目标函数和边界
            if current_n_blocks < real_dim:
                # 分块模式
                def current_objective(x_block):
                    x_full = block_to_full(x_block)
                    return objective_function(x_full)
                
                # 使用当前块数作为边界维度
                current_bounds = torch.zeros(2, current_n_blocks, **tkwargs)
                current_bounds[1] = 1
                print(f"分块模式，创建新的边界，维度: {current_bounds.shape[1]}")
            else:
                # 精细模式
                current_objective = objective_function
                current_bounds = bounds
                print(f"精细模式，使用原始边界，维度: {current_bounds.shape[1]}")
            
            # 初始化并训练模型
            mll, model = initialize_model(train_x, train_obj)
            fit_gpytorch_mll(mll)
            
            # 确保模型使用正确的当前维度
            # 重新创建模型，确保输入变换使用当前维度
            if train_x.shape[1] != dim:
                mll, model = create_gp_model(train_x, train_obj)
            
            # 定义QMC采样器
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            
            # 优化获取函数并获取新的观察值
            # 使用train_x的实际维度作为current_dim，确保决策变量维度匹配
            current_dim = train_x.shape[1]
            print(f"调用optimize_qnehvi_and_get_observation，current_dim: {current_dim}, current_bounds维度: {current_bounds.shape[1]}")
            new_x, new_obj, new_obj_true, new_info, cleanup_paths = optimize_qnehvi_and_get_observation(
                model, train_x, train_obj, sampler, current_dim, current_bounds
            )
            
            # 确保新生成的候选点维度与当前训练数据维度匹配
            if new_x.shape[1] != train_x.shape[1]:
                # 如果不匹配，进行转换
                if train_x.shape[1] < dim:
                    # 训练数据是分块模式，将新生成的候选点转换为分块模式
                    _, full_to_block = create_block_mapping(train_x.shape[1])
                    new_x = full_to_block(new_x)
                else:
                    # 训练数据是精细模式，将新生成的候选点转换为精细模式
                    block_to_full, _ = create_block_mapping(new_x.shape[1])
                    new_x = block_to_full(new_x)
            
            # 更新训练点
            train_x = torch.cat([train_x, new_x.to(train_x.device)])
            train_obj = torch.cat([train_obj, new_obj.to(train_obj.device)])
            train_obj_true = torch.cat([train_obj_true, new_obj_true.to(train_obj_true.device)])
            train_info.extend(new_info)
            
            # 计算新的超体积
            with torch.no_grad():
                new_hv = compute_pareto_hypervolume(train_obj_true, problem_ref_point)
            hvs.append(new_hv)
            
            # 打印进度
            if verbose:
                print(f"  超体积: {new_hv:.4f} (提升: {new_hv - hvs[-2]:.4f})")
                # 打印当前GPU内存使用情况
                if tkwargs['device'].type == 'cuda':
                    print(f"  GPU内存: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
            
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
            save_checkpoint(iteration, train_x, train_obj, train_obj_true, train_info, hvs, run_dir, tkwargs, current_n_blocks)
    except Exception as e:
        print(f"错误: 优化过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        # 即使出错也返回已有的结果
        return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
    
    # 打印最终结果
    if verbose:
        print("\n" + "="*60)
        print("优化完成!")
        print(f"总评估: {len(train_x)} 个点")
        print(f"最终超体积: {hvs[-1]:.6f}")
        print("="*60)
    
    return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
