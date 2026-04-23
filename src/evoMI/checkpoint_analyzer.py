#!/usr/bin/env python3
"""
Checkpoint Analyzer Tool

功能：
1. 读取任意checkpoint数据，在不同数据集上绘制指标
2. 读取不同迭代次数的checkpoint（用户设定间隔），绘制趋势
3. 生成均匀分布的偏好向量，选择帕累托前沿个体，生成筛选后的checkpoint
"""

import os
import sys
import argparse
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import json
import re

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



# 导入模型重现相关模块
from src.evoMI.model_reproduction import generate_model_cache_key, get_model_cache_path, load_cached_results

# 导入pymoo库用于生成参考方向
from pymoo.util.ref_dirs import get_reference_directions

# 导入botorch库用于计算超体积
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, 'checkpoints')
AUTO_ALGORITHM_ORDER = ['prior_sync', 'prior_async', 'priorbo', 'qnehvi', 'emm', 'momm', 'grid', 'saasbo']
AUTO_ALGORITHM_LABELS = {
    'prior_sync': 'Prior-Sync',
    'prior_async': 'Prior-Async',
    'priorbo': 'PriorBO',
    'qnehvi': 'qNEHVI',
    'emm': 'EMM',
    'momm': 'MOMM',
    'grid': 'Grid Search',
    'saasbo': 'SAASBO',
}


def load_checkpoint(checkpoint_path: str) -> dict:
    """
    加载checkpoint数据
    
    Args:
        checkpoint_path: checkpoint文件路径
        
    Returns:
        dict: 包含train_x, train_obj_true等数据的字典
    """
    print(f"从 {checkpoint_path} 加载checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def _normalize_path(path: str) -> str:
    if path is None:
        return None
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(PROJECT_ROOT, expanded))


def _extract_iteration_number(file_name: str):
    match = re.match(r'^checkpoint_iter_(\d+)\.pt$', file_name)
    if match:
        return int(match.group(1))
    return None


def _list_checkpoint_files(directory: str):
    if not os.path.isdir(directory):
        return []
    checkpoint_files = []
    for file_name in os.listdir(directory):
        full_path = os.path.join(directory, file_name)
        if not os.path.isfile(full_path):
            continue
        if file_name == 'checkpoint_latest.pt':
            checkpoint_files.append((float('inf'), full_path))
            continue
        iteration_number = _extract_iteration_number(file_name)
        if iteration_number is not None:
            checkpoint_files.append((iteration_number, full_path))
    return checkpoint_files


def _select_latest_checkpoint_file(directory: str):
    checkpoint_files = _list_checkpoint_files(directory)
    if not checkpoint_files:
        return None
    iter_checkpoints = [(iteration_number, path) for iteration_number, path in checkpoint_files if iteration_number != float('inf')]
    if iter_checkpoints:
        iter_checkpoints.sort(key=lambda item: item[0])
        return iter_checkpoints[-1][1]
    latest_path = os.path.join(directory, 'checkpoint_latest.pt')
    if os.path.exists(latest_path):
        return latest_path
    checkpoint_files.sort(key=lambda item: item[1])
    return checkpoint_files[-1][1]


def _is_algorithm_checkpoint_dir(directory: str):
    if not os.path.isdir(directory):
        return False
    return _select_latest_checkpoint_file(directory) is not None


def _infer_algorithm_key_from_name(name: str):
    name_lower = name.lower()
    for algorithm in sorted(AUTO_ALGORITHM_ORDER, key=len, reverse=True):
        if name_lower == algorithm or name_lower.endswith(f'_{algorithm}'):
            return algorithm
    return None


def _algorithm_sort_key(name: str):
    algorithm_key = _infer_algorithm_key_from_name(name)
    if algorithm_key in AUTO_ALGORITHM_ORDER:
        return (AUTO_ALGORITHM_ORDER.index(algorithm_key), name)
    return (len(AUTO_ALGORITHM_ORDER), name)


def _infer_algorithm_label(name: str):
    algorithm_key = _infer_algorithm_key_from_name(name)
    if algorithm_key is not None:
        return AUTO_ALGORITHM_LABELS.get(algorithm_key, algorithm_key)
    return name


def _collect_direct_algorithm_dirs(parent_dir: str):
    if not os.path.isdir(parent_dir):
        return []
    algorithm_dirs = []
    for entry_name in os.listdir(parent_dir):
        entry_path = os.path.join(parent_dir, entry_name)
        if _is_algorithm_checkpoint_dir(entry_path):
            algorithm_dirs.append(entry_path)
    algorithm_dirs.sort(key=lambda path: _algorithm_sort_key(os.path.basename(path)))
    return algorithm_dirs


def _find_latest_checkpoint_mtime(directory: str):
    latest_mtime = 0.0
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name == 'checkpoint_latest.pt' or _extract_iteration_number(file_name) is not None:
                try:
                    latest_mtime = max(latest_mtime, os.path.getmtime(os.path.join(root, file_name)))
                except OSError:
                    continue
    return latest_mtime


def _find_latest_settings_file(checkpoint_dir: str):
    output_dir = os.path.join(checkpoint_dir, 'output')
    if not os.path.isdir(output_dir):
        return None
    latest_settings_path = None
    latest_mtime = -1.0
    for root, _, files in os.walk(output_dir):
        if 'settings.json' not in files:
            continue
        settings_path = os.path.join(root, 'settings.json')
        try:
            settings_mtime = os.path.getmtime(settings_path)
        except OSError:
            continue
        if settings_mtime > latest_mtime:
            latest_mtime = settings_mtime
            latest_settings_path = settings_path
    return latest_settings_path


def _load_run_settings(checkpoint_dir: str):
    settings_path = _find_latest_settings_file(checkpoint_dir)
    if settings_path is None:
        return {}
    try:
        with open(settings_path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return {}


def _to_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compute_evaluation_count(iter_num: int, algorithm_dir_name: str, run_settings: dict):
    algorithm_key = _infer_algorithm_key_from_name(algorithm_dir_name)
    async_mode = bool(run_settings.get('async_mode', False))
    initial_samples = _to_int_or_none(run_settings.get('initial_samples'))
    if async_mode or algorithm_key == 'prior_async':
        if initial_samples is None:
            return iter_num
        # For async runs, checkpoint_iter_N is treated as aligned to evaluation N after init.
        # Keep iter_0 as the initial-design endpoint, and map later checkpoints directly.
        if iter_num == 0:
            return initial_samples
        return iter_num

    batch_size = _to_int_or_none(run_settings.get('batch_size'))
    if batch_size is None or initial_samples is None:
        return iter_num
    return initial_samples + iter_num * batch_size


def _format_evaluation_label(evaluation_count: int):
    return f'Evaluations {evaluation_count}'


def discover_algorithm_checkpoints(checkpoint_root: str = None, run_id: str = None, algorithms: list = None):
    checkpoint_root = _normalize_path(checkpoint_root or DEFAULT_CHECKPOINT_ROOT)
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(f'checkpoint 根目录不存在: {checkpoint_root}')

    candidate_run_dirs = []
    if run_id:
        requested_run_dir = _normalize_path(run_id)
        if not os.path.isdir(requested_run_dir):
            requested_run_dir = os.path.join(checkpoint_root, run_id)
        requested_run_dir = os.path.normpath(requested_run_dir)
        if not os.path.isdir(requested_run_dir):
            raise FileNotFoundError(f'未找到指定的 run 目录: {run_id}')
        candidate_run_dirs = [requested_run_dir]
    else:
        direct_algorithm_dirs = _collect_direct_algorithm_dirs(checkpoint_root)
        if direct_algorithm_dirs:
            candidate_run_dirs = [checkpoint_root]
        else:
            for entry_name in os.listdir(checkpoint_root):
                entry_path = os.path.join(checkpoint_root, entry_name)
                if not os.path.isdir(entry_path):
                    continue
                nested_algorithm_dirs = _collect_direct_algorithm_dirs(entry_path)
                if nested_algorithm_dirs:
                    candidate_run_dirs.append(entry_path)
        if not candidate_run_dirs:
            raise FileNotFoundError(f'在 {checkpoint_root} 下未找到可用的算法 checkpoint 目录')
        candidate_run_dirs.sort(
            key=lambda path: (
                _find_latest_checkpoint_mtime(path),
                len(_collect_direct_algorithm_dirs(path)),
            ),
            reverse=True,
        )

    selected_run_dir = candidate_run_dirs[0]
    algorithm_dirs = _collect_direct_algorithm_dirs(selected_run_dir)
    if not algorithm_dirs:
        raise FileNotFoundError(f'在 {selected_run_dir} 下未找到算法 checkpoint 目录')

    requested_algorithms = None
    if algorithms:
        requested_algorithms = {_infer_algorithm_key_from_name(item) or item.lower() for item in algorithms}

    checkpoint_paths = []
    legend_names = []
    for algorithm_dir in algorithm_dirs:
        algorithm_dir_name = os.path.basename(algorithm_dir)
        algorithm_key = _infer_algorithm_key_from_name(algorithm_dir_name)
        if requested_algorithms and algorithm_key not in requested_algorithms and algorithm_dir_name.lower() not in requested_algorithms:
            continue
        checkpoint_path = _select_latest_checkpoint_file(algorithm_dir)
        if checkpoint_path is None:
            continue
        checkpoint_paths.append(checkpoint_path)
        legend_names.append(_infer_algorithm_label(algorithm_dir_name))

    if not checkpoint_paths:
        requested_display = ', '.join(algorithms) if algorithms else '全部算法'
        raise FileNotFoundError(f'在 {selected_run_dir} 下未找到匹配 {requested_display} 的 checkpoint')

    return {
        'run_dir': selected_run_dir,
        'checkpoint_paths': checkpoint_paths,
        'legend_names': legend_names,
    }


def resolve_output_dir(checkpoint_paths: list, preferred_root: str = None):
    if preferred_root:
        base_dir = preferred_root
    else:
        checkpoint_dirs = [os.path.dirname(path) for path in checkpoint_paths if path]
        if not checkpoint_dirs:
            base_dir = DEFAULT_CHECKPOINT_ROOT
        elif len(checkpoint_dirs) == 1:
            base_dir = checkpoint_dirs[0]
        else:
            base_dir = os.path.commonpath(checkpoint_dirs)
    return os.path.normpath(os.path.join(base_dir, 'checkpoint_analysis_output'))


def get_pareto_optimal_points(points, return_indices=False):
    """
    从点集中筛选出帕累托最优解
    
    Args:
        points: 点集，可以是torch.Tensor或numpy.ndarray
        return_indices: 是否返回帕累托最优解的索引
        
    Returns:
        tuple: 如果return_indices为True，返回(pareto_points, indices)，否则返回pareto_points
    """
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
    
    # 获取帕累托最优解
    pareto_points = points[is_pareto]
    
    if return_indices:
        # 返回帕累托最优解的点和索引
        indices = np.where(is_pareto)[0]
        return pareto_points, indices
    
    return pareto_points


def compute_hypervolume(pareto_y, ref_point=None):
    """
    计算帕累托前沿的超体积
    
    Args:
        pareto_y: 帕累托前沿的目标值，可以是numpy.ndarray或torch.Tensor
        ref_point: 参考点，如果为None则使用默认值[-0.2, -0.2]
        
    Returns:
        float: 超体积值
    """
    # 1. 确保pareto_y是二维的
    if isinstance(pareto_y, np.ndarray):
        if pareto_y.ndim == 1:
            # 单个点的情况，转换为二维数组 (1, num_objectives)
            pareto_y = pareto_y.reshape(1, -1)
        pareto_y = torch.tensor(pareto_y, dtype=torch.float32)
    elif isinstance(pareto_y, torch.Tensor):
        if pareto_y.ndim == 1:
            # 单个点的情况，转换为二维张量 (1, num_objectives)
            pareto_y = pareto_y.unsqueeze(0)
    
    # 2. 设置参考点，与mi_opt_saasbo2.py保持一致
    if ref_point is None:
        # 使用固定参考点[-0.2, -0.2]，与mi_opt_saasbo2.py保持一致
        ref_point = torch.tensor([-0.2, -0.2], dtype=pareto_y.dtype)
    elif isinstance(ref_point, np.ndarray):
        ref_point = torch.tensor(ref_point, dtype=pareto_y.dtype)
    elif isinstance(ref_point, torch.Tensor):
        ref_point = ref_point.to(dtype=pareto_y.dtype)
    
    # 3. 确保ref_point是一维的
    if ref_point.ndim > 1:
        ref_point = ref_point.squeeze()
    
    # 4. 确保ref_point和pareto_y在同一个设备上
    if ref_point.device != pareto_y.device:
        ref_point = ref_point.to(pareto_y.device)
    
    try:
        # 5. 确保pareto_y至少有一个点
        if len(pareto_y) == 0:
            print("帕累托前沿没有点，无法计算超体积")
            return 0.0
        
        # 6. 使用FastNondominatedPartitioning计算超体积，参考saasbo_qnehvi_optimizer.py的实现
        # 确保pareto_y和ref_point形状匹配
        num_objectives = pareto_y.shape[1]
        if ref_point.shape[0] != num_objectives:
            print(f"参考点形状不匹配: ref_point形状={ref_point.shape}, 目标数={num_objectives}")
            # 调整参考点形状
            if ref_point.shape[0] == 1:
                # 如果参考点是标量扩展的，重复到目标数
                ref_point = ref_point.repeat(num_objectives)
            else:
                return 0.0
        
        bd = FastNondominatedPartitioning(ref_point=ref_point, Y=pareto_y)
        hypervolume = bd.compute_hypervolume().item()
        return hypervolume
    except Exception as e:
        print(f"计算超体积时出错: {e}")
        print(f"pareto_y形状: {pareto_y.shape}, ref_point形状: {ref_point.shape}")
        return 0.0


def _to_numpy_2d(points):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


def compute_spacing(pareto_y):
    """
    计算Spacing指标（越小越均匀）。

    使用Schott spacing定义：对每个点计算到最近邻的L1距离，再计算这些距离的标准差。
    """
    pareto_y = _to_numpy_2d(pareto_y)
    n_points = pareto_y.shape[0]
    if n_points <= 1:
        return 0.0

    nearest_distances = []
    for i in range(n_points):
        min_distance = None
        for j in range(n_points):
            if i == j:
                continue
            distance = np.sum(np.abs(pareto_y[i] - pareto_y[j]))
            if min_distance is None or distance < min_distance:
                min_distance = distance
        nearest_distances.append(min_distance if min_distance is not None else 0.0)

    nearest_distances = np.asarray(nearest_distances, dtype=float)
    if n_points == 2:
        return 0.0
    return float(np.sqrt(np.sum((nearest_distances - nearest_distances.mean()) ** 2) / (n_points - 1)))


def compute_max_gap(pareto_y):
    """
    计算排序后相邻帕累托点之间的最大欧氏间隙（越小越致密）。
    """
    pareto_y = _to_numpy_2d(pareto_y)
    n_points = pareto_y.shape[0]
    if n_points <= 1:
        return 0.0

    sort_indices = np.lexsort((pareto_y[:, 1], pareto_y[:, 0]))
    sorted_points = pareto_y[sort_indices]
    gaps = np.linalg.norm(np.diff(sorted_points, axis=0), axis=1)
    return float(np.max(gaps)) if len(gaps) else 0.0


def compute_spread(pareto_y, reference_front=None):
    """
    计算Deb spread指标 Delta（越小越均匀，且端点覆盖更完整）。

    说明：
    - 当前实现主要针对双目标前沿分析。
    - 若提供reference_front，则用其两端极值点作为端点参考；
      否则退化为只衡量内部间距均匀性。
    """
    pareto_y = _to_numpy_2d(pareto_y)
    n_points = pareto_y.shape[0]
    if n_points <= 1:
        return 0.0
    if pareto_y.shape[1] < 2:
        return 0.0

    sort_indices = np.lexsort((pareto_y[:, 1], pareto_y[:, 0]))
    sorted_points = pareto_y[sort_indices]
    adjacent_distances = np.linalg.norm(np.diff(sorted_points, axis=0), axis=1)
    if len(adjacent_distances) == 0:
        return 0.0

    d_bar = float(np.mean(adjacent_distances))
    interior_term = float(np.sum(np.abs(adjacent_distances - d_bar)))

    d_f = 0.0
    d_l = 0.0
    if reference_front is not None:
        reference_front = _to_numpy_2d(reference_front)
        if reference_front.shape[0] > 0 and reference_front.shape[1] >= 2:
            ref_sort_indices = np.lexsort((reference_front[:, 1], reference_front[:, 0]))
            ref_sorted = reference_front[ref_sort_indices]
            d_f = float(np.linalg.norm(sorted_points[0] - ref_sorted[0]))
            d_l = float(np.linalg.norm(sorted_points[-1] - ref_sorted[-1]))

    denominator = d_f + d_l + (n_points - 1) * d_bar
    if denominator <= 0:
        return 0.0
    return float((d_f + d_l + interior_term) / denominator)


def get_pareto_front(checkpoint: dict, filter_f2: bool = True) -> tuple:
    """
    从checkpoint中提取帕累托前沿
    
    Args:
        checkpoint: 加载的checkpoint字典
        filter_f2: 是否过滤掉f2 < -0.2的个体，默认为True
        
    Returns:
        tuple: (pareto_x, pareto_y)，帕累托前沿的决策变量和目标值
    """
    train_x = checkpoint['train_x']
    train_obj_true = checkpoint['train_obj_true']
    
    # 提取帕累托最优解
    # 转换为numpy数组（如果是torch.Tensor）
    if isinstance(train_x, torch.Tensor):
        train_x_np = train_x.cpu().numpy()
    else:
        train_x_np = train_x
    
    if isinstance(train_obj_true, torch.Tensor):
        train_obj_np = train_obj_true.cpu().numpy()
    else:
        train_obj_np = train_obj_true
    
    # 过滤掉f2 < -0.2的个体（仅当filter_f2为True时）
    # 假设f2是第二个目标（索引为1）
    if filter_f2 and train_obj_np.shape[1] >= 2:
        # 创建掩码，只保留f2 >= -0.2的个体
        mask = train_obj_np[:, 1] >= -0.2
        # 应用掩码过滤个体
        train_x_np = train_x_np[mask]
        train_obj_np = train_obj_np[mask]
        print(f"过滤掉f2 < -0.2的个体，剩余 {len(train_obj_np)} 个个体")
    
    # 计算帕累托最优解的索引
    _, pareto_indices = get_pareto_optimal_points(train_obj_np, return_indices=True)
    
    # 根据索引获取帕累托最优解的决策变量和目标值
    pareto_x = train_x_np[pareto_indices]
    pareto_y = train_obj_np[pareto_indices]
    
    return pareto_x, pareto_y


def plot_metrics_by_dataset(checkpoint_paths: list, output_dir: str, legend_names: list = None, algorithm_name_mapping: dict = None, original_models_results: dict = None, plot_settings: dict = None):
    """
    在不同数据集上绘制checkpoint的指标，可以绘制多个checkpoint进行对比
    
    Args:
        checkpoint_paths: checkpoint文件路径列表
        output_dir: 输出目录
        legend_names: checkpoint数据的legend名称列表，与checkpoint_path一一对应，在可视化环节中优先使用这些名称
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置默认的绘图参数
    if plot_settings is None:
        plot_settings = {
            # 字体大小设置
            'font_size': 12,  # 基础字体大小
            'title_font_size': 14,  # 标题字体大小
            'label_font_size': 18,  # 标签字体大小
            'tick_font_size': 16,  # 刻度字体大小
            'legend_font_size': 16,  # 图例字体大小
            
            # 标记和线宽设置
            'marker_size': 8,  # 线图标记大小
            'scatter_size': 50,  # 散点图大小
            'line_width': 2,  # 线宽
            'grid_line_width': 1,  # 网格线宽
            'vector_line_width': 0.8,  # 偏好向量线宽
            
            # 箭头大小设置
            'arrow_size': 10,  # 箭头大小
        }
    
    # 将单个checkpoint路径转换为列表
    if isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]
    
    # 为不同checkpoint使用不同颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # 加载所有checkpoint并提取帕累托前沿和所有解数据
    all_pareto_data = []
    all_solutions_data = []  # 保存所有解的数据，包括非帕累托解
    num_objectives = None
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        # 加载checkpoint
        checkpoint = load_checkpoint(checkpoint_path)
        
        # 提取帕累托前沿
        pareto_x, pareto_y = get_pareto_front(checkpoint)
        
        # 记录目标函数数量
        if num_objectives is None:
            num_objectives = pareto_y.shape[1]
        
        # 优先使用用户提供的legend_names
        if legend_names and i < len(legend_names) and legend_names[i]:
            checkpoint_name = legend_names[i]
        else:
            # 获取checkpoint标识（从文件名或路径中提取）
            checkpoint_dir_name = os.path.basename(os.path.dirname(checkpoint_path))
            # 解析checkpoint名称，区分model-level和默认算法
            if 'model_level' in checkpoint_dir_name:
                # 对于model-level结果，提取fusion_method
                # 名称格式通常为：model_level_{fusion_method}
                if '_' in checkpoint_dir_name:
                    fusion_method = "-".join(checkpoint_dir_name.split('_')[3:]) if len(checkpoint_dir_name.split('_')) > 2 else 'unknown'
                    checkpoint_name = f"{fusion_method}"
                else:
                    checkpoint_name = checkpoint_dir_name
            else:
                # 对于默认算法，显示为BAMBO (ours)
                checkpoint_name = "BAMBO (ours)"
        
        # 应用算法名称映射
        if algorithm_name_mapping and checkpoint_name in algorithm_name_mapping:
            checkpoint_name = algorithm_name_mapping[checkpoint_name]
        
        # 保存帕累托数据
        all_pareto_data.append((checkpoint_name, pareto_y))
        
        # 获取所有解的目标值数据
        all_obj = None
        if 'train_obj_true' in checkpoint:
            all_obj = checkpoint['train_obj_true']
            # 转换为numpy数组
            if isinstance(all_obj, torch.Tensor):
                all_obj = all_obj.cpu().numpy()
            # 绘制all_solution系列时，不需要删去任何解，所以这里不做过滤
        
        # 保存所有解的数据
        if all_obj is not None:
            all_solutions_data.append((checkpoint_name, all_obj))
    
    # 添加原始模型到帕累托数据和所有解数据中
    if original_models_results:
        # 解析原始模型结果
        original_models_list = []
        if isinstance(original_models_results, dict):
            if 'original_models' in original_models_results:
                original_models_list = original_models_results['original_models']
            elif 'metrics' in original_models_results:
                original_models_list = [original_models_results]
        elif isinstance(original_models_results, list):
            original_models_list = original_models_results
        
        print(f"将原始模型添加到图表数据中: {len(original_models_list)} 个模型")
        
        # 获取原始模型的目标值
        for model_result in original_models_list:
            metrics = model_result.get('metrics', {})
            # 从metrics中提取目标值
            f1 = 0
            f2 = 0
            f3 = 0
            
            # 根据目标数量构建目标值数组
            if num_objectives >= 3:
                obj_values = np.array([[f1, f2, f3]])
            else:
                obj_values = np.array([[f1, f2]])
            
            # 获取模型名称
            model_path = model_result.get('model_path', '')
            model_name = model_result.get('model_name', '')
            
            # 应用算法名称映射
            display_name = model_path
            for original_path, mapped_name in algorithm_name_mapping.items():
                if original_path in model_path or original_path in model_name:
                    display_name = mapped_name
                    break
            
            # 如果没有映射到任何名称，使用模型名称或路径的最后一部分
            if 'models/' in display_name:
                display_name = display_name.split('models/')[-1]
            if '/' in display_name:
                display_name = display_name.split('/')[-1]
            
            # 添加到帕累托数据中（原始模型被视为帕累托解）
            all_pareto_data.append((display_name, obj_values))
            # 添加到所有解数据中
            all_solutions_data.append((display_name, obj_values))
    
    print(f"绘制 {len(all_pareto_data)} 个checkpoint的帕累托前沿，每个checkpoint包含 {len(all_pareto_data[0][1])} 个个体...")
    
    # 收集所有解的目标值，用于全局帕累托排序
    all_combined_obj = []
    for _, all_obj in all_solutions_data:
        all_combined_obj.append(all_obj)
    
    # 将所有解的目标值合并为一个数组
    if all_combined_obj:
        all_combined_obj = np.concatenate(all_combined_obj, axis=0)
        print(f"合并所有解的目标值，共 {all_combined_obj.shape[0]} 个个体")
        # 进行全局帕累托排序
        global_pareto_obj, global_pareto_indices = get_pareto_optimal_points(all_combined_obj, return_indices=True)
        print(f"全局帕累托解数量: {global_pareto_obj.shape[0]}")
    else:
        all_combined_obj = np.array([])
        global_pareto_indices = np.array([])
    
    # 绘制目标空间图，根据目标数量选择2D或3D
    if num_objectives >= 3:
        # 绘制3D目标空间
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制每个checkpoint的帕累托前沿点
        for i, (checkpoint_name, pareto_y) in enumerate(all_pareto_data):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            scatter = ax.scatter(pareto_y[:, 0], pareto_y[:, 1], pareto_y[:, 2], 
                      c=color, marker=marker, alpha=0.7, s=plot_settings['scatter_size'], label=checkpoint_name)
            
            # 为BAMBO(ours)添加编号标注
            if checkpoint_name == 'BAMBO (ours)' or checkpoint_name == 'BAMBO (ours)':
                for j, (f1, f2, f3) in enumerate(pareto_y):
                    # 在3D图中添加文本标注
                    ax.text(f1, f2, f3, str(j), fontsize=10, ha='center', va='bottom',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))
        
        # 设置坐标轴标签
        ax.set_xlabel('F1 (Reasoning)')
        ax.set_ylabel('F2 (Efficiency)')
        ax.set_zlabel('F3 (IFEval)')
        
        ax.legend()
        plt.savefig(os.path.join(output_dir, 'pareto_front_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制2D投影，根据目标数量调整子图数量
    if num_objectives == 2:
        # 2个目标时，只绘制f1 vs f2
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 绘制每个checkpoint的帕累托前沿点
        for i, (checkpoint_name, pareto_y) in enumerate(all_pareto_data):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            scatter = ax.scatter(pareto_y[:, 0], pareto_y[:, 1], c=color, marker=marker, 
                      alpha=0.7, s=plot_settings['scatter_size'], label=checkpoint_name)
            
            # 将同一个算法的帕累托点用直线连接起来
            # 先按x轴排序，确保连接顺序正确
            sorted_indices = np.argsort(pareto_y[:, 0])
            sorted_pareto_y = pareto_y[sorted_indices]
            ax.plot(sorted_pareto_y[:, 0], sorted_pareto_y[:, 1], c=color, alpha=0.8, linewidth=plot_settings['line_width'], zorder=1)
            
            # 检查每个帕累托解是否是全局帕累托解，如果是则标记出来
            for j, (f1, f2) in enumerate(pareto_y):
                current_obj = np.array([f1, f2])
                is_global_pareto = False
                # 遍历全局帕累托解，检查当前解是否在全局帕累托集合中
                for gp_obj in global_pareto_obj:
                    if np.allclose(current_obj, gp_obj, atol=1e-6):
                        is_global_pareto = True
                        break
                
                # 如果是全局帕累托解，添加一个红色圆圈标记
                if is_global_pareto:
                    circle = plt.Circle((f1, f2), 0.02, color='red', fill=False, linewidth=2, zorder=2)
                    ax.add_patch(circle)
                
                # 为BAMBO(ours)添加编号标注
                if checkpoint_name == 'BAMBO (ours)' or checkpoint_name == 'BAMBO (ours)':
                    y_range = max(pareto_y[:, 1]) - min(pareto_y[:, 1])
                    y_offset = y_range * 0.02
                    ax.text(
                        f1, 
                        f2 + y_offset, 
                        str(j), 
                        fontsize=plot_settings['font_size'] * 0.8, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                    )
        
        ax.set_xlabel('F1 (Reasoning)', fontsize=plot_settings['label_font_size'])
        ax.set_ylabel('F2 (Efficiency)', fontsize=plot_settings['label_font_size'])
        ax.grid(True, linewidth=plot_settings['grid_line_width'])
        ax.legend(fontsize=plot_settings['legend_font_size'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pareto_front_2d_projections.png'), dpi=300, bbox_inches='tight')
        plt.close()
    elif num_objectives >= 3:
        # 3个目标时，绘制f1 vs f2, f1 vs f3, f2 vs f3
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        # 绘制f1 vs f2
        for i, (checkpoint_name, pareto_y) in enumerate(all_pareto_data):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            scatter = axs[0].scatter(pareto_y[:, 0], pareto_y[:, 1], c=color, marker=marker, 
                         alpha=0.7, s=plot_settings['scatter_size'], label=checkpoint_name if i == 0 else "")
            
            # 将同一个算法的帕累托点用直线连接起来
            sorted_indices = np.argsort(pareto_y[:, 0])
            sorted_pareto_y = pareto_y[sorted_indices]
            axs[0].plot(sorted_pareto_y[:, 0], sorted_pareto_y[:, 1], c=color, alpha=0.8, linewidth=plot_settings['line_width'], zorder=1)
            
            # 为BAMBO(ours)添加编号标注
            if  checkpoint_name == 'BAMBO (ours)' or checkpoint_name == 'BAMBO (ours)':
                for j, (f1, f2) in enumerate(pareto_y[:, :2]):
                    y_range = max(pareto_y[:, 1]) - min(pareto_y[:, 1])
                    y_offset = y_range * 0.02
                    axs[0].text(
                        f1, 
                        f2 + y_offset, 
                        str(j), 
                        fontsize=plot_settings['font_size'] * 0.8, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                    )
        axs[0].set_xlabel('F1 (Reasoning)', fontsize=plot_settings['label_font_size'])
        axs[0].set_ylabel('F2 (Efficiency)', fontsize=plot_settings['label_font_size'])
        axs[0].grid(True, linewidth=plot_settings['grid_line_width'])
        
        # 绘制f1 vs f3
        for i, (checkpoint_name, pareto_y) in enumerate(all_pareto_data):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            scatter = axs[1].scatter(pareto_y[:, 0], pareto_y[:, 2], c=color, marker=marker, 
                         alpha=0.7, s=plot_settings['scatter_size'], label=checkpoint_name if i == 0 else "")
            
            # 将同一个算法的帕累托点用直线连接起来
            sorted_indices = np.argsort(pareto_y[:, 0])
            sorted_pareto_y = pareto_y[sorted_indices]
            axs[1].plot(sorted_pareto_y[:, 0], sorted_pareto_y[:, 2], c=color, alpha=0.8, linewidth=plot_settings['line_width'], zorder=1)
            
            # 为BAMBO(ours)添加编号标注
            if  i==0:
                for j, (f1, f3) in enumerate(zip(pareto_y[:, 0], pareto_y[:, 2])):
                    y_range = max(pareto_y[:, 2]) - min(pareto_y[:, 2])
                    y_offset = y_range * 0.02
                    axs[1].text(
                        f1, 
                        f3 + y_offset, 
                        str(j), 
                        fontsize=plot_settings['font_size'] * 0.8, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                    )
        axs[1].set_xlabel('F1 (Reasoning)', fontsize=plot_settings['label_font_size'])
        axs[1].set_ylabel('F3 (IFEval)', fontsize=plot_settings['label_font_size'])
        axs[1].grid(True, linewidth=plot_settings['grid_line_width'])
        
        # 绘制f2 vs f3
        for i, (checkpoint_name, pareto_y) in enumerate(all_pareto_data):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            scatter = axs[2].scatter(pareto_y[:, 1], pareto_y[:, 2], c=color, marker=marker, 
                         alpha=0.7, s=plot_settings['scatter_size'], label=checkpoint_name if i == 0 else "")
            
            # 为BAMBO(ours)添加编号标注
            if checkpoint_name == 'BAMBO (ours)' or checkpoint_name == 'BAMBO (ours)':
                for j, (f2, f3) in enumerate(zip(pareto_y[:, 1], pareto_y[:, 2])):
                    y_range = max(pareto_y[:, 2]) - min(pareto_y[:, 2])
                    y_offset = y_range * 0.02
                    axs[2].text(
                        f2, 
                        f3 + y_offset, 
                        str(j), 
                        fontsize=plot_settings['font_size'] * 0.8, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                    )
        axs[2].set_xlabel('F2 (Efficiency)', fontsize=plot_settings['label_font_size'])
        axs[2].set_ylabel('F3 (IFEval)', fontsize=plot_settings['label_font_size'])
        axs[2].grid(True, linewidth=plot_settings['grid_line_width'])
        
        # 只在第一个子图显示图例
        axs[0].legend(fontsize=plot_settings['legend_font_size'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pareto_front_2d_projections.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 准备指标文本内容
    metrics_text = []
    metrics_text.append("\n===== 帕累托前沿个体指标 =====")
    metrics_text.append(f"{'编号':<8} {'方法':<20} {'f1':<8} {'f2':<8} {'GPQA_ACC':<10} {'GPQA_TOKENS':<12} {'AIME25_ACC':<12} {'AIME25_TOKENS':<14} {'IS_PARETO':<10}")
    metrics_text.append("-" * 95)
    
    # 遍历所有帕累托数据，生成指标文本
    for i, (checkpoint_name, pareto_y) in enumerate(all_pareto_data):
        # 处理所有checkpoint
        print(f"正在处理checkpoint: {checkpoint_name}")
        # 获取对应checkpoint的metrics数据
        if i < len(checkpoint_paths):
            checkpoint = load_checkpoint(checkpoint_paths[i])
            
            # 获取所有评估过的个体的指标
            all_metrics = []
            
            # 检查checkpoint中是否有evaluated_solutions字段
            if 'evaluated_solutions' in checkpoint:
                eval_solutions = checkpoint['evaluated_solutions']
                if isinstance(eval_solutions, dict) and 'metrics' in eval_solutions:
                    all_metrics = eval_solutions['metrics']
                    print(f"从evaluated_solutions中获取到 {len(all_metrics)} 个指标")
            elif 'train_info' in checkpoint:
                all_metrics = checkpoint['train_info']
                print(f"从train_info中获取到 {len(all_metrics)} 个指标")
            else:
                print("checkpoint中没有找到metrics数据")
            
            # 获取帕累托前沿个体的索引
            if 'train_obj_true' in checkpoint:
                all_obj = checkpoint['train_obj_true']
                _, pareto_indices = get_pareto_optimal_points(all_obj, return_indices=True)
                print(f"找到 {len(pareto_indices)} 个帕累托前沿个体")
                print(f"pareto_y形状: {pareto_y.shape}, 包含 {len(pareto_y)} 个个体")
                
                # 遍历帕累托前沿个体，生成指标文本
                if all_metrics:
                    # 确保pareto_y和all_metrics的对应关系
                    for j in range(len(pareto_y)):
                        # 获取帕累托前沿个体的目标值（f1, f2）
                        if len(pareto_y[j]) >= 2:
                            f1 = pareto_y[j][0]
                            f2 = pareto_y[j][1]
                        else:
                            f1 = 0
                            f2 = 0
                        
                        # 获取对应的metrics数据
                        if j < len(pareto_indices):
                            idx = pareto_indices[j]
                            if idx < len(all_metrics):
                                metrics = all_metrics[idx]
                                
                                # 提取gpqa和aime25的指标
                                gpqa_acc = metrics.get('gpqa_diamond', {}).get('mean_acc', 0)
                                gpqa_tokens = metrics.get('gpqa_diamond', {}).get('mean_tokens_num', 0)
                                aime25_acc = metrics.get('aime25', {}).get('mean_acc', 0)
                                aime25_tokens = metrics.get('aime25', {}).get('mean_tokens_num', 0)
                                
                                # 检查当前个体是否是全局帕累托解
                                is_global_pareto = "no"
                                current_obj = np.array([f1, f2])
                                # 考虑浮点精度，使用阈值检查
                                if len(global_pareto_obj) > 0:
                                    # 检查当前点是否在全局帕累托集合中
                                    for gp_obj in global_pareto_obj:
                                        # 只比较前两个目标值（因为当前是2D情况）
                                        if np.allclose(current_obj, gp_obj[:2], atol=1e-6):
                                            is_global_pareto = "yes"
                                            break
                                
                                # 生成指标行，添加是否为全局帕累托解的标记
                                metric_line = f"{j:<8} {checkpoint_name:<20} {f1:<8.4f} {f2:<8.4f} {gpqa_acc:<10.4f} {gpqa_tokens:<12.0f} {aime25_acc:<12.4f} {aime25_tokens:<14.0f} {is_global_pareto:<10}"
                                metrics_text.append(metric_line)
                else:
                    print("没有metrics数据，跳过打印指标")
            else:
                print("checkpoint中没有train_obj_true字段")
    metrics_text.append("=" * 85)
    
    # 打印指标到控制台
    for line in metrics_text:
        print(line)
    
    # 将指标保存到txt文件
    metrics_file_path = os.path.join(output_dir, 'pareto_front_metrics.txt')
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        for line in metrics_text:
            f.write(line + '\n')
    print(f"\n帕累托前沿个体指标已保存到: {metrics_file_path}")
    
    # 绘制包含所有解的2D投影图（对于其他算法绘制所有解，对于evoBMI只绘制帕累托解）
    print("绘制包含所有解的2D投影图...")
    
    # 根据目标数量选择绘制方式
    if num_objectives == 2:
        # 2个目标时，只绘制F1 vs F2
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 首先绘制其他checkpoint的所有解（在底层）
        other_algorithms_color = '#7f7f7f'  # 灰色
        other_algorithms_marker = 'x'  # 交叉标记
        first_other = True  # 用于控制是否添加图例
        for i, (checkpoint_name, all_obj) in enumerate(all_solutions_data):
            if i > 0:
                # 对于其他checkpoint，绘制所有解
                if first_other:
                    # 第一次绘制其他checkpoint，添加图例
                    ax.scatter(all_obj[:, 0], all_obj[:, 1], c=other_algorithms_color, marker=other_algorithms_marker, 
                              alpha=0.5, s=plot_settings['scatter_size'] * 0.5, label="Other algorithms' solutions")
                    first_other = False
                else:
                    # 后续的checkpoint不添加图例
                    ax.scatter(all_obj[:, 0], all_obj[:, 1], c=other_algorithms_color, marker=other_algorithms_marker, 
                              alpha=0.5, s=plot_settings['scatter_size'] * 0.5)
        
        # 绘制每个算法的所有解
        for i, (checkpoint_name, all_obj) in enumerate(all_solutions_data):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # 检查是否为原始模型
            is_original = False
            if any(keyword in checkpoint_name.lower() for keyword in ['qwen3-4b', 'original', 'base', 'expert']):
                is_original = True
            
            if i==0:
                # 对于BAMBO，绘制所有解（非帕累托解用浅色，帕累托解用深色）
                # 从all_solutions_data中找到对应的所有解
                for name, all_solutions in all_solutions_data:
                    if name == checkpoint_name or name == 'BAMBO (ours)':
                        # 获取帕累托解的索引
                        all_obj_np = all_solutions
                        pareto_obj_np = None
                        for pn, py in all_pareto_data:
                            if pn == checkpoint_name or pn == 'BAMBO (ours)':
                                pareto_obj_np = py
                                break
                        
                        if pareto_obj_np is not None:
                             # 找出非帕累托解
                             is_pareto_mask = np.zeros(len(all_obj_np), dtype=bool)
                             for pareto_pt in pareto_obj_np:
                                 for j in range(len(all_obj_np)):
                                     if not is_pareto_mask[j] and np.allclose(all_obj_np[j], pareto_pt, atol=1e-6):
                                         is_pareto_mask[j] = True
                             
                             # 绘制非帕累托解
                             non_pareto_obj = all_obj_np[~is_pareto_mask]
                             if len(non_pareto_obj) > 0:
                                 ax.scatter(non_pareto_obj[:, 0], non_pareto_obj[:, 1], c=color, marker=marker, 
                                           alpha=0.3, s=plot_settings['scatter_size'], label='non-Pareto solutions (Ours)')
                             
                             # 绘制帕累托解
                             ax.scatter(pareto_obj_np[:, 0], pareto_obj_np[:, 1], c=color, marker=marker, 
                                       alpha=0.7, s=plot_settings['scatter_size'], label='Pareto solutions (Ours)')
                        
                        # 生成偏好向量并绘制向量箭头
                        # 使用帕累托解的范围来计算偏好向量箭头
                        num_vectors = 5
                        obj_min = np.min(pareto_obj_np, axis=0)
                        obj_max = np.max(pareto_obj_np, axis=0)
                        obj_range = obj_max - obj_min
                        
                        # 生成均匀分布的偏好向量
                        angles = np.linspace(0, np.pi/2, num_vectors, endpoint=True)
                        preference_vectors = np.zeros((num_vectors, 2))
                        preference_vectors[:, 0] = np.cos(angles)
                        preference_vectors[:, 1] = np.sin(angles)
                        preference_vectors = np.abs(preference_vectors)
                        norms = np.linalg.norm(preference_vectors, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1e-10, norms)
                        preference_vectors = preference_vectors / norms
                        
                        # 定义颜色列表
                        vector_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                        
                        # 绘制偏好向量箭头
                        for i, vec in enumerate(preference_vectors):
                            color = vector_colors[i % len(vector_colors)]
                            theta = np.arctan2(vec[1], vec[0])
                            start_x = obj_min[0]
                            start_y = obj_min[1]
                            end_x = start_x + obj_range[0] * np.cos(theta)
                            end_y = start_y + obj_range[1] * np.sin(theta)
                            ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y, 
                                     head_width=0.01, head_length=0.01,
                                     fc=color, ec=color, linewidth=plot_settings['vector_line_width'], alpha=0.5)
                        
                        # 计算每个帕累托个体与偏好向量的余弦相似度，选择最接近的个体
                        selected_indices = []
                        selected_vec_indices = []
                        individual_to_vecs = {}
                        normalized_pareto_y = (pareto_obj_np - obj_min) / (obj_range + 1e-10)
                        
                        for i, pref_vec in enumerate(preference_vectors):
                            cos_similarities = []
                            for j, norm_obj_values in enumerate(normalized_pareto_y):
                                norm_obj_values_norm = norm_obj_values / (np.linalg.norm(norm_obj_values) + 1e-10)
                                similarity = np.dot(norm_obj_values_norm, pref_vec)
                                cos_similarities.append((similarity, j))
                            cos_similarities.sort(key=lambda x: x[0], reverse=True)
                            if cos_similarities:
                                best_idx = cos_similarities[0][1]
                                selected_indices.append(best_idx)
                                selected_vec_indices.append(i)
                                if best_idx not in individual_to_vecs:
                                    individual_to_vecs[best_idx] = []
                                individual_to_vecs[best_idx].append(i)
                        
                        # 绘制选中的个体为红色
                        first_selected = True
                        for sol_idx in individual_to_vecs:
                            sol_x, sol_y = pareto_obj_np[sol_idx]
                            if first_selected:
                                # 只为第一个选中的个体添加图例
                                ax.scatter(sol_x, sol_y, c='red', marker=marker, 
                                          alpha=1.0, s=plot_settings['scatter_size'], label='Selected solutions (Ours)')
                                first_selected = False
                            else:
                                # 其他选中的个体不添加图例
                                ax.scatter(sol_x, sol_y, c='red', marker=marker, 
                                          alpha=1.0, s=plot_settings['scatter_size'])
                            
                            # 为选中的个体添加相应颜色向量的圆框
                            vec_indices = individual_to_vecs[sol_idx]
                            base_size = plot_settings['scatter_size'] * 2.0
                            increment = base_size * 0.6
                            for i, vec_idx in enumerate(vec_indices):
                                vector_color = vector_colors[vec_idx % len(vector_colors)]
                                circle_size = base_size + i * increment
                                ax.scatter(sol_x, sol_y, c='none', marker='o', s=circle_size, 
                                          edgecolors=vector_color, linewidth=2, alpha=0.7)
                        
                        break
            elif is_original:
                # 对于原始模型，使用特殊标记和颜色
                ax.scatter(all_obj[:, 0], all_obj[:, 1], c='red', marker='^', 
                          alpha=0.8, s=plot_settings['scatter_size'], label=f"{checkpoint_name} (Original)")
            else:
                pass
                # 对于其他算法，绘制所有解
                # ax.scatter(all_obj[:, 0], all_obj[:, 1], c=color, marker=marker, 
                #          alpha=0.5, s=plot_settings['scatter_size'], label=f"{checkpoint_name} (All Solutions)")
        
        ax.set_xlabel('F1 (Reasoning)', fontsize=plot_settings['label_font_size'])
        ax.set_ylabel('F2 (Efficiency)', fontsize=plot_settings['label_font_size'])
        ax.grid(True, linewidth=plot_settings['grid_line_width'])
        ax.legend(fontsize=plot_settings['legend_font_size'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_solutions_2d_projections.png'), dpi=300, bbox_inches='tight')
        plt.close()
    elif num_objectives >= 3:
        # 3个目标时，绘制f1 vs f2, f1 vs f3, f2 vs f3
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        # 绘制f1 vs f2
        for i, (checkpoint_name, pareto_y) in enumerate(all_pareto_data):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            scatter = axs[0].scatter(pareto_y[:, 0], pareto_y[:, 1], c=color, marker=marker, 
                         alpha=0.7, s=60, label=checkpoint_name if i == 0 else "")
            
            # 为BAMBO(ours)添加编号标注
            if checkpoint_name == 'BAMBO (ours)' or checkpoint_name == 'BAMBO (ours)':
                for j, (f1, f2) in enumerate(pareto_y[:, :2]):
                    y_range = max(pareto_y[:, 1]) - min(pareto_y[:, 1])
                    y_offset = y_range * 0.02
                    axs[0].text(
                        f1, 
                        f2 + y_offset, 
                        str(j), 
                        fontsize=10, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                    )
        axs[0].set_xlabel('F1 (Reasoning)')
        axs[0].set_ylabel('F2 (Efficiency)')
        axs[0].grid(True)
        
        # 绘制f1 vs f3
        for i, (checkpoint_name, pareto_y) in enumerate(all_pareto_data):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            scatter = axs[1].scatter(pareto_y[:, 0], pareto_y[:, 2], c=color, marker=marker, 
                         alpha=0.7, s=60, label=checkpoint_name if i == 0 else "")
            
            # 为BAMBO(ours)添加编号标注
            if checkpoint_name == 'BAMBO (ours)' or checkpoint_name == 'BAMBO (ours)':
                for j, (f1, f3) in enumerate(zip(pareto_y[:, 0], pareto_y[:, 2])):
                    y_range = max(pareto_y[:, 2]) - min(pareto_y[:, 2])
                    y_offset = y_range * 0.02
                    axs[1].text(
                        f1, 
                        f3 + y_offset, 
                        str(j), 
                        fontsize=10, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                    )
        axs[1].set_xlabel('F1 (Reasoning)')
        axs[1].set_ylabel('F3 (IFEval)')
        axs[1].grid(True)
        
        # 绘制f2 vs f3
        for i, (checkpoint_name, pareto_y) in enumerate(all_pareto_data):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            scatter = axs[2].scatter(pareto_y[:, 1], pareto_y[:, 2], c=color, marker=marker, 
                         alpha=0.7, s=60, label=checkpoint_name if i == 0 else "")
            
            # 为BAMBO(ours)添加编号标注
            if checkpoint_name == 'BAMBO (ours)' or checkpoint_name == 'BAMBO (ours)':
                for j, (f2, f3) in enumerate(zip(pareto_y[:, 1], pareto_y[:, 2])):
                    y_range = max(pareto_y[:, 2]) - min(pareto_y[:, 2])
                    y_offset = y_range * 0.02
                    axs[2].text(
                        f2, 
                        f3 + y_offset, 
                        str(j), 
                        fontsize=10, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                    )
        axs[2].set_xlabel('F2 (Efficiency)')
        axs[2].set_ylabel('F3 (IFEval)')
        axs[2].grid(True)
        
        # 只在第一个子图显示图例
        axs[0].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pareto_front_2d_projections.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 新增：调用visualizer.plot_dataset_metrics绘制数据集指标
    # 准备metrics_list数据，与model_reproduction.py中的格式保持一致
    all_metrics_list = []
    all_solutions_metrics_list = []  # 保存所有解的数据，用于绘制包含所有解的图表
    
    # 遍历所有checkpoint，提取metrics数据
    # 只处理原始checkpoint数据，不包括我们添加的原始模型数据
    # 注意：all_pareto_data包含了原始checkpoint数据和我们添加的原始模型数据
    # 所以我们需要单独处理原始checkpoint数据
    for i in range(len(checkpoint_paths)):
        # 获取checkpoint路径
        checkpoint_path = checkpoint_paths[i]
        # 加载checkpoint
        checkpoint = load_checkpoint(checkpoint_path)
        # 从all_pareto_data中获取对应的checkpoint名称
        checkpoint_name = all_pareto_data[i][0]
        
        # 首先获取所有评估过的个体的指标
        all_metrics = []
        
        # 检查checkpoint中是否有evaluated_solutions字段
        if 'evaluated_solutions' in checkpoint:
            eval_solutions = checkpoint['evaluated_solutions']
            # 从evaluated_solutions中提取metrics
            if isinstance(eval_solutions, dict) and 'metrics' in eval_solutions:
                all_metrics = eval_solutions['metrics']
        elif 'train_info' in checkpoint:
            # 兼容旧版本的checkpoint格式，直接从train_info获取metrics
            all_metrics = checkpoint['train_info']
        
        # 然后获取帕累托前沿个体的索引
        # 从checkpoint中获取所有评估点的真实目标函数值
        if 'train_obj_true' in checkpoint and all_metrics:
            all_obj = checkpoint['train_obj_true']
            
            # 转换为numpy数组
            if isinstance(all_obj, torch.Tensor):
                all_obj_np = all_obj.cpu().numpy()
            else:
                all_obj_np = all_obj
            
            # 过滤掉f2 < -0.2的个体
            if all_obj_np.shape[1] >= 2:
                # 创建掩码，只保留f2 >= -0.2的个体
                mask = all_obj_np[:, 1] >= -0.2
                
                # 检查是否有符合条件的个体
                if np.any(mask):
                    # 应用掩码过滤个体
                    all_obj = all_obj[mask]
                    all_metrics = [all_metrics[i] for i, keep in enumerate(mask) if keep]
                    print(f"过滤掉f2 < -0.2的个体，剩余 {len(all_metrics)} 个个体")
                else:
                    # 如果没有符合条件的个体，跳过
                    print("没有找到f2 >= -0.2的个体，跳过该checkpoint")
                    continue
            
            # 计算所有评估点的帕累托最优解，返回索引
            _, pareto_indices = get_pareto_optimal_points(all_obj, return_indices=True)
            
            # 只保留帕累托前沿个体的指标
            for j, idx in enumerate(pareto_indices):
                if idx < len(all_metrics):
                    metrics = all_metrics[idx]
                    # 添加checkpoint名称作为类型标识
                    # 应用算法名称映射
                    display_name = checkpoint_name
                    if algorithm_name_mapping and checkpoint_name in algorithm_name_mapping:
                        display_name = algorithm_name_mapping[checkpoint_name]
                    
                    # 设置type字段，用于区分解决方案模型和原始模型
                    model_type = 'solution'  # 默认所有checkpoint中的模型都是解决方案模型
                    
                    metrics_list_entry = {
                        'type': display_name,
                        'model_type': model_type,  # 添加此字段用于绘图区分
                        'index': j,
                        **metrics
                    }
                    all_metrics_list.append(metrics_list_entry)
            
            # 保存所有解的数据（包括非帕累托解）
            # 对于evoBMI，只添加帕累托解；对于其他算法，添加所有解
            # 应用算法名称映射
            display_name = checkpoint_name
            if algorithm_name_mapping and checkpoint_name in algorithm_name_mapping:
                display_name = algorithm_name_mapping[checkpoint_name]
            
            if checkpoint_name == 'BAMBO (ours)':
                # 对于evoBMI，只添加帕累托解
                for j, idx in enumerate(pareto_indices):
                    if idx < len(all_metrics):
                        metrics = all_metrics[idx]
                        metrics_list_entry = {
                            'type': display_name,
                            'index': j,
                            **metrics
                        }
                        all_solutions_metrics_list.append(metrics_list_entry)
            else:
                # 对于其他算法，添加所有解
                for j, metrics in enumerate(all_metrics):
                    metrics_list_entry = {
                        'type': display_name,
                        'index': j,
                        **metrics
                    }
                    all_solutions_metrics_list.append(metrics_list_entry)
        
        # 处理model-level的情况，直接从metrics中获取数据
        elif 'metrics' in checkpoint and isinstance(checkpoint['metrics'], list):
            # 应用算法名称映射
            display_name = checkpoint_name
            if algorithm_name_mapping and checkpoint_name in algorithm_name_mapping:
                display_name = algorithm_name_mapping[checkpoint_name]
            
            # 处理model-level的帕累托解
            for j, metrics in enumerate(checkpoint['metrics']):
                metrics_list_entry = {
                    'type': display_name,
                    'index': j,
                    **metrics
                }
                all_metrics_list.append(metrics_list_entry)
            
            # 对于model-level结果，添加所有解（假设model-level结果已经是所有解）
            for j, metrics in enumerate(checkpoint['metrics']):
                metrics_list_entry = {
                    'type': display_name,
                    'index': j,
                    **metrics
                }
                all_solutions_metrics_list.append(metrics_list_entry)
    
    # 添加原始模型结果到metrics_list中
    if original_models_results:
        # 检查original_models_results的格式
        if isinstance(original_models_results, dict):
            # 格式1：包含original_models字段的字典
            if 'original_models' in original_models_results:
                original_models_list = original_models_results['original_models']
                for model_result in original_models_list:
                    if 'metrics' in model_result:
                        # 为原始模型创建metrics_list_entry
                        model_name = os.path.basename(model_result.get('model_path', 'Original Model'))
                        # 应用算法名称映射
                        display_name = model_name
                        if algorithm_name_mapping and model_name in algorithm_name_mapping:
                            display_name = algorithm_name_mapping[model_name]
                        
                        metrics_list_entry = {
                            'type': display_name,
                            'model_type': 'original',  # 明确标记为原始模型
                            'index': len(all_metrics_list),
                            **model_result['metrics']
                        }
                        all_metrics_list.append(metrics_list_entry)
                        # 同时添加到所有解列表中
                        all_solutions_metrics_list.append(metrics_list_entry)
            elif 'metrics' in original_models_results:
                # 格式2：直接包含metrics的字典（单个模型）
                model_name = os.path.basename(original_models_results.get('model_path', 'Original Model'))
                # 应用算法名称映射
                display_name = model_name
                if algorithm_name_mapping and model_name in algorithm_name_mapping:
                    display_name = algorithm_name_mapping[model_name]
                
                metrics_list_entry = {
                    'type': display_name,
                    'model_type': 'original',  # 明确标记为原始模型
                    'index': len(all_metrics_list),
                    **original_models_results['metrics']
                }
                all_metrics_list.append(metrics_list_entry)
                # 同时添加到所有解列表中
                all_solutions_metrics_list.append(metrics_list_entry)
        elif isinstance(original_models_results, list):
            # 格式3：直接是原始模型结果列表
            for model_result in original_models_results:
                if 'metrics' in model_result:
                    model_name = os.path.basename(model_result.get('model_path', 'Original Model'))
                    # 应用算法名称映射
                    display_name = model_name
                    if algorithm_name_mapping and model_name in algorithm_name_mapping:
                        display_name = algorithm_name_mapping[model_name]
                    
                    metrics_list_entry = {
                        'type': display_name,
                        'model_type': 'original',  # 明确标记为原始模型
                        'index': len(all_metrics_list),
                        **model_result['metrics']
                    }
                    all_metrics_list.append(metrics_list_entry)
                    # 同时添加到所有解列表中
                    all_solutions_metrics_list.append(metrics_list_entry)
    
    # 自定义实现数据集指标绘制，替代visualizer.plot_dataset_metrics
    if all_metrics_list:
        
        # 准备数据
        solution_data = {}
        other_methods = set()
        
        for metrics in all_metrics_list:
            method = metrics['type']
            other_methods.add(method)
            
            if method not in solution_data:
                solution_data[method] = []
            solution_data[method].append(metrics)
        
        # 绘制合并的tokens_vs_acc图，包含两个正方形子图
        # 创建一个尺寸与原来一张图相同的figure，使用正方形布局
        # 原来的单张图尺寸是10x6，现在我们创建2x2的grid，但只使用前两个子图，这样每个子图都是正方形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        
        # 定义颜色和标记
        solution_colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
        solution_markers = ['o', 's', '^', 'D', 'v', '<', '>']
        original_color = 'red'  # 原始模型使用红色
        original_marker = '^'  # 原始模型使用三角形标记
        
        # 分离解决方案模型和原始模型的数据
        all_data = []
        for method, data in solution_data.items():
            for d in data:
                # 检查是否有原始模型标记
                is_original = False
                # 优先检查model_type字段
                if d.get('model_type') == 'original':
                    is_original = True
                # 然后检查方法名称
                elif any(keyword in method.lower() for keyword in ['qwen3-4b', 'original', 'base', 'expert']):
                    is_original = True
                
                all_data.append({
                    'method': method,
                    'data': d,
                    'is_original': is_original
                })
        
        # 绘制aime25_tokens_vs_acc子图
        
        # 绘制解决方案模型
        solution_plotted = set()
        for item in all_data:
            if not item['is_original']:
                method = item['method']
                d = item['data']
                if 'aime25' in d and isinstance(d['aime25'], dict):
                    aime25_dict = d['aime25']
                    if 'mean_tokens_num' in aime25_dict and 'mean_acc' in aime25_dict:
                        tokens = aime25_dict['mean_tokens_num']
                        acc = aime25_dict['mean_acc']
                        
                        if method not in solution_plotted:
                            # 第一次绘制该方法，添加图例
                            i = len(solution_plotted)
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            ax1.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'], label=method)
                            solution_plotted.add(method)
                        else:
                            # 已经绘制过该方法，不添加图例
                            i = len(solution_plotted) - 1  # 使用最后一个颜色
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            ax1.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'])
                        
                        # 为evoBMI(ours)添加编号标注
                        if method == 'BAMBO (ours)' or method == 'BMI (ours)':
                            y_offset = 0.01  # 固定偏移量
                            ax1.text(
                                tokens, 
                                acc + y_offset, 
                                str(d.get('index', '')), 
                                ha='center', 
                                va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                            )
        
        # 绘制原始模型
        original_plotted = set()
        for item in all_data:
            if item['is_original']:
                method = item['method']
                d = item['data']
                if 'aime25' in d and isinstance(d['aime25'], dict):
                    aime25_dict = d['aime25']
                    if 'mean_tokens_num' in aime25_dict and 'mean_acc' in aime25_dict:
                        tokens = aime25_dict['mean_tokens_num']
                        acc = aime25_dict['mean_acc']
                        
                        if method not in original_plotted:
                            # 第一次绘制该原始模型，添加图例
                            ax1.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'], label=f'{method} (Original)')
                            original_plotted.add(method)
                        else:
                            # 已经绘制过该原始模型，不添加图例
                            ax1.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'])
                        
                        # 为原始模型添加名称标注
                        ax1.text(
                            tokens, 
                            acc, 
                            method.split('/')[-1],  # 只显示模型名称的最后一部分
                            ha='center', 
                            va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                        )
        
        # 设置aime25子图属性
        ax1.set_xlabel('Mean Tokens Count (AIME25)', fontsize=plot_settings['label_font_size'])
        ax1.set_ylabel('Accuracy', fontsize=plot_settings['label_font_size'])
        ax1.grid(True, alpha=0.3, linewidth=plot_settings['grid_line_width'])
        ax1.legend(fontsize=plot_settings['legend_font_size'])
        ax1.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
        
        # 绘制gpqa_tokens_vs_acc子图
        
        # 绘制解决方案模型
        solution_plotted = set()
        for item in all_data:
            if not item['is_original']:
                method = item['method']
                d = item['data']
                if 'gpqa_diamond' in d and isinstance(d['gpqa_diamond'], dict):
                    gpqa_dict = d['gpqa_diamond']
                    if 'mean_tokens_num' in gpqa_dict and 'mean_acc' in gpqa_dict:
                        tokens = gpqa_dict['mean_tokens_num']
                        acc = gpqa_dict['mean_acc']
                        
                        if method not in solution_plotted:
                            # 第一次绘制该方法，添加图例
                            i = len(solution_plotted)
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            ax2.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'], label=method)
                            solution_plotted.add(method)
                        else:
                            # 已经绘制过该方法，不添加图例
                            i = len(solution_plotted) - 1
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            ax2.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'])
                        
                        # 为evoBMI(ours)添加编号标注
                        if method == 'BAMBO (ours)' or method == 'BMI (ours)':
                            y_offset = 0.01  # 固定偏移量
                            ax2.text(
                                tokens, 
                                acc + y_offset, 
                                str(d.get('index', '')), 
                                fontsize=10, 
                                ha='center', 
                                va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                            )
        
        # 绘制原始模型
        original_plotted = set()
        for item in all_data:
            if item['is_original']:
                method = item['method']
                d = item['data']
                if 'gpqa_diamond' in d and isinstance(d['gpqa_diamond'], dict):
                    gpqa_dict = d['gpqa_diamond']
                    if 'mean_tokens_num' in gpqa_dict and 'mean_acc' in gpqa_dict:
                        tokens = gpqa_dict['mean_tokens_num']
                        acc = gpqa_dict['mean_acc']
                        
                        if method not in original_plotted:
                            # 第一次绘制该原始模型，添加图例
                            ax2.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'], label=f'{method} (Original)')
                            original_plotted.add(method)
                        else:
                            # 已经绘制过该原始模型，不添加图例
                            ax2.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'])
                        
                        # 为原始模型添加名称标注
                        ax2.text(
                            tokens, 
                            acc, 
                            method.split('/')[-1],  # 只显示模型名称的最后一部分
                            fontsize=9, 
                            ha='center', 
                            va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                        )
        
        # 设置gpqa子图属性
        ax2.set_xlabel('Mean Tokens Used (GPQA-Diamond)', fontsize=plot_settings['label_font_size'])
        ax2.set_ylabel('Accuracy', fontsize=plot_settings['label_font_size'])
        ax2.grid(True, alpha=0.3, linewidth=plot_settings['grid_line_width'])
        ax2.legend(fontsize=plot_settings['legend_font_size'])
        ax2.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
        
        # 保存合并后的图表
        output_path = os.path.join(output_dir, "combined_tokens_vs_acc.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"合并的Tokens vs Accuracy图已保存到: {output_path}")
        
        # 关闭主图
        plt.close(fig)
        
        # 重新绘制并保存单独的图表，保持向后兼容
        # 绘制aime25_tokens_vs_acc图
        plt.figure(figsize=(10, 6))
        
        # 绘制解决方案模型
        solution_plotted = set()
        for item in all_data:
            if not item['is_original']:
                method = item['method']
                d = item['data']
                if 'aime25' in d and isinstance(d['aime25'], dict):
                    aime25_dict = d['aime25']
                    if 'mean_tokens_num' in aime25_dict and 'mean_acc' in aime25_dict:
                        tokens = aime25_dict['mean_tokens_num']
                        acc = aime25_dict['mean_acc']
                        
                        if method not in solution_plotted:
                            # 第一次绘制该方法，添加图例
                            i = len(solution_plotted)
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            plt.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'], label=method)
                            solution_plotted.add(method)
                        else:
                            # 已经绘制过该方法，不添加图例
                            i = len(solution_plotted) - 1  # 使用最后一个颜色
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            plt.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'])
                        
                        # 为evoBMI(ours)添加编号标注
                        if method == 'BAMBO (ours)' or method == 'BMI (ours)':
                            y_offset = 0.01  # 固定偏移量
                            plt.text(
                                tokens, 
                                acc + y_offset, 
                                str(d.get('index', '')), 
                                ha='center', 
                                va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                            )
                            
        
        # 绘制原始模型
        original_plotted = set()
        for item in all_data:
            if item['is_original']:
                method = item['method']
                d = item['data']
                if 'aime25' in d and isinstance(d['aime25'], dict):
                    aime25_dict = d['aime25']
                    if 'mean_tokens_num' in aime25_dict and 'mean_acc' in aime25_dict:
                        tokens = aime25_dict['mean_tokens_num']
                        acc = aime25_dict['mean_acc']
                        
                        if method not in original_plotted:
                            # 第一次绘制该原始模型，添加图例
                            plt.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'], label=f'{method} (Original)')
                            original_plotted.add(method)
                        else:
                            # 已经绘制过该原始模型，不添加图例
                            plt.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'])
                        
                        # 为原始模型添加名称标注
                        plt.text(
                            tokens, 
                            acc, 
                            method.split('/')[-1],  # 只显示模型名称的最后一部分
                            ha='center', 
                            va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                        )
        
        # 设置图表属性
        plt.xlabel('AIME 25 Mean Tokens Used', fontsize=plot_settings['label_font_size'])
        plt.ylabel('Accuracy', fontsize=plot_settings['label_font_size'])
        plt.grid(True, alpha=0.3, linewidth=plot_settings['grid_line_width'])
        plt.legend(fontsize=plot_settings['legend_font_size'])
        plt.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, "aime25_tokens_vs_acc.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"AIME25 Tokens vs Accuracy图已保存到: {output_path}")
        
        # 绘制gpqa_tokens_vs_acc图
        plt.figure(figsize=(10, 6))
        
        # 绘制解决方案模型
        solution_plotted = set()
        for item in all_data:
            if not item['is_original']:
                method = item['method']
                d = item['data']
                if 'gpqa_diamond' in d and isinstance(d['gpqa_diamond'], dict):
                    gpqa_dict = d['gpqa_diamond']
                    if 'mean_tokens_num' in gpqa_dict and 'mean_acc' in gpqa_dict:
                        tokens = gpqa_dict['mean_tokens_num']
                        acc = gpqa_dict['mean_acc']
                        
                        if method not in solution_plotted:
                            # 第一次绘制该方法，添加图例
                            i = len(solution_plotted)
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            plt.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'], label=method)
                            solution_plotted.add(method)
                        else:
                            # 已经绘制过该方法，不添加图例
                            i = len(solution_plotted) - 1
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            plt.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'])
                        
                        # 为evoBMI(ours)添加编号标注
                        if method == 'BAMBO (ours)' or method == 'BMI (ours)':
                            y_offset = 0.01  # 固定偏移量
                            plt.text(
                                tokens, 
                                acc + y_offset, 
                                str(d.get('index', '')), 
                                fontsize=10, 
                                ha='center', 
                                va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                            )
        
        # 绘制原始模型
        original_plotted = set()
        for item in all_data:
            if item['is_original']:
                method = item['method']
                d = item['data']
                if 'gpqa_diamond' in d and isinstance(d['gpqa_diamond'], dict):
                    gpqa_dict = d['gpqa_diamond']
                    if 'mean_tokens_num' in gpqa_dict and 'mean_acc' in gpqa_dict:
                        tokens = gpqa_dict['mean_tokens_num']
                        acc = gpqa_dict['mean_acc']
                        
                        if method not in original_plotted:
                            # 第一次绘制该原始模型，添加图例
                            plt.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'], label=f'{method} (Original)')
                            original_plotted.add(method)
                        else:
                            # 已经绘制过该原始模型，不添加图例
                            plt.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'])
                        
                        # 为原始模型添加名称标注
                        plt.text(
                            tokens, 
                            acc, 
                            method.split('/')[-1],  # 只显示模型名称的最后一部分
                            fontsize=9, 
                            ha='center', 
                            va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                        )
        
        # 设置图表属性
        plt.xlabel('GPQA Diamond Mean Tokens Used', fontsize=plot_settings['label_font_size'])
        plt.ylabel('Accuracy', fontsize=plot_settings['label_font_size'])
        plt.grid(True, alpha=0.3, linewidth=plot_settings['grid_line_width'])
        plt.legend(fontsize=plot_settings['legend_font_size'])
        plt.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, "gpqa_tokens_vs_acc.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"GPQA Diamond Tokens vs Accuracy图已保存到: {output_path}")
        
        # 检查是否需要绘制ifeval相关内容（根据目标数量决定）
        # 从帕累托数据中判断目标数量
        num_objectives = 3  # 默认3个目标
        
        # 查找包含f3的checkpoint，判断是否有3个目标
        has_ifeval = False
        for data in all_metrics_list:
            if 'f3' in data or 'ifeval' in data:
                has_ifeval = True
                break
        
        # 如果有3个目标（包含ifeval），绘制相关图表
        if has_ifeval:
            # 绘制f1 vs f3图
            plt.figure(figsize=(10, 6))
            
            # 定义颜色和标记（与之前的图保持一致）
            solution_colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
            solution_markers = ['o', 's', '^', 'D', 'v', '<', '>']
            original_color = 'red'  # 原始模型使用红色
            original_marker = '^'  # 原始模型使用三角形标记
            
            # 分离解决方案模型和原始模型的数据
            all_data = []
            for method, data in solution_data.items():
                for d in data:
                    # 检查是否有原始模型标记
                    is_original = False
                    if any(keyword in method.lower() for keyword in ['qwen3-4b', 'original', 'base', 'expert']):
                        is_original = True
                    
                    all_data.append({
                        'method': method,
                        'data': d,
                        'is_original': is_original
                    })
            
            # 绘制解决方案模型
            solution_plotted = set()
            for item in all_data:
                if not item['is_original']:
                    method = item['method']
                    d = item['data']
                    
                    # 提取f1和f3数值
                    f1_val = None
                    f3_val = None
                    
                    if 'f1' in d:
                        f1_val = d['f1']
                        # 处理可能的嵌套结构
                        if isinstance(f1_val, dict):
                            if 'score' in f1_val:
                                f1_val = f1_val['score']
                            elif 'value' in f1_val:
                                f1_val = f1_val['value']
                            else:
                                for v in f1_val.values():
                                    if isinstance(v, (int, float)):
                                        f1_val = v
                                        break
                    
                    if 'f3' in d:
                        f3_val = d['f3']
                        # 处理可能的嵌套结构
                        if isinstance(f3_val, dict):
                            if 'score' in f3_val:
                                f3_val = f3_val['score']
                            elif 'value' in f3_val:
                                f3_val = f3_val['value']
                            else:
                                for v in f3_val.values():
                                    if isinstance(v, (int, float)):
                                        f3_val = v
                                        break
                    
                    if f1_val is not None and f3_val is not None and isinstance(f1_val, (int, float)) and isinstance(f3_val, (int, float)):
                        if method not in solution_plotted:
                            # 第一次绘制该方法，添加图例
                            i = len(solution_plotted)
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            plt.scatter([f1_val], [f3_val], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'], label=method)
                            solution_plotted.add(method)
                        else:
                            # 已经绘制过该方法，不添加图例
                            i = len(solution_plotted) - 1
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            plt.scatter([f1_val], [f3_val], c=color, marker=marker, 
                                      alpha=0.7, s=60)
                        
                        # 为evoBMI(ours)添加编号标注
                        if method == 'BAMBO (ours)' or method == 'BMI (ours)':
                            y_offset = 0.01  # 固定偏移量
                            plt.text(
                                f1_val, 
                                f3_val + y_offset, 
                                str(d.get('index', '')), 
                                fontsize=10, 
                                ha='center', 
                                va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                            )
            
            # 绘制原始模型
            original_plotted = set()
            for item in all_data:
                if item['is_original']:
                    method = item['method']
                    d = item['data']
                    
                    # 提取f1和f3数值
                    f1_val = None
                    f3_val = None
                    
                    if 'f1' in d:
                        f1_val = d['f1']
                        # 处理可能的嵌套结构
                        if isinstance(f1_val, dict):
                            if 'score' in f1_val:
                                f1_val = f1_val['score']
                            elif 'value' in f1_val:
                                f1_val = f1_val['value']
                            else:
                                for v in f1_val.values():
                                    if isinstance(v, (int, float)):
                                        f1_val = v
                                        break
                    
                    if 'f3' in d:
                        f3_val = d['f3']
                        # 处理可能的嵌套结构
                        if isinstance(f3_val, dict):
                            if 'score' in f3_val:
                                f3_val = f3_val['score']
                            elif 'value' in f3_val:
                                f3_val = f3_val['value']
                            else:
                                for v in f3_val.values():
                                    if isinstance(v, (int, float)):
                                        f3_val = v
                                        break
                    
                    if f1_val is not None and f3_val is not None and isinstance(f1_val, (int, float)) and isinstance(f3_val, (int, float)):
                        if method not in original_plotted:
                            # 第一次绘制该原始模型，添加图例
                            plt.scatter([f1_val], [f3_val], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=80, label=f'{method} (Original)')
                            original_plotted.add(method)
                        else:
                            # 已经绘制过该原始模型，不添加图例
                            plt.scatter([f1_val], [f3_val], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=80)
                        
                        # 为原始模型添加名称标注
                        plt.text(
                            f1_val, 
                            f3_val, 
                            method.split('/')[-1],  # 只显示模型名称的最后一部分
                            fontsize=9, 
                            ha='center', 
                            va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                        )
            
            plt.xlabel('F1 (Reasoning)', fontsize=12)
            plt.ylabel('F3 (IFEval)', fontsize=12)
            # 移除标题
            # plt.title('F1 (Reasoning) vs F3 (IFEval)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, "f1_vs_f3.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"f1 vs f3图已保存到: {output_path}")
    else:
        print("警告：未找到可用的metrics数据，无法绘制数据集指标图")
    
    print(f"指标图已保存到 {output_dir}")
    
    # 绘制包含所有解的图表（对于其他算法绘制所有解，对于evoBMI只绘制帕累托解）
    if all_solutions_metrics_list:
        print("绘制包含所有解的数据集指标图...")
        
        # 准备数据
        all_solutions_data = {}
        
        for metrics in all_solutions_metrics_list:
            method = metrics['type']
            if method not in all_solutions_data:
                all_solutions_data[method] = []
            all_solutions_data[method].append(metrics)
        
        # 绘制aime25所有解的图
        plt.figure(figsize=(10, 6))
        
        # 定义颜色和标记
        solution_colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
        solution_markers = ['o', 's', '^', 'D', 'v', '<', '>']
        original_color = 'red'  # 原始模型使用红色
        original_marker = '^'  # 原始模型使用三角形标记
        
        # 分离解决方案模型和原始模型的数据
        all_data = []
        for method, data in all_solutions_data.items():
            for d in data:
                # 检查是否有原始模型标记
                is_original = False
                # 优先检查model_type字段
                if d.get('model_type') == 'original':
                    is_original = True
                # 然后检查方法名称
                elif any(keyword in method.lower() for keyword in ['qwen3-4b', 'original', 'base', 'expert']):
                    is_original = True
                
                all_data.append({
                    'method': method,
                    'data': d,
                    'is_original': is_original
                })
        
        # 绘制解决方案模型
        solution_plotted = set()
        for item in all_data:
            if not item['is_original']:
                method = item['method']
                d = item['data']
                if 'aime25' in d and isinstance(d['aime25'], dict):
                    aime25_dict = d['aime25']
                    if 'mean_tokens_num' in aime25_dict and 'mean_acc' in aime25_dict:
                        tokens = aime25_dict['mean_tokens_num']
                        acc = aime25_dict['mean_acc']
                        
                        if method not in solution_plotted:
                            # 第一次绘制该方法，添加图例
                            i = len(solution_plotted)
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            plt.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'], label=method)
                            solution_plotted.add(method)
                        else:
                            # 已经绘制过该方法，不添加图例
                            i = len(solution_plotted) - 1
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            plt.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'])
                        
                        # 为evoBMI(ours)添加编号标注
                        if method == 'BAMBO (ours)' or method == 'BMI (ours)':
                            y_offset = 0.01  # 固定偏移量
                            plt.text(
                                tokens, 
                                acc + y_offset, 
                                str(d.get('index', '')), 
                                fontsize=10, 
                                ha='center', 
                                va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                            )
        
        # 绘制原始模型
        original_plotted = set()
        for item in all_data:
            if item['is_original']:
                method = item['method']
                d = item['data']
                if 'aime25' in d and isinstance(d['aime25'], dict):
                    aime25_dict = d['aime25']
                    if 'mean_tokens_num' in aime25_dict and 'mean_acc' in aime25_dict:
                        tokens = aime25_dict['mean_tokens_num']
                        acc = aime25_dict['mean_acc']
                        
                        if method not in original_plotted:
                            # 第一次绘制该原始模型，添加图例
                            plt.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'], label=f'{method} (Original)')
                            original_plotted.add(method)
                        else:
                            # 已经绘制过该原始模型，不添加图例
                            plt.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'])
                        
                        # 为原始模型添加名称标注
                        plt.text(
                            tokens, 
                            acc, 
                            method.split('/')[-1],  # 只显示模型名称的最后一部分
                            fontsize=plot_settings['font_size'] * 0.75, 
                            ha='center', 
                            va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                        )
        
        # 设置图表属性
        plt.xlabel('AIME 25 Mean Tokens Used', fontsize=plot_settings['label_font_size'])
        plt.ylabel('Accuracy', fontsize=plot_settings['label_font_size'])
        plt.grid(True, alpha=0.3, linewidth=plot_settings['grid_line_width'])
        plt.legend(fontsize=plot_settings['legend_font_size'])
        plt.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, "aime25_all_solutions.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"AIME25包含所有解的图已保存到: {output_path}")
        
        # 绘制gpqa所有解的图
        plt.figure(figsize=(10, 6))
        
        # 定义颜色和标记（与aime25图保持一致）
        solution_colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
        solution_markers = ['o', 's', '^', 'D', 'v', '<', '>']
        original_color = 'red'  # 原始模型使用红色
        original_marker = '^'  # 原始模型使用三角形标记
        
        # 分离解决方案模型和原始模型的数据
        all_data = []
        for method, data in all_solutions_data.items():
            for d in data:
                # 检查是否有原始模型标记
                is_original = False
                # 优先检查model_type字段
                if d.get('model_type') == 'original':
                    is_original = True
                # 然后检查方法名称
                elif any(keyword in method.lower() for keyword in ['qwen3-4b', 'original', 'base', 'expert']):
                    is_original = True
                
                all_data.append({
                    'method': method,
                    'data': d,
                    'is_original': is_original
                })
        
        # 绘制解决方案模型
        solution_plotted = set()
        for item in all_data:
            if not item['is_original']:
                method = item['method']
                d = item['data']
                if 'gpqa_diamond' in d and isinstance(d['gpqa_diamond'], dict):
                    gpqa_dict = d['gpqa_diamond']
                    if 'mean_tokens_num' in gpqa_dict and 'mean_acc' in gpqa_dict:
                        tokens = gpqa_dict['mean_tokens_num']
                        acc = gpqa_dict['mean_acc']
                        
                        if method not in solution_plotted:
                            # 第一次绘制该方法，添加图例
                            i = len(solution_plotted)
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            plt.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'], label=method)
                            solution_plotted.add(method)
                        else:
                            # 已经绘制过该方法，不添加图例
                            i = len(solution_plotted) - 1
                            color = solution_colors[i % len(solution_colors)]
                            marker = solution_markers[i % len(solution_markers)]
                            plt.scatter([tokens], [acc], c=color, marker=marker, 
                                      alpha=0.7, s=plot_settings['scatter_size'])
                        
                        # 为evoBMI(ours)添加编号标注
                        if method == 'BAMBO (ours)' or method == 'BMI (ours)':
                            y_offset = 0.01  # 固定偏移量
                            plt.text(
                                tokens, 
                                acc + y_offset, 
                                str(d.get('index', '')), 
                                fontsize=plot_settings['font_size'] * 0.8, 
                                ha='center', 
                                va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                            )
        
        # 绘制原始模型
        original_plotted = set()
        for item in all_data:
            if item['is_original']:
                method = item['method']
                d = item['data']
                if 'gpqa_diamond' in d and isinstance(d['gpqa_diamond'], dict):
                    gpqa_dict = d['gpqa_diamond']
                    if 'mean_tokens_num' in gpqa_dict and 'mean_acc' in gpqa_dict:
                        tokens = gpqa_dict['mean_tokens_num']
                        acc = gpqa_dict['mean_acc']
                        
                        if method not in original_plotted:
                            # 第一次绘制该原始模型，添加图例
                            plt.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'], label=f'{method} (Original)')
                            original_plotted.add(method)
                        else:
                            # 已经绘制过该原始模型，不添加图例
                            plt.scatter([tokens], [acc], c=original_color, marker=original_marker, 
                                      alpha=0.8, s=plot_settings['scatter_size'])
                        
                        # 为原始模型添加名称标注
                        plt.text(
                            tokens, 
                            acc, 
                            method.split('/')[-1],  # 只显示模型名称的最后一部分
                            fontsize=plot_settings['font_size'] * 0.75, 
                            ha='center', 
                            va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                        )
        
        # 设置图表属性
        plt.xlabel('GPQA Diamond Mean Tokens Used', fontsize=plot_settings['label_font_size'])
        plt.ylabel('Accuracy', fontsize=plot_settings['label_font_size'])
        plt.grid(True, alpha=0.3, linewidth=plot_settings['grid_line_width'])
        plt.legend(fontsize=plot_settings['legend_font_size'])
        plt.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, "gpqa_all_solutions.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"GPQA包含所有解的图已保存到: {output_path}")
    
    # 为各个算法的帕累托解计算HV、覆盖度、Spacing和Spread指标
    print("\n计算各个算法的帕累托解HV、覆盖度、Spacing和Spread指标...")
    
    # 计算所有算法的帕累托解总数
    total_pareto_count = len(global_pareto_obj)
    print(f"所有算法的全局帕累托解总数: {total_pareto_count}")
    
    # 准备算法指标内容
    hv_metrics_text = []
    hv_metrics_text.append("\n===== 算法帕累托解关键指标 =====")
    hv_metrics_text.append(
        f"{'算法名称':<25} {'HV值':<12} {'前沿覆盖':<12} {'帕累托数':<10} "
        f"{'Spacing':<12} {'Spread':<12} {'MaxGap':<12}"
    )
    hv_metrics_text.append("-" * 100)
    algorithm_metric_rows = []
    
    # 为每个算法计算HV指标和前沿覆盖指标
    for checkpoint_name, pareto_y in all_pareto_data:
        try:
            # 计算该算法帕累托解的HV指标
            hv_value = compute_hypervolume(pareto_y)
            
            # 计算该算法在全局帕累托解中的数量
            algo_global_pareto_count = 0
            for obj in pareto_y:
                is_global = False
                for gp_obj in global_pareto_obj:
                    if np.allclose(obj, gp_obj, atol=1e-6):
                        is_global = True
                        break
                if is_global:
                    algo_global_pareto_count += 1
            
            # 计算前沿覆盖指标（算法的全局帕累托解数 / 所有算法的全局帕累托解总数）
            if total_pareto_count > 0:
                front_coverage = algo_global_pareto_count / total_pareto_count
                coverage_str = f"{algo_global_pareto_count}/{total_pareto_count}"
            else:
                front_coverage = 0.0
                coverage_str = "0/0"

            spacing_value = compute_spacing(pareto_y)
            spread_value = compute_spread(pareto_y, reference_front=global_pareto_obj)
            max_gap_value = compute_max_gap(pareto_y)
            
            algorithm_metric_rows.append(
                {
                    "algorithm": checkpoint_name,
                    "hv": float(hv_value),
                    "front_coverage_count": int(algo_global_pareto_count),
                    "front_coverage_total": int(total_pareto_count),
                    "front_coverage_ratio": float(front_coverage),
                    "pareto_count": int(len(pareto_y)),
                    "spacing": float(spacing_value),
                    "spread": float(spread_value),
                    "max_gap": float(max_gap_value),
                }
            )

            # 添加到指标文本中
            hv_metrics_text.append(
                f"{checkpoint_name:<25} {hv_value:<12.6f} {coverage_str:<12} {len(pareto_y):<10} "
                f"{spacing_value:<12.6f} {spread_value:<12.6f} {max_gap_value:<12.6f}"
            )
        except Exception as e:
            print(f"计算算法 {checkpoint_name} 的关键指标时出错: {e}")
            hv_metrics_text.append(
                f"{checkpoint_name:<25} {'错误':<12} {'错误':<12} {'错误':<10} "
                f"{'错误':<12} {'错误':<12} {'错误':<12}"
            )
            continue
    
    hv_metrics_text.append("=" * 100)
    
    # 打印指标到控制台
    for line in hv_metrics_text:
        print(line)
    
    # 将关键指标保存到txt文件
    hv_metrics_file_path = os.path.join(output_dir, 'algorithm_hv_metrics.txt')
    with open(hv_metrics_file_path, 'w', encoding='utf-8') as f:
        for line in hv_metrics_text:
            f.write(line + '\n')
    
    print(f"\n算法帕累托解关键指标已保存到: {hv_metrics_file_path}")

    # 导出结构化表格，便于最终汇总
    algorithm_metrics_csv_path = os.path.join(output_dir, 'algorithm_metrics_table.csv')
    with open(algorithm_metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm',
            'hv',
            'front_coverage_count',
            'front_coverage_total',
            'front_coverage_ratio',
            'pareto_count',
            'spacing',
            'spread',
            'max_gap',
        ])
        for row in algorithm_metric_rows:
            writer.writerow([
                row['algorithm'],
                f"{row['hv']:.6f}",
                row['front_coverage_count'],
                row['front_coverage_total'],
                f"{row['front_coverage_ratio']:.6f}",
                row['pareto_count'],
                f"{row['spacing']:.6f}",
                f"{row['spread']:.6f}",
                f"{row['max_gap']:.6f}",
            ])

    algorithm_metrics_md_path = os.path.join(output_dir, 'algorithm_metrics_table.md')
    with open(algorithm_metrics_md_path, 'w', encoding='utf-8') as f:
        f.write("| Algorithm | HV | Coverage | Coverage Ratio | Pareto Count | Spacing | Spread | Max Gap |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in algorithm_metric_rows:
            coverage_str = f"{row['front_coverage_count']}/{row['front_coverage_total']}"
            f.write(
                f"| {row['algorithm']} | {row['hv']:.6f} | {coverage_str} | "
                f"{row['front_coverage_ratio']:.6f} | {row['pareto_count']} | "
                f"{row['spacing']:.6f} | {row['spread']:.6f} | {row['max_gap']:.6f} |\n"
            )

    print(f"算法关键指标CSV表已保存到: {algorithm_metrics_csv_path}")
    print(f"算法关键指标Markdown表已保存到: {algorithm_metrics_md_path}")


def get_iteration_checkpoints(checkpoint_dir: str, task_id: str, interval: int = 10) -> list:
    """
    获取不同迭代次数的checkpoint路径，按间隔筛选
    
    Args:
        checkpoint_dir: checkpoint根目录
        task_id: 任务ID
        interval: 迭代间隔
        
    Returns:
        list: 按间隔筛选后的checkpoint路径列表，包含迭代次数信息
    """
    # 1. 规范化checkpoint_dir
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    
    # 2. 尝试多种路径组合
    possible_task_dirs = [
        os.path.join(checkpoint_dir, task_id),  # 原始路径（checkpoint在task_id子目录下）
        checkpoint_dir,  # checkpoint直接在checkpoint_dir目录下
        os.path.join(os.getcwd(), checkpoint_dir, task_id),  # 绝对路径1
        os.path.join(os.getcwd(), checkpoint_dir),  # 绝对路径2
    ]
    
    task_dir = None
    for possible_dir in possible_task_dirs:
        possible_dir = os.path.normpath(possible_dir)
        if os.path.exists(possible_dir):
            # 检查该目录下是否有checkpoint文件
            has_checkpoints = any(f.startswith('checkpoint_iter_') and f.endswith('.pt') for f in os.listdir(possible_dir))
            if has_checkpoints:
                task_dir = possible_dir
                break
    
    if task_dir is None:
        # 尝试从checkpoint_path中获取目录（如果能访问到args的话）
        # 这里我们使用一个全局变量来传递checkpoint_path信息
        global args
        if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
            cp_path = args.checkpoint_path[0] if isinstance(args.checkpoint_path, list) else args.checkpoint_path
            cp_dir = os.path.dirname(cp_path)
            if os.path.exists(cp_dir):
                task_dir = cp_dir
    
    if task_dir is None:
        # 所有尝试都失败，抛出异常
        raise FileNotFoundError(f"任务目录不存在，尝试了以下路径: {possible_task_dirs}")
    
    # 查找所有迭代checkpoint文件
    checkpoint_files = []
    for file_name in os.listdir(task_dir):
        if file_name.startswith('checkpoint_iter_') and file_name.endswith('.pt'):
            # 先移除文件扩展名，再提取迭代次数
            base_name = os.path.splitext(file_name)[0]  # 移除 .pt 扩展名
            # 提取迭代次数
            iter_num = int(base_name.split('_')[2])
            checkpoint_files.append((iter_num, os.path.join(task_dir, file_name)))
    
    # 按迭代次数排序
    checkpoint_files.sort(key=lambda x: x[0])
    
    # 按间隔筛选
    selected_checkpoints = []
    for i, (iter_num, path) in enumerate(checkpoint_files):
        if i % interval == 0 or i == len(checkpoint_files) - 1:
            selected_checkpoints.append((iter_num, path))
    
    print(f"共找到 {len(checkpoint_files)} 个迭代checkpoint，按间隔 {interval} 筛选后保留 {len(selected_checkpoints)} 个")
    return selected_checkpoints


def plot_iteration_trends(checkpoint_dir: str, task_id: str, interval: int, output_dir: str, include_trend: bool = False, algorithm_name_mapping: dict = None, plot_settings: dict = None, checkpoint_paths: list = None, args=None):
    """
    绘制不同迭代次数的趋势
    
    Args:
        checkpoint_dir: checkpoint根目录
        task_id: 任务ID
        interval: 迭代间隔
        output_dir: 输出目录
        include_trend: 是否包含趋势图（每个目标的最优值趋势和帕累托点数量趋势）
        algorithm_name_mapping: 算法名称映射字典
        plot_settings: 绘图设置参数
        checkpoint_paths: 用户指定的checkpoint文件路径列表
        args: 命令行参数对象，包含checkpoint_path和legend_names
    """
    # 设置默认的绘图参数
    if plot_settings is None:
        plot_settings = {
            # 字体大小设置
            'font_size': 12,  # 基础字体大小
            'title_font_size': 14,  # 标题字体大小
            'label_font_size': 18,  # 标签字体大小
            'tick_font_size': 16,  # 刻度字体大小
            'legend_font_size': 16,  # 图例字体大小
            
            # 标记和线宽设置
            'marker_size': 8,  # 标记大小
            'line_width': 2,  # 线宽
            'grid_line_width': 1,  # 网格线宽
            'vector_line_width': 0.8,  # 偏好向量线宽
            
            # 箭头大小设置
            'arrow_size': 10,  # 箭头大小
        }
    
    if algorithm_name_mapping is None:
        algorithm_name_mapping = {}
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有算法的数据
    algorithms_data = {}
    
    # 使用传入的args参数或直接使用函数参数
    if args is not None:
        # 从args对象获取checkpoint_path和legend_names
        checkpoint_paths = args.checkpoint_path
        legend_names = args.legend_names
    else:
        # 如果没有传入args对象，使用函数参数
        if checkpoint_paths is None:
            checkpoint_paths = []
        legend_names = []
    
    # 遍历所有checkpoint路径，每个路径代表一个算法
    for i, checkpoint_path in enumerate(checkpoint_paths):
        algorithm_dir_name = os.path.basename(os.path.dirname(checkpoint_path))
        print(f"处理算法: {algorithm_dir_name}")
        
        # 获取算法名称
        algorithm_name = algorithm_dir_name
        if legend_names and i < len(legend_names):
            algorithm_name = legend_names[i]
        
        # 应用算法名称映射
        if algorithm_name_mapping and algorithm_name in algorithm_name_mapping:
            algorithm_name = algorithm_name_mapping[algorithm_name]
        
        try:
            # 获取checkpoint所在的目录
            checkpoint_dir = os.path.dirname(checkpoint_path)
            run_settings = _load_run_settings(checkpoint_dir)
            
            # 从用户指定的checkpoint文件中提取目标迭代次数
            target_checkpoint_file = os.path.basename(checkpoint_path)
            if target_checkpoint_file.startswith('checkpoint_iter_') and target_checkpoint_file.endswith('.pt'):
                base_name = os.path.splitext(target_checkpoint_file)[0]  # 移除 .pt 扩展名
                target_iter_num = int(base_name.split('_')[2])
                print(f"目标迭代次数: {target_iter_num}")
            else:
                print(f"无效的checkpoint文件名: {target_checkpoint_file}")
                continue
            
            # 扫描目录下所有checkpoint文件，只保留目标迭代次数及之前的文件
            checkpoint_files = []
            for file_name in os.listdir(checkpoint_dir):
                if file_name.startswith('checkpoint_iter_') and file_name.endswith('.pt'):
                    # 提取迭代次数
                    base_name = os.path.splitext(file_name)[0]  # 移除 .pt 扩展名
                    iter_num = int(base_name.split('_')[2])
                    # 只保留目标迭代次数及之前的文件
                    if iter_num <= target_iter_num:
                        checkpoint_files.append((iter_num, os.path.join(checkpoint_dir, file_name)))
            
            # 按迭代次数排序
            checkpoint_files.sort(key=lambda x: x[0])
            
            # 检查是否有checkpoint文件
            if not checkpoint_files:
                print(f"在目录 {checkpoint_dir} 中没有找到checkpoint文件或所有文件都超过目标迭代次数 {target_iter_num}")
                continue
            
            # 使用目标迭代次数及之前的所有文件
            selected_checkpoints = checkpoint_files
            print(f"找到 {len(selected_checkpoints)} 个checkpoint文件，将绘制到第 {target_iter_num} 代的数据")
            
            # 提取每个checkpoint的帕累托前沿和metrics数据
            iteration_data = []
            iteration_metrics = []
            evaluation_counts = []
            
            for iter_num, path in selected_checkpoints:
                checkpoint = load_checkpoint(path)
                
                # 获取帕累托解的索引
                train_obj_true = checkpoint['train_obj_true']
                _, pareto_indices = get_pareto_optimal_points(train_obj_true, return_indices=True)
                
                # 提取帕累托前沿的决策变量和目标值
                train_x = checkpoint['train_x']
                pareto_x = train_x[pareto_indices]
                pareto_y = train_obj_true[pareto_indices]
                iteration_data.append((iter_num, pareto_x, pareto_y))
                evaluation_counts.append(_compute_evaluation_count(iter_num, algorithm_dir_name, run_settings))
                
                # 提取该迭代的帕累托解的metrics数据
                metrics_list = []
                
                # 检查checkpoint中是否有evaluated_solutions字段
                if 'evaluated_solutions' in checkpoint:
                    eval_solutions = checkpoint['evaluated_solutions']
                    # 从evaluated_solutions中提取metrics
                    if isinstance(eval_solutions, dict) and 'metrics' in eval_solutions:
                        eval_metrics_list = eval_solutions['metrics']
                        # 只提取帕累托解的metrics
                        for i, idx in enumerate(pareto_indices):
                            if idx < len(eval_metrics_list):
                                metrics = eval_metrics_list[idx]
                                metrics_list.append({
                                    'type': 'solution',
                                    'index': i,
                                    **metrics
                                })
                elif 'train_info' in checkpoint:
                    # 兼容旧版本的checkpoint格式，直接从train_info获取metrics
                    train_info = checkpoint['train_info']
                    # 只提取帕累托解的metrics
                    for i, idx in enumerate(pareto_indices):
                        if idx < len(train_info):
                            metrics = train_info[idx]
                            metrics_list.append({
                                'type': 'solution',
                                'index': i,
                                **metrics
                            })
                iteration_metrics.append((iter_num, metrics_list))
            
            # 从最后一代的JSON文件中提取所有迭代的HV值
            hv_values = []
            if selected_checkpoints:
                # 获取最后一代的迭代次数
                last_iter_num = selected_checkpoints[-1][0]
                # 构建最后一代的JSON文件路径
                last_json_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{last_iter_num}.json')
                
                if os.path.exists(last_json_path):
                    print(f"从最后一代JSON文件中提取HV值: {last_json_path}")
                    import json
                    with open(last_json_path, 'r') as f:
                        last_checkpoint_json = json.load(f)
                    
                    # 提取hvs数组
                    if 'hvs' in last_checkpoint_json:
                        all_hvs = last_checkpoint_json['hvs']
                        
                        # 获取所有迭代次数
                        all_iter_nums = sorted([iter_num for iter_num, _ in selected_checkpoints])
                        
                        # 根据迭代次数提取对应的HV值
                        for iter_num in all_iter_nums:
                            # 迭代次数从0开始，所以索引是iter_num
                            if iter_num < len(all_hvs):
                                hv_values.append(all_hvs[iter_num])
                            else:
                                # 如果HV数组长度不足，使用最后一个值
                                hv_values.append(all_hvs[-1])
                        print(f"成功提取HV值，共 {len(hv_values)} 个")
                    else:
                        print("使用默认HV值")
                        hv_values = [0] * len(selected_checkpoints)
            
            # 保存该算法的数据
            algorithms_data[algorithm_name] = {
                'iteration_data': iteration_data,
                'iteration_metrics': iteration_metrics,
                'evaluation_counts': evaluation_counts,
                'hv_values': hv_values,
                'selected_checkpoints': selected_checkpoints,
                'run_settings': run_settings,
            }
        except Exception as e:
            print(f"处理算法 {algorithm_name} 失败: {e}")
            continue
    
    # 根据include_trend参数决定绘制哪些图
    if include_trend:
        # 绘制每个目标的最优值趋势和帕累托点数量趋势
        print("生成迭代趋势图...")
        
        # 设置更符合计算机顶会风格的样式
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 设置统一的字体，适合论文使用，使用通用字体避免依赖特定字体
        # 绘图参数已在main函数前统一设置，此处不再重复设置
        
        # 设置颜色主题
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # 顶会常用颜色
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # 1行2列布局，左边绘制HV值趋势，右边绘制帕累托解数量趋势
        fig = plt.figure(figsize=(15, 6))
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1], hspace=0.4, wspace=0.3)
        
        # 左列：HV值趋势
        ax_hv = plt.subplot(gs[0, 0])
        # 右列：帕累托解数量趋势
        ax_pareto = plt.subplot(gs[0, 1])
        
        # 遍历所有算法，绘制趋势图
        for i, (algorithm_dir, data) in enumerate(algorithms_data.items()):
            # 获取算法名称
            algorithm_name = algorithm_dir  # 使用目录名作为算法名称
            # 应用算法名称映射
            if algorithm_name_mapping and algorithm_name in algorithm_name_mapping:
                algorithm_name = algorithm_name_mapping[algorithm_name]
            elif algorithm_name_mapping and "BAMBO (ours)" in algorithm_name_mapping and "bambo" in algorithm_name.lower():
                algorithm_name = algorithm_name_mapping["BAMBO (ours)"]
            
            # 获取该算法的数据
            iteration_data = data['iteration_data']
            hv_values = data['hv_values']
            evaluation_counts = data.get('evaluation_counts', [iter_num for iter_num, _, _ in iteration_data])

            algorithm_key = _infer_algorithm_key_from_name(algorithm_dir)
            if algorithm_key == 'grid' or algorithm_name.strip().lower() == 'grid search':
                continue
            
            # 提取评估次数和帕累托解数量
            pareto_counts = [len(pareto_y) for _, _, pareto_y in iteration_data]
            
            # 获取颜色和标记
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # 绘制HV值趋势
            ax_hv.plot(evaluation_counts, hv_values, marker=marker, linestyle='-', color=color, 
                      linewidth=plot_settings['line_width'], markersize=plot_settings['marker_size'], 
                      alpha=0.7, label=algorithm_name)
            
            # 绘制帕累托解数量趋势
            ax_pareto.plot(evaluation_counts, pareto_counts, marker=marker, linestyle='-', color=color, 
                          linewidth=plot_settings['line_width'], markersize=plot_settings['marker_size'], 
                          alpha=0.7, label=algorithm_name)
        
        # 设置图表标签和网格
        ax_hv.set_xlabel('Evaluations', fontsize=plot_settings['label_font_size'])
        ax_hv.set_ylabel('Hypervolume', fontsize=plot_settings['label_font_size'])
        ax_hv.grid(True, linestyle='--', alpha=0.7, linewidth=plot_settings['grid_line_width'])
        ax_hv.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
        ax_hv.legend(loc='lower right', fontsize=plot_settings['legend_font_size'])
        
        ax_pareto.set_xlabel('Evaluations', fontsize=plot_settings['label_font_size'])
        ax_pareto.set_ylabel('Number of Pareto Points', fontsize=plot_settings['label_font_size'])
        ax_pareto.grid(True, linestyle='--', alpha=0.7, linewidth=plot_settings['grid_line_width'])
        ax_pareto.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
        ax_pareto.legend(loc='lower right', fontsize=plot_settings['legend_font_size'])
        
        # 移除主标题
        # fig.suptitle('Iteration Trends in Multi-Objective Optimization', fontsize=15, fontweight='bold', y=0.98)
        
        # 调整布局
        plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.3, wspace=0.3)
        
        # 保存图像，使用更高的DPI和更紧凑的边框
        plt.savefig(os.path.join(output_dir, 'iteration_trends.png'), dpi=500, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"迭代趋势图已保存到 {output_dir}")
    else:
        # 绘制函数空间的迭代变化 - 3D图
        # 仅当目标函数数量大于等于3时才生成3D图
        # 获取目标函数数量
        num_objectives = None
        
        # 获取第一个算法的数据
        if algorithms_data:
            # 选择第一个算法的数据
            first_algorithm_data = next(iter(algorithms_data.values()))
            iteration_data = first_algorithm_data['iteration_data']
            
            for _, _, pareto_y in iteration_data:
                if len(pareto_y) > 0:
                    if isinstance(pareto_y, torch.Tensor):
                        num_objectives = pareto_y.shape[1]
                    else:
                        num_objectives = pareto_y.shape[1]
                    break
        
        # 默认使用3个目标
        num_objectives = num_objectives if num_objectives else 3
        
        if algorithms_data:
            # 选择第一个算法的数据
            first_algorithm_data = next(iter(algorithms_data.values()))
            iteration_data = first_algorithm_data['iteration_data']
            iteration_metrics = first_algorithm_data['iteration_metrics']
            evaluation_counts = first_algorithm_data.get('evaluation_counts', [iter_num for iter_num, _, _ in iteration_data])
            
            if num_objectives >= 3:
                print("生成函数空间的迭代变化3D图...")
                
                # 绘图参数已在main函数前统一设置，此处不再重复设置
                
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # 设置更符合顶会风格的样式
                plt.style.use('seaborn-v0_8-paper')
                
                # 为不同迭代次数使用不同颜色和标记，使用tab20颜色图以获得更多颜色选择
                colors = plt.cm.tab20(np.linspace(0, 1, len(iteration_data)))
                markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X', '+', 'x', '|', '_']
                
                for i, (iter_num, _, pareto_y) in enumerate(iteration_data):
                    if len(pareto_y) > 0:
                        # 将pareto_y转换为numpy数组
                        if isinstance(pareto_y, torch.Tensor):
                            pareto_y_np = pareto_y.cpu().numpy()
                        else:
                            pareto_y_np = pareto_y
                        # 使用不同颜色和标记组合
                        color = colors[i % len(colors)]
                        marker = markers[i % len(markers)]
                        
                        # 3D散点图
                        ax.scatter(pareto_y_np[:, 0], pareto_y_np[:, 1], pareto_y_np[:, 2], 
                                 color=color, marker=marker, alpha=0.8, s=plot_settings['scatter_size'], label=_format_evaluation_label(evaluation_counts[i]))
                
                # 设置坐标轴标签，使用更大的字体和更好的样式
                ax.set_xlabel('F1 (Reasoning)', fontsize=plot_settings['label_font_size'])
                ax.set_ylabel('F2 (Efficiency)', fontsize=plot_settings['label_font_size'])
                ax.set_zlabel('F3 (IFEval)', fontsize=plot_settings['label_font_size'])
                # 移除标题
                # ax.set_title('Pareto Front Evolution in 3D Objective Space', fontsize=16)
                
                # 设置坐标轴刻度字体大小
                ax.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
                
                # 处理图例，使用更好的位置和样式
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc='upper right', fontsize=plot_settings['legend_font_size'], frameon=True, shadow=True)
                
                # 调整视角
                ax.view_init(elev=25, azim=45)
                
                # 添加网格线
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 保存图像，使用更高的DPI和更紧凑的边框
                plt.savefig(os.path.join(output_dir, 'pareto_evolution_3d.png'), dpi=500, bbox_inches='tight', pad_inches=0.1)
                plt.close()
        
        # 绘制函数空间的迭代变化 - 2D投影
        if algorithms_data:
            print("生成函数空间的迭代变化2D投影图...")
            
            # 绘图参数已在main函数前统一设置，此处不再重复设置
            
            # 根据目标函数数量设置绘图参数
            if num_objectives >= 3:
                # 3个目标时，绘制所有三对投影
                pairs = [(0, 1), (0, 2), (1, 2)]
                labels = [
                    ['F1 (Reasoning)', 'F2 (Efficiency)'],
                    ['F1 (Reasoning)', 'F3 (IFEval)'],
                    ['F2 (Efficiency)', 'F3 (IFEval)']
                ]
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            else:
                # 2个目标时，只绘制f1 vs f2
                pairs = [(0, 1)]
                labels = [
                    ['F1 (Reasoning)', 'F2 (Efficiency)']
                ]
                fig, axes = plt.subplots(1, 1, figsize=(10, 6))
                # 将axes转换为列表，以便统一处理
                axes = [axes]
            
            # 设置更符合顶会风格的样式
            plt.style.use('seaborn-v0_8-paper')
            
            # 为不同迭代次数使用不同颜色和标记，使用tab20颜色图以获得更多颜色选择
            colors = plt.cm.tab20(np.linspace(0, 1, len(iteration_data)))
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X', '+', 'x', '|', '_']
            
            for pair_idx, (idx1, idx2) in enumerate(pairs):
                ax = axes[pair_idx]
                ax.set_xlabel(labels[pair_idx][0], fontsize=plot_settings['label_font_size'])
                ax.set_ylabel(labels[pair_idx][1], fontsize=plot_settings['label_font_size'])
                # 移除标题
                # ax.set_title(f'Pareto Evolution: {labels[pair_idx][0]} vs {labels[pair_idx][1]}', fontsize=16)
                ax.grid(True, linestyle='--', alpha=0.7, linewidth=plot_settings['grid_line_width'])
                ax.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
                
                for i, (iter_num, _, pareto_y) in enumerate(iteration_data):
                    if len(pareto_y) > 0:
                        # 将pareto_y转换为numpy数组
                        if isinstance(pareto_y, torch.Tensor):
                            pareto_y_np = pareto_y.cpu().numpy()
                        else:
                            pareto_y_np = pareto_y
                        # 使用不同颜色和标记组合
                        color = colors[i % len(colors)]
                        marker = markers[i % len(markers)]
                        ax.scatter(pareto_y_np[:, idx1], pareto_y_np[:, idx2], 
                                  color=color, marker=marker, alpha=0.8, s=plot_settings['scatter_size'], label=_format_evaluation_label(evaluation_counts[i]))
            
            # 添加图例，改为垂直布局
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', ncol=1, fontsize=plot_settings['legend_font_size'], frameon=True, shadow=True)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95, right=0.85)
            # 保存图像，使用更高的DPI和更紧凑的边框
            plt.savefig(os.path.join(output_dir, 'pareto_evolution_2d.png'), dpi=500, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            # 为每个数据集绘制不同迭代次数的帕累托前沿在同一张图上
            print("为每个数据集绘制不同迭代次数的帕累托前沿图...")
            
            # 准备所有迭代的数据集指标数据，移除ifeval
            datasets = ['aime25', 'gpqa_diamond']
            dataset_names = {
                'aime25': 'AIME25',
                'gpqa_diamond': 'GPQA Diamond'
            }
            
            # 设置更符合顶会风格的样式
            plt.style.use('seaborn-v0_8-paper')
            
            # 为不同迭代次数使用不同颜色和标记，使用tab20颜色图以获得更多颜色选择
            colors = plt.cm.tab20(np.linspace(0, 1, len(iteration_metrics)))
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X', '+', 'x', '|', '_']
            
            # 为每个数据集创建一个图，显示不同迭代次数的帕累托前沿
            for dataset in datasets:
                plt.figure(figsize=(10, 6))
                
                # 为每个迭代次数绘制数据点
                for i, (iter_num, metrics_list) in enumerate(iteration_metrics):
                    if metrics_list:
                        # 提取当前数据集的token数量和准确度
                        tokens = []
                        acc = []
                        
                        for metrics in metrics_list:
                            if dataset in metrics:
                                dataset_metrics = metrics[dataset]
                                # 获取token数量
                                if 'mean_tokens_num' in dataset_metrics:
                                    tokens.append(dataset_metrics['mean_tokens_num'])
                                else:
                                    continue
                                
                                # 获取准确度
                                if 'mean_acc' in dataset_metrics:
                                    acc.append(dataset_metrics['mean_acc'])
                                elif dataset == 'ifeval' and 'mean_prompt_level_strict' in dataset_metrics:
                                    # ifeval的准确度计算方式不同
                                    acc_val = (dataset_metrics['mean_prompt_level_strict'] + 
                                              dataset_metrics['mean_inst_level_strict'] + 
                                              dataset_metrics['mean_prompt_level_loose'] + 
                                              dataset_metrics['mean_inst_level_loose']) / 4
                                    acc.append(acc_val)
                                else:
                                    continue
                        
                        # 绘制当前迭代的帕累托前沿点
                        if tokens and acc:
                            # 为当前迭代的所有点使用相同的颜色和标记，不同迭代次数使用不同颜色和标记
                            color = colors[i % len(colors)]
                            marker = markers[i % len(markers)]
                            plt.scatter(tokens, acc, color=color, marker=marker, alpha=0.8, s=plot_settings['scatter_size'], label=_format_evaluation_label(evaluation_counts[i]))
                
                # 设置图表属性
                if dataset == 'aime25':
                    plt.xlabel('AIME 25 Mean Tokens Used', fontsize=plot_settings['label_font_size'])
                elif dataset == 'gpqa_diamond':
                    plt.xlabel('GPQA Diamond Mean Tokens Used', fontsize=plot_settings['label_font_size'])
                else:
                    plt.xlabel('Mean Tokens Used', fontsize=plot_settings['label_font_size'])
                plt.ylabel('Accuracy', fontsize=plot_settings['label_font_size'])
                # 移除标题
                # plt.title(f'Token Usage vs Accuracy for {dataset_names[dataset]} (Pareto Front Evolution)', fontsize=16, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='upper right', fontsize=plot_settings['legend_font_size'], frameon=True, shadow=True)
                plt.tick_params(axis='both', which='major', labelsize=plot_settings['tick_font_size'])
                plt.tight_layout()
                
                # 保存图表，使用更高的DPI和更紧凑的边框
                output_path = os.path.join(output_dir, f'{dataset}_pareto_evolution.png')
                plt.savefig(output_path, dpi=500, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                
                print(f"{dataset_names[dataset]}数据集的帕累托前沿演化图已保存到: {output_path}")
    
    print(f"迭代趋势图已保存到 {output_dir}")


def generate_uniform_preference_vectors(num_vectors: int, num_objectives: int, obj_min: np.ndarray, obj_max: np.ndarray) -> np.ndarray:
    """
    生成均匀分布的偏好向量，直接均匀采样构成单位圆
    
    Args:
        num_vectors: 偏好向量数量
        num_objectives: 目标函数数量
        obj_min: 目标函数空间的最小值，形状为(num_objectives,)
        obj_max: 目标函数空间的最大值，形状为(num_objectives,)
        
    Returns:
        np.ndarray: 形状为(num_vectors, num_objectives)的偏好向量数组，值在[0,1]范围内或为单位向量
    """
    print(f"生成 {num_vectors} 个均匀分布的偏好向量，目标数为 {num_objectives}")
    
    # 直接均匀采样5个点，构成单位圆，角度从0°到90°均分（包括90°）
    angles = np.linspace(0, np.pi/2, num_vectors, endpoint=True)
    
    # 生成单位圆上的向量
    ref_dirs = np.zeros((num_vectors, num_objectives))
    ref_dirs[:, 0] = np.cos(angles)
    ref_dirs[:, 1] = np.sin(angles)
    
    # 确保所有分量为正（因为目标是最大化）
    ref_dirs = np.abs(ref_dirs)
    
    # 归一化到单位球面上
    norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    # 避免除以零
    norms = np.where(norms == 0, 1e-10, norms)
    ref_dirs = ref_dirs / norms
    
    return ref_dirs


def select_pareto_individuals(checkpoint_path: str, num_vectors: int, top_n: int, output_dir: str, algorithm_name_mapping: dict = None, plot_settings: dict = None):
    """
    根据偏好向量选择帕累托前沿个体，并生成筛选后的checkpoint
    
    Args:
        checkpoint_path: checkpoint文件路径
        num_vectors: 偏好向量数量
        top_n: 每个偏好向量选择最接近的前N个个体
        output_dir: 输出目录
        algorithm_name_mapping: 算法名称映射字典
        plot_settings: 绘图设置参数
    """
    # 设置默认的绘图参数
    if plot_settings is None:
        plot_settings = {
            # 字体大小设置
            'font_size': 12,  # 基础字体大小
            'title_font_size': 14,  # 标题字体大小
            'label_font_size': 18,  # 标签字体大小
            'tick_font_size': 16,  # 刻度字体大小
            'legend_font_size': 16,  # 图例字体大小
            
            # 标记和线宽设置
            'marker_size': 8,  # 线图标记大小
            'scatter_size': 50,  # 散点图大小
            'line_width': 2,  # 线宽
            'grid_line_width': 1,  # 网格线宽
            'vector_line_width': 0.8,  # 偏好向量线宽
            
            # 箭头大小设置
            'arrow_size': 10,  # 箭头大小
        }
    
    if algorithm_name_mapping is None:
        algorithm_name_mapping = {}
    
    # 检查是否是第一个checkpoint（我们的算法）
    # 从文件名或路径中判断是否包含model_level
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_basename = os.path.basename(checkpoint_dir)
    
    # 只处理第一个checkpoint，跳过所有包含model_level的checkpoint
    if 'model_level' in checkpoint_basename.lower():
        print(f"跳过model_level checkpoint: {checkpoint_path}")
        return
    
    # 加载checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # 提取帕累托前沿
    pareto_x, pareto_y = get_pareto_front(checkpoint)
    num_objectives = pareto_y.shape[1]
    
    # 计算目标函数空间的最小值和最大值
    obj_min = np.min(pareto_y, axis=0)
    obj_max = np.max(pareto_y, axis=0)
    
    # 生成偏好向量，不需要放缩到目标函数区间内
    preference_vectors = generate_uniform_preference_vectors(num_vectors, num_objectives, obj_min, obj_max)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择个体
    selected_indices = []
    selected_vec_indices = []  # 记录每个选中个体对应的偏好向量索引
    
    # 统计每个个体被哪些向量选择
    # 格式：{个体索引: [向量索引1, 向量索引2, ...]}
    individual_to_vecs = {}
    
    # 计算目标函数范围
    obj_range = obj_max - obj_min
    
    # 对帕累托解的目标函数值进行归一化处理（减去最小值，除以范围）
    # 注意：避免除以零的情况
    normalized_pareto_y = (pareto_y - obj_min) / (obj_range + 1e-10)
    
    for i, pref_vec in enumerate(preference_vectors):
        # 计算每个帕累托个体与偏好向量的余弦相似度
        cos_similarities = []
        for j, norm_obj_values in enumerate(normalized_pareto_y):
            # 将归一化后的目标值转换为单位向量
            norm_obj_values_norm = norm_obj_values / (np.linalg.norm(norm_obj_values) + 1e-10)
            # 计算真正的余弦相似度（点积，因为两个向量都是单位向量）
            similarity = np.dot(norm_obj_values_norm, pref_vec)
            cos_similarities.append((similarity, j))
        
        # 按相似度降序排序
        cos_similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 每个偏好向量只选择一个收敛性最佳的个体（目标值总和最大的个体）
        if cos_similarities:
            # 选择相似度最高的个体
            best_idx = cos_similarities[0][1]
            # 将选中的个体添加到选中列表
            selected_indices.append(best_idx)
            selected_vec_indices.append(i)  # 记录对应的偏好向量索引
            
            # 统计每个个体被哪些向量选择
            if best_idx not in individual_to_vecs:
                individual_to_vecs[best_idx] = []
            individual_to_vecs[best_idx].append(i)
    
    # 绘制帕累托解和偏好向量图（仅在2维情况下）
    if num_objectives == 2 and selected_indices:
        # 创建绘图，改为正方形画幅
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 只绘制所有帕累托解，不绘制非帕累托解
        ax.scatter(pareto_y[:, 0], pareto_y[:, 1], c='gray', marker='o', alpha=0.8, s=plot_settings['scatter_size'])
        
        # 定义颜色列表
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # 首先绘制所有选中的个体（保留解本身的散点，使用原始颜色）
        for sol_idx in individual_to_vecs:
            sol_x, sol_y = pareto_y[sol_idx]
            # 绘制解本身的散点，使用原始颜色和大小
            ax.scatter(sol_x, sol_y, c='blue', marker='o', s=plot_settings['scatter_size'], edgecolors='black', linewidth=1)  # 保留解本身的散点
        
        # 绘制偏好向量箭头
        for i, (vec_idx, sol_idx) in enumerate(zip(selected_vec_indices, selected_indices)):
            color = colors[vec_idx % len(colors)]
            
            # 绘制偏好向量箭头，终点位于由f1和f2范围构成的椭圆边上
            vec = preference_vectors[vec_idx]
            
            # 计算椭圆的长轴和短轴（f1范围和f2范围）
            a = obj_range[0]  # f1范围作为长轴
            b = obj_range[1]  # f2范围作为短轴
            
            # 计算向量终点，从所有帕累托解的最小值出发，落在椭圆边上
            start_x = obj_min[0]
            start_y = obj_min[1]
            
            # 计算向量方向角
            theta = np.arctan2(vec[1], vec[0])
            
            # 计算椭圆上的点，使用极坐标转换
            end_x = start_x + a * np.cos(theta)
            end_y = start_y + b * np.sin(theta)
            
            # 绘制向量箭头，保持宽度一致，透明度为0.5
            ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y, 
                     head_width=0.01,  # 固定小箭头宽度
                     head_length=0.01,  # 固定小箭头长度
                     fc=color, ec=color, linewidth=plot_settings['vector_line_width'], alpha=0.5)  # 保持宽度一致，透明度0.5
        
        # 为每个被选择的个体绘制圆框，支持显示被多个向量同时选择
        for sol_idx, vec_indices in individual_to_vecs.items():
            sol_x, sol_y = pareto_y[sol_idx]
            # 为每个选中该个体的向量绘制一个不同颜色的圆框，使用明显不同的半径
            base_size = plot_settings['scatter_size'] * 2.0  # 基础圆框大小
            increment = base_size * 0.6  # 每次增加的大小，确保明显不同
            
            for i, vec_idx in enumerate(vec_indices):
                color = colors[vec_idx % len(colors)]
                # 绘制圆框，半径逐渐增大，确保不相互覆盖
                circle_size = base_size + i * increment
                # 使用scatter绘制圆框，通过调整s参数控制大小
                ax.scatter(sol_x, sol_y, c='none', marker='o', s=circle_size, edgecolors=color, linewidth=2, alpha=0.7)
        
        # 设置图表属性，使用更大的字体和加粗
        ax.set_xlabel('F1 (Reasoning)', fontsize=plot_settings['label_font_size'], fontweight='bold')
        ax.set_ylabel('F2 (Efficiency)', fontsize=plot_settings['label_font_size'], fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=plot_settings['grid_line_width'])
        
        # 调整轴刻度，使用更大的刻度标记
        ax.tick_params(axis='both', which='major', length=8, width=2, labelsize=plot_settings['tick_font_size'])
        
        # 不显示图例
        # 调整布局，增加padding避免文字被截断
        plt.tight_layout(pad=3.0)
        
        # 保存图表
        output_path = os.path.join(output_dir, 'preference_vector_selection.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"帕累托解与偏好向量图已保存到: {output_path}")
    else:
        print(f"没有选中的个体或目标数不是2，跳过绘图")
    
    # 为每个benchmark视角下的token数和acc构成的二维空间生成帕累托集合
    # 准备所有解的metrics数据
    all_metrics = []
    
    # 检查checkpoint中是否有evaluated_solutions字段
    if 'evaluated_solutions' in checkpoint:
        eval_solutions = checkpoint['evaluated_solutions']
        # 从evaluated_solutions中提取metrics
        if isinstance(eval_solutions, dict) and 'metrics' in eval_solutions:
            all_metrics = eval_solutions['metrics']
    elif 'train_info' in checkpoint:
        # 兼容旧版本的checkpoint格式，直接从train_info获取metrics
        all_metrics = checkpoint['train_info']
    
    # 初始化数据集帕累托索引字典，只保留aime25和gpqa_diamond，删除ifeval
    datasets = ['aime25', 'gpqa_diamond']
    dataset_pareto_indices = {}
    
    if all_metrics:
        # 获取所有帕累托解的索引
        train_obj_true = checkpoint['train_obj_true']
        _, all_pareto_indices = get_pareto_optimal_points(train_obj_true, return_indices=True)
        
        for dataset in datasets:
            # 提取当前数据集的token数量和准确度
            token_acc_pairs = []
            indices = []
            
            for i, idx in enumerate(all_pareto_indices):
                if idx < len(all_metrics):
                    metrics = all_metrics[idx]
                    if dataset in metrics:
                        dataset_metrics = metrics[dataset]
                        # 获取token数量
                        if 'mean_tokens_num' in dataset_metrics:
                            tokens = dataset_metrics['mean_tokens_num']
                            # 获取准确度
                            acc = 0
                            if 'mean_acc' in dataset_metrics:
                                acc = dataset_metrics['mean_acc']
                            elif dataset == 'ifeval' and 'mean_prompt_level_strict' in dataset_metrics:
                                # ifeval的准确度计算方式不同
                                acc = (dataset_metrics['mean_prompt_level_strict'] + 
                                      dataset_metrics['mean_inst_level_strict'] + 
                                      dataset_metrics['mean_prompt_level_loose'] + 
                                      dataset_metrics['mean_inst_level_loose']) / 4
                            else:
                                continue
                            
                            # 存储token-acc对和对应的索引
                            token_acc_pairs.append([tokens, acc])
                            indices.append(i)  # 这里i是帕累托解的索引，不是全局索引
            
            # 生成当前数据集的二维帕累托集合
            if token_acc_pairs:
                token_acc_array = np.array(token_acc_pairs)
                # 计算二维帕累托前沿，注意这里我们需要的是最小化tokens和最大化acc，所以需要转换
                # 转换为最大化问题：-tokens（最小化tokens相当于最大化-tokens）和acc
                transformed_pairs = np.column_stack((-token_acc_array[:, 0], token_acc_array[:, 1]))
                _, pareto_indices = get_pareto_optimal_points(transformed_pairs, return_indices=True)
                
                # 存储当前数据集的帕累托个体索引
                dataset_pareto_indices[dataset] = [indices[i] for i in pareto_indices]
                
                # 将这些个体添加到选中列表
                selected_indices.extend([indices[i] for i in pareto_indices])
        
    # 初始化每个数据集的帕累托索引集合为空集合
    for dataset in datasets:
        if dataset not in dataset_pareto_indices:
            dataset_pareto_indices[dataset] = []
    
    # 去重
    selected_indices = list(set(selected_indices))
    selected_pareto_x = pareto_x[selected_indices]
    selected_pareto_y = pareto_y[selected_indices]
    
    print(f"从 {pareto_x.shape[0]} 个帕累托个体中选择了 {len(selected_indices)} 个个体")
    
    # 找到存在于三个解集视角下的最优个体
    # 首先获取每个数据集的帕累托个体索引集合
    aime25_pareto = set(dataset_pareto_indices.get('aime25', []))
    gpqa_pareto = set(dataset_pareto_indices.get('gpqa_diamond', []))
    ifeval_pareto = set(dataset_pareto_indices.get('ifeval', []))
    
    # 找到三个集合的交集，即存在于三个解集视角下的个体
    common_pareto = aime25_pareto.intersection(gpqa_pareto).intersection(ifeval_pareto)
    
    if common_pareto: 
        print(f"存在于三个解集视角下的最优个体：{common_pareto}")
        # 从交集中选择一个作为最收敛个体
        common_pareto_list = list(common_pareto)
        
        # 将common_pareto_list转换为selected_indices中的索引
        # 因为selected_indices是去重后的，我们需要找到common_pareto_list中的个体在selected_indices中的位置
        selected_common = []
        for idx in common_pareto_list:
            if idx in selected_indices:
                selected_common.append(idx)
        
        if selected_common:
            # 从selected_indices中找到对应的索引
            best_index_in_selected = selected_indices.index(selected_common[0])
            best_x = selected_pareto_x[best_index_in_selected]
            best_y = selected_pareto_y[best_index_in_selected]
            print(f"最收敛的个体：决策变量={best_x}, 目标值={best_y}")
        else:
            # 如果交集为空，选择目标值总和最大的个体
            print("警告：三个解集视角下没有共同的最优个体，使用默认方式选择最收敛个体")
            obj_sums = np.sum(selected_pareto_y, axis=1)
            best_index_in_selected = np.argmax(obj_sums)
            best_x = selected_pareto_x[best_index_in_selected]
            best_y = selected_pareto_y[best_index_in_selected]
            print(f"最收敛的个体：决策变量={best_x}, 目标值={best_y}")
    else: 
        print("警告：三个解集视角下没有共同的最优个体，使用默认方式选择最收敛个体")
        # 如果交集为空，选择目标值总和最大的个体
        obj_sums = np.sum(selected_pareto_y, axis=1)
        best_index_in_selected = np.argmax(obj_sums)
        best_x = selected_pareto_x[best_index_in_selected]
        best_y = selected_pareto_y[best_index_in_selected]
        print(f"最收敛的个体：决策变量={best_x}, 目标值={best_y}")
    
    # 生成筛选后的checkpoint
    selected_checkpoint = {
        'train_x': torch.tensor(selected_pareto_x),
        'train_obj_true': torch.tensor(selected_pareto_y),
        'best_x': torch.tensor(best_x),
        'best_y': torch.tensor(best_y),
        'preference_vectors': torch.tensor(preference_vectors),
        'selected_indices': selected_indices
    }
    
    # 保存筛选后的checkpoint
    output_path = os.path.join(output_dir, 'checkpoint_latest_selected.pt')
    torch.save(selected_checkpoint, output_path)
    print(f"筛选后的checkpoint已保存到 {output_path}")
    
    # 移除了保存selected_points_with_preference_vectors.png的功能
    
    # 移除了绘制不同benchmark指标空间解选择情况的功能，包括aime25_selected_vs_all.png、gpqa_diamond_selected_vs_all.png和ifeval相关绘图
    
    print(f"可视化结果已保存到 {output_dir}")


def setup_plot_params():
    """
    统一设置绘图参数，只包含字体相关设置
    """
    # 设置统一的字体，适合单栏论文使用
    plt.rcParams.update({
        # 字体设置
        'font.family': ['sans-serif'],
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana', 'Tahoma'],
        'font.weight': 'bold',  # 统一加粗字体
        
        # 标题设置
        'axes.titleweight': 'bold',  # 标题加粗
        
        # 标签设置
        'axes.labelweight': 'bold',  # 标签加粗
        
        # 网格设置
        'grid.alpha': 0.3,  # 网格透明度
    })


def main():
    # 统一设置所有大小参数
    plot_settings = {
        # 字体大小设置
        'font_size': 12,  # 基础字体大小
        'title_font_size': 18,  # 标题字体大小
        'label_font_size': 18,  # 标签字体大小
        'tick_font_size': 18,  # 刻度字体大小
        'legend_font_size': 16,  # 图例字体大小
        
        # 标记和线宽设置
        'marker_size': 15,  # 线图标记大小
        'scatter_size': 120,  # 散点图大小
        'line_width': 2,  # 线宽
        'grid_line_width': 1,  # 网格线宽
        'vector_line_width': 0.8,  # 偏好向量线宽
        
        # 箭头大小设置
        'arrow_size': 10,  # 箭头大小
    }
    
    # 算法名称映射字典
    algorithm_name_mapping = {
        'BAMBO (ours)': 'BAMBO (ours)',
        'task-arithmetic': 'TA',
        'ties': 'TIES',
        'dare-ties': 'DARE-TIES',
        'breadcrumbs': 'Breadcrumbs',
        'della': 'Della',
        # 原始模型名称映射
        'models/Qwen3-4B-Instruct-2507': 'Qwen3-4b-Instruct',
        'models/Qwen3-4B-thinking-2507': 'Qwen3-4b-thinking'
    }
    
    parser = argparse.ArgumentParser(description="Checkpoint Analyzer Tool")
    
    # 功能选择
    parser.add_argument('--plot_metrics', default=True, action='store_true', help='绘制不同数据集的指标')
    parser.add_argument('--plot_iterations', default=True, action='store_true', help='绘制迭代趋势（3D和2D函数空间图 + 数据集指标图）')
    parser.add_argument('--plot_trend', default=True, action='store_true', help='绘制迭代趋势图（每个目标的最优值趋势和帕累托点数量趋势）')
    parser.add_argument('--select_individuals', default=True, action='store_true', help='根据偏好向量选择个体')
    
    """
    parser.add_argument('--checkpoint_path', nargs='+', default=[
    './checkpoints/instruct_saasbo_qnehvi_prior_block_36/checkpoint_iter_20.pt',
    './checkpoints/instruct_saasbo_qnehvi_block_36/checkpoint_iter_20.pt'
    ], type=str, help='checkpoint文件路径列表，支持多个路径进行对比') 

    parser.add_argument('--legend_names', nargs='+', default=[
    'SIP-BMM',
    'BMM'
    ], type=str, help='checkpoint数据的legend名称列表，与checkpoint_path一一对应，在可视化环节中优先使用这些名称')
    """

    """
    parser.add_argument('--checkpoint_path', nargs='+', default=[
    './checkpoints/instruct_saasbo_qnehvi_prior_block_36/checkpoint_iter_20.pt',
    './checkpoints/model_level_test_ins_88_breadcrumbs/checkpoint_latest.pt',
    './checkpoints/model_level_test_ins_88_dare_linear/checkpoint_latest.pt',
    './checkpoints/model_level_test_ins_88_della_linear/checkpoint_latest.pt',
    './checkpoints/model_level_test_ins_88_ties/checkpoint_latest.pt',
    './checkpoints/model_level_test_ins_88_task_arithmetic/checkpoint_latest.pt'
    ], type=str, help='checkpoint文件路径列表，支持多个路径进行对比') 

    parser.add_argument('--legend_names', nargs='+', default=[
    'SIP-BMM (ours)',
    'Breadcrumbs',
    'DARE',
    'DELLA',
    'TIES',
    'TA'
    ], type=str, help='checkpoint数据的legend名称列表，与checkpoint_path一一对应，在可视化环节中优先使用这些名称')
    """
    
    
    
    parser.add_argument('--checkpoint_path', nargs='+', default=None, type=str, help='checkpoint文件路径列表，支持多个路径进行对比')
    parser.add_argument('--legend_names', nargs='+', default=None, type=str, help='checkpoint数据的legend名称列表，与checkpoint_path一一对应，在可视化环节中优先使用这些名称')
    parser.add_argument('--auto_checkpoint_root', type=str, default=DEFAULT_CHECKPOINT_ROOT, help='自动发现 checkpoint 时使用的根目录')
    parser.add_argument('--auto_run_id', type=str, default=None, help='自动发现 checkpoint 时指定 run 目录名或绝对路径')
    parser.add_argument('--algorithms', nargs='+', default=None, type=str, help='自动发现 checkpoint 时仅保留指定算法，例如 prior_sync qnehvi emm momm grid')
    parser.add_argument('--disable_auto_checkpoint_discovery', action='store_true', help='关闭自动发现 checkpoint，仅使用 --checkpoint_path')
    

    
    # 原始模型参数
    parser.add_argument('--original_models', nargs='+', default=None, type=str, help='原始模型路径列表，用于在图表中显示原始模型效果')
    parser.add_argument('--original_models_results', type=str, default=None, help='原始模型结果文件路径，用于在图表中显示原始模型效果')
    
    # 功能2参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='checkpoint根目录')
    parser.add_argument('--task_id', type=str, default='blcok_bi_exp3_instruct_beta005_block36', help='任务ID')
    parser.add_argument('--interval', type=int, default=1, help='迭代间隔')
    
    # 功能3参数
    parser.add_argument('--num_vectors', type=int, default=5, help='偏好向量数量')
    parser.add_argument('--top_n', type=int, default=1, help='每个偏好向量选择的前N个个体')
    
    args = parser.parse_args()

    auto_discovery_result = None
    if args.checkpoint_path:
        args.checkpoint_path = [_normalize_path(path) for path in args.checkpoint_path]
    elif not args.disable_auto_checkpoint_discovery:
        auto_discovery_result = discover_algorithm_checkpoints(
            checkpoint_root=args.auto_checkpoint_root,
            run_id=args.auto_run_id,
            algorithms=args.algorithms,
        )
        args.checkpoint_path = auto_discovery_result['checkpoint_paths']
        if not args.legend_names:
            args.legend_names = auto_discovery_result['legend_names']
        print('自动发现到以下 checkpoints:')
        for checkpoint_path, legend_name in zip(args.checkpoint_path, args.legend_names or []):
            print(f'  - {legend_name}: {checkpoint_path}')

    if args.legend_names and args.checkpoint_path and len(args.legend_names) != len(args.checkpoint_path):
        parser.error('--legend_names 的数量必须与 --checkpoint_path 一致')

    if not args.checkpoint_path:
        parser.error('未提供可用的 checkpoint。请使用 --checkpoint_path，或开启自动发现并确认 --auto_checkpoint_root/--auto_run_id 下存在算法 checkpoint')

    resolved_output_root = auto_discovery_result['run_dir'] if auto_discovery_result else None
    output_dir = resolve_output_dir(args.checkpoint_path, preferred_root=resolved_output_root)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 执行对应功能
    if args.plot_metrics:
        if not args.checkpoint_path:
            parser.error('--plot_metrics 需要提供 --checkpoint_path')
        
        # 读取原始模型结果
        original_models_results = None
        original_models_list = []
        
        # 1. 如果指定了--original_models参数，直接遍历所有缓存文件，找到匹配的模型
        if args.original_models is not None:
            print(f"根据指定的原始模型名称搜索缓存结果: {args.original_models}")
            
            # 遍历evaluation_cache/original目录中的所有文件
            # 直接使用当前目录下的evaluation_cache/original，不使用args.checkpoint_dir作为前缀
            original_dir = os.path.join('evaluation_cache', 'original')
            if os.path.exists(original_dir):
                for file_name in os.listdir(original_dir):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(original_dir, file_name)
                        # 读取文件内容
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                cached_result = json.load(f)
                                # 获取模型路径和名称
                                model_path = cached_result.get('model_path', '')
                                model_name = cached_result.get('model_name', '')
                                
                                print(model_name)
                                # 检查模型是否匹配
                                for target_model in args.original_models:
                                    # 检查模型路径或名称是否包含目标模型名称，忽略大小写
                                    if (target_model.lower() in model_path.lower() or 
                                        target_model.lower() in model_name.lower() or 
                                        target_model.replace('-', '').lower() in model_path.lower() or 
                                        target_model.replace('-', '').lower() in model_name.lower()):
                                        original_models_list.append(cached_result)
                                        print(f"找到模型 {target_model} 的缓存结果: {file_path}")
                                        break
                            except Exception as e:
                                print(f"读取缓存文件失败: {file_path}, 错误: {e}")
            else:
                print(f"原始模型缓存目录不存在: {original_dir}")
        
        # 2. 如果指定了--original_models_results参数，尝试从该路径加载结果
        elif args.original_models_results:
            if os.path.isfile(args.original_models_results):
                print(f"从文件读取原始模型结果: {args.original_models_results}")
                with open(args.original_models_results, 'r', encoding='utf-8') as f:
                    original_models_results = json.load(f)
            elif os.path.isdir(args.original_models_results):
                print(f"从目录读取原始模型结果: {args.original_models_results}")
                # 遍历目录中的所有json文件
                for file_name in os.listdir(args.original_models_results):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(args.original_models_results, file_name)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            model_result = json.load(f)
                            if isinstance(model_result, dict) and 'metrics' in model_result:
                                original_models_list.append(model_result)
        
        # 构建最终的原始模型结果格式
        if original_models_list:
            original_models_results = {'original_models': original_models_list}
        
        plot_metrics_by_dataset(args.checkpoint_path, output_dir, args.legend_names, algorithm_name_mapping, original_models_results, plot_settings)
    
    if args.plot_iterations:
        plot_iteration_trends(args.checkpoint_dir, args.task_id, args.interval, output_dir, include_trend=False, algorithm_name_mapping=algorithm_name_mapping, plot_settings=plot_settings, args=args)
    
    if args.plot_trend:
        # plot_trend功能固定间隔为1，忽略用户指定的interval
        plot_iteration_trends(args.checkpoint_dir, args.task_id, 1, output_dir, include_trend=True, algorithm_name_mapping=algorithm_name_mapping, plot_settings=plot_settings, args=args)
    
    # 独立执行个体选择功能，不依赖于plot_trend选项
    if args.select_individuals:
        if not args.checkpoint_path:
            parser.error('--select_individuals 需要提供 --checkpoint_path')
        # 遍历所有checkpoint路径，分别执行个体选择功能
        for checkpoint_path in args.checkpoint_path:
            select_pareto_individuals(checkpoint_path, args.num_vectors, args.top_n, output_dir, algorithm_name_mapping=algorithm_name_mapping, plot_settings=plot_settings)
    
    if not any([args.plot_metrics, args.plot_iterations, args.plot_trend, args.select_individuals]):
        parser.error('请至少选择一个功能：--plot_metrics, --plot_iterations, --plot_trend, 或 --select_individuals')


if __name__ == "__main__":
    # 在主函数调用前设置统一的绘图参数
    setup_plot_params()
    main()
