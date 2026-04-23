#!/usr/bin/env python3
"""
Solution Model Reproduction and Evaluation Module
Reads the latest solution set from checkpoint for the corresponding task ID,
reproduces and evaluates models for each solution, and generates visualization results.
Also includes testing of original thinking and instruct models for comparison.
With result caching functionality to avoid redundant evaluations.
"""
import re
import os
import sys
import time
import uuid
import json
import hashlib
import numpy as np
import torch
import shutil
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import tempfile
from evalscope import run_task
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# 导入可视化模块
from src.evoMI.optimization_reporting import reporter

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入所需模块
from src.evoMI.vllm_server_manager import VllmServerManager
from src.evoMI.result_processor import ResultProcessor
from src.evoMI.mi_block_fusion import mi_block_fusion, calculate_merged_blocks

available_gpus_global = [0,1,2,3]
def get_eval_profile_datasets(eval_profile: str = "aime_gpqa") -> List[str]:
    profile = str(eval_profile or "aime_gpqa").strip().lower()
    if profile == "gsm8k_gpqa":
        return ["gsm8k", "gpqa_diamond"]
    if profile == "math500_level5_gpqa":
        return ["math_500", "gpqa_diamond"]
    if profile == "aime_gpqa":
        return ["aime25", "gpqa_diamond"]
    raise ValueError(f"不支持的评测配置: {eval_profile}")


def normalize_eval_limit(eval_profile: str = "aime_gpqa", eval_limit: Optional[Dict[str, Optional[int]]] = None) -> Dict[str, Optional[int]]:
    datasets = get_eval_profile_datasets(eval_profile)
    if eval_limit is None:
        if eval_profile == "gsm8k_gpqa":
            eval_limit = {"gsm8k": 50, "gpqa_diamond": 50}
        elif eval_profile == "math500_level5_gpqa":
            eval_limit = {"math_500": None, "gpqa_diamond": None}
        else:
            eval_limit = {"aime25": 5, "gpqa_diamond": 60}
    normalized = {}
    for dataset in datasets:
        value = eval_limit.get(dataset, None) if isinstance(eval_limit, dict) else None
        normalized[dataset] = None if value is None else int(value)
    return normalized


def normalize_eval_repeats(eval_profile: str = "aime_gpqa", repeats: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    datasets = get_eval_profile_datasets(eval_profile)
    if repeats is None:
        repeats = {dataset: 1 for dataset in datasets}
    normalized = {}
    for dataset in datasets:
        normalized[dataset] = int(repeats.get(dataset, 1))
    return normalized


def build_eval_cache_config(
    eval_profile: str = "aime_gpqa",
    eval_limit: Optional[Dict[str, Optional[int]]] = None,
    repeats: Optional[Dict[str, int]] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    profile = str(eval_profile or "aime_gpqa").strip().lower()
    return {
        "eval_profile": profile,
        "datasets": get_eval_profile_datasets(profile),
        "limit": normalize_eval_limit(profile, eval_limit),
        "repeats": normalize_eval_repeats(profile, repeats),
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "seed": None if seed is None else int(seed),
    }


def build_eval_cache_namespace(eval_config: Optional[Dict[str, Any]] = None) -> str:
    if not isinstance(eval_config, dict) or not eval_config:
        return "legacy"
    normalized_payload = {
        "eval_profile": str(eval_config.get("eval_profile", "unknown")).strip().lower(),
        "datasets": [str(item) for item in eval_config.get("datasets", [])],
        "limit": eval_config.get("limit", {}),
        "repeats": eval_config.get("repeats", {}),
        "max_tokens": eval_config.get("max_tokens"),
        "seed": eval_config.get("seed"),
    }
    digest = hashlib.md5(
        json.dumps(normalized_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    dataset_part = "-".join(normalized_payload["datasets"]) if normalized_payload["datasets"] else "datasets"
    limit_part = "_".join(
        f"{key}{'all' if value is None else int(value)}"
        for key, value in sorted((normalized_payload["limit"] or {}).items())
    ) or "limit"
    repeat_part = "_".join(
        f"{key}x{int(value)}"
        for key, value in sorted((normalized_payload["repeats"] or {}).items())
    ) or "repeat"
    token_part = "tokall" if normalized_payload["max_tokens"] is None else f"tok{int(normalized_payload['max_tokens'])}"
    seed_part = "seednone" if normalized_payload["seed"] is None else f"seed{int(normalized_payload['seed'])}"
    raw_namespace = f"{normalized_payload['eval_profile']}__{dataset_part}__{limit_part}__{repeat_part}__{token_part}__{seed_part}__{digest}"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", raw_namespace)


def build_eval_setting_id(eval_config: Optional[Dict[str, Any]] = None) -> str:
    return build_eval_cache_namespace(eval_config)


def create_eval_task_config(
    model_path: str,
    max_tokens: Optional[int] = None,
    eval_profile: str = "aime_gpqa",
    eval_limit: Optional[Dict[str, Optional[int]]] = None,
    repeats: Optional[Dict[str, int]] = None,
    seed: Optional[int] = None,
):
    from src.config_manager import config_manager

    profile = str(eval_profile or "aime_gpqa").strip().lower()
    normalized_limit = normalize_eval_limit(profile, eval_limit)
    normalized_repeats = normalize_eval_repeats(profile, repeats)
    if profile == "gsm8k_gpqa":
        return config_manager.create_gsm8k_gpqa_task_config(
            model_path=model_path,
            max_tokens=max_tokens,
            limit=normalized_limit,
            repeats=normalized_repeats,
            seed=42 if seed is None else int(seed),
        )
    if profile == "math500_level5_gpqa":
        return config_manager.create_math500_level5_gpqa_task_config(
            model_path=model_path,
            max_tokens=max_tokens,
            limit=normalized_limit,
            repeats=normalized_repeats,
            seed=42 if seed is None else int(seed),
        )
    if profile == "aime_gpqa":
        return config_manager.create_aime_gpqa_task_config(
            model_path=model_path,
            max_tokens=max_tokens,
            limit=normalized_limit,
            repeats=normalized_repeats,
            seed=seed,
        )
    raise ValueError(f"不支持的评测配置: {eval_profile}")


def collect_dataset_metrics(result_obj: Any, dataset_names: List[str]) -> Dict[str, Dict[str, Any]]:
    metrics = {}
    if isinstance(result_obj, dict) and 'processed_results' in result_obj:
        results_list = result_obj['processed_results']
    elif isinstance(result_obj, list):
        results_list = result_obj
    else:
        results_list = [result_obj]

    for result in results_list:
        if not isinstance(result, dict):
            continue
        for dataset_name in dataset_names:
            dataset_metrics = result.get(dataset_name)
            if not isinstance(dataset_metrics, dict):
                continue
            if dataset_name not in metrics:
                metrics[dataset_name] = {}
            metrics[dataset_name].update(dataset_metrics)
    return metrics


def generate_model_cache_key(model_path: str, eval_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a unique cache key for a model based on its path
    
    Args:
        model_path: Path to the model
    
    Returns:
        str: Cache key
    """
    cache_payload = {"model_path": model_path}
    if eval_config is not None:
        cache_payload["eval_config"] = eval_config
    key_text = json.dumps(cache_payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(key_text.encode()).hexdigest()

def get_model_cache_path(checkpoint_dir: str, model_key: str, cache_type: str = 'original') -> str:
    """
    Get the cache path for a model evaluation result
    
    Args:
        checkpoint_dir: Checkpoint directory
        model_key: Model cache key
        cache_type: Type of cache ('original' or 'solution')
    
    Returns:
        str: Cache file path
    """
    cache_dir = os.path.join(checkpoint_dir, "evaluation_cache", cache_type)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{model_key}.json")


def get_model_cache_candidate_paths(
    checkpoint_dir: str,
    model_key: str,
    cache_type: str = 'original',
    eval_config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    candidates = []
    if eval_config is not None:
        namespaced_dir = os.path.join(
            checkpoint_dir,
            "evaluation_cache",
            cache_type,
            build_eval_cache_namespace(eval_config),
        )
        os.makedirs(namespaced_dir, exist_ok=True)
        candidates.append(os.path.join(namespaced_dir, f"{model_key}.json"))
    legacy_path = get_model_cache_path(checkpoint_dir, model_key, cache_type)
    if legacy_path not in candidates:
        candidates.append(legacy_path)
    return candidates


def get_compatible_model_cache_candidate_paths(
    checkpoint_dir: str,
    model_path: str,
    cache_type: str = 'original',
    eval_config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    candidates = []
    primary_key = generate_model_cache_key(model_path, eval_config=eval_config)
    for candidate in get_model_cache_candidate_paths(
        checkpoint_dir,
        primary_key,
        cache_type,
        eval_config=eval_config,
    ):
        if candidate not in candidates:
            candidates.append(candidate)
    legacy_key = generate_model_cache_key(model_path)
    legacy_path = get_model_cache_path(checkpoint_dir, legacy_key, cache_type)
    if legacy_path not in candidates:
        candidates.append(legacy_path)
    return candidates


def load_cached_results(cache_path: str, expected_eval_config: Optional[Dict[str, Any]] = None) -> Optional[Dict]:

    """
    Load cached evaluation results if they exist
    
    Args:
    
    Returns:
        Optional[Dict]: Cached results or None if not found
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                print(f"Loading cached results from: {cache_path}")
                cached = json.load(f)
                if expected_eval_config is None:
                    return cached
                cached_eval_config = cached.get("eval_config")
                if cached_eval_config is None:
                    return None
                if build_eval_cache_namespace(cached_eval_config) != build_eval_cache_namespace(expected_eval_config):
                    return None
                return cached
        except Exception as e:
            print(f"Error loading cache from {cache_path}: {e}")
    return None


def load_compatible_cached_results(
    cache_path: str,
    expected_eval_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict]:
    cached = load_cached_results(cache_path, expected_eval_config=expected_eval_config)
    if cached is not None:
        return cached
    if expected_eval_config is None or not os.path.exists(cache_path):
        return None
    legacy_cached = load_cached_results(cache_path, expected_eval_config=None)
    if not isinstance(legacy_cached, dict):
        return None
    if "metrics" not in legacy_cached:
        return None
    return legacy_cached

def save_results_to_cache(cache_path: str, results: Dict, eval_config: Optional[Dict[str, Any]] = None):
    """
    Save evaluation results to cache
    
    Args:
        cache_path: Path to cache file
        results: Results to cache
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Add timestamp to cached results
        cached_data = results.copy()
        cached_data['cached_at'] = datetime.now().isoformat()
        if eval_config is not None:
            cached_data['eval_config'] = eval_config
            cached_data['cache_namespace'] = build_eval_cache_namespace(eval_config)
            cached_data['cache_schema_version'] = 2
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cached_data, f, indent=2, ensure_ascii=False)
        print(f"Results cached to: {cache_path}")
    except Exception as e:
        print(f"Error saving cache to {cache_path}: {e}")


def load_model_level_results(model_level_dir: str) -> List[Dict]:
    """
    加载model_level_fusion_test.py生成的模型级融合结果
    
    参数:
        model_level_dir: model_level_fusion_test.py生成的结果目录
    
    返回:
        List[Dict]: 模型级融合结果列表
    """
    model_level_results = []
    
    # 读取results.json文件
    results_json_path = os.path.join(model_level_dir, "results.json")
    if os.path.exists(results_json_path):
        with open(results_json_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        
        # 提取fusion_method
        fusion_method = results_data.get("fusion_method", "unknown")
        
        # 遍历所有结果，转换为model_reproduction.py需要的格式
        for result in results_data.get("results", []):
            model_level_result = {
                "fusion_method": fusion_method,
                "weight": result.get("weight", 0.0),
                "decision_variables": result.get("decision_variables", []),
                "objectives": result.get("objectives", []),
                "metrics": result.get("metrics", {})
            }
            model_level_results.append(model_level_result)
    
    return model_level_results


def load_latest_checkpoint(checkpoint_dir: str, task_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从checkpoint目录中加载指定任务ID的最新解集
    
    参数:
        checkpoint_dir: 检查点根目录
        task_id: 任务ID
    
    返回:
        Tuple[torch.Tensor, torch.Tensor]: (决策变量, 目标函数值)
    """
    task_dir = os.path.join(checkpoint_dir, task_id)
    latest_checkpoint_path = os.path.join(task_dir, 'checkpoint_latest_selected.pt')
    if os.path.exists(latest_checkpoint_path):
        if not os.path.exists(latest_checkpoint_path):
            print(f"未找到任务 {task_id} 的最新检查点（选择后），尝试使用选择前: {latest_checkpoint_path}")
            latest_checkpoint_path = os.path.join(task_dir, 'checkpoint_latest.pt')
            if not os.path.exists(latest_checkpoint_path):
                raise FileNotFoundError(f"未找到任务 {task_id} 的最新检查点: {latest_checkpoint_path}") 
    print(f"从 {latest_checkpoint_path} 加载检查点...")
    checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
    
    # 获取决策变量和目标函数值
    train_x = checkpoint['train_x']
    train_obj_true = checkpoint['train_obj_true']
    
    print(f"成功加载 {train_x.shape[0]} 个候选解")
    return train_x, train_obj_true


def get_pareto_optimal_points(train_x: torch.Tensor, train_obj: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    从候选解中提取帕累托最优解
    
    参数:
        train_x: 所有候选解的决策变量
        train_obj: 所有候选解的目标函数值
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (帕累托决策变量, 帕累托目标函数值)
    """
    # 转换为numpy数组
    if isinstance(train_x, torch.Tensor):
        train_x = train_x.numpy()
    if isinstance(train_obj, torch.Tensor):
        train_obj = train_obj.numpy()
    
    # 标记是否为帕累托最优
    is_pareto = np.ones(train_x.shape[0], dtype=bool)
    for i in range(train_x.shape[0]):
        # 检查每个点是否被其他点支配
        for j in range(train_x.shape[0]):
            if i != j and is_pareto[j]:
                # 如果点j支配点i
                if np.all(train_obj[j] >= train_obj[i]) and np.any(train_obj[j] > train_obj[i]):
                    is_pareto[i] = False
                    break
    
    pareto_x = train_x[is_pareto]
    pareto_y = train_obj[is_pareto]
    print(f"从 {train_x.shape[0]} 个候选解中提取了 {pareto_x.shape[0]} 个帕累托最优解")
    
    return pareto_x, pareto_y


def weighted_fusion(pareto_x: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    根据权重融合帕累托最优解
    
    参数:
        pareto_x: 帕累托最优解的决策变量
        weights: 权重数组，默认为等权重
    
    返回:
        np.ndarray: 融合后的决策变量
    """
    if weights is None:
        # 默认使用等权重
        weights = np.ones(pareto_x.shape[0]) / pareto_x.shape[0]
    else:
        # 确保权重归一化
        weights = np.array(weights) / np.sum(weights)
    
    print(f"使用权重 {weights} 融合 {pareto_x.shape[0]} 个帕累托最优解")
    
    # 按权重融合
    fused_x = np.zeros(pareto_x.shape[1])
    for i in range(pareto_x.shape[0]):
        fused_x += weights[i] * pareto_x[i]
    
    print(f"融合后的决策变量: {fused_x}")
    return fused_x


def create_legacy_eval_task_config(model_path: str, max_tokens: int = 26000) -> Any:
    """
    创建评测任务配置
    
    参数:
        model_path: 模型路径
    
    返回:
        TaskConfig对象
    """
    # 从config_manager引入配置
    from src.config_manager import config_manager
    
    # 使用config_manager中的create_aime_gpqa_task_config方法
    task_cfg = create_eval_task_config(
        model_path=model_path,
        max_tokens=max_tokens,
        eval_profile="aime_gpqa",
        eval_limit={'aime25': None, 'gpqa_diamond': None},
        repeats={'aime25': 4, 'gpqa_diamond': 4},
    )
    
    # 设置work_dir
    task_cfg.work_dir = './output/evalscope_logs'
    
    return task_cfg


def run_task_with_server(port: int, task_cfg: Any, served_model_name: Optional[str] = None) -> Dict:
    """
    在指定端口的服务器上执行任务
    
    参数:
        port: 服务器端口
        task_cfg: 任务配置对象
    
    返回:
        任务执行结果
    """
    # 更新API URL以使用正确的端口
    task_cfg.api_url = f'http://127.0.0.1:{port}/v1/chat/completions'
    if served_model_name:
        task_cfg.model = served_model_name
    
    print(f"在端口 {port} 上执行任务: 模型={task_cfg.model}, 数据集={task_cfg.datasets}, repeats={task_cfg.repeats}")
    
    try:
        # 执行任务
        result = run_task(task_cfg=task_cfg)
        print(f"端口 {port} 上的任务执行完成")
        return result
    except Exception as e:
        print(f"端口 {port} 上的任务执行出错: {e}")
        return {"error": str(e)}


def extract_metrics(results: Dict) -> Dict:
    """
    从评测结果中提取详细指标
    
    参数:
        results: 评测结果字典
    
    返回:
        Dict: 提取的指标字典
    """
    metrics = {}
    
    # 检查results的结构
    if isinstance(results, dict) and 'processed_results' in results:
        results_list = results['processed_results']
    elif isinstance(results, list):
        results_list = results
    else:
        results_list = [results]
    
    for result in results_list:
        try:
            # 简单提取所有数据集的所有指标，直接使用key作为指标名称
            for dataset_name in ['aime25', 'gpqa_diamond', 'ifeval']:
                if dataset_name in result and isinstance(result[dataset_name], dict):
                    if dataset_name not in metrics:
                        metrics[dataset_name] = {}
                    # 直接将所有键值对复制到metrics中
                    for key, value in result[dataset_name].items():
                        metrics[dataset_name][key] = value
        except Exception as e:
            print(f"提取指标时出错: {e}")
    
    return metrics


def generate_fused_model(fused_x: np.ndarray, model_b_path: str, model_i_path: str, 
                        output_dir: str, merged_blocks: list = None) -> str:
    """
    使用融合后的决策变量生成模型
    
    参数:
        fused_x: 融合后的决策变量
        model_b_path: 基础模型路径
        model_i_path: 指令模型路径
        output_dir: 输出目录
        merged_blocks: 预计算的合并块列表，用于避免重复计算
    
    返回:
        str: 生成的模型路径
    """
    # 确保output_dir存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成模型ID
    model_id = f"fused_model_{uuid.uuid4().hex[:8]}"
    model_output_dir = os.path.join(output_dir, model_id)
    
    print(f"生成融合模型: {model_output_dir}")
    
    # 如果没有提供merged_blocks，则计算自动合并的块
    if merged_blocks is None:
        print("计算自动合并的块...")
        merged_blocks = calculate_merged_blocks(
            task_model_paths=[model_b_path, model_i_path],
            num_blocks=8,
            checkpoint_dir=output_dir
        )
    
    # 调用mi_block_fusion方法进行模型合并
    success = mi_block_fusion(
        base_model_path=model_b_path,
        task_model_paths=[model_b_path, model_i_path],
        block_weights=fused_x.tolist(),
        output_dir=model_output_dir,
        fusion_method="ties",
        copy_from_base=True,
        merged_blocks=merged_blocks,
        num_blocks=8
    )
    
    if not success:
        raise RuntimeError("模型融合失败")
    
    return model_output_dir


def evaluate_model(model_path: str, model_name: str = None, max_tokens: int = 26000, max_model_len: int = None) -> Dict:
    """
    Evaluate model performance (kept for backward compatibility)
    
    Args:
        model_path: Model path
        model_name: Optional model name for identification
    
    Returns:
        Dict: Evaluation metrics
    """
    model_display = model_name if model_name else model_path
    print(f"Starting model evaluation: {model_display}")
    
    # 创建评测任务配置
    task_cfg = create_eval_task_config(model_path, max_tokens)
    
    # 创建任务字典
    task = {
        'task_id': f'task_{uuid.uuid4().hex[:8]}',
        'model_path': model_path,
        'params_dict': {'task_cfg': task_cfg},
        'func_handle': run_task_with_server
    }
    
    # 使用VllmServerManager运行任务
    start_time = time.time()
    
    try:
        # 如果未指定max_model_len，则设置为max_tokens + 3000
        if max_model_len is None:
            max_model_len = max_tokens + 3000
            print(f"未指定max_model_len，将使用默认值: {max_model_len} (max_tokens + 3000)")
        
        with VllmServerManager(available_gpus=available_gpus_global, max_model_len=max_model_len) as server_manager:
            # 调用run_series_tasks方法执行任务
            results = server_manager.run_series_tasks([task])
            
        print(f"Evaluation completed, time taken: {time.time() - start_time:.2f} seconds")
        
        # 使用ResultProcessor处理结果
        print("处理评测结果...")
        result_processor = ResultProcessor()
        # 由于现在results是一个字典，直接处理
        res = result_processor.process_and_save(results)
        
        # 提取指标
        metrics = extract_metrics(res)
        
    except Exception as e:
        print(f"评测过程中发生错误: {e}")
        metrics = {"error": str(e)}
    
    return metrics


def save_solution_results(task_id: str, solutions: List[Dict], checkpoint_dir: str, original_models_results: List[Dict] = None, model_level_results: List[Dict] = None):
    """
    Save all solution results, original models results, and model-level fusion results
    
    Args:
        task_id: Task ID
        solutions: List of solution information dictionaries
        checkpoint_dir: Checkpoint directory
        original_models_results: Optional list of original models results
        model_level_results: Optional list of model-level fusion results
    """
    # 创建保存目录
    output_dir = os.path.join(checkpoint_dir, task_id, "solutions_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存所有解的信息
    all_results = {
        'task_id': task_id,
        'solutions': solutions,
        'timestamp': datetime.now().isoformat(),
        'total_solutions': len(solutions)
    }
    
    # Add original models results if provided
    if original_models_results:
        all_results['original_models'] = original_models_results
    
    # Add model-level fusion results if provided
    if model_level_results:
        all_results['model_level_results'] = model_level_results
    
    # 保存到JSON文件
    json_path = os.path.join(output_dir, "all_solutions_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"所有解的结果已保存到: {json_path}")
    
    # 初始化base和expert模型结果，用于动态归一化
    base_model_results = []
    expert_model_results = []
    
    # 从original_models_results中根据config_manager的固定模型列表分离base和expert模型
    if original_models_results:
        # 导入config_manager获取固定的base和expert模型列表
        from src.config_manager import config_manager
        
        # 遍历原始模型结果，将匹配base_model列表的模型添加到base_model_results
        for model_path in config_manager.base_model:
            for model_result in original_models_results:
                if model_result.get('model_path') == model_path:
                    base_model_results.append(model_result)
                    break
        
        # 遍历原始模型结果，将匹配expert_model列表的模型添加到expert_model_results
        for model_path in config_manager.expert_model:
            for model_result in original_models_results:
                if model_result.get('model_path') == model_path:
                    expert_model_results.append(model_result)
                    break
    
    # Extract all decision variables and metrics for visualization
    decision_variables_list = []
    objectives_list = []
    metrics_list = []
    
    # Add solutions results
    for solution in solutions:
        decision_variables_list.append(solution['decision_variables'])
        if 'metrics' in solution:
            metrics = solution['metrics']
            # 按照mi_opt.py中的动态归一化方式计算f1、f2、f3目标函数
            aime25_acc = 0
            aime25_tokens_num = 0
            gpqa_diamond_acc = 0
            gpqa_diamond_tokens_num = 0
            ifeval_acc = 0
            ifeval_tokens_num = 0
            
            if 'aime25' in metrics:
                aime25_acc = metrics['aime25'].get('mean_acc', 0)
                aime25_tokens_num = metrics['aime25'].get('mean_tokens_num', 0)
            if 'gpqa_diamond' in metrics:
                gpqa_diamond_acc = metrics['gpqa_diamond'].get('mean_acc', 0)
                gpqa_diamond_tokens_num = metrics['gpqa_diamond'].get('mean_tokens_num', 0)
            if 'ifeval' in metrics:
                ifeval_acc = (metrics['ifeval'].get('mean_prompt_level_strict', 0) + 
                             metrics['ifeval'].get('mean_inst_level_strict', 0) + 
                             metrics['ifeval'].get('mean_prompt_level_loose', 0) + 
                             metrics['ifeval'].get('mean_inst_level_loose', 0)) / 4
                ifeval_tokens_num = metrics['ifeval'].get('mean_tokens_num', 0)
            
            # 获取base模型的指标
            base_aime25_acc = base_model_results[0].get('metrics', {}).get('aime25', {}).get('mean_acc', 0.45) if len(base_model_results) > 0 else 0.45
            base_gpqa_diamond_acc = base_model_results[0].get('metrics', {}).get('gpqa_diamond', {}).get('mean_acc', 0.3) if len(base_model_results) > 0 else 0.3
            base_aime25_tokens = base_model_results[1].get('metrics', {}).get('aime25', {}).get('mean_tokens_num', 9000) if len(base_model_results) > 1 else 9000
            base_gpqa_diamond_tokens = base_model_results[1].get('metrics', {}).get('gpqa_diamond', {}).get('mean_tokens_num', 1000) if len(base_model_results) > 1 else 1000
            base_ifeval_tokens = base_model_results[1].get('metrics', {}).get('ifeval', {}).get('mean_tokens_num', 500) if len(base_model_results) > 1 else 500
            
            # 获取expert模型的指标（取第一个expert模型）
            expert_aime25_acc = expert_model_results[0].get('metrics', {}).get('aime25', {}).get('mean_acc', 0.8) if len(expert_model_results) > 0 else 0.8
            expert_gpqa_diamond_acc = expert_model_results[0].get('metrics', {}).get('gpqa_diamond', {}).get('mean_acc', 0.7) if len(expert_model_results) > 0 else 0.7
            expert_aime25_tokens = expert_model_results[1].get('metrics', {}).get('aime25', {}).get('mean_tokens_num', 22000) if len(expert_model_results) > 1 else 22000
            expert_gpqa_diamond_tokens = expert_model_results[1].get('metrics', {}).get('gpqa_diamond', {}).get('mean_tokens_num', 9000) if len(expert_model_results) > 1 else 9000
            expert_ifeval_tokens = expert_model_results[1].get('metrics', {}).get('ifeval', {}).get('mean_tokens_num', 2000) if len(expert_model_results) > 1 else 2000
            
            # 计算f1：使用aime25和gpqa_diamond的准确率进行归一化
            # 避免除以零的情况
            aime25_denominator = expert_aime25_acc - base_aime25_acc
            gpqa_diamond_denominator = expert_gpqa_diamond_acc - base_gpqa_diamond_acc
            aime25_norm = (aime25_acc - base_aime25_acc) / aime25_denominator
            gpqa_diamond_norm = (gpqa_diamond_acc - base_gpqa_diamond_acc) / gpqa_diamond_denominator
            f1 = np.mean([aime25_norm, gpqa_diamond_norm])
            
            # 计算f2：使用token数量进行归一化
            aime25_tokens_denominator = expert_aime25_tokens - base_aime25_tokens
            gpqa_diamond_tokens_denominator = expert_gpqa_diamond_tokens - base_gpqa_diamond_tokens
            ifeval_tokens_denominator = expert_ifeval_tokens - base_ifeval_tokens
            aime25_tokens_norm = (aime25_tokens_num - base_aime25_tokens) / aime25_tokens_denominator
            gpqa_diamond_tokens_norm = (gpqa_diamond_tokens_num - base_gpqa_diamond_tokens) / gpqa_diamond_tokens_denominator
            ifeval_tokens_norm = (ifeval_tokens_num - base_ifeval_tokens) / ifeval_tokens_denominator
            f2 = np.mean([aime25_tokens_norm, gpqa_diamond_tokens_norm, ifeval_tokens_norm])
            
            # 计算f3：直接使用ifeval的准确率
            f3 = ifeval_acc
            
            # 对于solution类型的结果，显示为evoBMI (ours)
            display_type = 'evoBMI (ours)'
            objectives_list.append([f1, f2, f3, display_type])
            # 确保添加type: 'evoBMI (ours)'标记，并保留原始metrics结构
            solution_metrics = {
                'type': display_type,
                'index': solution.get('index', -1),
                'f1': f1,
                'f2': f2,
                'f3': f3,
                **metrics
            }
            metrics_list.append(solution_metrics)
    
    # Add original models results if provided
    if original_models_results:
        for model_result in original_models_results:
            if 'metrics' in model_result:
                metrics = model_result['metrics']
                # 为原始模型使用与解决方案相同的动态归一化方式计算目标函数
                aime25_acc = 0
                aime25_tokens_num = 0
                gpqa_diamond_acc = 0
                gpqa_diamond_tokens_num = 0
                ifeval_acc = 0
                ifeval_tokens_num = 0
                
                if 'aime25' in metrics:
                    aime25_acc = metrics['aime25'].get('mean_acc', 0)
                    aime25_tokens_num = metrics['aime25'].get('mean_tokens_num', 0)
                if 'gpqa_diamond' in metrics:
                    gpqa_diamond_acc = metrics['gpqa_diamond'].get('mean_acc', 0)
                    gpqa_diamond_tokens_num = metrics['gpqa_diamond'].get('mean_tokens_num', 0)
                if 'ifeval' in metrics:
                    ifeval_acc = (metrics['ifeval'].get('mean_prompt_level_strict', 0) + 
                                 metrics['ifeval'].get('mean_inst_level_strict', 0) + 
                                 metrics['ifeval'].get('mean_prompt_level_loose', 0) + 
                                 metrics['ifeval'].get('mean_inst_level_loose', 0)) / 4
                    ifeval_tokens_num = metrics['ifeval'].get('mean_tokens_num', 0)
                
                # 获取base模型的指标
                base_aime25_acc = base_model_results[0].get('metrics', {}).get('aime25', {}).get('mean_acc', 0.45) if len(base_model_results) > 0 else 0.45
                base_gpqa_diamond_acc = base_model_results[0].get('metrics', {}).get('gpqa_diamond', {}).get('mean_acc', 0.3) if len(base_model_results) > 0 else 0.3
                base_aime25_tokens = base_model_results[1].get('metrics', {}).get('aime25', {}).get('mean_tokens_num', 9000) if len(base_model_results) > 1 else 9000
                base_gpqa_diamond_tokens = base_model_results[1].get('metrics', {}).get('gpqa_diamond', {}).get('mean_tokens_num', 1000) if len(base_model_results) > 1 else 1000
                base_ifeval_tokens = base_model_results[1].get('metrics', {}).get('ifeval', {}).get('mean_tokens_num', 500) if len(base_model_results) > 1 else 500
                
                # 获取expert模型的指标（取第一个expert模型）
                expert_aime25_acc = expert_model_results[0].get('metrics', {}).get('aime25', {}).get('mean_acc', 0.8) if len(expert_model_results) > 0 else 0.8
                expert_gpqa_diamond_acc = expert_model_results[0].get('metrics', {}).get('gpqa_diamond', {}).get('mean_acc', 0.7) if len(expert_model_results) > 0 else 0.7
                expert_aime25_tokens = expert_model_results[1].get('metrics', {}).get('aime25', {}).get('mean_tokens_num', 22000) if len(expert_model_results) > 1 else 22000
                expert_gpqa_diamond_tokens = expert_model_results[1].get('metrics', {}).get('gpqa_diamond', {}).get('mean_tokens_num', 9000) if len(expert_model_results) > 1 else 9000
                expert_ifeval_tokens = expert_model_results[1].get('metrics', {}).get('ifeval', {}).get('mean_tokens_num', 2000) if len(expert_model_results) > 1 else 2000
                
                # 计算f1：使用aime25和gpqa_diamond的准确率进行归一化
                # 避免除以零的情况
                aime25_denominator = expert_aime25_acc - base_aime25_acc
                gpqa_diamond_denominator = expert_gpqa_diamond_acc - base_gpqa_diamond_acc
                aime25_norm = (aime25_acc - base_aime25_acc) / aime25_denominator
                gpqa_diamond_norm = (gpqa_diamond_acc - base_gpqa_diamond_acc) / gpqa_diamond_denominator
                f1 = np.mean([aime25_norm, gpqa_diamond_norm])
                
                # 计算f2：使用token数量进行归一化
                aime25_tokens_denominator = expert_aime25_tokens - base_aime25_tokens
                gpqa_diamond_tokens_denominator = expert_gpqa_diamond_tokens - base_gpqa_diamond_tokens
                ifeval_tokens_denominator = expert_ifeval_tokens - base_ifeval_tokens
                aime25_tokens_norm = (aime25_tokens_num - base_aime25_tokens) / aime25_tokens_denominator
                gpqa_diamond_tokens_norm = (gpqa_diamond_tokens_num - base_gpqa_diamond_tokens) / gpqa_diamond_tokens_denominator
                ifeval_tokens_norm = (ifeval_tokens_num - base_ifeval_tokens) / ifeval_tokens_denominator
                f2 = np.mean([aime25_tokens_norm, gpqa_diamond_tokens_norm, ifeval_tokens_norm])
                
                # 计算f3：直接使用ifeval的准确率
                f3 = ifeval_acc
                
                model_type = model_result.get('model_type', 'original')
                # 对于默认算法，显示为evoBMI (ours)
                if model_type == 'solution':
                    display_type = 'evoBMI (ours)'
                else:
                    display_type = model_type
                objectives_list.append([f1, f2, f3, display_type])
                metrics_list.append({
                    'type': display_type,
                    'name': model_result.get('model_name', 'Unknown'),
                    'f1': f1,
                    'f2': f2,
                    'f3': f3,
                    **metrics
                })
    
    # Add model-level fusion results if provided
    if model_level_results:
        for model_level_result in model_level_results:
            if 'metrics' in model_level_result:
                metrics = model_level_result['metrics']
                # 为模型级融合结果使用相同的动态归一化方式计算目标函数
                aime25_acc = 0
                aime25_tokens_num = 0
                gpqa_diamond_acc = 0
                gpqa_diamond_tokens_num = 0
                ifeval_acc = 0
                ifeval_tokens_num = 0
                
                if 'aime25' in metrics:
                    aime25_acc = metrics['aime25'].get('mean_acc', 0)
                    aime25_tokens_num = metrics['aime25'].get('mean_tokens_num', 0)
                if 'gpqa_diamond' in metrics:
                    gpqa_diamond_acc = metrics['gpqa_diamond'].get('mean_acc', 0)
                    gpqa_diamond_tokens_num = metrics['gpqa_diamond'].get('mean_tokens_num', 0)
                if 'ifeval' in metrics:
                    ifeval_acc = (metrics['ifeval'].get('mean_prompt_level_strict', 0) + 
                                 metrics['ifeval'].get('mean_inst_level_strict', 0) + 
                                 metrics['ifeval'].get('mean_prompt_level_loose', 0) + 
                                 metrics['ifeval'].get('mean_inst_level_loose', 0)) / 4
                    ifeval_tokens_num = metrics['ifeval'].get('mean_tokens_num', 0)
                
                # 获取base模型的指标
                base_aime25_acc = base_model_results[0].get('metrics', {}).get('aime25', {}).get('mean_acc', 0.45) if len(base_model_results) > 0 else 0.45
                base_gpqa_diamond_acc = base_model_results[0].get('metrics', {}).get('gpqa_diamond', {}).get('mean_acc', 0.3) if len(base_model_results) > 0 else 0.3
                base_aime25_tokens = base_model_results[1].get('metrics', {}).get('aime25', {}).get('mean_tokens_num', 9000) if len(base_model_results) > 1 else 9000
                base_gpqa_diamond_tokens = base_model_results[1].get('metrics', {}).get('gpqa_diamond', {}).get('mean_tokens_num', 1000) if len(base_model_results) > 1 else 1000
                base_ifeval_tokens = base_model_results[1].get('metrics', {}).get('ifeval', {}).get('mean_tokens_num', 500) if len(base_model_results) > 1 else 500
                
                # 获取expert模型的指标（取第一个expert模型）
                expert_aime25_acc = expert_model_results[0].get('metrics', {}).get('aime25', {}).get('mean_acc', 0.8) if len(expert_model_results) > 0 else 0.8
                expert_gpqa_diamond_acc = expert_model_results[0].get('metrics', {}).get('gpqa_diamond', {}).get('mean_acc', 0.7) if len(expert_model_results) > 0 else 0.7
                expert_aime25_tokens = expert_model_results[1].get('metrics', {}).get('aime25', {}).get('mean_tokens_num', 22000) if len(expert_model_results) > 1 else 22000
                expert_gpqa_diamond_tokens = expert_model_results[1].get('metrics', {}).get('gpqa_diamond', {}).get('mean_tokens_num', 9000) if len(expert_model_results) > 1 else 9000
                expert_ifeval_tokens = expert_model_results[1].get('metrics', {}).get('ifeval', {}).get('mean_tokens_num', 2000) if len(expert_model_results) > 1 else 2000
                
                # 计算f1：使用aime25和gpqa_diamond的准确率进行归一化
                # 避免除以零的情况
                aime25_denominator = expert_aime25_acc - base_aime25_acc
                gpqa_diamond_denominator = expert_gpqa_diamond_acc - base_gpqa_diamond_acc
                aime25_norm = (aime25_acc - base_aime25_acc) / aime25_denominator
                gpqa_diamond_norm = (gpqa_diamond_acc - base_gpqa_diamond_acc) / gpqa_diamond_denominator
                f1 = np.mean([aime25_norm, gpqa_diamond_norm])
                
                # 计算f2：使用token数量进行归一化
                aime25_tokens_denominator = expert_aime25_tokens - base_aime25_tokens
                gpqa_diamond_tokens_denominator = expert_gpqa_diamond_tokens - base_gpqa_diamond_tokens
                ifeval_tokens_denominator = expert_ifeval_tokens - base_ifeval_tokens
                aime25_tokens_norm = (aime25_tokens_num - base_aime25_tokens) / aime25_tokens_denominator
                gpqa_diamond_tokens_norm = (gpqa_diamond_tokens_num - base_gpqa_diamond_tokens) / gpqa_diamond_tokens_denominator
                ifeval_tokens_norm = (ifeval_tokens_num - base_ifeval_tokens) / ifeval_tokens_denominator
                f2 = np.mean([aime25_tokens_norm, gpqa_diamond_tokens_norm, ifeval_tokens_norm])
                
                # 计算f3：直接使用ifeval的准确率
                f3 = ifeval_acc
                
                model_type = f"model_level_{model_level_result.get('fusion_method', 'unknown')}"
                objectives_list.append([f1, f2, f3, model_type])
                metrics_list.append({
                    'type': model_type,
                    'name': f"model_level_{model_level_result.get('weight', 'unknown')}",
                    'f1': f1,
                    'f2': f2,
                    'f3': f3,
                    **metrics
                })
    
    # 保存决策变量和目标值
    if decision_variables_list:
        np.save(os.path.join(output_dir, "decision_variables.npy"), np.array(decision_variables_list))
    if objectives_list:
        np.save(os.path.join(output_dir, "objectives.npy"), np.array(objectives_list))
    
    return output_dir, objectives_list, metrics_list

def extract_info_regex(text):
    # 使用正则表达式匹配
    pattern = r'merged-(.+?)-task_weight-([0-9.]+)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)+'-'+match.group(2)
    return text

def evaluate_original_models(models_dir: str = 'models', checkpoint_dir: str = None, max_tokens: int = 26000, max_model_len: int = None, strategy: str = 'models_first') -> List[Dict]:
    """
    Evaluate all original models with caching support
    
    Args:
        models_dir: Directory containing models to evaluate
        checkpoint_dir: Checkpoint directory for caching
        max_tokens: Max tokens parameter for generation config
        max_model_len: Max model length parameter for vllm server
        strategy: Model reading strategy, options: 
                 'models_first' - read models from models_dir and find corresponding checkpoints
                 'checkpoint_first' - read all models from checkpoint and create empty folders in models_dir if not exist
    
    Returns:
        List[Dict]: Results for original models
    """
    # 扫描模型，根据策略选择不同的方式
    original_models = []
    
    # 确保models_dir是绝对路径
    if not os.path.isabs(models_dir):
        models_dir = os.path.abspath(models_dir)
    
    if strategy == 'models_first':
        # 策略1：从models_dir读取模型，然后查找对应的checkpoints
        # 检查models_dir是否存在
        if not os.path.exists(models_dir):
            print(f"Warning: Models directory {models_dir} does not exist")
            return []
        
        # 扫描目录下的所有文件夹作为模型
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                # 尝试从目录名推断模型类型
                model_type = 'thinking' if 'thinking' in item.lower() else 'instruct' if 'instruct' in item.lower() else 'unknown'
                original_models.append({
                    'path': item_path,
                    'name': item,
                    'type': model_type
                })
        
        if not original_models:
            print(f"Warning: No models found in directory {models_dir}")
            return []
    else:  # strategy == 'checkpoint_first'
        # 策略2：从checkpoint的evaluation_cache/original目录读取所有JSON配置文件，
        # 解析model_path和其他属性，并在models_dir创建对应的空文件夹（如果不存在）
        if not checkpoint_dir:
            print(f"Error: checkpoint_dir is required for 'checkpoint_first' strategy")
            return []
        
        # 确保models_dir存在
        os.makedirs(models_dir, exist_ok=True)
        
        # 从checkpoint的evaluation_cache/original目录读取所有JSON配置文件
        checkpoint_cache_dir = os.path.join(checkpoint_dir, 'evaluation_cache', 'original')
        if not os.path.exists(checkpoint_cache_dir):
            print(f"Warning: Evaluation cache directory not found: {checkpoint_cache_dir}")
            return []
        
        # 扫描checkpoint下的所有JSON配置文件
        for item in os.listdir(checkpoint_cache_dir):
            if item.endswith('.json'):
                cache_file_path = os.path.join(checkpoint_cache_dir, item)
                try:
                    # 读取JSON配置文件
                    with open(cache_file_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    # 检查配置文件中是否包含model_path
                    if 'model_path' in cache_data:
                        model_path = cache_data['model_path']
                        model_name = cache_data.get('model_name', os.path.basename(model_path))
                        model_type = cache_data.get('model_type', 'unknown')
                        
                        # 确保model_path对应的文件夹存在于models_dir中
                        # 提取模型名称
                        model_basename = os.path.basename(model_path)
                        models_dir_model_path = os.path.join(models_dir, model_basename)
                        os.makedirs(models_dir_model_path, exist_ok=True)
                        
                        original_models.append({
                            'path': models_dir_model_path,
                            'name': model_name,
                            'type': model_type
                        })
                        print(f"Loaded model from cache: {model_name} (type: {model_type})")
                except Exception as e:
                    print(f"Error reading cache file {cache_file_path}: {e}")
                    continue
        
        if not original_models:
            print(f"Warning: No models found in checkpoint cache directory {checkpoint_cache_dir}")
            return []
    
    print(f"\n===== Evaluating {len(original_models)} original models =====")
    start_time = time.time()
    
    results = []
    # Models that need evaluation (not in cache)
    models_to_evaluate = []
    
    # Check cache first
    if checkpoint_dir:
        for model in original_models:
            model_key = generate_model_cache_key(model['path'])
            cache_path = get_model_cache_path(checkpoint_dir, model_key, 'original')
            cached_result = load_cached_results(cache_path)

            if cached_result and 'metrics' in cached_result:

                cached_result['model_name'] = extract_info_regex(cached_result['model_name'])
                # Use cached results
                # cached_result['model_type'] = model['type']
                # cached_result['model_name'] = model['name']
                # cached_result['model_path'] = model['path']
                results.append(cached_result)
                print(f"Using cached results for Original {model['name']}")
            else:
                # Add to models to evaluate
                models_to_evaluate.append(model)
    else:
        # No checkpoint dir provided, evaluate all models
        models_to_evaluate = original_models
    
    # Evaluate models not in cache
    if models_to_evaluate:
        print(f"Need to evaluate {len(models_to_evaluate)} original models...")
        
        # Create evaluation tasks
        all_tasks = []
        task_to_model = {}
        
        for model in models_to_evaluate:
            print(f"Preparing evaluation for Original {model['name']}")
            # Create evaluation task config
            task_cfg = create_eval_task_config(model['path'], max_tokens)
            
            # Create task dictionary
            task_id = f'task_original_{uuid.uuid4().hex[:8]}'
            task = {
                'task_id': task_id,
                'model_path': model['path'],
                'params_dict': {'task_cfg': task_cfg},
                'func_handle': run_task_with_server
            }
            
            all_tasks.append(task)
            task_to_model[task_id] = model
        
        # Use VllmServerManager to run all tasks in parallel
        try:
            # 如果未指定max_model_len，则设置为max_tokens + 3000
            if max_model_len is None:
                max_model_len = max_tokens + 3000
                print(f"未指定max_model_len，将使用默认值: {max_model_len} (max_tokens + 3000)")
            
            with VllmServerManager(available_gpus=available_gpus_global, max_model_len=max_model_len) as server_manager:
                # Call run_series_tasks once with all tasks
                task_results = server_manager.run_series_tasks(all_tasks)
            
            print(f"Parallel evaluation of original models completed, time taken: {time.time() - start_time:.2f} seconds")
            
            # Process results for each task
            print("Processing original model evaluation results...")
            result_processor = ResultProcessor()
            
            for task_id, result in task_results.items():
                if task_id in task_to_model:
                    model = task_to_model[task_id]
                    try:
                        # Process individual task result
                        res = result_processor.process_and_save({task_id: result})
                        
                        # Extract metrics
                        metrics = extract_metrics(res)
                        
                        model_result = {
                            'model_type': model['type'],
                            'model_name': model['name'],
                            'model_path': model['path'],
                            'metrics': metrics,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        results.append(model_result)
                        print(f"Original {model['name']} evaluation results processed successfully")
                        
                        # Save to cache if checkpoint_dir is provided
                        if checkpoint_dir:
                            model_key = generate_model_cache_key(model['path'])
                            cache_path = get_model_cache_path(checkpoint_dir, model_key, 'original')
                            save_results_to_cache(cache_path, model_result)
                            
                    except Exception as e:
                        print(f"Error processing results for original {model['name']}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Parallel evaluation of original models failed: {e}")
            # Fallback to sequential evaluation if parallel evaluation fails
            print("Falling back to sequential evaluation of original models...")
            
            for model in models_to_evaluate:
                try:
                    metrics = evaluate_model(model['path'], model['name'], max_tokens, max_model_len)
                    model_result = {
                        'model_type': model['type'],
                        'model_name': model['name'],
                        'model_path': model['path'],
                        'metrics': metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    results.append(model_result)
                    print(f"Original {model['name']} evaluation completed")
                    
                    # Save to cache if checkpoint_dir is provided
                    if checkpoint_dir:
                        model_key = generate_model_cache_key(model['path'])
                        cache_path = get_model_cache_path(checkpoint_dir, model_key, 'original')
                        save_results_to_cache(cache_path, model_result)
                        
                except Exception as e:
                    print(f"Error evaluating original {model['name']}: {e}")
                    continue
    
    return results


def main():
    """
    Main function - Execute solution model reproduction and evaluation process
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Solution Model Reproduction and Evaluation Tool")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", 
                       help="Checkpoint root directory")
    parser.add_argument("--task_id", default="blcok_test0", type=str, 
                       help="Task ID, corresponds to subdirectory name under checkpoint directory")
    parser.add_argument("--models_dir", type=str, default="models", 
                       help="Directory containing models to evaluate")
    parser.add_argument("--skip_evaluation", default=False, action="store_true", 
                       help="Skip model evaluation step")
    parser.add_argument("--use_pareto_only", default=True, action="store_true", 
                       help="Use only Pareto optimal solutions")
    parser.add_argument("--max_tokens", type=int, default=35000, 
                       help="Max tokens parameter for generation config (default: 26000)")
    parser.add_argument("--max_model_len", type=int, default=None, 
                       help="Max model length parameter for vllm server, default is max_tokens + 3000")
    parser.add_argument("--max_solutions", default='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]', 
                       help="Maximum number of solutions to process, can be an integer or a list of indices (e.g., '[3,4,5,7]')")
    parser.add_argument("--skip_original_models", default=False, action="store_true", 
                       help="Skip evaluation of original thinking and instruct models")
    parser.add_argument("--models_only", default=False, action="store_true", 
                       help="Only evaluate models in models_dir without loading checkpoints")
    parser.add_argument("--strategy", type=str, default="checkpoint_first", choices=["models_first", "checkpoint_first"],
                       help="Model reading strategy: 'models_first' (read from models_dir) or 'checkpoint_first' (read from checkpoint)")
    parser.add_argument("--model_level_dirs", type=str, nargs='+', default=None,
                       help="Directories containing model-level fusion results generated by model_level_fusion_test.py")

    args = parser.parse_args()
    
    print(f"Starting solution model reproduction and evaluation process")
    print(f"Task ID: {args.task_id}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Use only Pareto optimal solutions: {args.use_pareto_only}")
    print(f"Evaluate original models: {not args.skip_original_models}")
    print(f"Models only mode: {args.models_only}")
    
    # 处理models_only模式 - 仅评估原始模型，不加载checkpoint
    if args.models_only:
        print("Running in models_only mode - will only evaluate models in models_dir")
        
        # 确保不跳过原始模型评估
        if args.skip_original_models:
            print("Warning: --skip_original_models flag ignored in models_only mode")
            args.skip_original_models = False
        
        # 评估原始模型
        original_models_results = []
        # 无论是否跳过评估，都需要执行模型扫描逻辑
        original_models_results = evaluate_original_models(args.models_dir, args.checkpoint_dir, args.max_tokens, args.max_model_len, args.strategy)
        
        # 创建输出目录
        output_dir = os.path.join(args.checkpoint_dir, args.task_id, "models_only_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始模型结果
        output_dir, objectives_list, metrics_list = save_solution_results(
            args.task_id, [], args.checkpoint_dir, original_models_results
        )
        
        # 生成可视化结果
        print("\nGenerating visualization results...")
        # 使用visualization模块中的函数进行绘图
        reporter.plot_3d_objectives(objectives_list, output_dir)
        
        # 使用save_solution_results返回的metrics_list进行可视化
        reporter.plot_dataset_metrics(output_dir, metrics_list)
        
        # 添加新的可视化图表：所有指标的token平均值 vs 模型能力（f1+f3）
        reporter.plot_token_avg_vs_model_ability(output_dir, metrics_list)
        
        # 添加新的可视化图表：模型能力f1 vs f3
        reporter.plot_f1_vs_f3(output_dir, metrics_list)
        print(f"可视化图表已保存到: {output_dir}")
        
        print(f"\nModels only evaluation completed!")
        print(f"Successfully evaluated {len(original_models_results)} original models")
        print(f"Results saved to: {output_dir}")
        sys.exit(0)
    
    try:
        # 加载最新检查点
        train_x, train_obj_true = load_latest_checkpoint(args.checkpoint_dir, args.task_id)
        
        # 确定要处理的解
        if args.use_pareto_only:
            # 提取帕累托最优解
            solutions_x, solutions_y = get_pareto_optimal_points(train_x, train_obj_true)
            print(f"仅处理 {solutions_x.shape[0]} 个帕累托最优解")
        else:
            # 处理所有解
            solutions_x = train_x.numpy() if isinstance(train_x, torch.Tensor) else train_x
            solutions_y = train_obj_true.numpy() if isinstance(train_obj_true, torch.Tensor) else train_obj_true
            print(f"处理所有 {solutions_x.shape[0]} 个解")
        
        # Store original indices mapping for caching
        original_indices = list(range(solutions_x.shape[0]))
        
        # Limit number of solutions
        if args.max_solutions and args.max_solutions != 0:
            # Check if max_solutions is a list of indices
            if isinstance(args.max_solutions, str) and args.max_solutions.startswith('[') and args.max_solutions.endswith(']'):
                try:
                    # Convert string representation of list to actual list
                    indices = eval(args.max_solutions)
                    if isinstance(indices, list):
                        print(f"Processing specific solution indices: {indices}")
                        # Filter solutions by indices
                        valid_indices = []
                        for idx in indices:
                            if 0 <= idx < solutions_x.shape[0]:
                                valid_indices.append(idx)
                            else:
                                print(f"Warning: Index {idx} out of range, skipping")
                        
                        if valid_indices:
                            # Keep only the valid original indices
                            original_indices = valid_indices
                            # Filter solutions
                            solutions_x = solutions_x[valid_indices]
                            solutions_y = solutions_y[valid_indices] if len(solutions_y) > 0 else []
                            print(f"Successfully filtered to {len(solutions_x)} valid solutions")
                        else:
                            print("Warning: No valid indices provided, using all solutions")
                except Exception as e:
                    print(f"Error parsing max_solutions list: {e}, treating as integer")
                    # Fall back to integer handling
                    max_sols = int(args.max_solutions)
                    if max_sols > 0 and solutions_x.shape[0] > max_sols:
                        print(f"Limiting to maximum {max_sols} solutions")
                        solutions_x = solutions_x[:max_sols]
                        solutions_y = solutions_y[:max_sols]
                        original_indices = original_indices[:max_sols]
            else:
                # Integer handling
                max_sols = int(args.max_solutions)
                if max_sols > 0 and solutions_x.shape[0] > max_sols:
                    print(f"Limiting to maximum {max_sols} solutions")
                    solutions_x = solutions_x[:max_sols]
                    solutions_y = solutions_y[:max_sols]
                    original_indices = original_indices[:max_sols]
        
        # Create output directory
        models_output_dir = os.path.join(args.checkpoint_dir, args.task_id, "reproduced_models")
        os.makedirs(models_output_dir, exist_ok=True)
        
        # 只计算一次自动合并的块，所有模型生成过程复用这个结果
        print("\n===== 计算自动合并的块（仅运行一次）=====")
        merged_blocks = calculate_merged_blocks(
            task_model_paths=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"],
            num_blocks=8,
            checkpoint_dir=models_output_dir
        )
        
        # Evaluate original models if not skipped
        original_models_results = []
        if not args.skip_evaluation and not args.skip_original_models:
            original_models_results = evaluate_original_models(args.models_dir, args.checkpoint_dir, args.max_tokens, args.max_model_len, args.strategy)
        
        # Load model-level fusion results if provided
        model_level_results = []
        if args.model_level_dirs:
            print(f"\n===== 加载模型级融合结果 =====")
            for model_level_dir in args.model_level_dirs:
                print(f"从 {model_level_dir} 加载模型级融合结果...")
                results = load_model_level_results(model_level_dir)
                model_level_results.extend(results)
            print(f"成功加载 {len(model_level_results)} 个模型级融合结果")
        
        # 存储所有解的结果
        all_solutions = []
        
        # Process each solution
        total_solutions = solutions_x.shape[0]
        
        # Generate all models first
        models_to_evaluate = []
        for i in range(total_solutions):
            # Get the original index for caching purposes
            original_idx = original_indices[i]
            print(f"\n===== Processing solution {original_idx} ({i+1}/{total_solutions}) =====")
            
            # Get decision variables for current solution
            current_x = solutions_x[i]
            print(f"Decision variables: {current_x}")
            
            # Generate model with original index
            model_id = f"reproduced_model_{original_idx}"
            model_output_dir = os.path.join(models_output_dir, model_id)
            
            # 检查评测结果是否已存在，而不是检查模型文件夹
            # 创建解决方案缓存目录路径
            solutions_cache_dir = os.path.join(args.checkpoint_dir, args.task_id, "solution_evaluations")
            task_specific_cache_path = os.path.join(solutions_cache_dir, f"solution_{original_idx}_eval.json")
            
            # 生成模型缓存键和通用缓存路径
            model_key = generate_model_cache_key(model_output_dir)
            general_cache_path = get_model_cache_path(args.checkpoint_dir, model_key, 'solution')
            
            # 检查是否存在评测结果缓存
            if load_cached_results(task_specific_cache_path) is not None or load_cached_results(general_cache_path) is not None:
                print(f"评测结果已存在于缓存中，跳过模型融合和评测: solution_{original_idx}")
                success = True
            else:
                # 复用之前计算好的merged_blocks，不再重复计算
                print(f"开始生成融合模型到 {model_output_dir}")
                success = mi_block_fusion(
                    base_model_path="models/Qwen3-4B-thinking-2507",
                    task_model_paths=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"],
                    block_weights=current_x.tolist(),
                    output_dir=model_output_dir,
                    fusion_method="ties",
                    copy_from_base=True,
                    merged_blocks=merged_blocks,
                    num_blocks=8
                )
            
            if not success:
                print(f"Warning: Model generation failed for solution {original_idx}, skipping")
                continue
            
            # Prepare solution info and add to evaluation list if not skipping evaluation
            solution_info = {
                'index': original_idx,  # Use original index
                'decision_variables': current_x.tolist(),
                'objective_values': solutions_y[i].tolist() if i < len(solutions_y) else [],
                'model_path': model_output_dir,
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }
            
            if not args.skip_evaluation:
                models_to_evaluate.append((original_idx, model_output_dir, solution_info))
            else:
                # If skipping evaluation, add directly to all_solutions
                all_solutions.append(solution_info)
            
            print(f"Solution {original_idx} model generated successfully")
                
        
        # Evaluate all models in parallel if not skipping evaluation
        if not args.skip_evaluation and models_to_evaluate:
                print(f"\nEvaluating {len(models_to_evaluate)} models in parallel with caching support...")
                start_time = time.time()
                
                # Create a directory to store solution results in checkpoints
                solutions_cache_dir = os.path.join(args.checkpoint_dir, args.task_id, "solution_evaluations")
                os.makedirs(solutions_cache_dir, exist_ok=True)
                
                # First check for cached results
                solutions_to_evaluate = []
                
                for original_idx, model_path, solution_info in models_to_evaluate:
                    # Generate cache key for this solution
                    model_key = generate_model_cache_key(model_path)
                    cache_path = get_model_cache_path(args.checkpoint_dir, model_key, 'solution')
                    
                    # Also check in the task-specific solution evaluations directory using original index
                    task_specific_cache_path = os.path.join(solutions_cache_dir, f"solution_{original_idx}_eval.json")
                    
                    # Try task-specific cache first, then general cache
                    cached_result = load_cached_results(task_specific_cache_path) or load_cached_results(cache_path)
                    
                    if cached_result and 'metrics' in cached_result:
                        # Use cached results
                        solution_info['metrics'] = cached_result['metrics']
                        print(f"Using cached results for Solution {original_idx}")
                        all_solutions.append(solution_info)
                    else:
                        # Add to solutions to evaluate
                        solutions_to_evaluate.append((original_idx, model_path, solution_info))
                
                # Only evaluate solutions not in cache
                if solutions_to_evaluate:
                    print(f"Need to evaluate {len(solutions_to_evaluate)} solutions...")
                    
                    # Create evaluation tasks
                    all_tasks = []
                    task_to_info = {}
                    
                    for i, model_path, solution_info in solutions_to_evaluate:
                        # Create evaluation task config
                        task_cfg = create_eval_task_config(model_path, args.max_tokens)
                        
                        # Create task dictionary
                        task_id = f'task_{uuid.uuid4().hex[:8]}'
                        task = {
                            'task_id': task_id,
                            'model_path': model_path,
                            'params_dict': {'task_cfg': task_cfg},
                            'func_handle': run_task_with_server
                        }
                        
                        all_tasks.append(task)
                        task_to_info[task_id] = (i, model_path, solution_info)
                    
                    # Use VllmServerManager to run all tasks in parallel
                    try:
                        # 如果未指定max_model_len，则设置为max_tokens + 3000
                        max_model_len = args.max_model_len
                        if max_model_len is None:
                            max_model_len = args.max_tokens + 3000
                            print(f"未指定max_model_len，将使用默认值: {max_model_len} (max_tokens + 3000)")
                        
                        with VllmServerManager(available_gpus=available_gpus_global, max_model_len=max_model_len) as server_manager:
                            # Call run_series_tasks once with all tasks
                            results = server_manager.run_series_tasks(all_tasks)
                        
                        print(f"Parallel evaluation completed, time taken: {time.time() - start_time:.2f} seconds")
                        
                        # Process results for each task
                        print("Processing evaluation results...")
                        result_processor = ResultProcessor()
                        
                        for task_id, task_result in results.items():
                            if task_id in task_to_info:
                                original_idx, model_path, solution_info = task_to_info[task_id]
                                try:
                                    # Process individual task result
                                    res = result_processor.process_and_save({task_id: task_result})
                                    
                                    # Extract metrics
                                    metrics = extract_metrics(res)
                                    solution_info['metrics'] = metrics
                                    
                                    print(f"Solution {original_idx} evaluation results processed successfully")
                                    
                                    # Save to cache using original index
                                    cache_data = {'metrics': metrics, 'timestamp': datetime.now().isoformat()}
                                    
                                    # Save to task-specific directory
                                    task_specific_cache_path = os.path.join(solutions_cache_dir, f"solution_{original_idx}_eval.json")
                                    save_results_to_cache(task_specific_cache_path, cache_data)
                                    
                                    # Also save to general cache
                                    model_key = generate_model_cache_key(model_path)
                                    general_cache_path = get_model_cache_path(args.checkpoint_dir, model_key, 'solution')
                                    save_results_to_cache(general_cache_path, cache_data)
                                    
                                except Exception as e:
                                    print(f"Error processing results for solution {original_idx}: {e}")
                                    solution_info['metrics'] = {"error": str(e)}
                                
                                all_solutions.append(solution_info)
                    
                    except Exception as e:
                        print(f"Parallel evaluation process failed: {e}")
                        # Add solutions with error metrics if evaluation failed
                        for original_idx, model_path, solution_info in solutions_to_evaluate:
                            solution_info['metrics'] = {"error": f"Parallel evaluation failed: {str(e)}"}
                            all_solutions.append(solution_info)
                else:
                    print("All solutions found in cache, skipping evaluation")
        
        # Save all results
        output_dir, objectives_list, metrics_list = save_solution_results(
            args.task_id, all_solutions, args.checkpoint_dir, original_models_results, model_level_results
        )
        
        # Generate visualizations
        print("\nGenerating visualization results...")
        # 使用visualization模块中的函数进行绘图
        reporter.plot_3d_objectives(objectives_list, output_dir)
        
        # 使用save_solution_results返回的metrics_list进行可视化，它已经包含了正确的type标记
        reporter.plot_dataset_metrics(output_dir, metrics_list)
        
        # 添加新的可视化图表：所有指标的token平均值 vs 模型能力（f1+f3）
        reporter.plot_token_avg_vs_model_ability(output_dir, metrics_list)
        
        # 添加新的可视化图表：模型能力f1 vs f3
        reporter.plot_f1_vs_f3(output_dir, metrics_list)
        
        print(f"可视化图表已保存到: {output_dir}")
        
        print("\nSolution model reproduction and evaluation process completed!")
        print(f"Successfully processed {len(all_solutions)} solutions")
        if original_models_results:
            print(f"Successfully evaluated {len(original_models_results)} original models")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
