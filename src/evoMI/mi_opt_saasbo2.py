#!/usr/bin/env python3
"""
模型优化与合并模块 - 使用SAASBO+qNEHVI算法 - 版本2
提供模型合并和多目标优化功能的集成模块，支持权重和density双参数设置
"""

import os
import sys
import time
import uuid
import json
import copy
import shutil
import atexit
import concurrent.futures
import queue
import threading
import numpy as np
import requests
import torch
from evalscope import run_task

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入所需模块
from src.evoMI.vllm_server_manager import VllmServerManager
from src.evoMI.result_processor import ResultProcessor
from src.evoMI.saasbo_qnehvi_optimizer import saasbo_qnehvi_optimizer
from src.evoMI.saasbo_qnehvi_optimize_block import saasbo_qnehvi_two_stage
from src.evoMI.qehvi_optimizer import qehvi_optimizer
from src.evoMI.optimization_reporting import reporter
from src.evoMI.mi_block_fusion import mi_block_fusion
from src.evoMI.model_reproduction import (
    available_gpus_global,
    build_eval_cache_config,
    collect_dataset_metrics,
    create_eval_task_config,
    generate_model_cache_key,
    get_eval_profile_datasets,
    get_compatible_model_cache_candidate_paths,
    get_model_cache_path,
    load_compatible_cached_results,
    normalize_eval_limit,
    save_results_to_cache,
    run_task_with_server,
)
from src.config_manager import config_manager

# 全局变量用于存储base和expert模型的评测结果
base_model_results = None
expert_model_results = None
shared_vllm_manager = None
metrics_history = []

# 使用配置管理类获取模型配置
# 模型路径将通过命令行参数指定
# 缓存目录
checkpoint_dir = config_manager.checkpoint_dir


def set_available_gpus(gpus):
    available_gpus_global[:] = gpus
    print(f"评测可用GPU已更新: {available_gpus_global}")


def get_shared_vllm_manager(max_model_len: int = None):
    global shared_vllm_manager
    if max_model_len is None:
        max_model_len = 48000
    if shared_vllm_manager is not None and (not shared_vllm_manager._destroyed):
        return shared_vllm_manager
    shared_vllm_manager = VllmServerManager.get_shared_manager(
        available_gpus=available_gpus_global,
        max_model_len=max_model_len,
    )
    return shared_vllm_manager


def shutdown_shared_vllm_manager():
    global shared_vllm_manager
    if shared_vllm_manager is not None and (not shared_vllm_manager._destroyed):
        shared_vllm_manager.destroy()
    shared_vllm_manager = None


atexit.register(shutdown_shared_vllm_manager)


def submit_eval_tasks(tasks, max_model_len: int = None, reset_series: bool = False):
    manager = get_shared_vllm_manager(max_model_len=max_model_len)
    return manager.submit_tasks(tasks, reset_series=reset_series)


def wait_eval_tasks(task_ids, finalize_series: bool = True):
    manager = get_shared_vllm_manager()
    results = manager.wait_for_tasks(task_ids, clear_events=True)
    if finalize_series:
        manager._series_end_time = time.time()
        manager._update_progress_log(force=True)
        manager.export_gpu_usage_stats()
    return results


def get_idle_gpu_count() -> int:
    manager = get_shared_vllm_manager()
    return manager.get_idle_gpu_count()


def collect_newly_completed_tasks():
    manager = get_shared_vllm_manager()
    return manager.collect_newly_completed_tasks()


def create_optimizer_config(
    custom_initial_solutions=None,
    num_blocks=8,
    num_objectives=2,
    BATCH_SIZE=4,
    NUM_RESTARTS=10,
    RAW_SAMPLES=512,
    MC_SAMPLES=128,
    N_BATCH=50,
    verbose=True,
    device="cpu",
    dtype=torch.double,
    initial_samples=8,
    noise_level=0.0001,
    run_id="block_test0",
    checkpoint_dir="./checkpoints",
    optimize_density=1,
):
    if optimize_density == 1:
        dim = num_blocks + 1
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim]).to(dtype)
    elif optimize_density == 2:
        dim = (num_blocks + 1) * 2
        bounds = torch.tensor(
            [[0.0] * (num_blocks + 1) + [0.6] * (num_blocks + 1), [1.0] * (num_blocks + 1) + [1.0] * (num_blocks + 1)]
        ).to(dtype)
    elif optimize_density == 3:
        dim = (num_blocks + 1) * 3
        bounds = torch.tensor(
            [
                [0.0] * (num_blocks + 1) + [0.6] * (num_blocks + 1) + [0.00] * (num_blocks + 1),
                [1.0] * (num_blocks + 1) + [1.0] * (num_blocks + 1) + [0.05] * (num_blocks + 1),
            ]
        ).to(dtype)
    else:
        dim = num_blocks + 1
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim]).to(dtype)

    return {
        "dim": dim,
        "num_objectives": num_objectives,
        "bounds": bounds,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_RESTARTS": NUM_RESTARTS,
        "RAW_SAMPLES": RAW_SAMPLES,
        "MC_SAMPLES": MC_SAMPLES,
        "N_BATCH": N_BATCH,
        "verbose": verbose,
        "device": device,
        "dtype": dtype,
        "initial_samples": initial_samples,
        "noise_level": noise_level,
        "ref_point": torch.tensor([-0.2, -0.2]).to(dtype),
        "run_id": run_id,
        "checkpoint_dir": checkpoint_dir,
        "custom_initial_solutions": custom_initial_solutions,
    }


def estimate_eval_tokens(eval_limit, max_tokens, eval_profile="aime_gpqa"):
    normalized_limit = normalize_eval_limit(eval_profile, eval_limit)
    if normalized_limit is None:
        return float(max_tokens) * 65.0
    profile = str(eval_profile or "aime_gpqa").strip().lower()
    default_case_counts = {
        "aime25": 30.0,
        "gsm8k": 1319.0,
        "math_500": 100.0,
        "gpqa_diamond": 198.0,
    }
    if profile == "aime_gpqa":
        aime_limit = float(normalized_limit.get("aime25", default_case_counts["aime25"]) or default_case_counts["aime25"])
        gpqa_limit = float(normalized_limit.get("gpqa_diamond", default_case_counts["gpqa_diamond"]) or default_case_counts["gpqa_diamond"])
        estimated_cases = aime_limit * 2.0 + gpqa_limit
    else:
        estimated_cases = sum(
            float(default_case_counts.get(dataset_name, 1.0) if value is None else value or 0)
            for dataset_name, value in normalized_limit.items()
        )
    return max(1.0, estimated_cases) * float(max_tokens)

# 初始化函数 - 获取base和expert模型的评测结果
def initialize_model_evaluations(
    base_model,
    expert_model,
    max_tokens: int = 35000,
    max_model_len: int = None,
    eval_limit=None,
    eval_profile="aime_gpqa",
    eval_repeats=None,
    eval_seed=42,
    cache_root=None,
):
    """
    初始化base和expert模型的评测结果
    
    参数:
        base_model: base模型路径列表
        expert_model: expert模型路径列表
        max_tokens: 最大生成token数
        max_model_len: 最大模型长度
    
    返回:
        tuple: (base_model_results, expert_model_results)
    """
    print("\n===== 初始化模型评测结果 =====")
    eval_limit = normalize_eval_limit(eval_profile, eval_limit)
    dataset_names = get_eval_profile_datasets(eval_profile)
    eval_config = build_eval_cache_config(
        eval_profile=eval_profile,
        eval_limit=eval_limit,
        repeats=eval_repeats,
        max_tokens=max_tokens,
        seed=eval_seed,
    )
    cache_base_dir = os.path.abspath(cache_root or checkpoint_dir)
    resolved_model_results = {}

    model_eval_specs = []
    seen_cache_keys = set()
    for role_name, model_list in (("base", base_model), ("expert", expert_model)):
        for model_path in model_list:
            cache_key = (os.path.abspath(model_path), json.dumps(eval_config, sort_keys=True, ensure_ascii=False))
            if cache_key in resolved_model_results:
                print(f"复用本轮已加载的{role_name}模型结果: {model_path}")
                continue
            if cache_key in seen_cache_keys:
                continue
            seen_cache_keys.add(cache_key)
            cache_candidates = get_compatible_model_cache_candidate_paths(
                cache_base_dir,
                model_path,
                'original',
                eval_config=eval_config,
            )
            primary_cache_path = cache_candidates[0]
            legacy_cache_path = get_model_cache_path(
                cache_base_dir,
                generate_model_cache_key(model_path),
                'original',
            )
            cached_result = None
            for candidate_path in cache_candidates:
                cached_result = load_compatible_cached_results(
                    candidate_path,
                    expected_eval_config=eval_config,
                )
                if cached_result:
                    print(f"使用{role_name}模型的缓存结果: {model_path}")
                    resolved_model_results[cache_key] = cached_result
                    break
            if cached_result:
                continue
            print(f"未找到{role_name}模型的缓存结果，开始评测: {model_path}")
            model_id = f"original_{os.path.basename(model_path)}_{uuid.uuid4().hex[:8]}"
            task_cfg = create_eval_task_config(
                model_path,
                max_tokens=max_tokens,
                eval_profile=eval_profile,
                eval_limit=eval_limit,
                repeats=eval_repeats,
                seed=eval_seed,
            )
            model_eval_specs.append({
                'role_name': role_name,
                'model_path': model_path,
                'cache_key': cache_key,
                'cache_path': primary_cache_path,
                'legacy_cache_path': legacy_cache_path,
                'task': {
                    'task_id': f'task_{model_id}',
                    'model_path': model_path,
                    'gpu_count': 1,
                    'estimated_tokens': estimate_eval_tokens(eval_limit, max_tokens, eval_profile=eval_profile),
                    'params_dict': {
                        'task_cfg': task_cfg,
                    },
                    'func_handle': run_task_with_server
                },
            })

    if model_eval_specs:
        effective_max_model_len = max_model_len
        if effective_max_model_len is None:
            effective_max_model_len = max_tokens + 3000
            print(f"未指定max_model_len，将使用默认值: {effective_max_model_len} (max_tokens + 3000)")
        print(f"开始并行评测 {len(model_eval_specs)} 个未缓存模型")
        for spec in model_eval_specs:
            print(f"开始评测模型: {spec['model_path']}")
        start_time = time.time()
        try:
            submit_tasks = [spec['task'] for spec in model_eval_specs]
            server_manager = get_shared_vllm_manager(max_model_len=effective_max_model_len)
            results = server_manager.run_series_tasks(submit_tasks)
            print(f"模型初始化评测完成，耗时: {time.time() - start_time:.2f} 秒")
            result_processor = ResultProcessor()
            for spec in model_eval_specs:
                task_id = spec['task']['task_id']
                single_result = {task_id: results.get(task_id)}
                res = result_processor.process_and_save(single_result)
                metrics = collect_dataset_metrics(res, dataset_names)
                model_result = {
                    'model_type': 'thinking' if 'thinking' in spec['model_path'].lower() else 'instruct',
                    'model_name': os.path.basename(spec['model_path']),
                    'model_path': spec['model_path'],
                    'metrics': metrics,
                    'eval_config': eval_config,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                save_results_to_cache(spec['cache_path'], model_result, eval_config=eval_config)
                if spec['legacy_cache_path'] != spec['cache_path']:
                    save_results_to_cache(spec['legacy_cache_path'], model_result, eval_config=eval_config)
                resolved_model_results[spec['cache_key']] = model_result
        except Exception as e:
            print(f"评测过程中发生错误: {e}")
            for spec in model_eval_specs:
                model_result = {
                    'model_type': 'thinking' if 'thinking' in spec['model_path'].lower() else 'instruct',
                    'model_name': os.path.basename(spec['model_path']),
                    'model_path': spec['model_path'],
                    'metrics': {"error": str(e)},
                    'eval_config': eval_config,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                save_results_to_cache(spec['cache_path'], model_result, eval_config=eval_config)
                if spec['legacy_cache_path'] != spec['cache_path']:
                    save_results_to_cache(spec['legacy_cache_path'], model_result, eval_config=eval_config)
                resolved_model_results[spec['cache_key']] = model_result

    print(f"\n评测base模型: {base_model}")
    base_model_results = [
        resolved_model_results[(os.path.abspath(model_path), json.dumps(eval_config, sort_keys=True, ensure_ascii=False))]
        for model_path in base_model
    ]
    
    print(f"\n评测expert模型: {expert_model}")
    expert_model_results = [
        resolved_model_results[(os.path.abspath(model_path), json.dumps(eval_config, sort_keys=True, ensure_ascii=False))]
        for model_path in expert_model
    ]
    
    print("\n===== 模型评测初始化完成 =====")
    return base_model_results, expert_model_results


def _normalize_decision_matrix(decision_matrix, optimize_density, num_blocks):
    if isinstance(decision_matrix, list):
        decision_matrix = np.array(decision_matrix)
    if optimize_density == 1:
        expected_dim = num_blocks + 1
    elif optimize_density == 2:
        expected_dim = (num_blocks + 1) * 2
    elif optimize_density == 3:
        expected_dim = (num_blocks + 1) * 3
    else:
        expected_dim = num_blocks + 1
    if decision_matrix.ndim != 2 or decision_matrix.shape[1] != expected_dim:
        raise ValueError(f"决策变量矩阵必须是二维数组，每行包含 {expected_dim} 个决策变量")
    return decision_matrix


def _normalize_estimated_tokens_list(estimated_tokens):
    if estimated_tokens is None:
        return None
    if isinstance(estimated_tokens, torch.Tensor):
        return estimated_tokens.detach().cpu().reshape(-1).tolist()
    if isinstance(estimated_tokens, np.ndarray):
        return estimated_tokens.reshape(-1).tolist()
    if isinstance(estimated_tokens, list):
        return list(estimated_tokens)
    return [float(estimated_tokens)]


def _build_candidate_specs(
    decision_matrix,
    optimize_density,
    num_blocks,
    base_output_dir,
    eval_limit,
    max_tokens,
    eval_profile,
    estimated_tokens,
):
    decision_matrix = _normalize_decision_matrix(decision_matrix, optimize_density, num_blocks)
    estimated_tokens_list = _normalize_estimated_tokens_list(estimated_tokens)
    num_candidates = int(decision_matrix.shape[0])
    os.makedirs(base_output_dir, exist_ok=True)
    candidate_specs = []
    for i in range(num_candidates):
        if optimize_density == 1:
            block_weights = decision_matrix[i, :].tolist()
            block_densities = None
            gamma_params = None
            print(f"\n处理候选方案 {i+1}/{num_candidates}:")
            print(f"  权重参数: {block_weights}")
            print("  密度参数: 使用默认值")
        elif optimize_density == 2:
            weights_end = num_blocks + 1
            block_weights = decision_matrix[i, :weights_end].tolist()
            block_densities = decision_matrix[i, weights_end:].tolist()
            gamma_params = None
            print(f"\n处理候选方案 {i+1}/{num_candidates}:")
            print(f"  权重参数: {block_weights}")
            print(f"  密度参数: {block_densities}")
        else:
            weights_end = num_blocks + 1
            densities_end = weights_end * 2
            block_weights = decision_matrix[i, :weights_end].tolist()
            block_densities = decision_matrix[i, weights_end:densities_end].tolist()
            gamma_params = decision_matrix[i, densities_end:].tolist()
            print(f"\n处理候选方案 {i+1}/{num_candidates}:")
            print(f"  权重参数: {block_weights}")
            print(f"  密度参数: {block_densities}")
            print(f"  Gamma参数: {gamma_params}")
        model_id = f"merged_model_{i}_{uuid.uuid4().hex[:8]}"
        model_output_dir = os.path.join(base_output_dir, model_id)
        candidate_specs.append({
            'candidate_index': i,
            'model_id': model_id,
            'model_output_dir': model_output_dir,
            'block_weights': block_weights,
            'block_densities': block_densities,
            'gamma_params': gamma_params,
            'estimated_tokens': float(estimated_tokens_list[i]) if estimated_tokens_list is not None and i < len(estimated_tokens_list) else estimate_eval_tokens(eval_limit, max_tokens, eval_profile=eval_profile),
        })
    return decision_matrix, candidate_specs


class AsyncModelMergeEvaluationSession:
    def __init__(self, candidate_specs, finalize_series=True):
        self.candidate_specs = list(candidate_specs)
        self.total_candidates = len(self.candidate_specs)
        self.finalize_series = bool(finalize_series)
        self._result_queue = queue.Queue()
        self._done_event = threading.Event()
        self._error = None
        self._finalized = False

    def put_result(self, item):
        self._result_queue.put(item)

    def set_error(self, exc):
        self._error = exc

    def mark_done(self):
        self._done_event.set()

    def get_next_result(self, timeout=0.5):
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_finished(self):
        return self._done_event.is_set() and self._result_queue.empty()

    def finalize(self):
        if self._finalized:
            if self._error is not None:
                raise self._error
            return
        self._done_event.wait()
        if self.finalize_series:
            manager = get_shared_vllm_manager()
            manager._series_end_time = time.time()
            manager._update_progress_log(force=True)
            manager.export_gpu_usage_stats()
        self._finalized = True
        if self._error is not None:
            raise self._error


def _build_candidate_error_result(spec, error_message):
    return {
        'candidate_index': spec['candidate_index'],
        'objective': np.zeros((2,), dtype=float),
        'metric': {
            'status': 'error',
            'error': str(error_message),
            '_cleanup_model_dirs': [spec['model_output_dir']],
        },
    }


def _normalize_objective_row(objectives):
    array = np.asarray(objectives, dtype=float)
    if array.ndim == 0:
        normalized = np.zeros((2,), dtype=float)
        normalized[0] = float(array.item())
        return normalized
    if array.ndim >= 2 and array.shape[0] > 0:
        array = array[0]
    array = array.reshape(-1)
    normalized = np.zeros((2,), dtype=float)
    if array.size > 0:
        normalized[:min(2, array.size)] = array[:2]
    return normalized


def _get_results_list(results):
    if isinstance(results, dict) and 'processed_results' in results:
        return results['processed_results']
    if isinstance(results, list):
        return results
    return [results]


def _get_anchor_model(model_results, preferred_index):
    if not isinstance(model_results, list) or len(model_results) == 0:
        return {}
    if 0 <= preferred_index < len(model_results) and isinstance(model_results[preferred_index], dict):
        return model_results[preferred_index]
    for item in model_results:
        if isinstance(item, dict):
            return item
    return {}


def _get_dataset_metric(dataset_metrics, metric_name, fallback=0.0):
    if not isinstance(dataset_metrics, dict):
        return float(fallback)
    value = dataset_metrics.get(metric_name)
    if value is None and metric_name != 'score':
        value = dataset_metrics.get('score')
    if value is None:
        value = fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _get_model_dataset_metric(model_result, dataset_name, metric_name, fallback=0.0):
    if not isinstance(model_result, dict):
        return float(fallback)
    metrics = model_result.get('metrics', {})
    dataset_metrics = metrics.get(dataset_name, {}) if isinstance(metrics, dict) else {}
    return _get_dataset_metric(dataset_metrics, metric_name, fallback=fallback)


def _normalize_metric_value(value, lower_bound, upper_bound, fallback=0.0):
    denominator = float(upper_bound) - float(lower_bound)
    if abs(denominator) < 1e-12:
        return float(fallback)
    return (float(value) - float(lower_bound)) / denominator


def extract_objectives(results, base_model_results, expert_model_results, eval_profile="aime_gpqa"):
    default_bounds = {
        'aime25': {'acc': (0.45, 0.8), 'tokens': (22000.0, 9000.0)},
        'gpqa_diamond': {'acc': (0.3, 0.7), 'tokens': (9000.0, 1000.0)},
        'gsm8k': {'acc': (0.5, 0.9), 'tokens': (12000.0, 2000.0)},
        'math_500': {'acc': (0.2, 0.7), 'tokens': (15000.0, 3000.0)},
    }
    results_list = _get_results_list(results)
    dataset_names = get_eval_profile_datasets(eval_profile)
    accuracy_base_anchor = _get_anchor_model(base_model_results, 0)
    accuracy_expert_anchor = _get_anchor_model(expert_model_results, 0)
    token_base_anchor = _get_anchor_model(base_model_results, 1)
    token_expert_anchor = _get_anchor_model(expert_model_results, 1)
    objectives = []
    for result in results_list:
        try:
            if not isinstance(result, dict):
                raise TypeError(f"结果类型无效: {type(result).__name__}")
            accuracy_norms = []
            token_norms = []
            for dataset_name in dataset_names:
                dataset_metrics = result.get(dataset_name, {})
                bounds = default_bounds.get(dataset_name, {'acc': (0.0, 1.0), 'tokens': (10000.0, 1000.0)})
                accuracy_value = _get_dataset_metric(dataset_metrics, 'mean_acc', fallback=bounds['acc'][0])
                token_value = _get_dataset_metric(dataset_metrics, 'mean_tokens_num', fallback=bounds['tokens'][0])
                base_accuracy = _get_model_dataset_metric(
                    accuracy_base_anchor,
                    dataset_name,
                    'mean_acc',
                    fallback=bounds['acc'][0],
                )
                expert_accuracy = _get_model_dataset_metric(
                    accuracy_expert_anchor,
                    dataset_name,
                    'mean_acc',
                    fallback=bounds['acc'][1],
                )
                base_tokens = _get_model_dataset_metric(
                    token_base_anchor,
                    dataset_name,
                    'mean_tokens_num',
                    fallback=bounds['tokens'][0],
                )
                expert_tokens = _get_model_dataset_metric(
                    token_expert_anchor,
                    dataset_name,
                    'mean_tokens_num',
                    fallback=bounds['tokens'][1],
                )
                accuracy_norms.append(
                    _normalize_metric_value(
                        accuracy_value,
                        base_accuracy,
                        expert_accuracy,
                        fallback=0.0,
                    )
                )
                token_norms.append(
                    _normalize_metric_value(
                        token_value,
                        base_tokens,
                        expert_tokens,
                        fallback=0.0,
                    )
                )
            if len(accuracy_norms) == 0:
                objectives.append(np.array([-0.2, -0.2], dtype=float))
                continue
            objectives.append(
                np.array(
                    [
                        float(np.mean(accuracy_norms)),
                        float(np.mean(token_norms)),
                    ],
                    dtype=float,
                )
            )
        except Exception as e:
            print(f"提取目标函数值时出错: {e}")
            objectives.append(np.array([-0.2, -0.2], dtype=float))
    return np.asarray(objectives, dtype=float)


def _process_completed_task_result(task_id, task_result, spec, result_processor, base_model_results, expert_model_results, eval_profile):
    if isinstance(task_result, dict) and 'error' in task_result:
        return _build_candidate_error_result(spec, task_result.get('error', 'task_failed'))
    try:
        structured_results = result_processor.process_and_save({task_id: task_result})
        objectives = extract_objectives(
            structured_results,
            base_model_results,
            expert_model_results,
            eval_profile=eval_profile,
        )
        metric = structured_results[0] if len(structured_results) > 0 else {}
        if isinstance(metric, dict):
            metric['_cleanup_model_dirs'] = [spec['model_output_dir']]
            metric['_task_id'] = str(task_id)
        return {
            'candidate_index': spec['candidate_index'],
            'objective': _normalize_objective_row(objectives),
            'metric': metric if isinstance(metric, dict) else {},
            'task_id': str(task_id),
        }
    except Exception as e:
        print(f"处理任务 {task_id} 的评测结果时出错: {e}")
        return _build_candidate_error_result(spec, f"result_processing_error: {e}")


def start_async_model_merge_evaluation_session(
    decision_matrix,
    base_model_path="models/Qwen3-4B",
    task_model_paths=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B"],
    base_output_dir="output/mi_optimizer",
    max_tokens: int = 35000,
    max_model_len: int = None,
    merged_blocks: list = None,
    num_blocks: int = 8,
    fusion_method="breadcrumbs",
    base_model_results=None,
    expert_model_results=None,
    optimize_density=1,
    eval_limit=None,
    estimated_tokens=None,
    eval_profile="aime_gpqa",
    eval_repeats=None,
    eval_seed=42,
    finalize_series=True,
):
    eval_limit = normalize_eval_limit(eval_profile, eval_limit)
    decision_matrix, candidate_specs = _build_candidate_specs(
        decision_matrix=decision_matrix,
        optimize_density=optimize_density,
        num_blocks=num_blocks,
        base_output_dir=base_output_dir,
        eval_limit=eval_limit,
        max_tokens=max_tokens,
        eval_profile=eval_profile,
        estimated_tokens=estimated_tokens,
    )
    print(f"开始处理 {decision_matrix.shape[0]} 个候选方案")
    session = AsyncModelMergeEvaluationSession(candidate_specs=candidate_specs, finalize_series=finalize_series)
    if len(candidate_specs) == 0:
        session.mark_done()
        return session
    effective_max_model_len = max_tokens + 3000 if max_model_len is None else max_model_len

    def worker():
        submitted_model_dirs = set()
        try:
            server_manager = get_shared_vllm_manager(max_model_len=effective_max_model_len)
            fusion_worker_count = min(
                max(1, len(available_gpus_global) or 1),
                max(1, len(candidate_specs)),
            )
            max_pending_candidates = max(
                fusion_worker_count,
                min(max(2, fusion_worker_count * 2), max(1, len(candidate_specs))),
            )
            result_processor = ResultProcessor()
            pending_tasks = {}
            first_submission = True

            def fuse_candidate(spec):
                success = mi_block_fusion(
                    base_model_path=base_model_path,
                    task_model_paths=task_model_paths,
                    block_weights=spec['block_weights'],
                    block_densities=spec['block_densities'],
                    block_gammas=spec['gamma_params'],
                    output_dir=spec['model_output_dir'],
                    fusion_method=fusion_method,
                    copy_from_base=True,
                    merged_blocks=merged_blocks,
                    num_blocks=num_blocks,
                )
                return spec, success

            with concurrent.futures.ThreadPoolExecutor(max_workers=fusion_worker_count) as executor:
                pending_futures = {}
                remaining_specs = list(candidate_specs)

                def submit_more_fusions():
                    while remaining_specs and (len(pending_futures) + len(pending_tasks)) < max_pending_candidates:
                        next_spec = remaining_specs.pop(0)
                        pending_futures[executor.submit(fuse_candidate, next_spec)] = next_spec

                submit_more_fusions()
                while pending_futures or pending_tasks or remaining_specs:
                    done_futures = set()
                    if pending_futures:
                        done_futures, _ = concurrent.futures.wait(
                            list(pending_futures.keys()),
                            timeout=0.2,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                    for future in done_futures:
                        spec = pending_futures.pop(future)
                        try:
                            spec, success = future.result()
                        except Exception as exc:
                            print(f"警告: 候选方案 {spec['candidate_index'] + 1} 合并异常，跳过评测: {exc}")
                            session.put_result(_build_candidate_error_result(spec, exc))
                            continue
                        if not success:
                            print(f"警告: 候选方案 {spec['candidate_index'] + 1} 合并失败，跳过评测")
                            session.put_result(_build_candidate_error_result(spec, 'fusion_failed'))
                            continue
                        task_cfg = create_eval_task_config(
                            spec['model_output_dir'],
                            max_tokens=max_tokens,
                            eval_profile=eval_profile,
                            eval_limit=eval_limit,
                            repeats=eval_repeats,
                            seed=eval_seed,
                        )
                        task = {
                            'task_id': f"task_{spec['model_id']}",
                            'model_path': spec['model_output_dir'],
                            'gpu_count': 1,
                            'estimated_tokens': spec['estimated_tokens'],
                            'cleanup_model_dirs': [spec['model_output_dir']],
                            'cleanup_server_after_completion': False,
                            'params_dict': {
                                'task_cfg': task_cfg,
                            },
                            'func_handle': run_task_with_server,
                        }
                        task_ids = server_manager.submit_tasks([task], reset_series=first_submission)
                        first_submission = False
                        submitted_model_dirs.add(os.path.abspath(spec['model_output_dir']))
                        pending_tasks[task_ids[0]] = spec
                        print(f"创建并提交任务: {task['task_id']}")
                    submit_more_fusions()
                    if pending_tasks:
                        task_id, task_result = server_manager.wait_for_any_task(
                            list(pending_tasks.keys()),
                            clear_event=True,
                            poll_interval_sec=0.1,
                            timeout=0.1,
                        )
                        if task_id is not None:
                            spec = pending_tasks.pop(task_id)
                            session.put_result(
                                _process_completed_task_result(
                                    task_id=task_id,
                                    task_result=task_result,
                                    spec=spec,
                                    result_processor=result_processor,
                                    base_model_results=base_model_results,
                                    expert_model_results=expert_model_results,
                                    eval_profile=eval_profile,
                                )
                            )
                            submit_more_fusions()
            if first_submission:
                print("警告: 没有成功创建任何合并模型，无法进行评测")
        except Exception as exc:
            session.set_error(exc)
        finally:
            leftover_model_dirs = []
            for spec in candidate_specs:
                model_output_dir = spec.get('model_output_dir')
                if not isinstance(model_output_dir, str) or len(model_output_dir) == 0:
                    continue
                model_output_dir_abs = os.path.abspath(model_output_dir)
                if model_output_dir_abs in submitted_model_dirs:
                    continue
                if os.path.exists(model_output_dir_abs):
                    leftover_model_dirs.append(model_output_dir_abs)
            if leftover_model_dirs:
                try:
                    clean_generated_model_dirs(base_output_dir, leftover_model_dirs)
                except Exception as cleanup_exc:
                    print(f"警告: 清理异步评测残留模型目录失败: {cleanup_exc}")
            session.mark_done()

    worker_thread = threading.Thread(target=worker, name=f"async-merge-eval-{uuid.uuid4().hex[:8]}")
    worker_thread.daemon = True
    worker_thread.start()
    return session


def process_decision_variables(decision_matrix, base_model_path="models/Qwen3-4B", 
                              task_model_paths=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B"], 
                              base_output_dir="output/mi_optimizer",
                              max_tokens: int = 35000, max_model_len: int = None,
                              merged_blocks: list = None,
                              num_blocks: int = 8,
                              fusion_method="breadcrumbs",
                              base_model_results=None,
                              expert_model_results=None,
                              optimize_density=1,
                              eval_limit=None,
                              estimated_tokens=None,
                              eval_profile="aime_gpqa",
                              eval_repeats=None,
                              eval_seed=42):
    session = start_async_model_merge_evaluation_session(
        decision_matrix=decision_matrix,
        base_model_path=base_model_path,
        task_model_paths=task_model_paths,
        base_output_dir=base_output_dir,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        merged_blocks=merged_blocks,
        num_blocks=num_blocks,
        fusion_method=fusion_method,
        base_model_results=base_model_results,
        expert_model_results=expert_model_results,
        optimize_density=optimize_density,
        eval_limit=eval_limit,
        estimated_tokens=estimated_tokens,
        eval_profile=eval_profile,
        eval_repeats=eval_repeats,
        eval_seed=eval_seed,
        finalize_series=True,
    )
    result_by_index = {}
    while not session.is_finished():
        next_item = session.get_next_result(timeout=0.5)
        if next_item is None:
            continue
        result_by_index[int(next_item['candidate_index'])] = next_item
    session.finalize()
    objectives = []
    metrics = []
    for spec in session.candidate_specs:
        item = result_by_index.get(int(spec['candidate_index']))
        if item is None:
            item = _build_candidate_error_result(spec, 'missing_result')
        objectives.append(np.asarray(item['objective'], dtype=float).reshape(1, -1))
        metrics.append(item.get('metric', {}))
    if len(objectives) == 0:
        return np.zeros((0, 2)), []
    return np.concatenate(objectives, axis=0), metrics


def model_merge_optimization_function(x: np.ndarray, merged_blocks: list = None, num_blocks: int = 8, cache_dir: str = None, 
                                      base_model_path="models/Qwen3-4B", task_model_paths=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B"],
                                      fusion_method="breadcrumbs", base_model_results=None, expert_model_results=None, 
                                      optimize_density=1, max_tokens: int = 35000, max_model_len: int = None,
                                      eval_limit=None, eval_mode="full", estimated_tokens=None,
                                      eval_profile="aime_gpqa", eval_repeats=None, eval_seed=42) -> tuple:
    """
    模型合并优化目标函数
    将决策变量映射到目标函数值
    """
    n_samples = x.shape[0]
    print(f"\n处理 {n_samples} 个样本...")
    if cache_dir is None:
        cache_dir = os.path.join('output', 'mi_optimization_temp')
    output_dir = os.path.join(cache_dir, str(int(time.time())))
    os.makedirs(output_dir, exist_ok=True)
    if eval_mode == "submit_async":
        return start_async_model_merge_evaluation_session(
            decision_matrix=x,
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            base_output_dir=output_dir,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            merged_blocks=merged_blocks,
            num_blocks=num_blocks,
            fusion_method=fusion_method,
            base_model_results=base_model_results,
            expert_model_results=expert_model_results,
            optimize_density=optimize_density,
            eval_limit=eval_limit,
            estimated_tokens=estimated_tokens,
            eval_profile=eval_profile,
            eval_repeats=eval_repeats,
            eval_seed=eval_seed,
            finalize_series=True,
        )
    objectives, metrics = process_decision_variables(
        decision_matrix=x,
        base_model_path=base_model_path,
        task_model_paths=task_model_paths,
        base_output_dir=output_dir,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        merged_blocks=merged_blocks,
        num_blocks=num_blocks,
        fusion_method=fusion_method,
        base_model_results=base_model_results,
        expert_model_results=expert_model_results,
        optimize_density=optimize_density,
        eval_limit=eval_limit,
        estimated_tokens=estimated_tokens,
        eval_profile=eval_profile,
        eval_repeats=eval_repeats,
        eval_seed=eval_seed,
    )
    metrics_history.extend(metrics)
    expected_shape = (n_samples, 2)
    if objectives.shape != expected_shape:
        print(f"警告: process_decision_variables 返回的结果形状不正确: {objectives.shape}，预期 {expected_shape}")
        result = np.zeros(expected_shape)
        min_samples = min(n_samples, objectives.shape[0])
        if min_samples > 0:
            result[:min_samples] = objectives[:min_samples]
        return torch.from_numpy(result), metrics
    return torch.from_numpy(objectives), metrics



def create_iteration_callback(cache_dir: str, cleanup_interval: int = 1, 
                             keep_dirs: list = None, exclude_patterns: list = None):
    """
    创建迭代回调函数，用于在优化过程中清理缓存
    
    参数:
        cache_dir: 需要清理的缓存目录
        cleanup_interval: 清理间隔（迭代次数）
        keep_dirs: 需要保留的目录列表
        exclude_patterns: 需要排除的模式列表
    
    返回:
        Callable: 回调函数
    """
    if keep_dirs is None:
        keep_dirs = ['important', 'keep']
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    def callback(iteration, x, y, hypervolume):
        """优化迭代回调函数"""
        # 打印当前迭代信息
        print(f"\n===== 迭代 {iteration+1} ====")
        print(f"当前超体积: {hypervolume[-1]:.6f}")
        cleanup_paths = getattr(callback, "cleanup_paths", [])
        async_mode = bool(getattr(callback, "async_mode", False))
        callback.cleanup_paths = []
        if cleanup_paths:
            print(f"\n清理当前评估生成模型: {len(cleanup_paths)} 个目录")
            clean_generated_model_dirs(cache_dir, cleanup_paths)
            return
        
        # 定期清理缓存
        if (not async_mode) and (iteration + 1) % cleanup_interval == 0:
            print(f"\n清理缓存目录: {cache_dir}")
            clean_eval_task_dirs(cache_dir)
    
    callback.cleanup_paths = []
    callback.async_mode = False
    return callback


def clean_generated_model_dirs(cache_dir: str, model_dirs: list):
    cache_dir_abs = os.path.abspath(cache_dir)
    cleaned_count = 0
    pruned_parent_count = 0
    for model_dir in model_dirs:
        model_dir_abs = os.path.abspath(model_dir)
        if os.path.commonpath([cache_dir_abs, model_dir_abs]) != cache_dir_abs:
            continue
        if not os.path.isdir(model_dir_abs):
            continue
        shutil.rmtree(model_dir_abs)
        cleaned_count += 1
        print(f"已删除当前评估模型目录: {model_dir_abs}")
        parent_dir = os.path.dirname(model_dir_abs)
        while os.path.commonpath([cache_dir_abs, parent_dir]) == cache_dir_abs and parent_dir != cache_dir_abs:
            if os.path.isdir(parent_dir) and len(os.listdir(parent_dir)) == 0:
                os.rmdir(parent_dir)
                pruned_parent_count += 1
                print(f"已删除空目录: {parent_dir}")
                parent_dir = os.path.dirname(parent_dir)
                continue
            break
    print(f"当前评估模型清理完成: 删除 {cleaned_count} 个目录, 清理空目录 {pruned_parent_count} 个")


def clean_eval_task_dirs(cache_dir: str):
    cache_dir_abs = os.path.abspath(cache_dir)
    if not os.path.isdir(cache_dir_abs):
        print(f"目录不存在: {cache_dir_abs}")
        return {"deleted_task_dirs": 0, "deleted_empty_parents": 0}
    deleted_task_dirs = 0
    deleted_empty_parents = 0
    for entry in os.listdir(cache_dir_abs):
        parent_path = os.path.join(cache_dir_abs, entry)
        if not os.path.isdir(parent_path):
            continue
        child_names = [name for name in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, name))]
        eval_children = [name for name in child_names if name.startswith("merged_model_")]
        if len(eval_children) == 0:
            continue
        for child_name in eval_children:
            child_path = os.path.join(parent_path, child_name)
            shutil.rmtree(child_path)
            deleted_task_dirs += 1
            print(f"已删除评估任务目录: {child_path}")
        remaining_entries = os.listdir(parent_path)
        if len(remaining_entries) == 0:
            os.rmdir(parent_path)
            deleted_empty_parents += 1
            print(f"已删除空父目录: {parent_path}")
    print(
        f"评估任务缓存清理完成: 删除任务目录 {deleted_task_dirs} 个, 删除空父目录 {deleted_empty_parents} 个"
    )
    return {"deleted_task_dirs": deleted_task_dirs, "deleted_empty_parents": deleted_empty_parents}


def clean_temp_files(directory, keep_dirs=None, exclude_patterns=None):
    """
    清理临时文件和目录
    
    参数:
        directory: 要清理的目录
        keep_dirs: 需要保留的目录列表
        exclude_patterns: 需要排除的模式列表
    
    返回:
        dict: 清理统计信息
    """
    if keep_dirs is None:
        keep_dirs = []
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    deleted_files = 0
    deleted_dirs = 0
    
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return {'deleted_files': 0, 'deleted_dirs': 0}
    
    try:
        # 获取目录中的所有项目
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            should_keep = False
            
            # 检查是否在保留目录列表中
            if any(keep_dir in item_path for keep_dir in keep_dirs):
                should_keep = True
            
            # 检查是否匹配排除模式
            if not should_keep:
                for pattern in exclude_patterns:
                    if pattern in item:
                        should_keep = True
                        break
            
            # 如果不应该保留，则删除
            if not should_keep:
                if os.path.isdir(item_path):
                    try:
                        import shutil
                        shutil.rmtree(item_path)
                        deleted_dirs += 1
                        print(f"已删除目录: {item_path}")
                    except Exception as e:
                        print(f"删除目录 {item_path} 失败: {e}")
                elif os.path.isfile(item_path):
                    try:
                        os.remove(item_path)
                        deleted_files += 1
                        print(f"已删除文件: {item_path}")
                    except Exception as e:
                        print(f"删除文件 {item_path} 失败: {e}")
    
    except Exception as e:
        print(f"清理缓存时发生错误: {e}")
    
    stats = {'deleted_files': deleted_files, 'deleted_dirs': deleted_dirs}
    print(f"缓存清理完成: 删除了 {deleted_files} 个文件和 {deleted_dirs} 个目录")
    return stats


def visualize_optimization_results(result_dict: dict, output_dir: str):
    """
    可视化优化结果（使用VisualizationTool）
    
    参数:
        result_dict: 优化结果字典
        output_dir: 输出目录
    """
    # 使用可视化工具类进行绘图
    reporter.visualize_optimization_results(result_dict, output_dir)
    print(f"可视化结果已保存到 {output_dir}")


def save_optimization_results(result_dict: dict, output_dir: str):
    """
    保存优化结果到文件
    
    参数:
        result_dict: 优化结果字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取帕累托前沿
    pareto_x = result_dict.get('pareto_x', np.array([]))
    pareto_y = result_dict.get('pareto_y', np.array([]))
    
    # 保存帕累托决策变量
    np.save(os.path.join(output_dir, 'pareto_decision_variables.npy'), pareto_x)
    
    # 保存帕累托目标值
    np.save(os.path.join(output_dir, 'pareto_objectives.npy'), pareto_y)
    
    # 保存所有评估过的点
    np.save(os.path.join(output_dir, 'all_evaluated_variables.npy'), result_dict.get('all_x', np.array([])))
    np.save(os.path.join(output_dir, 'all_evaluated_objectives.npy'), result_dict.get('all_y', np.array([])))
    
    # 保存为JSON格式（便于人类阅读），不包含参数
    import json
    results = {
        'pareto_solutions': [
            {
                'decision_variables': x.tolist(),
                'objectives': y.tolist()
            }
            for x, y in zip(pareto_x, pareto_y)
        ],
        'hypervolume_history': result_dict.get('hypervolume_history', []),
        'total_evaluations': len(result_dict.get('all_y', [])),
        'best_hypervolume': max(result_dict.get('hypervolume_history', [0])) if result_dict.get('hypervolume_history') else 0
    }
    
    with open(os.path.join(output_dir, 'optimization_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"优化结果已保存到 {output_dir}")


def save_settings(params: dict, output_dir: str):
    """
    保存优化设置到setting.json文件
    
    参数:
        params: 优化参数字典，包含所有配置参数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    settings_path = os.path.join(output_dir, 'setting.json')
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"优化设置已保存到 {settings_path}")


def compute_layer_importance(merged_blocks, num_blocks, optimize_density=1):
    """
    计算各层的重要性，基于任务模型各层的差异
    
    参数:
        merged_blocks: 合并后的块列表，每个块包含层列表和差异值
        num_blocks: block数量
        optimize_density: 密度优化模式
    
    返回:
        torch.Tensor: 各层的重要性权重，归一化到0-1区间
                     - 模式1: 形状为(num_blocks+1,)
                     - 模式2: 形状为((num_blocks+1)*2,)
                     - 模式3: 形状为((num_blocks+1)*3,)
    """
    print("\n===== 计算层重要性 =====")
    
    # 提取层差异
    layer_differences = []
    for block_info in merged_blocks:
        # calculate_merged_blocks返回的是元组列表，每个元组包含(层索引列表, 差异分数)
        # block_info[0]是层索引列表，block_info[1]是差异分数
        diff = abs(block_info[1])
        layer_differences.append(diff)
    
    # 添加最后一个维度（norm和lm_head）的差异，使用平均值
    layer_differences.append(np.mean(layer_differences))
    
    print(f"层差异原始值: {layer_differences}")
    
    # 归一化到0-1区间
    diff_array = np.array(layer_differences)
    min_diff = diff_array.min()
    max_diff = diff_array.max()
    
    # 避免除以零
    if max_diff - min_diff < 1e-8:
        importance = np.ones_like(diff_array) * 0.5
    else:
        importance = (diff_array - min_diff) / (max_diff - min_diff)
    
    print(f"原始层重要性: {importance}")
    
    if optimize_density == 1:
        # 模式1: 仅优化权重，返回原始重要性数组
        final_importance = importance
        print(f"层重要性（模式1: 仅权重）: {final_importance}")
        print(f"重要性形状: {final_importance.shape}")
    elif optimize_density == 2:
        # 模式2: 优化权重和密度，返回两倍长度的重要性数组
        final_importance = np.concatenate([importance, importance])
        print(f"层重要性（模式2: 权重+密度）: {final_importance}")
        print(f"重要性形状: {final_importance.shape}")
    elif optimize_density == 3:
        # 模式3: 优化权重、密度和gamma，返回三倍长度的重要性数组
        final_importance = np.concatenate([importance, importance, importance])
        print(f"层重要性（模式3: 权重+密度+gamma）: {final_importance}")
        print(f"重要性形状: {final_importance.shape}")
    else:
        # 默认为模式1
        final_importance = importance
        print(f"无效的optimize_density值，使用默认模式1")
        print(f"层重要性: {final_importance}")
        print(f"重要性形状: {final_importance.shape}")
    
    print("===== 层重要性计算完成 =====")
    
    return torch.tensor(final_importance, dtype=torch.float64)


def main_optimization(
    custom_initial_solutions=None,
    num_blocks=8,
    num_objectives=2,
    BATCH_SIZE=4,
    NUM_RESTARTS=10,
    RAW_SAMPLES=512,
    MC_SAMPLES=128,
    N_BATCH=50,
    verbose=True,
    device='cpu',
    dtype=torch.double,
    initial_samples=8,
    noise_level=0.0001,
    run_id="blcok_test0",
    cache_dir="output/mi_optimization_temp",
    alpha=1.0,
    beta=0.005,
    # 模型路径参数
    base_model=['models/Qwen3-4B','models/Qwen3-4B-thinking-2507'],
    expert_model=['models/Qwen3-4B-thinking-2507', 'models/Qwen3-4B'],
    base_model_path="models/Qwen3-4B",
    task_model_paths=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B"],
    # 算法参数
    algorithm="saasbo_qnehvi",  # 新增参数：算法选择，可选值："saasbo_qnehvi"、"qehvi"或"saasbo_qnehvi_block"
    use_saas=True,
    enable_importance_prior=True,
    enable_importance_update=True,
    enable_importance_guidance=True,
    enable_importance_weighted_acq=False,
    fusion_method="breadcrumbs",
    # 块优化算法参数
    block_configs=None,
    # 新增参数
    optimize_density=1,
    # 距离度量和划分方法参数
    metric="L2-norm",
    partition_method="hybrid"
):
    """
    主函数 - 执行模型合并优化流程
    
    参数:
        custom_initial_solutions: list, optional
            用户自定义的初始解列表，例如[0.55, 0.8]，则将N个初始解里面的两个初始为全是0.55的，和全是0.8，剩下再按原算法生成N-2个初始解
        num_blocks: int, optional
            block数
        num_objectives: int, optional
            目标函数维度：2个目标
        BATCH_SIZE: int, optional
            每批评估的样本数
        NUM_RESTARTS: int, optional
            优化重启次数
        RAW_SAMPLES: int, optional
            初始采样点数量
        MC_SAMPLES: int, optional
            MC采样数量
        N_BATCH: int, optional
            迭代次数
        verbose: bool, optional
            详细输出
        device: str, optional
            计算设备
        dtype: torch.dtype, optional
            数据类型
        initial_samples: int, optional
            初始采样点数
        noise_level: float, optional
            噪声水平
        run_id: str, optional
            运行ID
        # 模型路径参数
        base_model: list, optional
            base模型路径列表
        expert_model: list, optional
            expert模型路径列表
        base_model_path: str, optional
            基础模型路径
        task_model_paths: list, optional
            任务模型路径列表
        # 算法参数
        algorithm: str, optional
            优化算法选择，可选值：
            - "saasbo_qnehvi": 使用SAASBO+qNEHVI算法
            - "qehvi": 使用普通qNEHVI算法
            - "saasbo_qnehvi_block": 使用块优化SAASBO+qNEHVI算法，根据block_configs配置在不同迭代阶段使用不同的分块数
        use_saas: bool, optional
            是否使用SAAS先验（仅当algorithm="saasbo_qnehvi"或"saasbo_qnehvi_block"时有效）
        enable_importance_prior: bool, optional
            是否启用重要性先验（仅当algorithm="saasbo_qnehvi"或"saasbo_qnehvi_block"时有效）
        enable_importance_update: bool, optional
            是否启用重要性更新（仅当algorithm="saasbo_qnehvi"或"saasbo_qnehvi_block"时有效）
        enable_importance_guidance: bool, optional
            是否启用重要性指导（仅当algorithm="saasbo_qnehvi"或"saasbo_qnehvi_block"时有效）
        enable_importance_weighted_acq: bool, optional
            是否启用获取函数加权（仅当algorithm="saasbo_qnehvi"或"saasbo_qnehvi_block"时有效）
        fusion_method: str, optional
            模型融合方法
        block_configs: list, optional
            块划分配置列表，每个元素是一个字典，包含：
            - start_iter: 开始迭代次数（包含）
            - end_iter: 结束迭代次数（包含）
            - n_blocks: 该阶段使用的分块数
            例如：[{"start_iter": 0, "end_iter": 9, "n_blocks": 6}, {"start_iter": 10, "end_iter": 19, "n_blocks": 12}, {"start_iter": 20, "end_iter": 49, "n_blocks": 36}]
            如果为None，则使用默认配置：前10次迭代使用num_blocks块，之后使用36块（精细搜索）
        optimize_density: int, optional
            密度优化模式：
            - 1: 不开启密度优化，仅优化现有权重
            - 2: 优化密度和权重
            - 3: 优化权重、密度和gamma参数，决策变量为（块数+1）的三倍
    """
    # 设置checkpoint目录，与saasbo_qnehvi_optimizer保持一致
    checkpoint_dir = "./checkpoints"
    
    # 构建完整的checkpoint路径：./checkpoints/[run_id]
    checkpoint_run_dir = os.path.join(checkpoint_dir, run_id)
    
    # 创建输出目录，放在checkpoint目录下
    output_root = os.path.join(checkpoint_run_dir, 'output')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_root, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始模型合并优化流程，输出目录: {output_dir}")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 构建参数字典，保存到setting.json文件
    params_dict = {
        'custom_initial_solutions': custom_initial_solutions,
        'num_blocks': num_blocks,
        'num_objectives': num_objectives,
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_RESTARTS': NUM_RESTARTS,
        'RAW_SAMPLES': RAW_SAMPLES,
        'MC_SAMPLES': MC_SAMPLES,
        'N_BATCH': N_BATCH,
        'verbose': verbose,
        'device': device,
        'initial_samples': initial_samples,
        'noise_level': noise_level,
        'run_id': run_id,
        'checkpoint_dir': checkpoint_dir,
        'cache_dir': cache_dir,
        'alpha': alpha,
        'beta': beta,
        'base_model': base_model,
        'expert_model': expert_model,
        'base_model_path': base_model_path,
        'task_model_paths': task_model_paths,
        'algorithm': algorithm,
        'use_saas': use_saas,
        'enable_importance_prior': enable_importance_prior,
        'enable_importance_update': enable_importance_update,
        'enable_importance_guidance': enable_importance_guidance,
        'enable_importance_weighted_acq': enable_importance_weighted_acq,
        'fusion_method': fusion_method,
        'dtype': str(dtype),
        'optimize_density': optimize_density,
        'block_configs': block_configs,
        'metric': metric,
        'partition_method': partition_method
    }
    
    # 保存设置到saasbo_qnehvi_optimizer的checkpoint目录中
    print("\n保存设置到checkpoint目录...")
    save_settings(params_dict, checkpoint_run_dir)
    
    # 初始化时计算所有需要的合并块，并保存图形到output_dir
    print("\n初始化时计算所有需要的合并块...")
    from src.evoMI.mi_block_fusion import calculate_merged_blocks
    
    # 收集所有需要的块数
    required_block_counts = set()
    
    # 算法特定的块数收集
    if algorithm == "saasbo_qnehvi_block" and block_configs is not None:
        # 从block_configs中提取所有n_blocks值
        for config in block_configs:
            required_block_counts.add(config["n_blocks"])
    else:
        # 对于非块算法，只使用固定的num_blocks
        required_block_counts.add(num_blocks)
    
    # 确保36（真实层数）也被计算，用于精细搜索
    required_block_counts.add(36)
    
    # 计算并存储每个块数对应的merged_blocks
    merged_blocks_dict = {}
    initial_importance_dict = {}
    
    # 转换为列表并按升序排列
    sorted_block_counts = sorted(required_block_counts)
    
    if algorithm == "saasbo_qnehvi_block":
        # 仅对saasbo_qnehvi_block算法使用自细向粗的划分方法
        print(f"\n使用自细向粗的划分方法计算所有块数: {sorted_block_counts}")
        
        # 使用calculate_merged_blocks的新功能，一次计算所有块数的划分
        merged_blocks_dict = calculate_merged_blocks(
            task_model_paths=task_model_paths,
            num_blocks=min(sorted_block_counts),  # 最小块数作为默认值
            alpha=alpha,
            beta=beta,
            checkpoint_dir=output_dir,  # 保存图形到输出目录
            block_numbers=sorted_block_counts,  # 传入所有需要的块数，实现自细向粗划分
            metric=metric,
            partition_method=partition_method
        )
    else:
        # 其他算法保持之前的调用方式，逐个计算每个块数
        print(f"\n逐个计算所有块数: {sorted_block_counts}")
        
        for n_blocks in sorted_block_counts:
            print(f"\n计算 {n_blocks} 个块的合并结果...")
            merged_blocks = calculate_merged_blocks(
                task_model_paths=task_model_paths,
                num_blocks=n_blocks,
                alpha=alpha,
                beta=beta,
                checkpoint_dir=output_dir,  # 保存图形到输出目录
                metric=metric,
                partition_method=partition_method
            )
            merged_blocks_dict[n_blocks] = merged_blocks
    
    # 计算每个块数对应的层重要性
    for n_blocks in sorted_block_counts:
        merged_blocks = merged_blocks_dict[n_blocks]
        initial_importance = compute_layer_importance(
            merged_blocks=merged_blocks,
            num_blocks=n_blocks,
            optimize_density=optimize_density
        )
        initial_importance_dict[n_blocks] = initial_importance
    
    # 默认使用第一个块配置的块数或固定num_blocks
    default_n_blocks = block_configs[0]["n_blocks"] if (algorithm == "saasbo_qnehvi_block" and block_configs is not None) else num_blocks
    initial_importance = initial_importance_dict[default_n_blocks]
    
    # 创建优化器配置，添加checkpoint_dir参数
    print("\n初始化优化配置...")
    config = create_optimizer_config(
        custom_initial_solutions=custom_initial_solutions,
        num_blocks=num_blocks,
        num_objectives=num_objectives,
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
        run_id=run_id,
        checkpoint_dir=checkpoint_dir,
        optimize_density=optimize_density
    )
    

    
    # 运行优化
    print("\n开始优化过程...")
    start_time = time.time()
    
    # 创建缓存清理回调函数，每5轮迭代清理一次
    exclude_patterns = ['pareto', 'best', 'important']  # 需要排除的模式
    iteration_callback = create_iteration_callback(cache_dir, cleanup_interval=1, exclude_patterns=exclude_patterns)
    
    # 初始化模型评测结果
    print("\n初始化前清理评估任务缓存目录...")
    clean_eval_task_dirs(cache_dir)
    base_model_results, expert_model_results = initialize_model_evaluations(base_model, expert_model)
    
    # 创建包装函数，将所有必要参数传递给model_merge_optimization_function
    def wrapped_optimization_function(x):
        # 根据x的维度动态调整num_blocks
        # 计算当前num_blocks：
        # - optimize_density=1: num_blocks = x.shape[1] - 1
        # - optimize_density=2: num_blocks = (x.shape[1] // 2) - 1
        # - optimize_density=3: num_blocks = (x.shape[1] // 3) - 1
        if len(x.shape) > 1:
            current_dim = x.shape[1]
            if optimize_density == 1:
                current_num_blocks = current_dim - 1
            elif optimize_density == 2:
                current_num_blocks = (current_dim // 2) - 1
            elif optimize_density == 3:
                current_num_blocks = (current_dim // 3) - 1
            else:
                current_num_blocks = num_blocks  # 使用默认值
        else:
            current_num_blocks = num_blocks  # 使用默认值
        
        # 确保current_num_blocks与传入的块数匹配
        print(f"wrapped_optimization_function: x.shape[1]={x.shape[1]}, current_num_blocks={current_num_blocks}")
        
        # 从merged_blocks_dict中获取对应的merged_blocks
        if current_num_blocks in merged_blocks_dict:
            current_merged_blocks = merged_blocks_dict[current_num_blocks]
            print(f"使用块数 {current_num_blocks} 对应的merged_blocks，包含 {len(current_merged_blocks)} 个块")
            
            # 检查merged_blocks包含的块数是否与current_num_blocks匹配
            if len(current_merged_blocks) != current_num_blocks:
                print(f"警告：merged_blocks包含的块数 {len(current_merged_blocks)} 与current_num_blocks {current_num_blocks} 不匹配，重新计算")
                # 重新计算merged_blocks
                from src.evoMI.mi_block_fusion import calculate_merged_blocks
                current_merged_blocks = calculate_merged_blocks(
                    task_model_paths=task_model_paths,
                    num_blocks=current_num_blocks,
                    alpha=alpha,
                    beta=beta,
                    checkpoint_dir=cache_dir  # 保存图形到缓存目录
                )
                # 更新merged_blocks_dict
                merged_blocks_dict[current_num_blocks] = current_merged_blocks
                print(f"重新计算后merged_blocks包含 {len(current_merged_blocks)} 个块")
        else:
            # 如果没有对应的merged_blocks，使用默认的
            default_n_blocks = block_configs[0]["n_blocks"] if (algorithm == "saasbo_qnehvi_block" and block_configs is not None) else num_blocks
            current_merged_blocks = merged_blocks_dict[default_n_blocks]
            print(f"使用默认块数 {default_n_blocks} 对应的merged_blocks，包含 {len(current_merged_blocks)} 个块")
            
            # 检查merged_blocks包含的块数是否与current_num_blocks匹配
            if len(current_merged_blocks) != current_num_blocks:
                print(f"警告：merged_blocks包含的块数 {len(current_merged_blocks)} 与current_num_blocks {current_num_blocks} 不匹配，重新计算")
                # 重新计算merged_blocks
                from src.evoMI.mi_block_fusion import calculate_merged_blocks
                current_merged_blocks = calculate_merged_blocks(
                    task_model_paths=task_model_paths,
                    num_blocks=current_num_blocks,
                    alpha=alpha,
                    beta=beta,
                    checkpoint_dir=cache_dir  # 保存图形到缓存目录
                )
                # 更新merged_blocks_dict
                merged_blocks_dict[current_num_blocks] = current_merged_blocks
                print(f"重新计算后merged_blocks包含 {len(current_merged_blocks)} 个块")
        
        return model_merge_optimization_function(
            x, 
            merged_blocks=current_merged_blocks, 
            num_blocks=current_num_blocks, 
            cache_dir=cache_dir,
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            fusion_method=fusion_method,
            base_model_results=base_model_results,
            expert_model_results=expert_model_results,
            optimize_density=optimize_density
        )
    
    # 根据选择的算法调用相应的优化器
    if algorithm == "saasbo_qnehvi":
        # 调用saasbo_qnehvi_optimizer
        result = saasbo_qnehvi_optimizer(
            wrapped_optimization_function, 
            iteration_callback=iteration_callback, 
            initial_importance=initial_importance,  # 传入层重要性
            use_saas=use_saas,  # 使用SAAS先验
            enable_importance_prior=enable_importance_prior,  # 启用重要性先验
            enable_importance_update=enable_importance_update,  # 启用重要性更新
            enable_importance_guidance=enable_importance_guidance,  # 启用重要性指导
            enable_importance_weighted_acq=enable_importance_weighted_acq,  # 启用或禁用获取函数加权
            **config
        )
    elif algorithm == "qehvi":
        # 调用普通的qehvi_optimizer
        result = qehvi_optimizer(
            wrapped_optimization_function, 
            iteration_callback=iteration_callback, 
            **config
        )
    elif algorithm == "saasbo_qnehvi_block":
        # 检查是否提供了block_configs
        if block_configs is None:
            # 如果没有提供block_configs，则使用默认的块配置
            block_configs = [
                {"start_iter": 0, "end_iter": 9, "n_blocks": num_blocks},
                {"start_iter": 10, "end_iter": N_BATCH - 1, "n_blocks": 36}  # 36是真实的层数，用于精细搜索
            ]
        
        # 调用块优化SAASBO+qNEHVI优化器
        result = saasbo_qnehvi_two_stage(
            wrapped_optimization_function, 
            iteration_callback=iteration_callback, 
            initial_importance=initial_importance,  # 传入层重要性
            initial_importance_dict=initial_importance_dict,  # 传入所有块数对应的重要性
            use_saas=use_saas,  # 使用SAAS先验
            enable_importance_prior=enable_importance_prior,  # 启用重要性先验
            enable_importance_update=enable_importance_update,  # 启用重要性更新
            enable_importance_guidance=enable_importance_guidance,  # 启用重要性指导
            enable_importance_weighted_acq=enable_importance_weighted_acq,  # 启用或禁用获取函数加权
            block_configs=block_configs,  # 块划分配置
            merged_blocks_dict=merged_blocks_dict,  # 传入不同块数对应的合并块
            real_dim=36,  # 真实的层数，用于精细搜索
            **config
        )
    else:
        raise ValueError(f"无效的算法选择: {algorithm}，可选值为'saasbo_qnehvi'、'qehvi'或'saasbo_qnehvi_block'")
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"\n优化完成！总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/3600:.2f} 小时)")
    
    # 构建结果字典
    # saasbo_qnehvi_optimizer返回值：train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
    result_dict = {
        'pareto_x': result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0],
        'pareto_y': result[1].cpu().numpy() if isinstance(result[1], torch.Tensor) else result[1],
        'all_x': result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0],
        'all_y': result[1].cpu().numpy() if isinstance(result[1], torch.Tensor) else result[1],
        'all_metrics': result[2],  # 保存所有评测指标
        'hypervolume_history': result[3] if len(result) > 3 else [],
        'problem_ref_point': result[4].tolist() if isinstance(result[4], torch.Tensor) else result[4],
        'run_id': result[5] if len(result) > 5 else None
    }
    

    
    # 保存结果，不再包含参数
    print("\n保存优化结果...")
    save_optimization_results(result_dict, output_dir)
    
    # 可视化结果
    print("\n生成可视化结果...")
    visualize_optimization_results(result_dict, output_dir)
    
    # 获取帕累托前沿
    pareto_x = result_dict.get('pareto_x', np.array([]))
    pareto_y = result_dict.get('pareto_y', np.array([]))
    print(f"\n找到 {len(pareto_x)} 个帕累托最优解")
    
    # 打印优化统计信息
    print("\n=== 优化统计信息 ===")
    print(f"总评估次数: {len(result_dict.get('all_x', []))}")
    print(f"初始样本数: {config['initial_samples']}")
    print(f"迭代次数: {config['N_BATCH']}")
    print(f"批次大小: {config['BATCH_SIZE']}")
    try:
        hypervolume_history = result_dict.get('hypervolume_history', [0])
        best_hypervolume = max(hypervolume_history) if hypervolume_history else 0
        print(f"最佳超体积: {best_hypervolume}")
    except:
        print("最佳超体积: 计算失败")
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="模型合并优化工具 - 使用SAASBO+qNEHVI算法 - 版本2")
    
    # 添加命令行参数
    parser.add_argument('--custom-initial-solutions', type=str, default=None,
                        help='用户自定义的初始解列表，例如"0.55,0.8"，则将N个初始解里面的两个初始为全是0.55的，和全是0.8，剩下再按原算法生成N-2个初始解')
    parser.add_argument('--num-blocks', type=int, default=36,
                        help='块数，优化参数数量为(block数+1)*2（前(block数+1)个为权重参数，后(block数+1)个为density参数）')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='方差权重，默认1.0')
    parser.add_argument('--beta', type=float, default=0.000,
                        help='均衡权重，默认0.005')
    parser.add_argument('--num-objectives', type=int, default=2,
                        help='目标函数维度：2个目标')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='每批评估的样本数')
    parser.add_argument('--num-restarts', type=int, default=10,
                        help='优化重启次数')
    parser.add_argument('--raw-samples', type=int, default=512,
                        help='初始采样点数量')
    parser.add_argument('--mc-samples', type=int, default=128,
                        help='MC采样数量')
    parser.add_argument('--n-batch', type=int, default=50,
                        help='迭代次数')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='详细输出')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--initial-samples', type=int, default=8,
                        help='初始采样点数')
    parser.add_argument('--noise-level', type=float, default=0.0001,
                        help='噪声水平')
    parser.add_argument('--run-id', type=str, default="instruct_saasbo_qnehvi_prior_grassmann_wasserstein_block_36",
                        help='运行ID')
    parser.add_argument('--cache-dir', type=str, default="output/mi_optimization_temp",
                        help='缓存目录位置 /mnt/data/output/mi_optimization_temp')
    parser.add_argument('--available-gpus', type=str, default="0,1,2,3",
                        help='可用GPU列表，以逗号分隔，例如：0,1,2')
    
    # 模型路径参数
    # 模型路径参数
    parser.add_argument('--base-model', type=str, default="models/Qwen3-4B-Instruct-2507,models/Qwen3-4B-thinking-2507",
                        help='base模型路径列表，以逗号分隔，例如：models/Qwen3-4B-Instruct-2507,models/Qwen3-4B-thinking-2507')
    parser.add_argument('--expert-model', type=str, default="models/Qwen3-4B-thinking-2507,models/Qwen3-4B-Instruct-2507",
                        help='expert模型路径列表，以逗号分隔，例如：models/Qwen3-4B-thinking-2507,models/Qwen3-4B-Instruct-2507')
    parser.add_argument('--base-model-path', type=str, default="models/Qwen3-4B-Instruct-2507",
                        help='基础模型路径')
    parser.add_argument('--task-model-paths', type=str, default="models/Qwen3-4B-thinking-2507,models/Qwen3-4B-Instruct-2507",
                        help='任务模型路径列表，以逗号分隔，例如：models/Qwen3-4B-thinking-2507,models/Qwen3-4B-Instruct-2507')
    
    
    # 算法选择参数
    parser.add_argument('--algorithm', type=str, default="saasbo_qnehvi",
                        help='优化算法选择，可选值："saasbo_qnehvi"、"qehvi"或"saasbo_qnehvi_block"')
    
    # 块优化算法参数
    parser.add_argument('--block-configs', type=str, default="0-2-12;3-5-24;6-20-36",
                        help='块划分配置，格式为：start_iter1-end_iter1-n_blocks1;start_iter2-end_iter2-n_blocks2，例如：0-9-6;10-19-12;20-49-36')

    # 算法参数（仅当--algorithm=saasbo_qnehvi时有效）
    parser.add_argument('--use-saas', type=bool, default=True,
                        help='是否使用SAAS先验（仅当--algorithm=saasbo_qnehvi时有效）')
    parser.add_argument('--enable-importance-prior', type=bool, default=True,
                        help='是否启用重要性先验（仅当--algorithm=saasbo_qnehvi时有效）')
    parser.add_argument('--enable-importance-update', type=bool, default=False,
                        help='是否启用重要性更新（仅当--algorithm=saasbo_qnehvi时有效）')
    parser.add_argument('--enable-importance-guidance', type=bool, default=False,
                        help='是否启用重要性指导（仅当--algorithm=saasbo_qnehvi时有效）')
    parser.add_argument('--enable-importance-weighted-acq', type=bool, default=False,
                        help='是否启用获取函数加权（仅当--algorithm=saasbo_qnehvi时有效）')
    parser.add_argument('--fusion-method', type=str, default="task_arithmetic",
                        help='模型融合方法，例如：breadcrumbs, task_arithmetic等')
    
    # 新增参数
    parser.add_argument('--optimize-density', type=int, default=1,
                        help='密度优化模式：1-仅优化权重，2-优化权重和密度，3-优化权重、密度和gamma参数')
    
    # 距离度量和划分方法参数
    parser.add_argument('--metric', type=str, default='L2-norm',
                        help='距离度量方法，默认值："Grassmann"，可选值："Fisher", "Grassmann", "L2-norm", "Block", "Grassmann-Wasserstein", "LayerNorm-Wasserstein"')
    parser.add_argument('--partition-method', type=str, default='hybrid',
                        help='划分方法，默认值："hybrid"，可选值："hybrid", "balance", "variance"')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置可用GPU列表
    available_gpus_global = [int(gpu.strip()) for gpu in args.available_gpus.split(',')]
    
    # 处理自定义初始解参数
    custom_initial_solutions = None
    if args.custom_initial_solutions:
        try:
            custom_initial_solutions = [float(val.strip()) for val in args.custom_initial_solutions.split(',')]
            print(f"使用自定义初始解: {custom_initial_solutions}")
        except ValueError:
            print(f"警告: 自定义初始解参数格式错误: {args.custom_initial_solutions}，将使用默认初始解生成方式")
    
    # 解析模型路径列表
    base_model = [model.strip() for model in args.base_model.split(',')]
    expert_model = [model.strip() for model in args.expert_model.split(',')]
    task_model_paths = [model.strip() for model in args.task_model_paths.split(',')]
    
    # 解析块划分配置
    block_configs = None
    if args.block_configs:
        try:
            block_configs = []
            for config_str in args.block_configs.split(';'):
                parts = config_str.strip().split('-')
                if len(parts) == 3:
                    start_iter = int(parts[0])
                    end_iter = int(parts[1])
                    n_blocks = int(parts[2])
                    block_configs.append({
                        "start_iter": start_iter,
                        "end_iter": end_iter,
                        "n_blocks": n_blocks
                    })
            print(f"使用自定义块配置: {block_configs}")
        except ValueError:
            print(f"警告: 块配置参数格式错误: {args.block_configs}，将使用默认配置")
            block_configs = None
    
    # 调用主函数，传递所有参数
    main_optimization(
        custom_initial_solutions=custom_initial_solutions,
        num_blocks=args.num_blocks,
        num_objectives=args.num_objectives,
        BATCH_SIZE=args.batch_size,
        NUM_RESTARTS=args.num_restarts,
        RAW_SAMPLES=args.raw_samples,
        MC_SAMPLES=args.mc_samples,
        N_BATCH=args.n_batch,
        verbose=args.verbose,
        device=args.device,
        initial_samples=args.initial_samples,
        cache_dir=args.cache_dir,
        noise_level=args.noise_level,
        run_id=args.run_id,
        alpha=args.alpha,
        beta=args.beta,
        base_model=base_model,
        expert_model=expert_model,
        base_model_path=args.base_model_path,
        task_model_paths=task_model_paths,
        algorithm=args.algorithm,
        use_saas=args.use_saas,
        enable_importance_prior=args.enable_importance_prior,
        enable_importance_update=args.enable_importance_update,
        enable_importance_guidance=args.enable_importance_guidance,
        enable_importance_weighted_acq=args.enable_importance_weighted_acq,
        fusion_method=args.fusion_method,
        block_configs=block_configs,
        optimize_density=args.optimize_density,
        metric=args.metric,
        partition_method=args.partition_method
    )
