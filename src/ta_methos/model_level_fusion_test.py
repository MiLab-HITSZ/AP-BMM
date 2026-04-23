#!/usr/bin/env python3
"""
模型级融合测试脚本
用于测试不同权重[0.1-0.9]下的模型级融合效果，支持多种融合方法
"""

import os
import sys
import time
import uuid
import json
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Any, Tuple
import tempfile

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 导入所需模块
from src.evoMI.vllm_server_manager import VllmServerManager
from src.evoMI.result_processor import ResultProcessor
from src.evoMI.mi_block_fusion import mi_block_fusion, calculate_merged_blocks
from src.evoMI.model_reproduction import generate_model_cache_key, get_model_cache_path, load_cached_results, save_results_to_cache
from src.config_manager import config_manager

# 全局变量
base_model = config_manager._base_model_bi
expert_model = config_manager._expert_model_bi
checkpoint_dir = config_manager.checkpoint_dir
available_gpus_global = [0,1,2,3]

def initialize_model_evaluations(max_tokens: int = 35000, max_model_len: int = None):
    """
    初始化base和expert模型的评测结果
    
    参数:
        max_tokens: 最大生成token数
        max_model_len: 最大模型长度
    """
    global base_model_results, expert_model_results
    
    print("\n===== 初始化模型评测结果 =====")
    
    # 评测base模型
    print(f"\n评测base模型: {base_model}")
    base_model_results = []
    
    for model_path in base_model:
        model_key = generate_model_cache_key(model_path)
        cache_path = get_model_cache_path(checkpoint_dir, model_key, 'original')
        cached_result = load_cached_results(cache_path)
        
        if cached_result:
            print(f"使用base模型的缓存结果: {model_path}")
            base_model_results.append(cached_result)
        else:
            print(f"未找到base模型的缓存结果，开始评测: {model_path}")
            
            # 直接实现评测逻辑，使用双目标配置
            model_id = f"original_{os.path.basename(model_path)}_{uuid.uuid4().hex[:8]}"
            
            # 创建评测任务配置 - 使用双目标配置
            task_cfg = config_manager.create_aime_gpqa_task_config(model_path, max_tokens)
            
            # 创建任务字典
            task = {
                'task_id': f'task_{model_id}',
                'model_path': model_path,
                'params_dict': {'task_cfg': task_cfg},
                'func_handle': run_task_with_server
            }
            
            # 使用VllmServerManager运行任务
            print(f"开始评测模型: {model_path}")
            start_time = time.time()
            try:
                # 如果未指定max_model_len，则设置为max_tokens + 3000
                if max_model_len is None:
                    max_model_len = max_tokens + 3000
                    print(f"未指定max_model_len，将使用默认值: {max_model_len} (max_tokens + 3000)")
                
                with VllmServerManager(available_gpus=available_gpus_global, 
                                     max_model_len=max_model_len) as server_manager:
                    # 调用run_series_tasks方法执行任务
                    results = server_manager.run_series_tasks([task])
                    
                print(f"评测完成，耗时: {time.time() - start_time:.2f} 秒")
                
                # 使用ResultProcessor处理结果
                print("处理评测结果...")
                result_processor = ResultProcessor()
                res = result_processor.process_and_save(results)
                
                # 提取指标
                metrics = {}
                if isinstance(res, dict) and 'processed_results' in res:
                    results_list = res['processed_results']
                elif isinstance(res, list):
                    results_list = res
                else:
                    results_list = [res]
                
                for result in results_list:
                    try:
                        # 简单提取所有数据集的所有指标，直接使用key作为指标名称
                        for dataset_name in ['aime25', 'gpqa_diamond']:
                            if dataset_name in result and isinstance(result[dataset_name], dict):
                                if dataset_name not in metrics:
                                    metrics[dataset_name] = {}
                                # 直接将所有键值对复制到metrics中
                                for key, value in result[dataset_name].items():
                                    metrics[dataset_name][key] = value
                    except Exception as e:
                        print(f"提取指标时出错: {e}")
                
            except Exception as e:
                print(f"评测过程中发生错误: {e}")
                metrics = {"error": str(e)}
            
            model_result = {
                'model_type': 'thinking' if 'thinking' in model_path.lower() else 'instruct',
                'model_name': os.path.basename(model_path),
                'model_path': model_path,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            base_model_results.append(model_result)
            # 保存到缓存
            save_results_to_cache(cache_path, model_result)
    
    # 评测expert模型
    print(f"\n评测expert模型: {expert_model}")
    expert_model_results = []
    
    for model_path in expert_model:
        model_key = generate_model_cache_key(model_path)
        cache_path = get_model_cache_path(checkpoint_dir, model_key, 'original')
        cached_result = load_cached_results(cache_path)
        
        if cached_result:
            print(f"使用expert模型的缓存结果: {model_path}")
            expert_model_results.append(cached_result)
        else:
            print(f"未找到expert模型的缓存结果，开始评测: {model_path}")
            
            # 直接实现评测逻辑，使用双目标配置
            model_id = f"original_{os.path.basename(model_path)}_{uuid.uuid4().hex[:8]}"
            
            # 创建评测任务配置 - 使用双目标配置
            task_cfg = config_manager.create_aime_gpqa_task_config(model_path, max_tokens)
            
            # 创建任务字典
            task = {
                'task_id': f'task_{model_id}',
                'model_path': model_path,
                'params_dict': {'task_cfg': task_cfg},
                'func_handle': run_task_with_server
            }
            
            # 使用VllmServerManager运行任务
            print(f"开始评测模型: {model_path}")
            start_time = time.time()
            try:
                # 如果未指定max_model_len，则设置为max_tokens + 3000
                if max_model_len is None:
                    max_model_len = max_tokens + 3000
                    print(f"未指定max_model_len，将使用默认值: {max_model_len} (max_tokens + 3000)")
                
                with VllmServerManager(available_gpus=available_gpus_global, 
                                     max_model_len=max_model_len) as server_manager:
                    # 调用run_series_tasks方法执行任务
                    results = server_manager.run_series_tasks([task])
                    
                print(f"评测完成，耗时: {time.time() - start_time:.2f} 秒")
                
                # 使用ResultProcessor处理结果
                print("处理评测结果...")
                result_processor = ResultProcessor()
                res = result_processor.process_and_save(results)
                
                # 提取指标
                metrics = {}
                if isinstance(res, dict) and 'processed_results' in res:
                    results_list = res['processed_results']
                elif isinstance(res, list):
                    results_list = res
                else:
                    results_list = [res]
                
                for result in results_list:
                    try:
                        # 简单提取所有数据集的所有指标，直接使用key作为指标名称
                        for dataset_name in ['aime25', 'gpqa_diamond']:
                            if dataset_name in result and isinstance(result[dataset_name], dict):
                                if dataset_name not in metrics:
                                    metrics[dataset_name] = {}
                                # 直接将所有键值对复制到metrics中
                                for key, value in result[dataset_name].items():
                                    metrics[dataset_name][key] = value
                    except Exception as e:
                        print(f"提取指标时出错: {e}")
                
            except Exception as e:
                print(f"评测过程中发生错误: {e}")
                metrics = {"error": str(e)}
            
            model_result = {
                'model_type': 'thinking' if 'thinking' in model_path.lower() else 'instruct',
                'model_name': os.path.basename(model_path),
                'model_path': model_path,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            expert_model_results.append(model_result)
            # 保存到缓存
            save_results_to_cache(cache_path, model_result)
    
    print("\n===== 模型评测初始化完成 =====")
    return base_model_results, expert_model_results

def run_task_with_server(port, task_cfg, served_model_name=None):
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
    
    print(f"在端口 {port} 上执行任务: 模型={task_cfg.model}, 数据集={task_cfg.datasets}")
    
    try:
        # 执行任务
        from evalscope import run_task
        result = run_task(task_cfg=task_cfg)
        print(f"端口 {port} 上的任务执行完成")
        return result
    except Exception as e:
        print(f"端口 {port} 上的任务执行出错: {e}")
        return {"error": str(e)}

def extract_objectives(results, base_model_results=None, expert_model_results=None):
    """
    从评测结果中提取目标函数值（使用动态归一化）
    
    参数:
        results: 评测结果字典
        base_model_results: base模型评测结果
        expert_model_results: expert模型评测结果
    
    返回:
        numpy数组: 形状为(n, 2)的数组，每行包含两个目标函数值
    """
    objectives = []
    
    # 检查results的结构
    if isinstance(results, dict) and 'processed_results' in results:
        results_list = results['processed_results']
    elif isinstance(results, list):
        results_list = results
    else:
        results_list = [results]
    
    for result in results_list:
        try:
            # 提取各个指标
            aime25_acc = result['aime25'].get('mean_acc', 0) if 'aime25' in result else 0
            aime25_tokens_num = result['aime25'].get('mean_tokens_num', 0) if 'aime25' in result else 0
            gpqa_diamond_acc = result['gpqa_diamond'].get('mean_acc', 0) if 'gpqa_diamond' in result else 0
            gpqa_diamond_tokens_num = result['gpqa_diamond'].get('mean_tokens_num', 0) if 'gpqa_diamond' in result else 0
            
            # 确保base_model_results和expert_model_results有效
            if not base_model_results or not expert_model_results:
                print("警告: base_model_results或expert_model_results未初始化，使用默认值")
                # 使用默认值作为备用
                f1 = np.mean([(aime25_acc-0.45)/(0.8-0.45),(gpqa_diamond_acc-0.3)/(0.7-0.3)])
                f2 = -np.mean([(aime25_tokens_num-9000)/(22000-9000),(gpqa_diamond_tokens_num-1000)/(9000-1000)])
            else:
                # 使用动态计算的归一化数值
                # 获取base模型的指标
                base_aime25_acc = base_model_results[0].get('metrics', {}).get('aime25', {}).get('mean_acc', 0.45)
                base_gpqa_diamond_acc = base_model_results[0].get('metrics', {}).get('gpqa_diamond', {}).get('mean_acc', 0.3)
                base_aime25_tokens = base_model_results[1].get('metrics', {}).get('aime25', {}).get('mean_tokens_num', 9000)
                base_gpqa_diamond_tokens = base_model_results[1].get('metrics', {}).get('gpqa_diamond', {}).get('mean_tokens_num', 1000)
                
                # 获取expert模型的指标（取第一个expert模型）
                expert_aime25_acc = expert_model_results[0].get('metrics', {}).get('aime25', {}).get('mean_acc', 0.8)
                expert_gpqa_diamond_acc = expert_model_results[0].get('metrics', {}).get('gpqa_diamond', {}).get('mean_acc', 0.7)
                expert_aime25_tokens = expert_model_results[1].get('metrics', {}).get('aime25', {}).get('mean_tokens_num', 22000)
                expert_gpqa_diamond_tokens = expert_model_results[1].get('metrics', {}).get('gpqa_diamond', {}).get('mean_tokens_num', 9000)
                
                # 计算f1：使用aime25和gpqa_diamond的准确率进行归一化
                # 避免除以零的情况
                aime25_denominator = expert_aime25_acc - base_aime25_acc
                gpqa_diamond_denominator = expert_gpqa_diamond_acc - base_gpqa_diamond_acc
                aime25_norm = (aime25_acc - base_aime25_acc) / aime25_denominator
                gpqa_diamond_norm = (gpqa_diamond_acc - base_gpqa_diamond_acc) / gpqa_diamond_denominator
                f1 = np.mean([aime25_norm, gpqa_diamond_norm])
                
                # 计算f2：使用token数量进行归一化，不再考虑ifeval
                aime25_tokens_denominator = expert_aime25_tokens - base_aime25_tokens
                gpqa_diamond_tokens_denominator = expert_gpqa_diamond_tokens - base_gpqa_diamond_tokens
                aime25_tokens_norm = (aime25_tokens_num - base_aime25_tokens) / aime25_tokens_denominator
                gpqa_diamond_tokens_norm = (gpqa_diamond_tokens_num - base_gpqa_diamond_tokens) / gpqa_diamond_tokens_denominator
                f2 = np.mean([aime25_tokens_norm, gpqa_diamond_tokens_norm])
            
            objectives.append([f1, f2])
        except Exception as e:
            print(f"提取目标函数值时出错: {e}")
            # 使用默认值
            objectives.append([-0.2, -0.2])
    
    return np.array(objectives)

def test_model_level_fusion(
    fusion_method: str = "task_arithmetic",
    num_blocks: int = 8,
    max_tokens: int = 35000,
    max_model_len: int = None,
    run_id: str = "model_level_test",
    batch_size: int = 3,
    weight_min: float = 0.1,
    weight_max: float = 0.9,
    density_min: float = 0.1,
    density_max: float = 0.9,
    budget: int = 10
):
    """
    测试模型级融合效果，使用不同的权重和density参数
    
    参数:
        fusion_method: 融合方法
        num_blocks: block数
        max_tokens: 最大生成token数
        max_model_len: 最大模型长度
        run_id: 运行ID
        batch_size: 批次大小，每次处理的模型数量
        weight_min: 权重最小值
        weight_max: 权重最大值
        density_min: density最小值（非task_arithmetic方法使用）
        density_max: density最大值（非task_arithmetic方法使用）
        budget: 总的计算预算次数
    """
    # 模型路径设置
    base_model_path = "models/Qwen3-4B-Instruct-2507"
    task_model_paths = ["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"]
    
    # 输出目录，包含fusion_method标识
    output_root = f"checkpoints/{run_id}_{fusion_method}"
    os.makedirs(output_root, exist_ok=True)
    
    # 创建临时模型存储目录
    temp_model_dir = "output/model_velvel_temp"
    os.makedirs(temp_model_dir, exist_ok=True)
    
    # 初始化时计算一次merged_blocks
    print(f"\n===== 计算自动合并的块（仅运行一次）=====")
    merged_blocks = calculate_merged_blocks(
        task_model_paths=task_model_paths,
        num_blocks=num_blocks,
        checkpoint_dir=output_root
    )
    
    # 初始化base和expert模型评测结果
    base_results, expert_results = initialize_model_evaluations(max_tokens, max_model_len)
    
    # 生成网格搜索参数
    print(f"\n===== 生成网格搜索参数 =====")
    print(f"融合方法: {fusion_method}")
    print(f"权重范围: [{weight_min}, {weight_max}]")
    if fusion_method != 'task_arithmetic':
        print(f"Density范围: [{density_min}, {density_max}]")
    print(f"计算预算: {budget}")
    
    # 生成参数组合
    param_combinations = []
    
    if fusion_method == 'task_arithmetic':
        # 只需要权重参数
        num_weight_points = budget
        weights = np.linspace(weight_min, weight_max, num_weight_points)
        for weight in weights:
            param_combinations.append({
                'weight': weight,
                'density': None
            })
    else:
        # 需要权重和density参数
        # 计算权重和density的范围
        weight_range = weight_max - weight_min
        density_range = density_max - density_min

        print(f"权重范围大小: {weight_range:.4f}")
        print(f"Density范围大小: {density_range:.4f}")
        
        # 计算范围比例
        if weight_range == 0:
            weight_range = 1.0
        if density_range == 0:
            density_range = 1.0
        
        range_ratio = weight_range / density_range
        print(f"范围比例 (权重:density): {range_ratio:.4f}")
        
        # 根据范围比例分配网格点，确保总组合数接近预算
        # 使用几何分配方法，使得权重和密度的网格点乘积接近预算
        total_grid_points = max(4, budget)  # 确保至少有4个点(2x2)
        
        # 根据范围比例计算每个参数的点数
        if range_ratio >= 1.0:
            # 权重范围更大，分配更多点
            num_density_points = max(2, int(np.sqrt(total_grid_points / range_ratio)))
            num_weight_points = max(2, int(total_grid_points / num_density_points))
        else:
            # density范围更大，分配更多点
            num_weight_points = max(2, int(np.sqrt(total_grid_points * range_ratio)))
            num_density_points = max(2, int(total_grid_points / num_weight_points))
        
        # 调整确保乘积不超过预算且接近预算
        actual_total = num_weight_points * num_density_points
        while actual_total > budget and (num_weight_points > 2 or num_density_points > 2):
            if num_weight_points >= num_density_points:
                num_weight_points -= 1
            else:
                num_density_points -= 1
            actual_total = num_weight_points * num_density_points
        
        print(f"分配的网格点数量: 权重={num_weight_points}, Density={num_density_points}")
        print(f"预计总组合数: {num_weight_points * num_density_points}")
        
        # 生成权重和density的网格
        weights = np.linspace(weight_min, weight_max, num_weight_points)
        densities = np.linspace(density_min, density_max, num_density_points)
        
        # 创建网格并展平
        weight_grid, density_grid = np.meshgrid(weights, densities)
        weight_flat = weight_grid.flatten()
        density_flat = density_grid.flatten()
        
        # 添加所有组合（现在总数已经接近预算）
        for weight, density in zip(weight_flat, density_flat):
            param_combinations.append({
                'weight': weight,
                'density': density
            })
    
    print(f"生成了 {len(param_combinations)} 个参数组合")
    
    # 存储所有结果
    all_decision_variables = []
    all_objectives = []
    all_metrics = []
    skipped_params = []
    
    # 将参数组合分成多个批次
    param_batches = [param_combinations[i:i+batch_size] for i in range(0, len(param_combinations), batch_size)]
    
    # 处理每个批次
    for batch_idx, batch_params in enumerate(param_batches):
        print(f"\n===== 处理批次 {batch_idx+1}/{len(param_batches)}，包含 {len(batch_params)} 个参数组合 =====")
        
        # 创建任务列表和相关映射
        batch_tasks = []
        batch_task_info_map = {}
        
        # 第一阶段：创建当前批次的模型和任务
        for params in batch_params:
            weight = params['weight']
            density = params['density']
            
            # 生成模型级决策变量：所有块使用相同的权重
            # 决策变量格式：num_blocks个块权重 + 1个embedding权重 + 1个norm/lm_head权重
            # 模型级融合时，所有块使用相同的权重
            decision_vars = [weight] * (num_blocks + 1)
            
            print(f"\n===== 准备参数 ====")
            print(f"权重: {weight:.3f}")
            if density is not None:
                print(f"Density: {density:.3f}")
            print(f"决策变量: {decision_vars}")
            
            # 生成唯一的模型ID，包含权重和density参数
            if density is None:
                model_id = f"model_level_{fusion_method}_w{weight:.3f}_{uuid.uuid4().hex[:8]}"
            else:
                model_id = f"model_level_{fusion_method}_w{weight:.3f}_d{density:.3f}_{uuid.uuid4().hex[:8]}"
            
            # 将模型保存到临时目录
            model_output_dir = os.path.join(temp_model_dir, model_id)
            
            # 检查是否存在评测结果缓存
            model_key = generate_model_cache_key(model_output_dir)
            general_cache_path = get_model_cache_path(checkpoint_dir, model_key, 'solution')
            cached_result = load_cached_results(general_cache_path)
            
            if cached_result:
                print(f"使用缓存的评测结果: {model_id}")
                # 提取目标函数值
                metrics = cached_result['metrics']
                # 转换为结果格式
                results = {'processed_results': [metrics]}
                objectives = extract_objectives(results, base_results, expert_results)
                
                # 保存结果
                all_decision_variables.append(decision_vars)
                all_objectives.append(objectives[0])  # 只有一个结果
                all_metrics.append(metrics)
                skipped_weights.append(weight)
                print(f"权重 {weight} 评测完成，目标值: {objectives[0]}")
            else:
                # 调用mi_block_fusion方法进行模型融合
                success = mi_block_fusion(
                    base_model_path=base_model_path,
                    task_model_paths=task_model_paths,
                    block_weights=decision_vars,
                    output_dir=model_output_dir,
                    fusion_method=fusion_method,
                    copy_from_base=True,
                    merged_blocks=merged_blocks,
                    num_blocks=num_blocks
                )
                
                if not success:
                    print(f"警告: 权重 {weight} 融合失败，跳过评测")
                    skipped_weights.append(weight)
                    continue
                
                # 创建评测任务配置 - 使用双目标配置
                task_cfg = config_manager.create_aime_gpqa_task_config(model_output_dir, max_tokens)
                
                # 创建任务字典
                task = {
                    'task_id': f'task_{model_id}',
                    'model_path': model_output_dir,
                    'params_dict': {'task_cfg': task_cfg},
                    'func_handle': run_task_with_server
                }
                
                # 添加任务到列表
                batch_tasks.append(task)
                
                # 保存任务相关信息
            batch_task_info_map[task['task_id']] = {
                'weight': weight,
                'density': density,
                'decision_vars': decision_vars,
                'model_id': model_id,
                'model_output_dir': model_output_dir,
                'general_cache_path': general_cache_path
            }
            
            if density is None:
                print(f"已创建任务: {task['task_id']} 对应权重: {weight:.3f}")
            else:
                print(f"已创建任务: {task['task_id']} 对应权重: {weight:.3f}, Density: {density:.3f}")
        
        # 第二阶段：执行当前批次的所有任务
        if batch_tasks:
            print(f"\n===== 开始执行批次 {batch_idx+1} 的 {len(batch_tasks)} 个评测任务 =====")
            batch_start_time = time.time()
            
            try:
                # 如果未指定max_model_len，则设置为max_tokens + 3000
                if max_model_len is None:
                    batch_max_model_len = max_tokens + 3000
                    print(f"未指定max_model_len，将使用默认值: {batch_max_model_len} (max_tokens + 3000)")
                else:
                    batch_max_model_len = max_model_len
                
                with VllmServerManager(available_gpus=available_gpus_global, 
                                     max_model_len=batch_max_model_len) as server_manager:
                    # 调用run_series_tasks方法执行当前批次的所有任务
                    batch_results = server_manager.run_series_tasks(batch_tasks)
                    
                print(f"批次 {batch_idx+1} 的所有评测任务完成，耗时: {time.time() - batch_start_time:.2f} 秒")
                
                # 使用ResultProcessor处理结果
                print(f"处理批次 {batch_idx+1} 的所有评测结果...")
                result_processor = ResultProcessor()
                processed_batch_results = result_processor.process_and_save(batch_results)
                
                # 检查processed_results的结构
                if isinstance(processed_batch_results, dict) and 'processed_results' in processed_batch_results:
                    batch_results_list = processed_batch_results['processed_results']
                elif isinstance(processed_batch_results, list):
                    batch_results_list = processed_batch_results
                else:
                    batch_results_list = [processed_batch_results]
                
                # 处理每个结果
                for i, result in enumerate(batch_results_list):
                    if i >= len(batch_tasks):
                        break
                    
                    task = batch_tasks[i]
                    task_id = task['task_id']
                    task_info = batch_task_info_map[task_id]
                    weight = task_info['weight']
                    decision_vars = task_info['decision_vars']
                    general_cache_path = task_info['general_cache_path']
                    
                    try:
                        # 提取目标函数值
                        objectives = extract_objectives([result], base_results, expert_results)
                        
                        # 提取详细指标
                        metrics = {}
                        for dataset_name in ['aime25', 'gpqa_diamond']:
                            if dataset_name in result and isinstance(result[dataset_name], dict):
                                metrics[dataset_name] = result[dataset_name]
                        
                        # 保存结果到缓存
                        model_result = {
                            'model_type': f'model_level_{fusion_method}',
                            'model_name': f'model_level_{fusion_method}_{weight:.1f}',
                            'model_path': task_info['model_output_dir'],
                            'metrics': metrics,
                            'timestamp': datetime.now().isoformat()
                        }
                        save_results_to_cache(general_cache_path, model_result)
                        
                        # 保存结果
                        all_decision_variables.append(decision_vars)
                        all_objectives.append(objectives[0])  # 只有一个结果
                        all_metrics.append(metrics)
                        
                        print(f"批次 {batch_idx+1} - 权重 {weight} 评测完成，目标值: {objectives[0]}")
                        
                    except Exception as e:
                        print(f"处理批次 {batch_idx+1} - 任务 {task_id} 结果时出错: {e}")
                        continue
            
            except Exception as e:
                print(f"批次 {batch_idx+1} 评测过程中发生错误: {e}")
        
        # 清空当前批次的临时模型目录，释放存储空间
        print(f"\n===== 清空批次 {batch_idx+1} 的临时模型目录 {temp_model_dir} =====")
        import shutil
        try:
            shutil.rmtree(temp_model_dir)
            # 重新创建空的临时目录
            os.makedirs(temp_model_dir, exist_ok=True)
            print(f"临时模型目录已清空: {temp_model_dir}")
        except Exception as e:
            print(f"清空临时模型目录时出错: {e}")
    
    # 保存所有结果到文件，兼容checkpoint_analyzer
    print(f"\n===== 保存结果到 {output_root} =====")
    
    # 保存为numpy格式
    np.save(os.path.join(output_root, "decision_variables.npy"), np.array(all_decision_variables))
    np.save(os.path.join(output_root, "objectives.npy"), np.array(all_objectives))
    
    # 保存为JSON格式
    results_json = {
        "fusion_method": fusion_method,
        "run_id": run_id,
        "weights": weights.tolist(),
        "results": [
            {
                "weight": float(weights[i]),
                "decision_variables": all_decision_variables[i] if i < len(all_decision_variables) else [],
                "objectives": all_objectives[i].tolist() if i < len(all_objectives) else [],
                "metrics": all_metrics[i] if i < len(all_metrics) else {}
            }
            for i in range(len(weights))
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_root, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    # 保存为类似checkpoint的格式，兼容checkpoint_analyzer
    checkpoint_data = {
        'train_x': torch.tensor(np.array(all_decision_variables)),
        'train_obj_true': torch.tensor(np.array(all_objectives)),
        'train_info': all_metrics,
        'fusion_method': fusion_method,
        'run_id': run_id,
        'weights': weights.tolist()
    }
    
    # 保存为PyTorch checkpoint文件
    torch.save(checkpoint_data, os.path.join(output_root, "checkpoint_iter_0.pt"))
    torch.save(checkpoint_data, os.path.join(output_root, "checkpoint_latest.pt"))
    
    # 清空临时模型目录
    import shutil
    try:
        shutil.rmtree(temp_model_dir)
        print(f"临时模型目录已清空: {temp_model_dir}")
    except Exception as e:
        print(f"清空临时模型目录时出错: {e}")
    
    print(f"所有结果已保存到: {output_root}")
    return output_root

def main():
    """
    主函数
    """
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="模型级融合测试工具")
    
    # 添加命令行参数
    parser.add_argument('--fusion_method', type=str, default="dare_linear",
                        choices=["task_arithmetic", "ties", "dare_ties", "dare_linear", 
                                 "breadcrumbs", "breadcrumbs_ties", "della", "della_linear"],
                        help='融合方法')
    parser.add_argument('--num_blocks', type=int, default=8,
                        help='块数')
    parser.add_argument('--max_tokens', type=int, default=35000,
                        help='最大生成token数')
    parser.add_argument('--max_model_len', type=int, default=None,
                        help='最大模型长度')
    parser.add_argument('--run_id', type=str, default="model_level_test_ins_88",
                        help='运行ID')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小，每次处理的模型数量')
    
    # 网格搜索参数
    parser.add_argument('--weight_min', type=float, default=0.05,
                        help='权重最小值')
    parser.add_argument('--weight_max', type=float, default=0.95,
                        help='权重最大值')
    parser.add_argument('--density_min', type=float, default=0.7,
                        help='density最小值（非task_arithmetic方法使用）')
    parser.add_argument('--density_max', type=float, default=0.9,
                        help='density最大值（非task_arithmetic方法使用）')
    parser.add_argument('--budget', type=int, default=88,
                        help='总的计算预算次数')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用测试函数
    test_model_level_fusion(
        fusion_method=args.fusion_method,
        num_blocks=args.num_blocks,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        run_id=args.run_id,
        batch_size=args.batch_size,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        density_min=args.density_min,
        density_max=args.density_max,
        budget=args.budget
    )

if __name__ == "__main__":
    main()
