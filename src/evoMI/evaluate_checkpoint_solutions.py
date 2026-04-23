#!/usr/bin/env python3
"""
Checkpoint Solutions Evaluator
功能：
1. 读取指定的checkpoint文件
2. 从checkpoint中选取5个解
3. 分别运行对应的评测（AIME和GPQA数据集，各1个问题）
4. 获取evalscope自动输出到outputs目录的结果
5. 提取问题回答，整理为txt输出到结果
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import uuid
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入所需模块
from src.evoMI.vllm_server_manager import VllmServerManager
from src.evoMI.result_processor import ResultProcessor
from src.evoMI.mi_block_fusion import mi_block_fusion
from src.evoMI.checkpoint_analyzer import load_checkpoint, get_pareto_optimal_points
from src.config_manager import config_manager

# 全局变量
outputs_dir = os.path.abspath("./outputs")
results_dir = os.path.abspath("./results")


def generate_uniform_preference_vectors(num_vectors: int, num_objectives: int, obj_min: np.ndarray, obj_max: np.ndarray) -> np.ndarray:
    """
    生成均匀分布的偏好向量，直接均匀采样构成单位圆
    
    Args:
        num_vectors: 偏好向量数量
        num_objectives: 目标函数数量
        obj_min: 目标函数空间的最小值，形状为(num_objectives,)
        obj_max: 目标函数空间的最大值，形状为(num_objectives,)
        
    Returns:
        np.ndarray: 形状为(num_vectors, num_objectives)的偏好向量数组
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


def select_individuals_by_preference(pareto_x, pareto_y, num_vectors=5):
    """
    根据偏好向量选择帕累托前沿个体
    
    Args:
        pareto_x: 帕累托前沿的决策变量
        pareto_y: 帕累托前沿的目标值
        num_vectors: 偏好向量数量
        
    Returns:
        tuple: (selected_solutions, selected_objectives)
    """
    print(f"\n根据偏好向量选择 {num_vectors} 个个体")
    
    # 计算目标函数空间的最小值和最大值
    obj_min = np.min(pareto_y, axis=0)
    obj_max = np.max(pareto_y, axis=0)
    obj_range = obj_max - obj_min
    
    # 生成均匀分布的偏好向量
    preference_vectors = generate_uniform_preference_vectors(num_vectors, pareto_y.shape[1], obj_min, obj_max)
    
    # 归一化帕累托目标值到[0, 1]区间
    normalized_pareto_y = (pareto_y - obj_min) / (obj_range + 1e-10)
    
    # 计算每个帕累托个体与偏好向量的余弦相似度，选择最接近的个体
    selected_indices = []
    individual_to_vecs = {}
    
    for i, pref_vec in enumerate(preference_vectors):
        cos_similarities = []
        for j, norm_obj_values in enumerate(normalized_pareto_y):
            # 归一化目标值向量
            norm_obj_values_norm = norm_obj_values / (np.linalg.norm(norm_obj_values) + 1e-10)
            # 计算余弦相似度
            similarity = np.dot(norm_obj_values_norm, pref_vec)
            cos_similarities.append((similarity, j))
        
        # 按相似度降序排序
        cos_similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 选择相似度最高的个体
        if cos_similarities:
            best_idx = cos_similarities[0][1]
            selected_indices.append(best_idx)
            
            # 记录个体对应的偏好向量
            if best_idx not in individual_to_vecs:
                individual_to_vecs[best_idx] = []
            individual_to_vecs[best_idx].append(i)
    
    # 去重，确保每个个体只被选择一次
    unique_selected_indices = list(set(selected_indices))
    
    # 如果去重后数量不足，补充其他个体
    if len(unique_selected_indices) < num_vectors:
        # 获取所有未被选择的个体索引
        all_indices = set(range(len(pareto_x)))
        unselected_indices = list(all_indices - set(unique_selected_indices))
        
        # 补充未被选择的个体，直到达到所需数量
        num_to_add = num_vectors - len(unique_selected_indices)
        unique_selected_indices.extend(unselected_indices[:num_to_add])
    
    # 按索引排序
    unique_selected_indices.sort()
    
    print(f"根据偏好向量选择了 {len(unique_selected_indices)} 个个体，索引为: {unique_selected_indices}")
    
    # 返回选择的解和对应的目标值
    selected_solutions = pareto_x[unique_selected_indices]
    selected_objectives = pareto_y[unique_selected_indices]
    
    return selected_solutions, selected_objectives


def get_solutions_from_checkpoint(checkpoint_path, num_solutions=5, select_individuals=True):
    """
    从checkpoint中获取指定数量的解
    
    Args:
        checkpoint_path: checkpoint文件路径
        num_solutions: 要获取的解的数量
        select_individuals: 是否根据偏好向量选择个体
        
    Returns:
        tuple: (solutions, objectives)
    """
    print(f"从 {checkpoint_path} 加载checkpoint...")
    checkpoint = load_checkpoint(checkpoint_path)
    
    # 获取所有解和目标值
    train_x = checkpoint['train_x']
    train_obj_true = checkpoint['train_obj_true']
    
    # 转换为numpy数组
    if isinstance(train_x, torch.Tensor):
        train_x_np = train_x.cpu().numpy()
    else:
        train_x_np = train_x
    
    if isinstance(train_obj_true, torch.Tensor):
        train_obj_np = train_obj_true.cpu().numpy()
    else:
        train_obj_np = train_obj_true
    
    # 获取帕累托最优解
    pareto_obj, pareto_indices = get_pareto_optimal_points(train_obj_np, return_indices=True)
    pareto_x = train_x_np[pareto_indices]
    
    print(f"找到 {len(pareto_x)} 个帕累托最优解")
    
    # 如果帕累托解数量小于所需数量，直接返回所有帕累托解
    if len(pareto_x) <= num_solutions:
        print(f"帕累托解数量不足 {num_solutions}，返回所有 {len(pareto_x)} 个解")
        return pareto_x, pareto_obj
    
    if select_individuals:
        # 根据偏好向量选择个体
        return select_individuals_by_preference(pareto_x, pareto_obj, num_solutions)
    else:
        # 否则，从帕累托解中均匀选取指定数量的解
        selected_indices = np.linspace(0, len(pareto_x) - 1, num_solutions, dtype=int)
        selected_solutions = pareto_x[selected_indices]
        selected_objectives = pareto_obj[selected_indices]
        
        print(f"从帕累托解中均匀选取了 {num_solutions} 个解")
        return selected_solutions, selected_objectives


def create_single_question_eval_task_config(model_path, dataset_name, max_tokens=35000):
    """
    创建单个问题的评测任务配置
    
    Args:
        model_path: 模型路径
        dataset_name: 数据集名称
        max_tokens: 最大生成token数
        
    Returns:
        task_cfg: 任务配置对象
    """
    # 获取原始的双目标配置
    task_cfg = config_manager.create_aime_gpqa_task_config(model_path, max_tokens)
    
    # 修改配置，只使用指定数据集的1个问题
    # 正确的做法是修改limit参数，而不是datasets属性
    task_cfg.datasets = [dataset_name]  # 只保留指定的数据集
    task_cfg.limit = {dataset_name: 1}   # 只评估1个问题
    
    return task_cfg


def evaluate_solution(solution, solution_idx, checkpoint_info, base_model_path, task_model_paths, fusion_method="breadcrumbs", num_blocks=8):
    """
    评估单个解
    
    Args:
        solution: 决策变量
        solution_idx: 解的索引
        checkpoint_info: checkpoint信息，用于输出文件名
        base_model_path: 基础模型路径
        task_model_paths: 任务模型路径列表
        fusion_method: 模型融合方法
        num_blocks: block数
        
    Returns:
        dict: 包含评测结果的字典
    """
    print(f"\n===== 评估解 {solution_idx+1} ====")
    
    # 生成唯一的模型ID
    model_id = f"solution_{solution_idx}_{uuid.uuid4().hex[:8]}_{checkpoint_info}"
    output_dir = os.path.join("output", "eval_temp", model_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 调用mi_block_fusion方法进行模型融合
    print(f"开始模型融合，输出目录: {output_dir}")
    
    # 解析决策变量
    optimize_density = 1
    block_weights = None
    block_densities = None
    gamma_params = None
    
    # 确保solution是list类型
    if isinstance(solution, np.ndarray):
        solution_list = solution.tolist()
    else:
        solution_list = solution
    
    if len(solution_list) == num_blocks + 1:
        # 仅优化权重，参数数量 = (block数 + 1)
        block_weights = solution_list
        optimize_density = 1
    elif len(solution_list) == (num_blocks + 1) * 2:
        # 优化权重和密度，参数数量 = (block数 + 1) * 2
        weights_end = num_blocks + 1
        block_weights = solution_list[:weights_end]
        block_densities = solution_list[weights_end:]
        optimize_density = 2
    elif len(solution_list) == (num_blocks + 1) * 3:
        # 优化权重、密度和gamma，参数数量 = (block数 + 1) * 3
        weights_end = num_blocks + 1
        densities_end = weights_end * 2
        block_weights = solution_list[:weights_end]
        block_densities = solution_list[weights_end:densities_end]
        gamma_params = solution_list[densities_end:]
        optimize_density = 3
    
    success = mi_block_fusion(
        base_model_path=base_model_path,
        task_model_paths=task_model_paths,
        block_weights=block_weights,
        block_densities=block_densities,
        block_gammas=gamma_params,
        output_dir=output_dir,
        fusion_method=fusion_method,
        copy_from_base=True,
        num_blocks=num_blocks
    )
    
    if not success:
        print(f"警告: 解 {solution_idx+1} 融合失败，跳过评测")
        return None
    
    # 仅运行评测 - GPQA数据集
    print(f"开始在GPQA数据集上评测解 {solution_idx+1}")
    gpqa_results = run_evaluation(output_dir, model_id, "gpqa_diamond")
    
    # 删除临时模型文件，释放磁盘空间
    print(f"删除临时模型文件: {output_dir}")
    import shutil
    try:
        shutil.rmtree(output_dir)
        print(f"成功删除临时模型文件: {output_dir}")
    except Exception as e:
        print(f"删除临时模型文件时出错: {e}")
    
    return {
        "solution_idx": solution_idx,
        "model_id": model_id,
        "gpqa_results": gpqa_results
    }


def run_evaluation(model_path, model_id, dataset_name):
    """
    运行单个数据集的评测
    
    Args:
        model_path: 模型路径
        model_id: 模型ID
        dataset_name: 数据集名称
        
    Returns:
        dict: 评测结果
    """
    # 创建评测任务配置 - 使用双目标配置，参考mi_opt_saasbo2.py
    max_tokens = 35000
    task_cfg = config_manager.create_aime_gpqa_task_config(model_path, max_tokens)
    
    # 修改配置，只使用指定数据集的1个问题
    # 正确的做法是修改datasets和limit参数，而不是使用字典格式
    task_cfg.datasets = [dataset_name]  # 只保留指定的数据集
    task_cfg.limit = {dataset_name: 1}   # 只评估1个问题
    
    # 创建任务字典
    task = {
        'task_id': f'{dataset_name}_{model_id}',
        'model_path': model_path,
        'params_dict': {'task_cfg': task_cfg},
        'func_handle': run_task_with_server
    }
    
    # 使用VllmServerManager运行任务 - 只使用1个GPU
    try:
        max_model_len = max_tokens + 3000
        
        # 只使用第一个GPU
        with VllmServerManager(available_gpus=[0], 
                              max_model_len=max_model_len) as server_manager:
            # 调用run_series_tasks方法执行任务
            results = server_manager.run_series_tasks([task])
        
        # 使用ResultProcessor处理结果
        result_processor = ResultProcessor()
        res = result_processor.process_and_save(results)
        
        return res
    except Exception as e:
        print(f"评测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def run_task_with_server(port, task_cfg, served_model_name=None):
    """
    在指定端口的服务器上执行任务
    
    Args:
        port: 服务器端口
        task_cfg: 任务配置对象
        
    Returns:
        任务执行结果
    """
    from evalscope import run_task
    
    # 更新API URL以使用正确的端口
    task_cfg.api_url = f'http://127.0.0.1:{port}/v1/chat/completions'
    if served_model_name:
        task_cfg.model = served_model_name
    
    print(f"在端口 {port} 上执行任务: 模型={task_cfg.model}, 数据集={task_cfg.datasets}")
    
    try:
        # 执行任务
        result = run_task(task_cfg=task_cfg)
        print(f"端口 {port} 上的任务执行完成")
        return result
    except Exception as e:
        print(f"端口 {port} 上的任务执行出错: {e}")
        return {"error": str(e)}


def extract_answers_from_outputs(model_id, dataset_name):
    """
    从outputs目录中提取指定模型和数据集的回答，包括问题、答案、正确答案和字符数
    
    Args:
        model_id: 模型ID
        dataset_name: 数据集名称
        
    Returns:
        dict: 提取的完整结果字典，包含问题、回答、正确答案、完整回答和字符数
    """
    print(f"从outputs目录提取 {dataset_name} 结果...")
    
    # 查找最新的输出目录
    if not os.path.exists(outputs_dir):
        print(f"警告: outputs目录不存在: {outputs_dir}")
        return {}
    
    # 获取所有输出目录，按修改时间排序
    timestamp_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    timestamp_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)), reverse=True)
    
    if not timestamp_dirs:
        print(f"警告: outputs目录中没有时间戳子目录")
        return {}
    
    # 遍历所有时间戳目录，直到找到匹配的结果文件
    for timestamp_dir in timestamp_dirs:
        timestamp_path = os.path.join(outputs_dir, timestamp_dir)
        
        # 检查是否存在reviews和reports子目录
        reviews_path = os.path.join(timestamp_path, "reviews")
        reports_path = os.path.join(timestamp_path, "reports")
        if not os.path.exists(reviews_path) or not os.path.exists(reports_path):
            continue
        
        # 遍历reviews目录下的所有模型目录
        for review_model_dir in os.listdir(reviews_path):
            if model_id in review_model_dir:
                review_model_path = os.path.join(reviews_path, review_model_dir)
                report_model_path = os.path.join(reports_path, review_model_dir)
                
                # 遍历模型目录下的所有JSONL文件
                for review_file in os.listdir(review_model_path):
                    if review_file.endswith('.jsonl') and dataset_name.lower() in review_file.lower():
                        review_file_path = os.path.join(review_model_path, review_file)
                        print(f"找到reviews文件: {review_file_path}")
                        
                        # 查找对应的reports文件
                        report_file_path = None
                        for report_file in os.listdir(report_model_path):
                            if report_file.endswith('.json') and dataset_name.lower() in report_file.lower():
                                report_file_path = os.path.join(report_model_path, report_file)
                                print(f"找到reports文件: {report_file_path}")
                                break
                        
                        # 提取字符数（从reports目录）
                        tokens_num = 0
                        if report_file_path:
                            try:
                                with open(report_file_path, 'r', encoding='utf-8') as rf:
                                    report_data = json.load(rf)
                                    
                                    # 提取mean_tokens_num指标的score
                                    if 'metrics' in report_data and isinstance(report_data['metrics'], list):
                                        for metric in report_data['metrics']:
                                            if metric.get('name') == 'mean_tokens_num':
                                                # 直接使用指标的top-level score作为字符数
                                                tokens_num = metric.get('score', 0)
                                                print(f"提取到字符数: {tokens_num}")
                                                break
                            except Exception as e:
                                print(f"读取reports文件时出错: {e}")
                        
                        try:
                            with open(review_file_path, 'r', encoding='utf-8') as f:
                                # 读取整个文件内容，然后按行分割
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        try:
                                            review_data = json.loads(line)
                                            
                                            # 提取问题
                                            question = ""
                                            final_answer = ""
                                            full_answer = ""
                                            correct_answer = ""
                                            
                                            # 提取问题
                                            if 'input' in review_data:
                                                input_content = review_data['input']
                                                # 从input中提取问题，去除**User**: 前缀
                                                if "**User**:" in input_content:
                                                    question = input_content.split("**User**:")[-1].strip()
                                                else:
                                                    question = input_content.strip()
                                            
                                            # 提取正确答案
                                            if 'target' in review_data:
                                                correct_answer = review_data['target']
                                            
                                            # 提取模型回答
                                            if 'sample_score' in review_data and isinstance(review_data['sample_score'], dict):
                                                sample_score = review_data['sample_score']
                                                score_data = sample_score.get('score', {})
                                                
                                                # 提取完整回答
                                                full_answer = ""
                                                if isinstance(score_data, dict):
                                                    # 从score_data中提取完整回答
                                                    if 'prediction' in score_data:
                                                        full_answer = score_data['prediction']
                                                        print(f"提取到prediction字段")
                                                    elif 'extracted_prediction' in score_data:
                                                        full_answer = score_data['extracted_prediction']
                                                        print(f"提取到extracted_prediction字段")
                                                
                                                # 提取最终答案
                                                if dataset_name.lower() == "aime25":
                                                    # AIME答案通常在\boxed{xxx}中
                                                    if full_answer:
                                                        boxed_start = full_answer.find('\\boxed{')
                                                        if boxed_start != -1:
                                                            boxed_end = full_answer.find('}', boxed_start + 7)
                                                            if boxed_end != -1:
                                                                final_answer = full_answer[boxed_start + 7:boxed_end]
                                                elif dataset_name.lower().startswith("gpqa"):
                                                    # GPQA答案通常在"ANSWER: X"中
                                                    if full_answer:
                                                        answer_start = full_answer.rfind('ANSWER: ')
                                                        if answer_start != -1:
                                                            final_answer = full_answer[answer_start + 8:].strip()
                                                        # 也检查extracted_prediction
                                                        elif isinstance(score_data, dict) and 'extracted_prediction' in score_data:
                                                            final_answer = score_data['extracted_prediction']
                                                
                                                # 如果没有找到格式化答案，使用完整回答的最后部分
                                                if not final_answer and full_answer:
                                                    final_answer = full_answer.split('\n')[-1].strip()
                                                # 或者直接使用extracted_prediction作为最终答案
                                                elif not final_answer and isinstance(score_data, dict) and 'extracted_prediction' in score_data:
                                                    final_answer = score_data['extracted_prediction']
                                            
                                            # 如果找到了完整的信息，返回结果字典
                                            if question and full_answer:
                                                # 返回完整的结果字典
                                                return {
                                                    "dataset": dataset_name,
                                                    "question": question,
                                                    "model_answer": final_answer,
                                                    "correct_answer": correct_answer,
                                                    "character_count": tokens_num,
                                                    "full_answer": full_answer
                                                }
                                        except json.JSONDecodeError as e:
                                            print(f"解析JSON行时出错: {e}")
                                            continue
                        except Exception as e:
                            print(f"读取reviews文件时出错: {e}")
                            continue
    
    print(f"未找到匹配的结果文件，model_id: {model_id}, dataset_name: {dataset_name}")
    return {}


def generate_results_txt(eval_results, checkpoint_path, output_file):
    """
    整理评测结果为单个LaTeX表格
    
    Args:
        eval_results: 评测结果列表
        checkpoint_path: checkpoint路径，用于输出信息
        output_file: 输出文件路径
    """
    print(f"\n===== 整理结果为单个LaTeX表格 ====")
    
    # 创建结果目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 准备结果内容
    result_lines = []
    
    # LaTeX 文档头部
    result_lines.append(r"\documentclass{article}")
    result_lines.append(r"\usepackage{longtable}")
    result_lines.append(r"\usepackage{ltablex}")
    result_lines.append(r"\usepackage{booktabs}")
    result_lines.append(r"\usepackage{geometry}")
    result_lines.append(r"\geometry{a4paper, margin=1in}")
    result_lines.append(r"\begin{document}")
    result_lines.append("")
    
    # 标题和基本信息
    result_lines.append(r"\section*{Checkpoint Solutions Evaluation Results}")
    result_lines.append(f"Checkpoint: {checkpoint_path}")
    result_lines.append(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result_lines.append("")
    
    # 创建单个综合LaTeX表格
    result_lines.append(r"\begin{longtable}{|c|p{0.2\textwidth}|p{0.35\textwidth}|c|c|c|p{0.35\textwidth}|}")
    result_lines.append(r"\hline")
    result_lines.append(r"\textbf{Solution} & \textbf{Dataset} & \textbf{Question} & \textbf{Model Answer} & \textbf{Correct Answer} & \textbf{Character Count} & \textbf{Full Answer} \\ ")
    result_lines.append(r"\hline")
    result_lines.append(r"\endfirsthead")
    result_lines.append(r"\hline")
    result_lines.append(r"\textbf{Solution} & \textbf{Dataset} & \textbf{Question} & \textbf{Model Answer} & \textbf{Correct Answer} & \textbf{Character Count} & \textbf{Full Answer} \\ ")
    result_lines.append(r"\hline")
    result_lines.append(r"\endhead")
    result_lines.append(r"\hline")
    result_lines.append(r"\endfoot")
    result_lines.append(r"\endlastfoot")
    
    # 添加每个解决方案的数据
    for eval_result in eval_results:
        if eval_result is None:
            continue
        
        solution_idx = eval_result["solution_idx"]
        model_id = eval_result["model_id"]
        
        # 仅处理GPQA结果
        gpqa_answer = extract_answers_from_outputs(model_id, "gpqa_diamond")
        if gpqa_answer:
            # 直接使用字典数据
            result_lines.append(str(solution_idx+1) + r" & GPQA & " + gpqa_answer.get('question', '') + r" & " + gpqa_answer.get('model_answer', '') + r" & " + gpqa_answer.get('correct_answer', '') + r" & " + str(gpqa_answer.get('character_count', '')) + r" & " + gpqa_answer.get('full_answer', '') + r" \\ ")
            result_lines.append(r"\hline")
    
    result_lines.append(r"\end{longtable}")
    result_lines.append("")
    
    # LaTeX文档结束
    result_lines.append(r"\end{document}")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in result_lines:
            f.write(line + '\n')
    
    print(f"单个LaTeX表格结果已保存到: {output_file}")


def main():
    """
    主函数
    """
    # 测试模式：允许用户在main函数中直接设定checkpoints
    test_mode = True  # 设置为True时，使用下面的测试配置
    
    if test_mode:
        # 测试配置 - 用户可以在这里直接修改
        checkpoints = [
            "./checkpoints/instruct_saasbo_qnehvi_prior_block_36/checkpoint_iter_20.pt"
        ]
        num_solutions = 5
        base_model = "models/Qwen3-4B-Instruct-2507"
        task_models = ["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"]
        num_blocks = 36
        fusion_method = "task_arithmetic"
        select_individuals = True  # 是否根据偏好向量选择个体
    else:
        # 命令行参数模式
        parser = argparse.ArgumentParser(description='Checkpoint Solutions Evaluator')
        parser.add_argument('--checkpoints', type=str, nargs='+', required=True, help='Checkpoint文件路径列表')
        parser.add_argument('--num-solutions', type=int, default=5, help='每个checkpoint要评估的解的数量')
        parser.add_argument('--base-model', type=str, default="models/Qwen3-4B", help='基础模型路径')
        parser.add_argument('--task-models', type=str, nargs='+', default=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B"], help='任务模型路径列表')
        parser.add_argument('--num-blocks', type=int, default=8, help='block数')
        parser.add_argument('--fusion-method', type=str, default="breadcrumbs", help='模型融合方法')
        parser.add_argument('--select-individuals', default=True, action='store_true', help='根据偏好向量选择个体')
        parser.add_argument('--no-select-individuals', dest='select_individuals', action='store_false', help='不根据偏好向量选择个体')
        
        args = parser.parse_args()
        
        checkpoints = args.checkpoints
        num_solutions = args.num_solutions
        base_model = args.base_model
        task_models = args.task_models
        num_blocks = args.num_blocks
        fusion_method = args.fusion_method
        select_individuals = args.select_individuals
    
    print(f"\n===== Checkpoint Solutions Evaluator ====")
    print(f"测试模式: {test_mode}")
    print(f"要评估的checkpoints: {checkpoints}")
    print(f"每个checkpoint评估的解数量: {num_solutions}")
    print(f"是否根据偏好向量选择个体: {select_individuals}")
    
    if not checkpoints:
        print("警告: 没有指定checkpoints，请在main函数中设置checkpoints列表")
        return
    
    for checkpoint_path in checkpoints:
        if not os.path.exists(checkpoint_path):
            print(f"警告: Checkpoint文件不存在: {checkpoint_path}")
            continue
        
        # 从checkpoint中获取解
        solutions, objectives = get_solutions_from_checkpoint(checkpoint_path, num_solutions, select_individuals)
        
        # 按照目标函数1（第一个目标）排序
        print(f"\n按照目标函数1排序解决方案...")
        # 组合解和目标值，以便排序
        solutions_with_obj = list(zip(solutions, objectives))
        # 按第一个目标函数值排序
        solutions_with_obj.sort(key=lambda x: x[1][0])
        # 重新解包排序后的解和目标值
        solutions = [s for s, obj in solutions_with_obj]
        objectives = [obj for s, obj in solutions_with_obj]
        
        print(f"排序后各解的目标值:")
        for i, obj in enumerate(objectives):
            print(f"解 {i+1} 的目标值: {obj}")
        
        # 准备checkpoint信息，用于输出文件名
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
        checkpoint_dir = os.path.basename(os.path.dirname(checkpoint_path))
        checkpoint_info = f"{checkpoint_dir}_{checkpoint_name}"
        
        # 评估每个解
        eval_results = []
        for i, (solution, objective) in enumerate(zip(solutions, objectives)):
            print(f"\n解 {i+1} 的目标值: {objective}")
            eval_result = evaluate_solution(
                solution=solution,
                solution_idx=i,
                checkpoint_info=checkpoint_info,
                base_model_path=base_model,
                task_model_paths=task_models,
                fusion_method=fusion_method,
                num_blocks=num_blocks
            )
            eval_results.append(eval_result)
        
        # 整理结果为txt
        output_file = os.path.join(results_dir, f"eval_results_{checkpoint_info}.txt")
        generate_results_txt(eval_results, checkpoint_path, output_file)
    
    print(f"\n===== 所有评估完成 ====")


if __name__ == "__main__":
    main()
