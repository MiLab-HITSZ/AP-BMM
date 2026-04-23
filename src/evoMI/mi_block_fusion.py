#!/usr/bin/env python3
"""
模型块融合模块
提供基于自动块合并的模型融合功能
"""

import os
import sys
import json
import re
import uuid
from typing import List, Dict, Tuple
import numpy as np
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所需模块
from src.evoMI.task_diff_analyzer import TaskDiffAnalyzer
from src.ta_methos.model_fusion_layer import LayerwiseModelFusion, LayerFusionConfig


def calculate_merged_blocks(task_model_paths: List[str], num_blocks: int = 8, alpha: float = 1.0, beta: float = 0.005, checkpoint_dir: str = None, block_numbers: List[int] = None, metric: str = "L2-norm", partition_method: str = "hybrid"):
    """
    计算自动合并的块，只需在初始化时运行一次
    
    参数:
        task_model_paths: 任务模型路径列表
        num_blocks: 要合并的块数，默认为8
        alpha: 方差权重，默认为1.0
        beta: 均衡权重，默认为0.005
        checkpoint_dir: 检查点目录，用于保存图形结果
        block_numbers: 可选，用于生成从细到粗的分块配置列表（如[6, 12, 24, 36]）
        metric: 距离度量方法，默认为"L2-norm"
        partition_method: 划分方法，可选值："hybrid"、"balance"、"variance"，默认为"hybrid"
    
    返回:
        当block_numbers为None时: List[Tuple[List[int], float]] - 合并后的块列表
        当block_numbers不为None时: Dict[int, List[Tuple[List[int], float]]] - 不同块数对应的合并块字典
    """
    # 初始化TaskDiffAnalyzer获取自动合并的块
    print("\n初始化TaskDiffAnalyzer获取自动合并的块...")
    visualizer = TaskDiffAnalyzer(device="cpu", alpha=alpha, beta=beta)
    
    # 使用前两个任务模型来确定块合并
    if len(task_model_paths) < 2:
        raise ValueError("至少需要两个任务模型来确定块合并")
    
    model1_path = task_model_paths[0]
    model2_path = task_model_paths[1]
    
    if block_numbers is None:
        # 获取单个自动合并的块配置
        merged_blocks = visualizer.run(
            model1_path, 
            model2_path, 
            output_dir=checkpoint_dir,  # 保存图形到检查点目录
            num_blocks=num_blocks,
            metric=metric,
            partition_method=partition_method
        )
        
        print(f"\n自动合并结果: 共生成 {len(merged_blocks)} 个块")
        for i, (block_layers, block_diff) in enumerate(merged_blocks):
            print(f"块 {i+1}: 层 {block_layers[0]}-{block_layers[-1]} (差异: {block_diff:.6f})")
        
        return merged_blocks
    else:
        # 生成从细到粗的多个块配置
        print(f"\n生成从细到粗的块配置: {block_numbers}")
        block_configs = visualizer.generate_multiple_block_configs(
            model1_path, 
            model2_path, 
            block_numbers=block_numbers, 
            output_dir=checkpoint_dir,
            metric=metric,
            partition_method=partition_method
        )
        
        # 打印所有块配置
        for block_count, merged_blocks in block_configs.items():
            print(f"\n自动合并结果 ({block_count} 块): 共生成 {len(merged_blocks)} 个块")
            for i, (block_layers, block_diff) in enumerate(merged_blocks):
                print(f"块 {i+1}: 层 {block_layers[0]}-{block_layers[-1]} (差异: {block_diff:.6f})")
        
        return block_configs


def create_fusion_configs(merged_blocks: List[Tuple[List[int], float]], block_weights: List[float], 
                         block_densities: List[float] = None, fusion_method: str = "ties", num_task_models: int = 2, block_gammas: List[float] = None, num_blocks: int = None):
    """
    根据合并的块和权重创建融合配置
    
    参数:
        merged_blocks: 合并后的块列表
        block_weights: 块权重列表，包含num_blocks+1个决策变量
        block_densities: 块密度列表，包含num_blocks+1个决策变量
        fusion_method: 融合方法
        num_task_models: 任务模型数量，默认为2
        num_blocks: 块数，用于计算expected_length，如果为None则使用len(merged_blocks)
    
    返回:
        List[LayerFusionConfig]: 融合配置列表
    """
    # 确保block_weights是列表
    if not isinstance(block_weights, list):
        raise ValueError("block_weights必须是列表类型")
    
    # 确定使用的块数，优先使用传入的num_blocks参数
    if num_blocks is None:
        num_blocks = len(merged_blocks)
    
    # block_weights长度必须为num_blocks + 1（num_blocks个块权重 + 1个norm/lm_head权重）
    expected_length = num_blocks + 1
    if len(block_weights) != expected_length:
        raise ValueError(f"block_weights必须包含 {expected_length} 个权重值，当前长度为 {len(block_weights)}")
    
    # 如果提供了block_densities，确保其长度正确
    if block_densities is not None:
        if len(block_densities) != expected_length:
            raise ValueError(f"block_densities必须包含 {expected_length} 个密度值，当前长度为 {len(block_densities)}")
    
    print(f"\n创建融合配置...")
    configs = []
    
    # 为每个自动合并的块创建配置
    for i, (block_layers, _) in enumerate(merged_blocks):
        # 生成层模式正则表达式
        if len(block_layers) == 1:
            # 单个层
            layer_pattern = rf"layers\.{block_layers[0]}"
        else:
            # 层范围
            start_layer = block_layers[0]
            end_layer = block_layers[-1]
            # 创建匹配该范围的正则表达式
            layer_pattern = rf"layers\.({start_layer}|{end_layer}|({start_layer+1}-{end_layer-1}))"
        
        # 为当前块创建model_weights: [x, 1-x]，其中x是block_weights[i]
        # 对于每个块，使用独立的model_weights，对应到决策变量x是[x, 1-x]
        current_model_weights = [block_weights[i], 1 - block_weights[i]]
        
        # 从block_densities获取当前块的密度值，默认0.8
        current_density = 0.8
        if block_densities is not None:
            current_density = block_densities[i]
        
        # 获取当前块的gamma值，默认0.0
        current_gamma = 0.0
        if block_gammas is not None:
            current_gamma = block_gammas[i]
        
        # 为当前块创建配置
        config = LayerFusionConfig(
            method=fusion_method,
            params={
                "density": current_density, 
                "layer_weight": 1.0,
                "normalize": True,
                "gamma": current_gamma
            },
            layer_pattern=layer_pattern,
            apply_to_embeddings=False,
            apply_to_norm=False,
            apply_to_lm_head=False,
            model_weights=current_model_weights  # 为每个块配置独立的model_weights
        )
        configs.append(config)
        print(f"创建块 {i+1} 配置: 层 {block_layers[0]}-{block_layers[-1]}, 权重: {1.0}, model_weights: {current_model_weights}, density: {current_density}, gamma: {current_gamma}")
    
    # 处理embedding层，固定权重为1.0
    embedding_weight = 1.0  # 固定embedding权重为1.0
    embedding_model_weights = [embedding_weight, 1 - embedding_weight]
    
    # 处理embeddings层
    emb_config = LayerFusionConfig(
        method="task_arithmetic",
        params={
            "layer_weight": 1.0
        },
        apply_to_embeddings=True,
        apply_to_norm=False,
        apply_to_lm_head=False,
        model_weights=embedding_model_weights  # 为embeddings层配置独立的model_weights
    )
    configs.append(emb_config)
    print(f"创建embeddings层配置: 权重: 1.0, model_weights: {embedding_model_weights}")
    
    # 处理norm层和lm_head层，使用第num_blocks个参数
    norm_lm_head_weight = block_weights[num_blocks]  # 第num_blocks个参数用于norm和lm_head层
    norm_lm_head_model_weights = [norm_lm_head_weight, 1 - norm_lm_head_weight]
    
    # 获取norm层的密度值，默认1.0
    norm_density = 1.0
    if block_densities is not None:
        norm_density = block_densities[num_blocks]
    
    # 处理norm层
    norm_config = LayerFusionConfig(
        method=fusion_method,
        params={
            "density": norm_density, 
            "layer_weight": 1.0
        },
        apply_to_embeddings=False,
        apply_to_norm=True,
        apply_to_lm_head=False,
        model_weights=norm_lm_head_model_weights  # norm层使用共用的model_weights
    )
    configs.append(norm_config)
    print(f"创建norm层配置: 权重: {1.0}, model_weights: {norm_lm_head_model_weights}, density: {norm_density}")
    
    # 处理lm_head层，获取密度值，默认1.0
    lm_head_density = 1.0
    if block_densities is not None:
        lm_head_density = block_densities[num_blocks]
    
    lm_head_config = LayerFusionConfig(
        method="task_arithmetic",
        params={
            "density": lm_head_density, 
            "layer_weight": 1.0
        },
        apply_to_embeddings=False,
        apply_to_norm=False,
        apply_to_lm_head=True,
        model_weights=norm_lm_head_model_weights  # lm_head层使用共用的model_weights
    )
    configs.append(lm_head_config)
    print(f"创建lm_head层配置: 权重: {1.0}, model_weights: {norm_lm_head_model_weights}, density: {lm_head_density}")
    
    return configs


def mi_block_fusion(base_model_path: str, task_model_paths: List[str], 
                   block_weights: List[float], output_dir: str, 
                   fusion_method: str = "task_arithmetic", 
                   copy_from_base: bool = True, 
                   merged_blocks: List[Tuple[List[int], float]] = None, 
                   num_blocks: int = 8,
                   block_densities: List[float] = None,
                   block_gammas: List[float] = None,
                   fusion_device: str = "cpu"):
    """
    基于自动块合并的模型融合方法
    
    参数:
        base_model_path: 基础模型路径
        task_model_paths: 任务模型路径列表
        block_weights: 块权重列表，包含num_blocks + 1个决策变量：
                      - 前num_blocks个值用于num_blocks个自动合并的transformer块
                      - 最后1个值用于正则层和输出层(lm_head)共用
        output_dir: 输出目录
        fusion_method: 融合方法，默认为"task_arithmetic"
        copy_from_base: 是否从基础模型复制非权重文件
        merged_blocks: 预计算的合并块列表，如果为None则自动计算
        num_blocks: 要合并的块数，默认为8
        block_densities: 块密度列表，包含num_blocks + 1个决策变量：
                        - 前num_blocks个值用于num_blocks个自动合并的transformer块
                        - 最后1个值用于正则层和输出层(lm_head)共用
        block_gammas: 块gamma参数列表，包含num_blocks + 1个决策变量：
                     - 前num_blocks个值用于num_blocks个自动合并的transformer块
                     - 最后1个值用于正则层和输出层(lm_head)共用
    
    返回:
        bool: 融合是否成功
    """
    # 确保block_weights是列表
    if not isinstance(block_weights, list):
        raise ValueError("block_weights必须是列表类型")
    
    print(f"使用的块权重数量: {len(block_weights)}")
    if block_densities is not None:
        print(f"使用的块密度数量: {len(block_densities)}")
    print(f"融合策略: {num_blocks}个自动合并的transformer块，正则层和输出层各一组")
    if fusion_device not in ["cpu", "cuda"]:
        raise ValueError(f"fusion_device必须是'cpu'或'cuda'，当前值: {fusion_device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录 '{output_dir}' 已准备就绪。")
    
    # 如果没有提供预计算的块，则计算块合并
    if merged_blocks is None:
        merged_blocks = calculate_merged_blocks(task_model_paths, num_blocks)
    
    # 创建融合配置，传递任务模型数量、块密度和gamma参数
    configs = create_fusion_configs(merged_blocks, block_weights, block_densities, fusion_method, len(task_model_paths), block_gammas, num_blocks)
    
    # 初始化融合器
    print("\n初始化LayerwiseModelFusion...")
    fusion = LayerwiseModelFusion()
    
    # 执行融合，不再传递全局model_weights，每个配置有自己的model_weights
    print("\n开始执行层融合...")
    runtime_fusion_device = fusion_device
    if runtime_fusion_device == "cuda" and not torch.cuda.is_available():
        print("警告: 请求使用cuda进行融合，但当前不可用，自动切换为cpu")
        runtime_fusion_device = "cpu"
    print(f"融合设备: {runtime_fusion_device}")
    success = fusion.layer_fusion(
        base_model_path=base_model_path,
        task_model_paths=task_model_paths,
        output_path=output_dir,
        layer_configs=configs,
        device=runtime_fusion_device
    )
    
    if success:
        print(f"\n模型融合成功完成！")
        print(f"融合模型已保存到: {output_dir}")
        print(f"使用的融合方法: {fusion_method}")
        print(f"自动合并的块数: {len(merged_blocks)}")
    else:
        print(f"\n模型融合失败！")
        return False
    return True


def process_decision_variables(decision_matrix, base_model_path, 
                              task_model_paths, base_output_dir,
                              fusion_method="ties",
                              num_blocks=8):
    """
    处理决策变量矩阵，为每个决策变量创建融合模型
    
    参数:
        decision_matrix: numpy数组，形状为 (n, num_blocks+1)，每行代表一个候选解决方案的num_blocks+1个决策变量
        base_model_path: 基础模型路径
        task_model_paths: 任务模型路径列表
        base_output_dir: 基础输出目录
        fusion_method: 融合方法
        num_blocks: 自动合并的块数
    
    返回:
        list: 成功创建的模型路径列表
    """
    # 确保决策矩阵是numpy数组
    if isinstance(decision_matrix, list):
        decision_matrix = np.array(decision_matrix)
    
    # 验证决策矩阵的维度
    expected_dim = num_blocks + 1
    if decision_matrix.ndim != 2 or decision_matrix.shape[1] != expected_dim:
        raise ValueError(f"决策变量矩阵必须是二维数组，每行包含 {expected_dim} 个决策变量")
    
    num_candidates = decision_matrix.shape[0]
    print(f"开始处理 {num_candidates} 个候选方案")
    
    # 确保输出目录存在
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 创建任务列表
    model_paths = []
    
    # 为每个候选方案创建融合模型
    for i in range(num_candidates):
        # 提取当前行的决策变量
        block_weights = decision_matrix[i].tolist()
        print(f"\n处理候选方案 {i+1}/{num_candidates}: {block_weights}")
        
        # 生成唯一的模型ID
        model_id = f"merged_model_{i}_{uuid.uuid4().hex[:8]}"
        model_output_dir = os.path.join(base_output_dir, model_id)
        
        # 调用mi_block_fusion方法进行模型融合
        success = mi_block_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            block_weights=block_weights,
            output_dir=model_output_dir,
            fusion_method=fusion_method,
            copy_from_base=True,
            num_blocks=num_blocks
        )
        
        if success:
            model_paths.append(model_output_dir)
            print(f"融合模型创建成功: {model_output_dir}")
        else:
            print(f"警告: 候选方案 {i+1} 融合失败")
    
    return model_paths


def main():
    """
    命令行接口示例
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="基于自动块合并的模型融合工具")
    parser.add_argument("--base_model", type=str, default="models/Qwen3-4B-Base", help="基础模型路径")
    parser.add_argument("--task_models", type=str, nargs='+', 
                        default=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"], 
                        help="任务模型路径列表")
    parser.add_argument("--block_weights", type=str, 
                        default='[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]', 
                        help="块权重列表的JSON字符串，格式如 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]'")
    parser.add_argument("--output_dir", type=str, default="models/test-merged-block", help="输出目录")
    parser.add_argument("--fusion_method", type=str, default="ties", 
                        choices=["task_arithmetic", "ties", "dare_ties", "dare_linear", 
                                 "breadcrumbs", "breadcrumbs_ties", "della", "della_linear"],
                        help="融合方法")
    parser.add_argument("--copy_from_base", type=bool, default=True, help="是否从基础模型复制非权重文件")
    parser.add_argument("--num_blocks", type=int, default=8, help="要合并的块数")
    
    args = parser.parse_args()
    
    # 解析块权重
    try:
        block_weights = json.loads(args.block_weights)
        print(f"加载了 {len(block_weights)} 个块权重")
    except Exception as e:
        print(f"错误: 解析block_weights失败: {e}")
        return
    
    # 执行融合
    success = mi_block_fusion(
        base_model_path=args.base_model,
        task_model_paths=args.task_models,
        block_weights=block_weights,
        output_dir=args.output_dir,
        fusion_method=args.fusion_method,
        copy_from_base=args.copy_from_base,
        num_blocks=args.num_blocks
    )
    
    if success:
        print(f"\n融合完成! 可以在 {args.output_dir} 找到融合后的模型。")
    else:
        print(f"\n融合过程中发生错误。")


if __name__ == "__main__":
    main()
