#!/usr/bin/env python3
"""
绘图工具类 - 集中管理所有绘图相关功能
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
from typing import Tuple, List, Optional, Dict, Any


class OptimizationReporter:
    """绘图工具类，提供各种优化结果可视化功能"""
    
    @staticmethod
    def render_optimization_results(result_dict: dict, output_dir: str) -> None:
        """
        可视化优化结果
        
        参数:
            result_dict: 优化结果字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取帕累托前沿和对应的决策变量
        pareto_x = result_dict.get('pareto_x', np.array([]))
        pareto_y = result_dict.get('pareto_y', np.array([]))
        
        if len(pareto_y) == 0:
            print("没有找到帕累托前沿")
            return
        
        if pareto_y.ndim == 2 and pareto_y.shape[1] >= 3:
            OptimizationReporter._plot_pareto_3d(pareto_y, result_dict.get('all_y', np.array([])), output_dir)
        
        # 绘制2D帕累托前沿对
        OptimizationReporter._plot_pareto_2d_pairs(pareto_y, result_dict.get('all_y', np.array([])), output_dir)
        
        # 绘制超体积随迭代变化的图
        hypervolumes = result_dict.get('hypervolume_history', [])
        if hypervolumes:
            OptimizationReporter.plot_hypervolume_history(hypervolumes, "多目标优化", os.path.join(output_dir, 'hypervolume_history.png'))
        
        print(f"可视化结果已保存到 {output_dir}")

    visualize_optimization_results = render_optimization_results
    
    @staticmethod
    def _plot_pareto_3d(pareto_y: np.ndarray, all_y: np.ndarray, output_dir: str) -> None:
        """
        绘制3D帕累托前沿
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制帕累托前沿点
        scatter = ax.scatter(
            pareto_y[:, 0], 
            pareto_y[:, 1], 
            pareto_y[:, 2], 
            c='r', 
            marker='o', 
            s=50, 
            label='帕累托前沿'
        )
        
        # 为每个帕累托点添加编号标注
        for i in range(len(pareto_y)):
            ax.text(
                pareto_y[i, 0], 
                pareto_y[i, 1], 
                pareto_y[i, 2], 
                str(i), 
                fontsize=10, 
                ha='center', 
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
            )
        
        # 绘制所有评估点
        if len(all_y) > 0 and all_y.ndim == 2:
            # 确保all_y是2D数组且有足够的列
            if all_y.shape[1] >= 3:
                ax.scatter(
                    all_y[:, 0], 
                    all_y[:, 1], 
                    all_y[:, 2], 
                    c='b', 
                    marker='x', 
                    s=20, 
                    alpha=0.5, 
                    label='评估点'
                )
        
        # 设置坐标轴标签
        ax.set_xlabel('F1: aime25 acc')
        ax.set_ylabel('F2: apqa acc')
        ax.set_zlabel('F3: tokens + ifeval')
        
        ax.set_title('多目标优化的帕累托前沿')
        ax.legend()
        
        # 保存3D图
        plt.savefig(os.path.join(output_dir, 'pareto_front_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _plot_pareto_2d_pairs(pareto_y: np.ndarray, all_y: np.ndarray, output_dir: str) -> None:
        """
        绘制每两个目标之间的2D帕累托前沿
        """
        if pareto_y.ndim != 2 or pareto_y.shape[1] < 2:
            return

        pairs = list(itertools.combinations(range(pareto_y.shape[1]), 2))
        labels = [[f"F{idx1 + 1}", f"F{idx2 + 1}"] for idx1, idx2 in pairs]
        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
        if len(pairs) == 1:
            axes = [axes]
        
        for pair_idx, (idx1, idx2) in enumerate(pairs):
            ax = axes[pair_idx]
            
            # 绘制所有评估点
            if len(all_y) > 0 and all_y.ndim == 2:
                if all_y.shape[1] >= max(idx1, idx2) + 1:
                    ax.scatter(all_y[:, idx1], all_y[:, idx2], c='b', marker='x', s=20, alpha=0.3, label='评估点')
            
            # 绘制帕累托前沿点
            ax.scatter(pareto_y[:, idx1], pareto_y[:, idx2], c='r', marker='o', s=50, label='帕累托前沿')
            
            # 为每个帕累托点添加编号标注
            for point_idx in range(len(pareto_y)):
                ax.text(
                    pareto_y[point_idx, idx1], 
                    pareto_y[point_idx, idx2], 
                    str(point_idx), 
                    fontsize=10, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                )
            
            # 连接帕累托点（按第一个指标排序）
            sorted_indices = np.argsort(pareto_y[:, idx1])
            ax.plot(
                pareto_y[sorted_indices, idx1], 
                pareto_y[sorted_indices, idx2], 
                'r-', 
                alpha=0.7
            )
            
            ax.set_xlabel(labels[pair_idx][0])
            ax.set_ylabel(labels[pair_idx][1])
            ax.set_title(f'帕累托前沿: {labels[pair_idx][0]} vs {labels[pair_idx][1]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pareto_front_2d_pairs.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_pareto_results(train_obj_true: torch.Tensor, 
                          problem_name: str = "Multi-objective Problem", 
                          true_pareto_front: Optional[Tuple[np.ndarray, ...]] = None,
                          ref_point: Optional[torch.Tensor] = None,
                          save_path: Optional[str] = None) -> None:
        """
        绘制帕累托前沿结果
        
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
        from .qnehvi_optimizer import get_pareto_optimal_points
        
        # 确保张量在CPU上，以便matplotlib绘图
        if isinstance(train_obj_true, torch.Tensor):
            train_obj_true = train_obj_true.cpu()
        
        # 获取帕累托最优解
        pareto_points = get_pareto_optimal_points(train_obj_true)
        
        # 确保pareto_points也在CPU上
        if isinstance(pareto_points, torch.Tensor):
            pareto_points = pareto_points.cpu()
        
        # 处理三目标问题
        if true_pareto_front is not None and len(true_pareto_front) == 3:
            OptimizationReporter._plot_pareto_3d_comparison(pareto_points, true_pareto_front, problem_name, save_path)
            return
        
        # 处理双目标问题
        plt.figure(figsize=(8, 6))
        
        # 绘制真实帕累托前沿（如果提供）
        if true_pareto_front is not None and len(true_pareto_front) == 2:
            f1, f2 = true_pareto_front
            plt.scatter(f1, f2, c='red', s=20, label='True Pareto Front', alpha=0.6)
        
        # 绘制获取的帕累托最优解
        plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='blue', s=40, 
                   label='Obtained Pareto Solutions', edgecolors='black', linewidths=0.5)
        
        # 为每个帕累托解添加编号标注
        for i in range(len(pareto_points)):
            plt.text(
                pareto_points[i, 0], 
                pareto_points[i, 1], 
                str(i), 
                fontsize=10, 
                ha='center', 
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.3)
            )
        
        # 如果有参考点，也绘制出来
        if ref_point is not None and len(ref_point) == 2:
            # 确保参考点在CPU上
            if isinstance(ref_point, torch.Tensor):
                ref_point = ref_point.cpu()
            plt.scatter(ref_point[0], ref_point[1], c='green', s=100, 
                       label='Reference Point', marker='*', edgecolors='black')
        
        # 设置图形属性
        plt.xlabel('Objective Function 1', fontsize=12)
        plt.ylabel('Objective Function 2', fontsize=12)
        plt.title(f'Pareto Solutions for {problem_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results figure saved as '{save_path}'")
        
        plt.close()
    
    @staticmethod
    def _plot_pareto_3d_comparison(pareto_points: np.ndarray, 
                                 true_pareto_front: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                 problem_name: str, 
                                 save_path: Optional[str]) -> None:
        """
        绘制3D帕累托前沿比较
        """
        ax = plt.figure(figsize=(10, 8)).add_subplot(111, projection='3d')
        f1, f2, f3 = true_pareto_front
        
        # 绘制真实帕累托前沿
        ax.scatter(f1, f2, f3, c='red', s=20, label='True Pareto Front', alpha=0.6)
        
        # 绘制获取的帕累托解
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], 
                  c='blue', s=40, label='Obtained Pareto Solutions', 
                  edgecolors='black', linewidths=0.5)
        
        # 为每个帕累托解添加编号标注
        for i in range(len(pareto_points)):
            ax.text(
                pareto_points[i, 0], 
                pareto_points[i, 1], 
                pareto_points[i, 2], 
                str(i), 
                fontsize=10, 
                ha='center', 
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.3)
            )
        
        # 设置坐标轴标签
        ax.set_xlabel('Objective Function 1', fontsize=12)
        ax.set_ylabel('Objective Function 2', fontsize=12)
        ax.set_zlabel('Objective Function 3', fontsize=12)
        ax.set_title(f'Pareto Solutions for {problem_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results figure saved as '{save_path}'")
        
        plt.close()
    
    @staticmethod
    def plot_hypervolume_history(hvs: List[float], 
                               problem_name: str = "Multi-objective Problem",
                               save_path: Optional[str] = None) -> None:
        """
        绘制超体积随迭代的变化
        
        参数:
        ----------
        hvs : list
            每轮迭代的超体积值
        problem_name : str, optional
            问题名称
        save_path : str, optional
            保存图像的路径
        """
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(hvs)), hvs, 'b-o', linewidth=2, markersize=5)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Hypervolume', fontsize=12)
        plt.title(f'Hypervolume History for {problem_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hypervolume history saved as '{save_path}'")
        
        plt.close()
    
    @staticmethod
    def plot_zdt3_results(train_obj_true: torch.Tensor, 
                         hvs: List[float],
                         compute_true_pareto_func,
                         get_pareto_optimal_func) -> None:
        """
        绘制ZDT3问题的结果
        
        参数:
        ----------
        train_obj_true : torch.Tensor
            所有评估点的真实目标函数值
        hvs : list
            每轮迭代的超体积值
        compute_true_pareto_func : callable
            计算真实帕累托前沿的函数
        get_pareto_optimal_func : callable
            获取帕累托最优解的函数
        """
        # 计算真实帕累托前沿
        true_f1, true_f2 = compute_true_pareto_func()
        
        # 获取帕累托最优解
        pareto_points = get_pareto_optimal_func(train_obj_true)
        
        # 创建图形
        plt.figure(figsize=(8, 6))
        
        # 绘制真实帕累托前沿
        plt.scatter(true_f1, true_f2, c='red', s=20, label='True Pareto Front', alpha=0.6)
        
        # 绘制获取的帕累托最优解
        plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='blue', s=40, 
                   label='Obtained Pareto Solutions', edgecolors='black', linewidths=0.5)
        
        # 设置图形属性
        plt.xlabel('Objective Function 1', fontsize=12)
        plt.ylabel('Objective Function 2', fontsize=12)
        plt.title('Ideal vs Obtained Pareto Solutions for ZDT3 Problem', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # 调整轴以确保所有点都可见
        plt.tight_layout()
        
        # 保存图形
        plt.savefig('qehvi_zdt3_results.png', dpi=300, bbox_inches='tight')
        print("Results figure saved as 'qehvi_zdt3_results.png'")
        plt.close()
        
        # 如果提供了超体积历史，绘制超体积曲线
        if hvs:
            OptimizationReporter.plot_hypervolume_history(hvs, "ZDT3 Problem", "qehvi_zdt3_hypervolume.png")


    @staticmethod
    def plot_3d_objectives(objectives_list: List[List], output_dir: str) -> None:
        """
        可视化3D目标空间中的所有解，并为解决方案点添加编号标注
        
        参数:
            objectives_list: 目标函数值列表，格式为[[f1, f2, f3, type], ...]
            output_dir: 输出目录
        """
        if not objectives_list:
            print("没有目标值数据，跳过3D可视化")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 分离不同类型的目标点
        solution_obj = []
        original_obj = []
        solution_indices = []
        
        for idx, obj in enumerate(objectives_list):
            if len(obj) > 3 and obj[3] == 'solution':
                solution_obj.append(obj[:3])
                solution_indices.append(idx)
            else:
                original_obj.append(obj[:3])
        
        solution_obj = np.array(solution_obj) if solution_obj else np.array([])
        original_obj = np.array(original_obj) if original_obj else np.array([])
        
        # 绘制解决方案点
        if len(solution_obj) > 0:
            f1_sol = solution_obj[:, 0]
            f2_sol = solution_obj[:, 1]
            f3_sol = solution_obj[:, 2]
            scatter_sol = ax.scatter(f1_sol, f2_sol, f3_sol, c=f1_sol, cmap='viridis', 
                                    s=50, alpha=0.7, label='Merged Models')
            
            # 为每个解决方案点添加编号标注
            for i, idx in enumerate(solution_indices):
                ax.text(
                    solution_obj[i, 0]+0.01, 
                    solution_obj[i, 1]+0.01, 
                    solution_obj[i, 2]+0.01, 
                    str(i), 
                    fontsize=10, 
                    ha='center', 
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.1)
                )
            
            # 添加颜色条
            cbar = plt.colorbar(scatter_sol, ax=ax)
            cbar.set_label('F1 Score')
        
        # 绘制原始模型点
        if len(original_obj) > 0:
            f1_orig = original_obj[:, 0]
            f2_orig = original_obj[:, 1]
            f3_orig = original_obj[:, 2]
            scatter_orig = ax.scatter(f1_orig, f2_orig, f3_orig, c='red', 
                                     s=80, alpha=0.9, marker='^', label='Original Models')
        
        # 设置标签
        ax.set_xlabel('F1 (AIME25 & GPQA Avg Accuracy)', fontsize=12, labelpad=10)
        ax.set_ylabel('F2 (-tokens_num/20000)', fontsize=12, labelpad=10)
        ax.set_zlabel('F3 (IFEval Avg Accuracy)', fontsize=12, labelpad=10)
        
        # 设置标题
        ax.set_title('Distribution of Solutions in 3D Objective Space', fontsize=14, pad=20)
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 设置视角
        ax.view_init(30, 45)
        
        # 保存图像
        plt.tight_layout()
        output_path = os.path.join(output_dir, "3d_objectives_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"3D目标空间图已保存到: {output_path}")
    
    @staticmethod
    def plot_dataset_metrics(output_dir: str, metrics_list: List[Dict]) -> None:
        """
        可视化每个数据集的token数量与准确度的关系
        
        参数:
            output_dir: 输出目录
            metrics_list: 所有解决方案和原始模型的指标列表
        """
        if not metrics_list:
            print("没有指标数据，跳过数据集可视化")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        datasets = ['aime25', 'gpqa_diamond', 'ifeval']
        dataset_names = {
            'aime25': 'AIME25',
            'gpqa_diamond': 'GPQA Diamond',
            'ifeval': 'IFEval'
        }
        
        for dataset in datasets:
            # 提取当前数据集的token数量和准确度，区分解决方案和原始模型
            solution_tokens = []
            solution_acc = []
            original_tokens = []
            original_acc = []
            original_names = []
            
            for metrics in metrics_list:
                if dataset in metrics:
                    tokens = metrics[dataset].get('mean_tokens_num', 5000)
                    # 根据数据集获取准确度
                    if 'mean_acc' in metrics[dataset]:
                        acc = metrics[dataset]['mean_acc']
                    elif dataset == 'ifeval' and 'mean_prompt_level_strict' in metrics[dataset]:
                        acc = (metrics[dataset]['mean_prompt_level_strict'] + 
                              metrics[dataset]['mean_inst_level_strict'] + 
                              metrics[dataset]['mean_prompt_level_loose'] + 
                              metrics[dataset]['mean_inst_level_loose']) / 4
                    else:
                        continue
                    # 按类型分离
                    if metrics.get('type') == 'solution':
                        solution_tokens.append(tokens)
                        solution_acc.append(acc)
                    else:
                        original_tokens.append(tokens)
                        original_acc.append(acc)
                        original_names.append(metrics.get('name', f'Original_{len(original_names)}'))
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 绘制解决方案点
            if solution_tokens and solution_acc:
                scatter_sol = plt.scatter(solution_tokens, solution_acc, c='black', 
                                         cmap='plasma', s=50, alpha=0.7, label='evoMI(ours)')
                
                # 为每个解决方案点添加编号标注
                for i, (tokens, acc) in enumerate(zip(solution_tokens, solution_acc)):
                    # 计算y轴上移的偏移量（根据图表范围动态调整）
                    if solution_acc:
                        y_range = max(solution_acc) - min(solution_acc)
                        y_offset = y_range * 0.02  # 向上偏移2%的y轴范围
                    else:
                        y_offset = 0.02
                        
                    plt.text(
                        tokens, 
                        acc + y_offset,  # 向上移动一点
                        str(i), 
                        fontsize=10, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                    )
            
            # 绘制原始模型点
            if original_tokens and original_acc:
                scatter_orig = plt.scatter(original_tokens, original_acc, c='red', 
                                          s=70, alpha=0.8, marker='^', label='Original Models')
                
                # 为原始模型点添加名称标注
                for i, (tokens, acc, name) in enumerate(zip(original_tokens, original_acc, original_names)):
                    plt.text(
                        tokens, 
                        acc, 
                        name, 
                        fontsize=9, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                    )
            
            # 设置图表属性
            plt.xlabel('Mean Tokens Used', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Token Usage vs Accuracy for {dataset_names[dataset]} Dataset', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            # 保存图表
            output_path = os.path.join(output_dir, f"{dataset}_tokens_vs_accuracy.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{dataset_names[dataset]}数据集可视化已保存到: {output_path}")
    
    @staticmethod
    def plot_token_avg_vs_model_ability(output_dir: str, metrics_list: List[Dict]) -> None:
        """
        可视化所有指标的token平均值与模型能力（f1+f3）的关系
        
        参数:
            output_dir: 输出目录
            metrics_list: 所有解决方案和原始模型的指标列表
        """
        if not metrics_list:
            print("没有指标数据，跳过token平均值vs模型能力可视化")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        solution_avg_tokens = []
        solution_model_ability = []
        original_avg_tokens = []
        original_model_ability = []
        original_names = []
        
        for metrics in metrics_list:
            # 计算所有数据集的token平均值
            tokens_list = []
            if 'aime25' in metrics:
                tokens_list.append(metrics['aime25'].get('mean_tokens_num', 0))
            if 'gpqa_diamond' in metrics:
                tokens_list.append(metrics['gpqa_diamond'].get('mean_tokens_num', 0))
            if 'ifeval' in metrics:
                tokens_list.append(metrics['ifeval'].get('mean_tokens_num', 0))
            
            if not tokens_list:
                continue
            
            avg_tokens = np.mean(tokens_list)
            
            # 计算模型能力：f1 + f3
            # 从metrics中获取f1, f2, f3（这些是在save_solution_results中添加的）
            f1 = metrics.get('f1', 0)
            f3 = metrics.get('f3', 0)
            model_ability = f1 + f3
            
            # 区分解决方案和原始模型
            if metrics.get('type') == 'solution':
                solution_avg_tokens.append(avg_tokens)
                solution_model_ability.append(model_ability)
            else:
                original_avg_tokens.append(avg_tokens)
                original_model_ability.append(model_ability)
                original_names.append(metrics.get('name', f'Original_{len(original_names)}'))
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制解决方案点
        if solution_avg_tokens and solution_model_ability:
            scatter_sol = plt.scatter(solution_avg_tokens, solution_model_ability, c='black', 
                                     cmap='plasma', s=50, alpha=0.7, label='evoMI(ours)')
            
            # 为每个解决方案点添加编号标注
            for i, (tokens, ability) in enumerate(zip(solution_avg_tokens, solution_model_ability)):
                y_range = max(solution_model_ability) - min(solution_model_ability)
                y_offset = y_range * 0.02  # 向上偏移2%的y轴范围
                
                plt.text(
                    tokens, 
                    ability + y_offset,  # 向上移动一点
                    str(i), 
                    fontsize=10, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                )
        
        # 绘制原始模型点
        if original_avg_tokens and original_model_ability:
            scatter_orig = plt.scatter(original_avg_tokens, original_model_ability, c='red', 
                                      s=70, alpha=0.8, marker='^', label='Original Models')
            
            # 为原始模型点添加名称标注
            for tokens, ability, name in zip(original_avg_tokens, original_model_ability, original_names):
                plt.text(
                    tokens, 
                    ability, 
                    name, 
                    fontsize=9, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                )
        
        # 设置图表属性
        plt.xlabel('Average Tokens Used Across All Datasets', fontsize=12)
        plt.ylabel('Model Ability (F1 + F3)', fontsize=12)
        plt.title('Average Tokens vs Model Ability (F1 + F3)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, "token_avg_vs_model_ability.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"token平均值vs模型能力可视化已保存到: {output_path}")
    
    @staticmethod
    def plot_f1_vs_f3(output_dir: str, metrics_list: List[Dict]) -> None:
        """
        可视化模型能力f1与f3的关系
        
        参数:
            output_dir: 输出目录
            metrics_list: 所有解决方案和原始模型的指标列表
        """
        if not metrics_list:
            print("没有指标数据，跳过f1 vs f3可视化")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        solution_f1 = []
        solution_f3 = []
        original_f1 = []
        original_f3 = []
        original_names = []
        
        for metrics in metrics_list:
            # 从metrics中获取f1和f3
            f1 = metrics.get('f1', 0)
            f3 = metrics.get('f3', 0)
            
            # 区分解决方案和原始模型
            if metrics.get('type') == 'solution':
                solution_f1.append(f1)
                solution_f3.append(f3)
            else:
                original_f1.append(f1)
                original_f3.append(f3)
                original_names.append(metrics.get('name', f'Original_{len(original_names)}'))
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制解决方案点
        if solution_f1 and solution_f3:
            scatter_sol = plt.scatter(solution_f1, solution_f3, c='black', 
                                     cmap='plasma', s=50, alpha=0.7, label='evoMI(ours)')
            
            # 为每个解决方案点添加编号标注
            for i, (f1_val, f3_val) in enumerate(zip(solution_f1, solution_f3)):
                y_range = max(solution_f3) - min(solution_f3)
                y_offset = y_range * 0.02  # 向上偏移2%的y轴范围
                
                plt.text(
                    f1_val, 
                    f3_val + y_offset,  # 向上移动一点
                    str(i), 
                    fontsize=10, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                )
        
        # 绘制原始模型点
        if original_f1 and original_f3:
            scatter_orig = plt.scatter(original_f1, original_f3, c='red', 
                                      s=70, alpha=0.8, marker='^', label='Original Models')
            
            # 为原始模型点添加名称标注
            for f1_val, f3_val, name in zip(original_f1, original_f3, original_names):
                plt.text(
                    f1_val, 
                    f3_val, 
                    name, 
                    fontsize=9, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                )
        
        # 设置图表属性
        plt.xlabel('Model Ability F1', fontsize=12)
        plt.ylabel('Model Ability F3', fontsize=12)
        plt.title('Model Ability F1 vs F3', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, "f1_vs_f3.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"f1 vs f3可视化已保存到: {output_path}")

# 创建可视化工具实例供外部使用
reporter = OptimizationReporter()
