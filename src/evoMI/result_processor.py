#!/usr/bin/env python3
"""
结果处理工具类 - 用于解析和保存任务执行结果
"""
import json
import datetime
import os
import sys

# 确保可以正确导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ResultProcessor:
    """
    结果处理器，用于解析和保存任务执行结果
    支持将复杂的报告对象转换为可序列化格式，并保存为JSON和TXT文件
    """
    
    def __init__(self, base_output_dir=None):
        """
        初始化结果处理器
        
        参数:
            base_output_dir: 基础输出目录，默认为evoMI项目目录下的outputs子目录
        """
        if base_output_dir is None:
            # 保存到evoMI项目目录下的outputs目录
            current_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # 从src/evoMI/result_processor.py获取evoMI目录
            evoMI_dir = os.path.dirname(current_script_dir)
            self.base_output_dir = os.path.join(evoMI_dir, 'output/results_logs')
        else:
            self.base_output_dir = base_output_dir
        
    def process_and_save(self, results):
        """
        处理任务执行结果并保存为JSON和TXT文件
        
        参数:
            results: 任务执行结果字典
            
        返回:
            list: 结构化的结果数组，每个元素是一个任务的结果字典
                  格式: [{问题1: {指标1: 值1, 指标2: 值2, ...}}, {问题2: {指标1: 值1, ...}}, ...]
        """
        # 创建输出目录
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(self.base_output_dir, timestamp)
        os.makedirs(result_dir, exist_ok=True)
        
        # 解析结果为可序列化格式
        parsed_results = self._parse_results(results)
        
        # 保存为JSON文件
        json_path = os.path.join(result_dir, 'results.json')
        self._save_as_json(parsed_results, json_path)
        
        # 生成并保存TXT文件
        txt_path = os.path.join(result_dir, 'results_summary.txt')
        self._save_as_txt(parsed_results, txt_path, timestamp)
        
        print(f"\n结果已保存:")
        print(f"  JSON文件: {json_path}")
        print(f"  文本摘要: {txt_path}")
        
        # 生成结构化结果数组
        structured_results = self._generate_structured_results(results)
        
        return structured_results
    
    def _parse_results(self, results):
        """
        解析结果为可序列化格式
        
        参数:
            results: 原始任务执行结果
            
        返回:
            dict: 解析后的结果字典
        """
        parsed_results = {}
        
        for task_id, task_result in results.items():
            if isinstance(task_result, dict) and 'error' in task_result:
                parsed_results[task_id] = {'error': task_result['error']}
                continue
            
            task_parsed = {}
            
            # 遍历数据集报告
            if hasattr(task_result, '__dict__'):
                # 处理单个报告对象的情况
                dataset_key = 'single_report'
                report = task_result
                task_parsed[dataset_key] = self._parse_report(report)
            else:
                # 处理多个数据集报告的情况
                try:
                    for dataset_key, report in task_result.items():
                        if hasattr(report, '__dict__'):
                            task_parsed[dataset_key] = self._parse_report(report)
                        else:
                            task_parsed[dataset_key] = str(report)
                except (AttributeError, TypeError):
                    # 如果task_result不是可迭代的字典类型，直接转换为字符串
                    task_parsed['unknown_result'] = str(task_result)
            
            parsed_results[task_id] = task_parsed
        
        return parsed_results
    
    def _parse_report(self, report):
        """
        解析Report对象为字典格式
        
        参数:
            report: Report对象
            
        返回:
            dict: 解析后的报告字典
        """
        parsed = {
            'name': getattr(report, 'name', 'N/A'),
            'dataset_name': getattr(report, 'dataset_name', 'N/A'),
            'dataset_pretty_name': getattr(report, 'dataset_pretty_name', 'N/A'),
            'dataset_description': getattr(report, 'dataset_description', 'N/A'),
            'model_name': getattr(report, 'model_name', 'N/A'),
            'score': getattr(report, 'score', 'N/A'),
            'analysis': getattr(report, 'analysis', 'N/A'),
            'metrics': []
        }
        
        # 解析metrics
        metrics = getattr(report, 'metrics', [])
        for metric in metrics:
            if hasattr(metric, '__dict__'):
                metric_dict = {
                    'name': getattr(metric, 'name', 'Unknown'),
                    'num': getattr(metric, 'num', 'N/A'),
                    'score': getattr(metric, 'score', 'N/A'),
                    'macro_score': getattr(metric, 'macro_score', 'N/A')
                }
                parsed['metrics'].append(metric_dict)
        
        return parsed
    
    def _generate_structured_results(self, results):
        """
        生成结构化的结果数组
        
        参数:
            results: 原始任务执行结果
            
        返回:
            list: 结构化的结果数组
                  格式: [{问题1: {指标1: 值1, 指标2: 值2, ...}}, {问题2: {指标1: 值1, ...}}, ...]
        """
        structured_results = []
        
        # 按任务ID排序，确保结果数组的顺序一致
        sorted_task_ids = sorted(results.keys())
        
        for task_id in sorted_task_ids:
            task_result = results[task_id]
            task_structured = {}
            
            # 跳过错误任务
            if isinstance(task_result, dict) and 'error' in task_result:
                structured_results.append({task_id: {'status': 'error', 'error': task_result['error']}})
                continue
            
            # 处理报告对象
            if hasattr(task_result, '__dict__'):
                # 单个报告对象
                report = task_result
                # 使用任务ID作为问题标识
                metrics_dict = {}
                
                # 提取所有指标
                metrics = getattr(report, 'metrics', [])
                for metric in metrics:
                    if hasattr(metric, '__dict__'):
                        metric_name = getattr(metric, 'name', 'Unknown')
                        metric_score = getattr(metric, 'score', 'N/A')
                        metrics_dict[metric_name] = metric_score
                
                # 如果没有指标，添加基本信息
                if not metrics_dict:
                    metrics_dict['score'] = getattr(report, 'score', 'N/A')
                
                task_structured[task_id] = metrics_dict
            else:
                # 多个数据集报告的情况
                try:
                    # 假设第一个数据集作为主要问题标识
                    for dataset_key, report in task_result.items():
                        if hasattr(report, '__dict__'):
                            metrics_dict = {}
                            
                            # 提取所有指标
                            metrics = getattr(report, 'metrics', [])
                            for metric in metrics:
                                if hasattr(metric, '__dict__'):
                                    metric_name = getattr(metric, 'name', 'Unknown')
                                    metric_score = getattr(metric, 'score', 'N/A')
                                    metrics_dict[metric_name] = metric_score
                            
                            # 如果没有指标，添加基本信息
                            if not metrics_dict:
                                metrics_dict['score'] = getattr(report, 'score', 'N/A')
                            
                            task_structured[dataset_key] = metrics_dict
                        else:
                            # 如果不是报告对象，直接保存
                            task_structured[dataset_key] = {'value': str(report)}
                except (AttributeError, TypeError):
                    # 如果无法迭代，将整个结果作为单个问题
                    task_structured[task_id] = {'value': str(task_result)}
            
            structured_results.append(task_structured)
        
        return structured_results
    
    def _save_as_json(self, parsed_results, json_path):
        """
        将解析后的结果保存为JSON文件
        
        参数:
            parsed_results: 解析后的结果字典
            json_path: JSON文件路径
        """
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_results, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_as_txt(self, parsed_results, txt_path, timestamp):
        """
        将解析后的结果保存为TXT摘要文件
        
        参数:
            parsed_results: 解析后的结果字典
            txt_path: TXT文件路径
            timestamp: 时间戳字符串
        """
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"任务执行结果汇总 - {timestamp}\n")
            f.write("="*80 + "\n\n")
            
            for task_id, task_data in parsed_results.items():
                f.write(f"任务ID: {task_id}\n")
                f.write("-" * 60 + "\n")
                
                if 'error' in task_data:
                    f.write(f"状态: 失败\n")
                    f.write(f"错误信息: {task_data['error']}\n")
                else:
                    f.write(f"状态: 成功\n")
                    
                    # 遍历每个数据集的报告
                    for dataset_key, report_data in task_data.items():
                        if report_data and isinstance(report_data, dict):
                            f.write(f"\n数据集: {dataset_key}\n")
                            f.write(f"模型: {report_data.get('model_name', 'N/A')}\n")
                            f.write(f"总得分: {report_data.get('score', 'N/A')}\n")
                            
                            # 写入指标信息
                            if 'metrics' in report_data:
                                f.write("指标详情:\n")
                                for metric in report_data['metrics']:
                                    f.write(f"  - {metric.get('name', 'Unknown')}: {metric.get('score', 'N/A')} (样本数: {metric.get('num', 'N/A')})\n")
                        elif report_data:
                            f.write(f"\n数据集: {dataset_key}\n")
                            f.write(f"结果: {str(report_data)}\n")
                
                f.write("\n" + "="*80 + "\n\n")


if __name__ == "__main__":
    # 示例用法
    processor = ResultProcessor()
    # 这里可以添加示例代码来演示如何使用这个工具类
    print("ResultProcessor类已加载，可用于处理任务执行结果")