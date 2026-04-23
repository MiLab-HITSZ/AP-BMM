import os
from typing import Dict, List, Any, Optional


class EvoMIConfig:
    """
    EvoMI配置管理类
    提供统一的配置管理，包括任务配置生成和模型设置
    """
    
    def __init__(self):
        # 默认配置
        self._base_model = ['models/Qwen3-4B-Instruct-2507','models/Qwen3-4B-thinking-2507','models/Qwen3-4B-thinking-2507']
        self._expert_model = ['models/Qwen3-4B-thinking-2507', 'models/Qwen3-4B-Instruct-2507', 'models/Qwen3-4B-Instruct-2507']
        self._base_model_bi = self._base_model[:2]
        self._expert_model_bi = self._expert_model[:2]
        self._checkpoint_dir = './checkpoints'
        self._default_max_tokens = 20000
        self._default_max_model_len = None
        
        # 确保缓存目录存在
        os.makedirs(self._checkpoint_dir, exist_ok=True)
    
    @property
    def base_model(self) -> List[str]:
        """获取基础模型路径列表"""
        return self._base_model
    
    @base_model.setter
    def base_model(self, value: List[str]) -> None:
        """设置基础模型路径列表"""
        self._base_model = value
        self._base_model_bi = value[:2]
    
    @property
    def expert_model(self) -> List[str]:
        """获取专家模型路径列表"""
        return self._expert_model
    
    @expert_model.setter
    def expert_model(self, value: List[str]) -> None:
        """设置专家模型路径列表"""
        self._expert_model = value
        self._expert_model_bi = value[:2]
    
    @property
    def checkpoint_dir(self) -> str:
        """获取检查点目录"""
        return self._checkpoint_dir
    
    @checkpoint_dir.setter
    def checkpoint_dir(self, value: str) -> None:
        """设置检查点目录"""
        self._checkpoint_dir = value
        # 确保目录存在
        os.makedirs(value, exist_ok=True)
    
    def _get_generation_config(self, model_path: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        if max_tokens is None:
            max_tokens = self._default_max_tokens
        if 'thinking' in model_path or 'merged' in model_path:
            return {
                'max_tokens': max_tokens,
                'temperature': 0.0,
                'top_p': 0.95,
                'top_k': 20
            }
        return {
            'max_tokens': max_tokens,
            'temperature': 0.0,
            'top_p': 0.8,
            'top_k': 20
        }

    def _create_task_config(
        self,
        model_path: str,
        datasets: List[str],
        dataset_args: Dict[str, Any],
        max_tokens: Optional[int] = None,
        limit: Optional[Dict[str, Optional[int]]] = None,
        repeats: Optional[Dict[str, int]] = None,
        seed: Optional[int] = None,
        stream: bool = False,
    ):
        from evalscope import TaskConfig

        task_config_kwargs = {
            'model': model_path,
            'api_url': '',
            'eval_type': 'server',
            'datasets': datasets,
            'dataset_args': dataset_args,
            'eval_batch_size': 64,
            'ignore_errors': True,
            'generation_config': self._get_generation_config(model_path, max_tokens),
            'timeout': 60000,
            'stream': stream,
            'repeats': repeats,
            'limit': limit,
        }
        if seed is not None:
            task_config_kwargs['seed'] = seed
        return TaskConfig(**task_config_kwargs)

    def create_aime_gpqa_task_config(
        self,
        model_path: str,
        max_tokens: Optional[int] = None,
        limit: Optional[Dict[str, Optional[int]]] = None,
        repeats: Optional[Dict[str, int]] = None,
        seed: Optional[int] = None,
        stream: bool = False,
    ):
        if limit is None:
            limit = {'aime25': None, 'gpqa_diamond': None}
        if repeats is None:
            repeats = {'aime25': 1, 'gpqa_diamond': 1}
        return self._create_task_config(
            model_path=model_path,
            datasets=['aime25', 'gpqa_diamond'],
            dataset_args={
                'aime25': {
                    'aggregation': 'mean_and_pass_at_k',
                    'metric_list': [{'acc': {'numeric': True}}, {'tokens_num': {'tokenizer_path': model_path}}, 'think_num']
                },
                'gpqa_diamond': {
                    'aggregation': 'mean_and_vote_at_k',
                    'metric_list': ['acc', {'tokens_num': {'tokenizer_path': model_path}}, 'think_num']
                }
            },
            max_tokens=max_tokens,
            limit=limit,
            repeats=repeats,
            seed=seed,
            stream=stream,
        )

    def create_gsm8k_gpqa_task_config(
        self,
        model_path: str,
        max_tokens: Optional[int] = None,
        limit: Optional[Dict[str, Optional[int]]] = None,
        repeats: Optional[Dict[str, int]] = None,
        seed: int = 42,
        stream: bool = False,
    ):
        if limit is None:
            limit = {'gsm8k': 50, 'gpqa_diamond': 50}
        if repeats is None:
            repeats = {'gsm8k': 1, 'gpqa_diamond': 1}
        return self._create_task_config(
            model_path=model_path,
            datasets=['gsm8k', 'gpqa_diamond'],
            dataset_args={
                'gsm8k': {
                    'aggregation': 'mean',
                    'few_shot_num': 0,
                    'shuffle': True,
                    'metric_list': [{'acc': {'numeric': True}}, {'tokens_num': {'tokenizer_path': model_path}}, 'think_num']
                },
                'gpqa_diamond': {
                    'aggregation': 'mean',
                    'shuffle': True,
                    'metric_list': ['acc', {'tokens_num': {'tokenizer_path': model_path}}, 'think_num']
                }
            },
            max_tokens=max_tokens,
            limit=limit,
            repeats=repeats,
            seed=seed,
            stream=stream,
        )

    def create_math500_level5_gpqa_task_config(
        self,
        model_path: str,
        max_tokens: Optional[int] = None,
        limit: Optional[Dict[str, Optional[int]]] = None,
        repeats: Optional[Dict[str, int]] = None,
        seed: Optional[int] = None,
        stream: bool = False,
    ):
        if limit is None:
            limit = {'math_500': None, 'gpqa_diamond': None}
        if repeats is None:
            repeats = {'math_500': 1, 'gpqa_diamond': 1}
        return self._create_task_config(
            model_path=model_path,
            datasets=['math_500', 'gpqa_diamond'],
            dataset_args={
                'math_500': {
                    'subset_list': ['Level 5'],
                    'few_shot_num': 0,
                    'metric_list': [{'acc': {'numeric': True}}, {'tokens_num': {'tokenizer_path': model_path}}, 'think_num']
                },
                'gpqa_diamond': {
                    'aggregation': 'mean',
                    'shuffle': True,
                    'metric_list': ['acc', {'tokens_num': {'tokenizer_path': model_path}}, 'think_num']
                }
            },
            max_tokens=max_tokens,
            limit=limit,
            repeats=repeats,
            seed=seed,
            stream=stream,
        )

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        获取特定模型的配置
        
        参数:
            model_name: 模型名称
        
        返回:
            模型配置字典
        """
        # 这里可以根据不同模型返回特定的配置
        return {
            "name": model_name,
            "checkpoint_dir": os.path.join(self._checkpoint_dir, model_name),
            "base_model": model_name in self._base_model
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式
        
        返回:
            配置字典
        """
        return {
            "base_model": self._base_model,
            "expert_model": self._expert_model,
            "checkpoint_dir": self._checkpoint_dir,
            "default_max_tokens": self._default_max_tokens,
            "default_max_model_len": self._default_max_model_len
        }
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        从字典加载配置
        
        参数:
            config_dict: 包含配置的字典
        """
        if "base_model" in config_dict:
            self._base_model = config_dict["base_model"]
        if "expert_model" in config_dict:
            self._expert_model = config_dict["expert_model"]
        if "checkpoint_dir" in config_dict:
            self._checkpoint_dir = config_dict["checkpoint_dir"]
        if "default_max_tokens" in config_dict:
            self._default_max_tokens = config_dict["default_max_tokens"]
        if "default_max_model_len" in config_dict:
            self._default_max_model_len = config_dict["default_max_model_len"]


# 创建全局配置实例
config_manager = EvoMIConfig()
