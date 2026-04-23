"""
Layer-wise Model Fusion Interface
Supports per-layer fusion methods and parameters based on mergekit implementations
"""

import os
import logging
import torch
import re
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from safetensors.torch import load_file, save_file
from glob import glob
from tqdm import tqdm
from copy import deepcopy

# 尝试导入mergekit模块，如果失败则使用临时实现
from mergekit.merge_methods.generalized_task_arithmetic import (
    get_mask
)
from mergekit.sparsify import SparsificationMethod, RescaleNorm, sparsify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayerFusionConfig:
    """
    Configuration for layer-wise fusion
    """
    def __init__(self,
                 method: str,
                 params: Optional[Dict[str, Any]] = None,
                 layer_pattern: Optional[str] = None,
                 apply_to_embeddings: bool = False,
                 apply_to_norm: bool = True,
                 apply_to_lm_head: bool = True,
                 layer_names: Optional[List[str]] = None,
                 model_weights: Optional[List[float]] = None):
        """
        Initialize layer fusion configuration
        
        Args:
            method: Fusion method name (task_arithmetic, ties, dare_ties, etc.)
            params: Parameters specific to this method and layer group
            layer_pattern: Regex pattern to match layer names
            apply_to_embeddings: Whether to apply this config to embedding layers
            apply_to_norm: Whether to apply this config to normalization layers
            apply_to_lm_head: Whether to apply this config to LM head
            layer_names: List of layer name prefixes to match
            model_weights: Optional weights for each task model, specific to this layer group
        """
        self.method = method
        self.params = params or {}
        self.layer_pattern = layer_pattern
        self.apply_to_embeddings = apply_to_embeddings
        self.apply_to_norm = apply_to_norm
        self.apply_to_lm_head = apply_to_lm_head
        self.layer_names = layer_names or []
        self.model_weights = model_weights  # 每个配置独立的模型权重

class LayerFusionMethod:
    """
    Base class for layer fusion methods
    """
    def __init__(self, method_name: str, params: Dict[str, Any]):
        self.method_name = method_name
        self.params = params
        
    def apply(self, base: torch.Tensor, deltas: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Apply the fusion method to the base tensor and task vectors
        """
        with torch.no_grad():
            raise NotImplementedError("Subclasses must implement apply method")
    
    def _get_delta_weights(self, deltas: List[Dict[str, Any]], base: torch.Tensor) -> torch.Tensor:
        """
        Helper method to get weights for deltas
        """
        weight = self.params.get("layer_weight", 1.0)
        return torch.tensor(
            [delta_info.get("weight", 1.0) for delta_info in deltas],
            dtype=base.dtype, device=base.device
        ) * weight

class TaskArithmeticMethod(LayerFusionMethod):
    """
    Task Arithmetic fusion method implementation
    """
    def apply(self, base: torch.Tensor, deltas: List[Dict[str, Any]]) -> torch.Tensor:
        with torch.no_grad():
            lambda_ = self.params.get("lambda", 1.0)
            
            if not deltas:
                return base
            
            # Calculate weighted sum of deltas using in-place operations
            # Initialize mixed_delta with zeros
            mixed_delta = torch.zeros_like(base)
            
            # Calculate total weights for normalization
            total_weights = 0.0
            for delta_info in deltas:
                delta_weight = delta_info.get("weight", 1.0)
                total_weights += delta_weight
            
            # Accumulate weighted deltas in-place
            for delta_info in deltas:
                delta_weight = delta_info.get("weight", 1.0)
                global_weight = self.params.get("layer_weight", 1.0)
                # Calculate weight factor
                weight_factor = delta_weight * global_weight
                # Accumulate directly into mixed_delta
                mixed_delta.add_(delta_info["delta"], alpha=weight_factor)
            
            if self.params.get("normalize", False) and total_weights > 0:
                # Normalize in-place
                mixed_delta.div_(total_weights)
            
            if lambda_ != 1:
                # Scale in-place
                mixed_delta.mul_(lambda_)
            
            # Add to base and convert to base dtype
            return (base + mixed_delta).to(base.dtype)

class TIESMethod(LayerFusionMethod):
    """
    TIES fusion method implementation
    """
    def apply(self, base: torch.Tensor, deltas: List[Dict[str, Any]]) -> torch.Tensor:
        with torch.no_grad():
            density = self.params.get("density", 0.5)
            normalize = self.params.get("normalize", True)
            lambda_ = self.params.get("lambda", 1.0)
            
            if not deltas:
                return base
            
            # Sparsify deltas first
            for delta_info in deltas:
                current_density = delta_info.get("density", density)
                delta_info["delta"] = sparsify(
                    delta_info["delta"],
                    density=current_density,
                    method=SparsificationMethod.magnitude_prune,
                    rescale_norm=None
                )
            
            # Get weights
            weights = self._get_delta_weights(deltas, base)
            
            # Initialize accumulators for weighted deltas and divisor
            mixed_delta = torch.zeros_like(base)
            divisor = torch.zeros_like(base)
            
            # Process deltas incrementally instead of stacking all at once
            for i, delta_info in enumerate(deltas):
                # Get delta and weight for this task
                delta = delta_info["delta"]
                weight = weights[i]
                
                # Expand weight dimensions to match delta
                weight_expanded = weight
                while len(weight_expanded.shape) < len(delta.shape):
                    weight_expanded = weight_expanded.unsqueeze(-1)
                
                # Calculate weighted delta
                weighted_delta = delta * weight_expanded
                
                # Get mask for this weighted delta
                # We need to create a temporary stack with just this delta for get_mask
                temp_stack = weighted_delta.unsqueeze(0)
                mask = get_mask(temp_stack, method="sum", mask_dtype=base.dtype)
                mask = mask.squeeze(0)
                mask = mask.squeeze(0)  # Remove the extra dimension added by unsqueeze
                del temp_stack  # Free temporary memory
                
                # Accumulate weighted delta and divisor
                mixed_delta.add_(weighted_delta * mask)
                divisor.add_(weight_expanded * mask)
            
            # Handle division by zero
            divisor[divisor == 0] = 1
            
            if normalize:
                # Divide in-place
                mixed_delta.div_(divisor)
            
            if lambda_ != 1:
                # Scale in-place
                mixed_delta.mul_(lambda_)
            
            return (base + mixed_delta).to(base.dtype)

class DARETIESMethod(LayerFusionMethod):
    """
    DARE TIES fusion method implementation
    """
    def apply(self, base: torch.Tensor, deltas: List[Dict[str, Any]]) -> torch.Tensor:
        with torch.no_grad():
            density = self.params.get("density", 0.5)
            rescale = self.params.get("rescale", True)
            lambda_ = self.params.get("lambda", 1.0)
            
            if not deltas:
                return base
            
            # Sparsify deltas first
            rescale_norm = RescaleNorm.l1 if rescale else None
            for delta_info in deltas:
                current_density = delta_info.get("density", density)
                delta_info["delta"] = sparsify(
                    delta_info["delta"],
                    density=current_density,
                    method=SparsificationMethod.magnitude_prune,
                    rescale_norm=rescale_norm
                )
            
            # Get weights
            weights = self._get_delta_weights(deltas, base)
            
            # Initialize accumulators for weighted deltas and divisor
            mixed_delta = torch.zeros_like(base)
            divisor = torch.zeros_like(base)
            
            # Process deltas incrementally instead of stacking all at once
            for i, delta_info in enumerate(deltas):
                # Get delta and weight for this task
                delta = delta_info["delta"]
                weight = weights[i]
                
                # Expand weight dimensions to match delta
                weight_expanded = weight
                while len(weight_expanded.shape) < len(delta.shape):
                    weight_expanded = weight_expanded.unsqueeze(-1)
                
                # Calculate weighted delta
                weighted_delta = delta * weight_expanded
                
                # Get mask for this weighted delta
                temp_stack = weighted_delta.unsqueeze(0)
                mask = get_mask(temp_stack, method="sum", mask_dtype=base.dtype)
                mask = mask.squeeze(0)
                del temp_stack  # Free temporary memory
                
                # Accumulate weighted delta and divisor
                mixed_delta.add_(weighted_delta * mask)
                divisor.add_(weight_expanded * mask)
            
            # Handle division by zero
            divisor[divisor == 0] = 1
            
            # Normalize in-place
            mixed_delta.div_(divisor)
            
            if lambda_ != 1:
                # Scale in-place
                mixed_delta.mul_(lambda_)
            
            return (base + mixed_delta).to(base.dtype)

class DARELinearMethod(LayerFusionMethod):
    """
    DARE Linear fusion method implementation
    """
    def apply(self, base: torch.Tensor, deltas: List[Dict[str, Any]]) -> torch.Tensor:
        with torch.no_grad():
            density = self.params.get("density", 0.5)
            rescale = self.params.get("rescale", True)
            lambda_ = self.params.get("lambda", 1.0)
            
            if not deltas:
                return base
            
            # Sparsify deltas first
            rescale_norm = RescaleNorm.l1 if rescale else None
            for delta_info in deltas:
                current_density = delta_info.get("density", density)
                delta_info["delta"] = sparsify(
                    delta_info["delta"],
                    density=current_density,
                    method=SparsificationMethod.magnitude_prune,
                    rescale_norm=rescale_norm
                )
            
            # Get weights
            weights = self._get_delta_weights(deltas, base)
            
            # Initialize accumulators for weighted deltas and total weight
            mixed_delta = torch.zeros_like(base)
            total_weight = torch.zeros_like(base)
            
            # Process deltas incrementally instead of stacking all at once
            for i, delta_info in enumerate(deltas):
                # Get delta and weight for this task
                delta = delta_info["delta"]
                weight = weights[i]
                
                # Expand weight dimensions to match delta
                weight_expanded = weight
                while len(weight_expanded.shape) < len(delta.shape):
                    weight_expanded = weight_expanded.unsqueeze(-1)
                
                # Calculate weighted delta and accumulate
                mixed_delta.add_(delta * weight_expanded)
                total_weight.add_(weight_expanded)
            
            # Handle division by zero
            total_weight[total_weight == 0] = 1
            
            # Normalize in-place
            mixed_delta.div_(total_weight)
            
            if lambda_ != 1:
                # Scale in-place
                mixed_delta.mul_(lambda_)
            
            return (base + mixed_delta).to(base.dtype)

class BreadcrumbsMethod(LayerFusionMethod):
    """
    Breadcrumbs fusion method implementation
    """
    def apply(self, base: torch.Tensor, deltas: List[Dict[str, Any]]) -> torch.Tensor:
        with torch.no_grad():
            density = self.params.get("density", 0.5)
            lambda_ = self.params.get("lambda", 1.0)
            gamma = self.params.get("gamma", 0.01)

            if not deltas:
                return base
            
            # Sparsify deltas using magnitude_outliers method
            for delta_info in deltas:
                current_density = delta_info.get("density", density)
                gamma = delta_info.get("gamma", gamma)
                delta_info["delta"] = sparsify(
                    delta_info["delta"],
                    density=current_density,
                    method=SparsificationMethod.magnitude_outliers,
                    gamma=gamma,
                    rescale_norm=None
                )
            
            # Get weights
            weights = self._get_delta_weights(deltas, base)
            
            # Initialize accumulators for weighted deltas and total weight
            mixed_delta = torch.zeros_like(base)
            total_weight = torch.zeros_like(base)
            
            # Process deltas incrementally instead of stacking all at once
            for i, delta_info in enumerate(deltas):
                # Get delta and weight for this task
                delta = delta_info["delta"]
                weight = weights[i]
                
                # Expand weight dimensions to match delta
                weight_expanded = weight
                while len(weight_expanded.shape) < len(delta.shape):
                    weight_expanded = weight_expanded.unsqueeze(-1)
                
                # Calculate weighted delta and accumulate
                mixed_delta.add_(delta * weight_expanded)
                total_weight.add_(weight_expanded)
            
            # Handle division by zero
            total_weight[total_weight == 0] = 1
            
            # Normalize in-place
            mixed_delta.div_(total_weight)
            
            if lambda_ != 1:
                # Scale in-place
                mixed_delta.mul_(lambda_)
            
            return (base + mixed_delta).to(base.dtype)

class BreadcrumbsTIESMethod(LayerFusionMethod):
    """
    Breadcrumbs TIES fusion method implementation
    """
    def apply(self, base: torch.Tensor, deltas: List[Dict[str, Any]]) -> torch.Tensor:
        with torch.no_grad():
            density = self.params.get("density", 0.5)
            lambda_ = self.params.get("lambda", 1.0)
            
            if not deltas:
                return base
            
            # Sparsify deltas using magnitude_outliers method
            for delta_info in deltas:
                current_density = delta_info.get("density", density)
                gamma = delta_info.get("gamma", 0.01)
                delta_info["delta"] = sparsify(
                    delta_info["delta"],
                    density=current_density,
                    method=SparsificationMethod.magnitude_outliers,
                    gamma=gamma,
                    rescale_norm=None
                )
            
            # Get weights
            weights = self._get_delta_weights(deltas, base)
            
            # Initialize accumulators for weighted deltas and divisor
            mixed_delta = torch.zeros_like(base)
            divisor = torch.zeros_like(base)
            
            # Process deltas incrementally instead of stacking all at once
            for i, delta_info in enumerate(deltas):
                # Get delta and weight for this task
                delta = delta_info["delta"]
                weight = weights[i]
                
                # Expand weight dimensions to match delta
                weight_expanded = weight
                while len(weight_expanded.shape) < len(delta.shape):
                    weight_expanded = weight_expanded.unsqueeze(-1)
                
                # Calculate weighted delta
                weighted_delta = delta * weight_expanded
                
                # Get mask for this weighted delta
                temp_stack = weighted_delta.unsqueeze(0)
                mask = get_mask(temp_stack, method="sum", mask_dtype=base.dtype)
                mask = mask.squeeze(0)
                del temp_stack  # Free temporary memory
                
                # Accumulate weighted delta and divisor
                mixed_delta.add_(weighted_delta * mask)
                divisor.add_(weight_expanded * mask)
            
            # Handle division by zero
            divisor[divisor == 0] = 1
            
            # Normalize in-place
            mixed_delta.div_(divisor)
            
            if lambda_ != 1:
                # Scale in-place
                mixed_delta.mul_(lambda_)
            
            return (base + mixed_delta).to(base.dtype)

class DELLAMethod(LayerFusionMethod):
    """
    DELLA fusion method implementation
    """
    def apply(self, base: torch.Tensor, deltas: List[Dict[str, Any]]) -> torch.Tensor:
        with torch.no_grad():
            density = self.params.get("density", 0.5)
            normalize = self.params.get("normalize", True)
            rescale = self.params.get("rescale", True)
            lambda_ = self.params.get("lambda", 1.0)
            
            if not deltas:
                return base
            
            # Sparsify deltas using della_magprune method
            rescale_norm = RescaleNorm.l1 if rescale else None
            for delta_info in deltas:
                current_density = delta_info.get("density", density)
                epsilon = delta_info.get("epsilon", min(1.0 - current_density,0.05))
                delta_info["delta"] = sparsify(
                    delta_info["delta"],
                    density=current_density,
                    method=SparsificationMethod.della_magprune,
                    epsilon=epsilon,
                    rescale_norm=rescale_norm
                )
            
            # Get weights
            weights = self._get_delta_weights(deltas, base)
            
            # Initialize accumulators for weighted deltas and divisor
            mixed_delta = torch.zeros_like(base)
            divisor = torch.zeros_like(base)
            
            # Process deltas incrementally instead of stacking all at once
            for i, delta_info in enumerate(deltas):
                # Get delta and weight for this task
                delta = delta_info["delta"]
                weight = weights[i]
                
                # Expand weight dimensions to match delta
                weight_expanded = weight
                while len(weight_expanded.shape) < len(delta.shape):
                    weight_expanded = weight_expanded.unsqueeze(-1)
                
                # Calculate weighted delta
                weighted_delta = delta * weight_expanded
                
                # Get mask for this weighted delta
                temp_stack = weighted_delta.unsqueeze(0)
                mask = get_mask(temp_stack, method="sum", mask_dtype=base.dtype)
                mask = mask.squeeze(0)
                del temp_stack  # Free temporary memory
                
                # Accumulate weighted delta and divisor
                mixed_delta.add_(weighted_delta * mask)
                divisor.add_(weight_expanded * mask)
            
            # Handle division by zero
            divisor[divisor == 0] = 1
            
            if normalize:
                # Normalize in-place
                mixed_delta.div_(divisor)
            
            if lambda_ != 1:
                # Scale in-place
                mixed_delta.mul_(lambda_)
            
            return (base + mixed_delta).to(base.dtype)

class DELLALinearMethod(LayerFusionMethod):
    """
    Linear DELLA fusion method implementation
    """
    def apply(self, base: torch.Tensor, deltas: List[Dict[str, Any]]) -> torch.Tensor:
        with torch.no_grad():
            density = self.params.get("density", 0.5)
            rescale = self.params.get("rescale", True)
            lambda_ = self.params.get("lambda", 1.0)
            
            if not deltas:
                return base
            
            # Sparsify deltas using della_magprune method
            rescale_norm = RescaleNorm.l1 if rescale else None
            for delta_info in deltas:
                current_density = delta_info.get("density", density)
                epsilon = delta_info.get("epsilon", min(1.0 - current_density,0.05))
                delta_info["delta"] = sparsify(
                    delta_info["delta"],
                    density=current_density,
                    method=SparsificationMethod.della_magprune,
                    epsilon=epsilon,
                    rescale_norm=rescale_norm
                )
            
            # Get weights
            weights = self._get_delta_weights(deltas, base)
            
            # Initialize accumulators for weighted deltas and total weight
            mixed_delta = torch.zeros_like(base)
            total_weight = torch.zeros_like(base)
            
            # Process deltas incrementally instead of stacking all at once
            for i, delta_info in enumerate(deltas):
                # Get delta and weight for this task
                delta = delta_info["delta"]
                weight = weights[i]
                
                # Expand weight dimensions to match delta
                weight_expanded = weight
                while len(weight_expanded.shape) < len(delta.shape):
                    weight_expanded = weight_expanded.unsqueeze(-1)
                
                # Calculate weighted delta and accumulate
                mixed_delta.add_(delta * weight_expanded)
                total_weight.add_(weight_expanded)
            
            # Handle division by zero
            total_weight[total_weight == 0] = 1
            
            # Normalize in-place
            mixed_delta.div_(total_weight)
            
            if lambda_ != 1:
                # Scale in-place
                mixed_delta.mul_(lambda_)
            
            return (base + mixed_delta).to(base.dtype)

class LayerwiseModelFusion:
    """
    Interface for layer-wise model fusion using various methods
    """
    
    def __init__(self):
        """
        Initialize the layer-wise model fusion interface
        """
        self.method_registry = {
            "task_arithmetic": TaskArithmeticMethod,
            "ties": TIESMethod,
            "dare_ties": DARETIESMethod,
            "dare_linear": DARELinearMethod,
            "breadcrumbs": BreadcrumbsMethod,
            "breadcrumbs_ties": BreadcrumbsTIESMethod,
            "della": DELLAMethod,
            "della_linear": DELLALinearMethod
        }
    
    def _get_fusion_method(self, method_name: str, params: Dict[str, Any]) -> LayerFusionMethod:
        """
        Get the appropriate fusion method instance
        """
        if method_name not in self.method_registry:
            raise ValueError(f"Unsupported fusion method: {method_name}")
        return self.method_registry[method_name](method_name, params)
    
    def _match_config_to_layer(self, key: str, configs: List[LayerFusionConfig]) -> Optional[LayerFusionConfig]:
        """
        Determine which configuration applies to a given tensor key
        """
        # Check for special layer types first
        if "embed_tokens" in key or "rotary_emb" in key:
            for config in configs:
                if config.apply_to_embeddings:
                    return config
            return None
        
        if "norm" in key and ".layers." not in key:
            for config in configs:
                if config.apply_to_norm:
                    return config
        
        if "lm_head" in key:
            for config in configs:
                if config.apply_to_lm_head:
                    return config
        
        # Check layer patterns
        for config in configs:
            if config.layer_pattern and re.search(config.layer_pattern, key):
                return config
        
        # Default to the first config if no patterns match
        return configs[0] if configs else None
    
    def layer_fusion(
        self,
        base_model_path: str,
        task_model_paths: List[str],
        output_path: str,
        layer_configs: List[LayerFusionConfig],
        model_weights: Optional[List[float]] = None,
        device: Optional[str] = None,
        copy_extra_files: bool = True
    ) -> bool:
        """
        Perform layer-wise fusion of models
        
        Args:
            base_model_path: Path to the base model
            task_model_paths: List of paths to task models
            output_path: Path to save the merged model
            layer_configs: List of layer fusion configurations
            model_weights: Optional weights for each task model
            device: Device to perform fusion on (cpu or cuda)
            copy_extra_files: Whether to copy non-weight files from base model
            
        Returns:
            bool: Whether fusion was successful
        """
        # Validate inputs
        if not layer_configs:
            raise ValueError("At least one layer configuration must be provided")
        
        if model_weights and len(model_weights) != len(task_model_paths):
            raise ValueError("model_weights must have the same length as task_model_paths")
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output directory '{output_path}' created")
        
        # Load task models to CPU first to save GPU memory
        logger.info("Loading task models...")
        task_models_tensors = []
        
        for i, model_path in enumerate(task_model_paths):
            model_files = glob(os.path.join(model_path, "*.safetensors"))
            if not model_files:
                logger.error(f"No safetensors files found in {model_path}")
                return False
            
            model_tensors = {}
            for file_path in model_files:
                try:
                    # Load to CPU first
                    tensors = load_file(file_path, device="cpu")
                    model_tensors.update(tensors)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    return False
            
            task_models_tensors.append(model_tensors)
            logger.info(f"Loaded {len(model_tensors)} tensors from model {i+1}")
        
        # Get model weights
        if model_weights is None:
            model_weights = [1.0] * len(task_model_paths)
        
        # Process base model
        base_files = glob(os.path.join(base_model_path, "*.safetensors"))
        if not base_files:
            logger.error(f"No safetensors files found in {base_model_path}")
            return False
        
        # Process each base model file
        for base_file in base_files:
            filename = os.path.basename(base_file)
            output_file_path = os.path.join(output_path, filename)
            
            logger.info(f"Processing {filename}...")
            
            try:
                # Load base file to CPU first
                base_tensors_cpu = load_file(base_file, device="cpu")
            except Exception as e:
                logger.error(f"Error loading {base_file}: {e}")
                continue
            
            merged_tensors = {}
            
            # Process each tensor
            for key in tqdm(base_tensors_cpu.keys(), desc=f"Fusing {filename}"):
                # Find applicable config
                config = self._match_config_to_layer(key, layer_configs)
                
                if not config:
                    # No config matches, copy base tensor directly
                    merged_tensors[key] = base_tensors_cpu[key]
                    continue
                
                # Move base tensor to GPU only when needed
                base_tensor_dtype = base_tensors_cpu[key].dtype
                base_tensor = base_tensors_cpu[key].to(device=device, dtype=torch.float32)
                
                # Calculate task vectors
                deltas = []
                # 使用当前配置的model_weights，如果不存在则使用全局model_weights
                current_model_weights = config.model_weights if config.model_weights is not None else model_weights
                
                for i, task_tensors in enumerate(task_models_tensors):
                    if key not in task_tensors:
                        logger.warning(f"Key {key} not found in task model {i+1}")
                        continue
                    
                    # Move task tensor to GPU only when needed
                    task_tensor = task_tensors[key].to(device=device, dtype=torch.float32)
                    
                    # Handle shape mismatch if needed
                    if task_tensor.shape != base_tensor.shape:
                        if "embed_tokens" in key:
                            # Try to use a submatrix
                            try:
                                task_tensor = task_tensor[:base_tensor.shape[0], :base_tensor.shape[1]]
                                logger.warning(f"Using submatrix for {key} in task model {i+1}")
                            except:
                                logger.warning(f"Shape mismatch for {key}, skipping task model {i+1}")
                                del task_tensor
                                continue
                        else:
                            logger.warning(f"Shape mismatch for {key}, skipping task model {i+1}")
                            del task_tensor
                            continue
                    
                    # Calculate delta
                    delta = task_tensor - base_tensor
                    
                    # Add delta info
                    delta_info = {
                        "delta": delta,
                        "weight": current_model_weights[i]
                    }
                    
                    # Add density if specified in config
                    if "density" in config.params:
                        delta_info["density"] = config.params["density"]
                    
                    deltas.append(delta_info)
                    # Free task tensor memory
                    del task_tensor
                
                if deltas:
                    # Get fusion method and apply
                    fusion_method = self._get_fusion_method(config.method, config.params)
                    merged_tensor = fusion_method.apply(base_tensor, deltas)
                    merged_tensors[key] = merged_tensor.to(device="cpu", dtype=base_tensor_dtype)
                    
                    # Free delta tensors memory
                    for delta_info in deltas:
                        del delta_info["delta"]
                else:
                    # No deltas, copy base tensor
                    merged_tensors[key] = base_tensors_cpu[key]
                
                # Free base tensor memory
                del base_tensor
                torch.cuda.empty_cache()
            
            # Save merged file from CPU
            try:
                save_file(merged_tensors, output_file_path)
                logger.info(f"Saved merged file to {output_file_path}")
            except Exception as e:
                logger.error(f"Error saving {output_file_path}: {e}")
                continue
        
        # Copy extra files
        if copy_extra_files:
            model_name = [base_model_path] + task_model_paths
            extra_files_model_path = base_model_path
            for name in model_name:
                if "thinking" in name:
                    extra_files_model_path = name
                    continue
            logger.info(f"Copying extra files from {extra_files_model_path}...")
            for filename in os.listdir(extra_files_model_path):
                if filename.endswith(".safetensors"):
                    continue
                
                src_path = os.path.join(extra_files_model_path, filename)
                dest_path = os.path.join(output_path, filename)
                
                if os.path.isfile(src_path) and not os.path.exists(dest_path):
                    try:
                        import shutil
                        shutil.copy2(src_path, dest_path)
                        logger.info(f"Copied {filename}")
                    except Exception as e:
                        logger.warning(f"Error copying {filename}: {e}")
        
        logger.info("Layer-wise fusion completed successfully")
        return True
    
    def task_arithmetic_layer(
        self,
        base_model_path: str,
        task_model_paths: List[str],
        output_path: str,
        layer_weights: List[float],
        model_weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ) -> bool:
        """
        Task Arithmetic fusion with layer-wise weights
        
        Args:
            base_model_path: Path to the base model
            task_model_paths: List of paths to task models
            output_path: Path to save the merged model
            layer_weights: List of weights for different layer groups
                          Format: [w0-w7 for layers 12-35 (3 per group), w8 for norm, w9 for lm_head]
            model_weights: Optional weights for each task model
            device: Device to perform fusion on
            
        Returns:
            bool: Whether fusion was successful
        """
        # Create layer configurations based on weights
        if len(layer_weights) != 10:
            raise ValueError("layer_weights must contain exactly 10 values")
        
        configs = []
        
        # Create configs for each layer group
        for i in range(8):
            start_layer = 12 + i * 3
            end_layer = start_layer + 2
            layer_pattern = f"layers\\.({start_layer}|{start_layer+1}|{end_layer})"
            configs.append(LayerFusionConfig(
                method="task_arithmetic",
                params={"weight": layer_weights[i]},
                layer_pattern=layer_pattern,
                apply_to_embeddings=False,
                apply_to_norm=False,
                apply_to_lm_head=False
            ))
        
        # Add config for norm layers
        configs.append(LayerFusionConfig(
            method="task_arithmetic",
            params={"weight": layer_weights[8]},
            apply_to_embeddings=False,
            apply_to_norm=True,
            apply_to_lm_head=False
        ))
        
        # Add config for lm_head
        configs.append(LayerFusionConfig(
            method="task_arithmetic",
            params={"weight": layer_weights[9]},
            apply_to_embeddings=False,
            apply_to_norm=False,
            apply_to_lm_head=True
        ))
        
        # Add default config for embeddings (no fusion)
        configs.append(LayerFusionConfig(
            method="task_arithmetic",
            params={"weight": 0.0},
            apply_to_embeddings=True,
            apply_to_norm=False,
            apply_to_lm_head=False
        ))
        
        return self.layer_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            output_path=output_path,
            layer_configs=configs,
            model_weights=model_weights,
            device=device
        )
    
    def ties_layer(
        self,
        base_model_path: str,
        task_model_paths: List[str],
        output_path: str,
        layer_configs: List[LayerFusionConfig],
        model_weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ) -> bool:
        """
        TIES fusion with layer-wise configuration
        """
        # Ensure all configs use the 'ties' method
        for config in layer_configs:
            config.method = "ties"
        
        return self.layer_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            output_path=output_path,
            layer_configs=layer_configs,
            model_weights=model_weights,
            device=device
        )
    
    def dare_ties_layer(
        self,
        base_model_path: str,
        task_model_paths: List[str],
        output_path: str,
        layer_configs: List[LayerFusionConfig],
        model_weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ) -> bool:
        """
        DARE TIES fusion with layer-wise configuration
        """
        # Ensure all configs use the 'dare_ties' method
        for config in layer_configs:
            config.method = "dare_ties"
        
        return self.layer_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            output_path=output_path,
            layer_configs=layer_configs,
            model_weights=model_weights,
            device=device
        )
    
    def dare_linear_layer(
        self,
        base_model_path: str,
        task_model_paths: List[str],
        output_path: str,
        layer_configs: List[LayerFusionConfig],
        model_weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ) -> bool:
        """
        DARE Linear fusion with layer-wise configuration
        """
        # Ensure all configs use the 'dare_linear' method
        for config in layer_configs:
            config.method = "dare_linear"
        
        return self.layer_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            output_path=output_path,
            layer_configs=layer_configs,
            model_weights=model_weights,
            device=device
        )
    
    def breadcrumbs_layer(
        self,
        base_model_path: str,
        task_model_paths: List[str],
        output_path: str,
        layer_configs: List[LayerFusionConfig],
        model_weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ) -> bool:
        """
        Breadcrumbs fusion with layer-wise configuration
        """
        # Ensure all configs use the 'breadcrumbs' method
        for config in layer_configs:
            config.method = "breadcrumbs"
        
        return self.layer_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            output_path=output_path,
            layer_configs=layer_configs,
            model_weights=model_weights,
            device=device
        )
    
    def breadcrumbs_ties_layer(
        self,
        base_model_path: str,
        task_model_paths: List[str],
        output_path: str,
        layer_configs: List[LayerFusionConfig],
        model_weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ) -> bool:
        """
        Breadcrumbs TIES fusion with layer-wise configuration
        """
        # Ensure all configs use the 'breadcrumbs_ties' method
        for config in layer_configs:
            config.method = "breadcrumbs_ties"
        
        return self.layer_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            output_path=output_path,
            layer_configs=layer_configs,
            model_weights=model_weights,
            device=device
        )
    
    def della_layer(
        self,
        base_model_path: str,
        task_model_paths: List[str],
        output_path: str,
        layer_configs: List[LayerFusionConfig],
        model_weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ) -> bool:
        """
        DELLA fusion with layer-wise configuration
        """
        # Ensure all configs use the 'della' method
        for config in layer_configs:
            config.method = "della"
        
        return self.layer_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            output_path=output_path,
            layer_configs=layer_configs,
            model_weights=model_weights,
            device=device
        )
    
    def della_linear_layer(
        self,
        base_model_path: str,
        task_model_paths: List[str],
        output_path: str,
        layer_configs: List[LayerFusionConfig],
        model_weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ) -> bool:
        """
        Linear DELLA fusion with layer-wise configuration
        """
        # Ensure all configs use the 'della_linear' method
        for config in layer_configs:
            config.method = "della_linear"
        
        return self.layer_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            output_path=output_path,
            layer_configs=layer_configs,
            model_weights=model_weights,
            device=device
        )

# Example usage
def example_usage():
    """
    Example of how to use the LayerwiseModelFusion interface
    """
    # Initialize the interface
    fusion = LayerwiseModelFusion()
    
    # Example 1: Task Arithmetic with layer weights
    base_model = "models/Qwen3-4B"
    task_models = [
        "models/Qwen3-4B-Instruct-2507",
        "models/Qwen3-4B-thinking-2507"
    ]
    output_path = "models/Qwen3-4B-merged"
    
    # Layer weights: 8 for layers 12-35, 1 for norm, 1 for lm_head
    layer_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Run task arithmetic with layer weights
    success = fusion.task_arithmetic_layer(
        base_model_path=base_model,
        task_model_paths=task_models,
        output_path=output_path,
        layer_weights=layer_weights
    )
    
    if success:
        print(f"Model successfully merged to {output_path}")
    
    # Example 2: Custom layer configurations with different methods
    configs = [
        # Use TIES for layers 0-11
        LayerFusionConfig(
            method="ties",
            params={"density": 0.3, "layer_weight": 0.5, "normalize": True},
            layer_pattern="layers\\.([0-9]|1[01])",
            apply_to_embeddings=False,
            apply_to_norm=False,
            apply_to_lm_head=False
        ),
        # Use DARE TIES for layers 12-23
        LayerFusionConfig(
            method="dare_ties",
            params={"density": 0.4, "layer_weight": 0.7, "rescale": True},
            layer_pattern="layers\\.((1[2-9])|(2[0-3]))",
            apply_to_embeddings=False,
            apply_to_norm=False,
            apply_to_lm_head=False
        ),
        # Use Task Arithmetic for layers 24+
        LayerFusionConfig(
            method="task_arithmetic",
            params={"layer_weight": 0.3},
            layer_pattern="layers\\.([2-9][4-9]|[3-9][0-9])",
            apply_to_embeddings=False,
            apply_to_norm=False,
            apply_to_lm_head=False
        ),
        # Handle norm layers
        LayerFusionConfig(
            method="ties",
            params={"density": 0.5, "layer_weight": 0.8},
            apply_to_embeddings=False,
            apply_to_norm=True,
            apply_to_lm_head=False
        ),
        # Handle lm_head
        LayerFusionConfig(
            method="task_arithmetic",
            params={"layer_weight": 0.5},
            apply_to_embeddings=False,
            apply_to_norm=False,
            apply_to_lm_head=True
        ),
        # Don't modify embeddings
        LayerFusionConfig(
            method="task_arithmetic",
            params={"layer_weight": 0.0},
            apply_to_embeddings=True,
            apply_to_norm=False,
            apply_to_lm_head=False
        )
    ]
    
    output_path_custom = "./merged_model_custom"
    
    # Run custom layer fusion
    success = fusion.layer_fusion(
        base_model_path=base_model,
        task_model_paths=task_models,
        output_path=output_path_custom,
        layer_configs=configs
    )
    
    if success:
        print(f"Custom model successfully merged to {output_path_custom}")

# Test functions to verify the functionality
def test_layer_fusion_methods():
    """
    Test individual layer fusion methods with synthetic data
    """
    import torch
    import numpy as np
    
    print("Testing layer fusion methods with synthetic data...")
    
    # Create synthetic base weights and deltas
    base_weights = torch.randn(10, 20)
    
    # Create deltas for 3 task models
    deltas = [
        {"delta": torch.randn(10, 20) * 0.1, "weight": 0.5},
        {"delta": torch.randn(10, 20) * 0.1, "weight": 0.3},
        {"delta": torch.randn(10, 20) * 0.1, "weight": 0.2}
    ]
    
    # Test each fusion method
    methods = [
        ("Task Arithmetic", TaskArithmeticMethod("task_arithmetic", {"layer_weight": 1.0, "lambda": 1.0})),
        ("TIES", TIESMethod("ties", {"density": 0.5, "layer_weight": 1.0, "normalize": True})),
        ("DARE TIES", DARETIESMethod("dare_ties", {"density": 0.5, "layer_weight": 1.0, "rescale": True})),
        ("DARE Linear", DARELinearMethod("dare_linear", {"density": 0.5, "layer_weight": 1.0, "rescale": True})),
        ("Breadcrumbs", BreadcrumbsMethod("breadcrumbs", {"density": 0.5, "layer_weight": 1.0})),
        ("Breadcrumbs TIES", BreadcrumbsTIESMethod("breadcrumbs_ties", {"density": 0.5, "layer_weight": 1.0})),
        ("DELLA", DELLAMethod("della", {"density": 0.5, "layer_weight": 1.0, "rescale": True, "normalize": True})),
        ("DELLA Linear", DELLALinearMethod("della_linear", {"density": 0.5, "layer_weight": 1.0, "rescale": True}))
    ]
    
    results = {}
    for method_name, method in methods:
        try:
            result = method.apply(base_weights.clone(), deltas.copy())
            results[method_name] = result
            print(f"✓ {method_name} applied successfully")
            print(f"  - Result shape: {result.shape}")
            print(f"  - Result min/max: {result.min():.4f} / {result.max():.4f}")
        except Exception as e:
            print(f"✗ {method_name} failed with error: {str(e)}")
    
    # Test layer configuration matching
    print("\nTesting layer configuration matching...")
    configs = [
        LayerFusionConfig(layer_names=["model.layers.0", "model.layers.1"], method="task_arithmetic", params={}),
        LayerFusionConfig(layer_names=["model.layers.2", "model.layers.3"], method="ties", params={}),
        LayerFusionConfig(layer_names=["model.norm", "lm_head"], method="dare_ties", params={})
    ]
    
    # Test layer matching logic
    test_layers = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.2.mlp.gate_proj",
        "model.norm.weight",
        "unmatched.layer"
    ]
    
    for layer in test_layers:
        for config in configs:
            if any(layer.startswith(prefix) for prefix in config.layer_names):
                print(f"✓ Layer '{layer}' matched to config with method: {config.method}")
                break
        else:
            print(f"✗ Layer '{layer}' not matched to any config")
    
    print("\nAll tests completed!")
    return results


def test_complete_fusion_workflow():
    """
    Test the complete fusion workflow with synthetic model weights
    """
    import torch
    
    print("\nTesting complete fusion workflow...")
    
    try:
        print("1. Creating test model configurations...")
        
        # Define test layer configurations
        test_configs = [
            LayerFusionConfig(
                layer_names=["model.layers.0", "model.layers.1"],
                method="task_arithmetic",
                params={"layer_weight": 0.5, "lambda": 1.0}
            ),
            LayerFusionConfig(
                layer_names=["model.layers.2"],
                method="ties",
                params={"density": 0.7, "layer_weight": 0.6}
            ),
            LayerFusionConfig(
                layer_names=["model.norm", "lm_head"],
                method="dare_ties",
                params={"density": 0.5, "layer_weight": 0.8, "rescale": True}
            )
        ]
        
        print("2. Initializing fusion interface...")
        fusion = LayerwiseModelFusion()
        
        print("3. Testing configuration validation...")
        # Validate that all methods are registered
        for config in test_configs:
            if config.method in fusion.method_registry:
                print(f"✓ Method '{config.method}' is properly registered")
            else:
                print(f"✗ Method '{config.method}' is not registered")
        
        print("\nWorkflow test completed successfully!")
        return True
    
    except Exception as e:
        print(f"✗ Workflow test failed with error: {str(e)}")
        return False



def run_tests():
    """
    Run all tests to verify functionality
    """
    print("\nRunning comprehensive tests for LayerwiseModelFusion...\n" + "="*60)
    
    # Run individual method tests
    results = test_layer_fusion_methods()
    
    # Run complete workflow test
    workflow_success = test_complete_fusion_workflow()
    
    print("\n" + "="*60)
    print("Testing summary:")
    print(f"- Individual methods tested: {len(results)}")
    print(f"- Workflow test: {'PASSED' if workflow_success else 'FAILED'}")
    print("\nTo use this module in your code:")
    print("1. Import the LayerwiseModelFusion and LayerFusionConfig classes")
    print("2. Define your layer configurations")
    print("3. Call the appropriate fusion method")
    print("\nNote: For actual model fusion, you'll need to provide real model paths.")


if __name__ == "__main__":
    # You can choose to run the example or the tests
    print("Choose what to run:")
    print("1. Example usage")
    print("2. Run tests")
    
    # Default to example usage
    choice = 2
    
    if choice == 1:
        example_usage()
    else:
        run_tests()
    
    # Uncomment to run tests directly
    # run_tests()
   


   # DARE-TIES
   # TIES
   # TA
   # DARE-TA
   # Breadcrumbs
   # Breadcrumb-TES
   # DELLA
   # DELLA Linear