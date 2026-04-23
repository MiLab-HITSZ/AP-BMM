"""
Model Fusion Interface for Generalized Task Arithmetic Methods
Based on mergekit library implementations
"""

import os
import logging
import torch
from typing import Dict, List, Optional, Union
from pathlib import Path

from mergekit.config import MergeConfiguration, InputModelDefinition
from mergekit.merge import run_merge
from mergekit.options import MergeOptions
from mergekit.merge_methods.generalized_task_arithmetic import (
    ConsensusMethod,
    GeneralizedTaskArithmeticMerge,
)
from mergekit.sparsify import SparsificationMethod
from mergekit.common import ModelReference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFusionInterface:
    """
    Interface for model fusion using various Generalized Task Arithmetic methods.
    """
    
    def __init__(self):
        """Initialize the model fusion interface."""
        pass
    
    def _create_merge_config(
        self,
        base_model: str,
        task_models: List[str],
        method_name: str,
        parameters: Optional[Dict] = None,
        model_parameters: Optional[List[Dict]] = None
    ) -> MergeConfiguration:
        """
        Create a merge configuration for the specified method.
        
        Args:
            base_model: Path or HuggingFace ID of the base model
            task_models: List of paths or HuggingFace IDs of task models
            method_name: Name of the merge method to use
            parameters: Additional parameters for the merge method
            model_parameters: Parameters for each individual model (list of dicts)
            
        Returns:
            MergeConfiguration object
        """
        # Create model definitions
        models = []
        for i, task_model in enumerate(task_models):
            # Get model-specific parameters if provided
            model_params = {"weight": 1.0}  # Default weight
            if model_parameters and i < len(model_parameters):
                model_params.update(model_parameters[i])
                
            models.append(InputModelDefinition(
                model=ModelReference.parse(task_model),
                parameters=model_params
            ))
        
        # Set default parameters if none provided
        if parameters is None:
            parameters = {}
            
        # Create merge configuration
        config = MergeConfiguration(
            models=models,
            merge_method=method_name,
            base_model=ModelReference.parse(base_model),
            parameters=parameters
        )
        
        return config
    
    def _run_merge_operation(
        self,
        config: MergeConfiguration,
        output_path: str,
        options: Optional[MergeOptions] = None
    ) -> None:
        """
        Execute the merge operation.
        
        Args:
            config: Merge configuration
            output_path: Path to save the merged model
            options: Merge options
        """
        if options is None:
            options = MergeOptions(
                cuda=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
                low_cpu_memory=False,
                safe_serialization=True,
                copy_tokenizer=True
            )
        
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Run the merge
        run_merge(
            merge_config=config,
            out_path=output_path,
            options=options
        )
    
    def task_arithmetic(
        self,
        base_model: str,
        task_models: List[str],
        output_path: str,
        density: Optional[float] = None,
        weight: float = 1.0,
        model_weights: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Task Arithmetic merging method.
        
        Reference: https://arxiv.org/abs/2212.04089
        
        Args:
            base_model: Path or HuggingFace ID of the base model
            task_models: List of paths or HuggingFace IDs of task models
            output_path: Path to save the merged model
            density: Density parameter for sparsification (optional)
            weight: Weight for the task vectors (global)
            model_weights: Individual weights for each task model
            **kwargs: Additional parameters to pass to the merge method
        """
        parameters = {"weight": weight}
        if density is not None:
            parameters["density"] = density
            
        # Add any additional parameters
        parameters.update(kwargs)
        
        # Prepare model-specific parameters
        model_parameters = None
        if model_weights:
            model_parameters = [{"weight": w} for w in model_weights]
            
        config = self._create_merge_config(
            base_model=base_model,
            task_models=task_models,
            method_name="task_arithmetic",
            parameters=parameters,
            model_parameters=model_parameters
        )
        
        self._run_merge_operation(config, output_path)
    
    def ties(
        self,
        base_model: str,
        task_models: List[str],
        output_path: str,
        density: float = 0.5,
        weight: float = 1.0,
        normalize: bool = True,
        model_weights: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        TIES merging method.
        
        Reference: https://arxiv.org/abs/2306.01708
        
        Args:
            base_model: Path or HuggingFace ID of the base model
            task_models: List of paths or HuggingFace IDs of task models
            output_path: Path to save the merged model
            density: Density parameter for sparsification
            weight: Weight for the task vectors (global)
            normalize: Whether to normalize the task vectors
            model_weights: Individual weights for each task model
            **kwargs: Additional parameters to pass to the merge method
        """
        parameters = {
            "density": density,
            "weight": weight,
            "normalize": normalize
        }
        
        # Add any additional parameters
        parameters.update(kwargs)
        
        # Prepare model-specific parameters
        model_parameters = None
        if model_weights:
            model_parameters = [{"weight": w} for w in model_weights]
        
        config = self._create_merge_config(
            base_model=base_model,
            task_models=task_models,
            method_name="ties",
            parameters=parameters,
            model_parameters=model_parameters
        )
        
        self._run_merge_operation(config, output_path)
    
    def dare_ties(
        self,
        base_model: str,
        task_models: List[str],
        output_path: str,
        density: float = 0.5,
        weight: float = 1.0,
        rescale: bool = True,
        model_weights: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        DARE TIES merging method.
        
        Reference: https://arxiv.org/abs/2311.03099
        
        Args:
            base_model: Path or HuggingFace ID of the base model
            task_models: List of paths or HuggingFace IDs of task models
            output_path: Path to save the merged model
            density: Density parameter for sparsification
            weight: Weight for the task vectors (global)
            rescale: Whether to rescale the task vectors
            model_weights: Individual weights for each task model
            **kwargs: Additional parameters to pass to the merge method
        """
        parameters = {
            "density": density,
            "weight": weight,
            "rescale": rescale
        }
        
        # Add any additional parameters
        parameters.update(kwargs)
        
        # Prepare model-specific parameters
        model_parameters = None
        if model_weights:
            model_parameters = [{"weight": w} for w in model_weights]
        
        config = self._create_merge_config(
            base_model=base_model,
            task_models=task_models,
            method_name="dare_ties",
            parameters=parameters,
            model_parameters=model_parameters
        )
        
        self._run_merge_operation(config, output_path)
    
    def dare_linear(
        self,
        base_model: str,
        task_models: List[str],
        output_path: str,
        density: float = 0.5,
        weight: float = 1.0,
        rescale: bool = True,
        model_weights: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Linear DARE merging method.
        
        Reference: https://arxiv.org/abs/2311.03099
        
        Args:
            base_model: Path or HuggingFace ID of the base model
            task_models: List of paths or HuggingFace IDs of task models
            output_path: Path to save the merged model
            density: Density parameter for sparsification
            weight: Weight for the task vectors (global)
            rescale: Whether to rescale the task vectors
            model_weights: Individual weights for each task model
            **kwargs: Additional parameters to pass to the merge method
        """
        parameters = {
            "density": density,
            "weight": weight,
            "rescale": rescale
        }
        
        # Add any additional parameters
        parameters.update(kwargs)
        
        # Prepare model-specific parameters
        model_parameters = None
        if model_weights:
            model_parameters = [{"weight": w} for w in model_weights]
        
        config = self._create_merge_config(
            base_model=base_model,
            task_models=task_models,
            method_name="dare_linear",
            parameters=parameters,
            model_parameters=model_parameters
        )
        
        self._run_merge_operation(config, output_path)
    
    def breadcrumbs(
        self,
        base_model: str,
        task_models: List[str],
        output_path: str,
        density: float = 0.5,
        weight: float = 1.0,
        model_weights: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Model Breadcrumbs merging method.
        
        Reference: https://arxiv.org/abs/2312.06795
        
        Args:
            base_model: Path or HuggingFace ID of the base model
            task_models: List of paths or HuggingFace IDs of task models
            output_path: Path to save the merged model
            density: Density parameter for sparsification
            weight: Weight for the task vectors (global)
            model_weights: Individual weights for each task model
            **kwargs: Additional parameters to pass to the merge method
        """
        parameters = {
            "density": density,
            "weight": weight
        }
        
        # Add any additional parameters
        parameters.update(kwargs)
        
        # Prepare model-specific parameters
        model_parameters = None
        if model_weights:
            model_parameters = [{"weight": w} for w in model_weights]
        
        config = self._create_merge_config(
            base_model=base_model,
            task_models=task_models,
            method_name="breadcrumbs",
            parameters=parameters,
            model_parameters=model_parameters
        )
        
        self._run_merge_operation(config, output_path)
    
    def breadcrumbs_ties(
        self,
        base_model: str,
        task_models: List[str],
        output_path: str,
        density: float = 0.5,
        weight: float = 1.0,
        model_weights: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Model Breadcrumbs with TIES merging method.
        
        Reference: https://arxiv.org/abs/2312.06795
        
        Args:
            base_model: Path or HuggingFace ID of the base model
            task_models: List of paths or HuggingFace IDs of task models
            output_path: Path to save the merged model
            density: Density parameter for sparsification
            weight: Weight for the task vectors (global)
            model_weights: Individual weights for each task model
            **kwargs: Additional parameters to pass to the merge method
        """
        parameters = {
            "density": density,
            "weight": weight
        }
        
        # Add any additional parameters
        parameters.update(kwargs)
        
        # Prepare model-specific parameters
        model_parameters = None
        if model_weights:
            model_parameters = [{"weight": w} for w in model_weights]
        
        config = self._create_merge_config(
            base_model=base_model,
            task_models=task_models,
            method_name="breadcrumbs_ties",
            parameters=parameters,
            model_parameters=model_parameters
        )
        
        self._run_merge_operation(config, output_path)
    
    def della(
        self,
        base_model: str,
        task_models: List[str],
        output_path: str,
        density: float = 0.5,
        weight: float = 1.0,
        normalize: bool = True,
        rescale: bool = True,
        model_weights: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        DELLA merging method.
        
        Reference: https://arxiv.org/abs/2406.11617
        
        Args:
            base_model: Path or HuggingFace ID of the base model
            task_models: List of paths or HuggingFace IDs of task models
            output_path: Path to save the merged model
            density: Density parameter for sparsification
            weight: Weight for the task vectors (global)
            normalize: Whether to normalize the task vectors
            rescale: Whether to rescale the task vectors
            model_weights: Individual weights for each task model
            **kwargs: Additional parameters to pass to the merge method
        """
        parameters = {
            "density": density,
            "weight": weight,
            "normalize": normalize,
            "rescale": rescale
        }
        
        # Add any additional parameters
        parameters.update(kwargs)
        
        # Prepare model-specific parameters
        model_parameters = None
        if model_weights:
            model_parameters = [{"weight": w} for w in model_weights]
        
        config = self._create_merge_config(
            base_model=base_model,
            task_models=task_models,
            method_name="della",
            parameters=parameters,
            model_parameters=model_parameters
        )
        
        self._run_merge_operation(config, output_path)
    
    def della_linear(
        self,
        base_model: str,
        task_models: List[str],
        output_path: str,
        density: float = 0.5,
        weight: float = 1.0,
        rescale: bool = True,
        model_weights: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Linear DELLA merging method.
        
        Reference: https://arxiv.org/abs/2406.11617
        
        Args:
            base_model: Path or HuggingFace ID of the base model
            task_models: List of paths or HuggingFace IDs of task models
            output_path: Path to save the merged model
            density: Density parameter for sparsification
            weight: Weight for the task vectors (global)
            rescale: Whether to rescale the task vectors
            model_weights: Individual weights for each task model
            **kwargs: Additional parameters to pass to the merge method
        """
        parameters = {
            "density": density,
            "weight": weight,
            "rescale": rescale
        }
        
        # Add any additional parameters
        parameters.update(kwargs)
        
        # Prepare model-specific parameters
        model_parameters = None
        if model_weights:
            model_parameters = [{"weight": w} for w in model_weights]
        
        config = self._create_merge_config(
            base_model=base_model,
            task_models=task_models,
            method_name="della_linear",
            parameters=parameters,
            model_parameters=model_parameters
        )
        
        self._run_merge_operation(config, output_path)

# Example usage function
def example_usage():
    """
    Example of how to use the ModelFusionInterface
    """
    # Initialize the interface
    fusion = ModelFusionInterface()
    
    # Define model paths
    base_model = "Qwen3-4B-Base"
    task_models = [
        "Qwen3-4B-Instruct-2507'",
        "Qwen3-4B-thinking-2507"
    ]
    output_path = "./models"
    
    # Example: Using TIES method
    fusion.ties(
        base_model=base_model,
        task_models=task_models,
        output_path=output_path,
        density=0.5,
        weight=1.0,
        normalize=True
    )

if __name__ == "__main__":
    example_usage()