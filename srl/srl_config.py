#!/usr/bin/env python3
"""
SRL Configuration Module

Centralized configuration for SRL training with hardware constraints.
Optimized for RTX 5060 (8GB VRAM) + 32GB RAM.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model and LoRA configuration."""
    model_name: str = "unsloth/Qwen2.5-7B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # Auto-detect (bf16 if supported)
    
    # LoRA parameters
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    
    num_epochs: int = 1
    batch_size: int = 2  # Small for VRAM constraints
    gradient_accumulation_steps: int = 4  # Simulate larger batch
    max_grad_norm: float = 1.0
    
    # GRPO specific
    num_rollouts: int = 4  # Must be >= 2 for non-zero advantage
    clip_epsilon: float = 0.2
    kl_coef: float = 0.1


@dataclass
class VLLMConfig:
    """vLLM inference configuration."""
    gpu_memory_utilization: float = 0.80  # 80% of VRAM for vLLM
    sleep_level: int = 1  # Level 1: offload to CPU RAM
    max_completion_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    # DISABLED: vLLM has version incompatibility with current Unsloth
    # Set to True only after upgrading vLLM/Unsloth to compatible versions
    fast_inference: bool = True  # Enable vLLM integration


@dataclass
class RewardConfig:
    """Reward function configuration."""
    format_check: bool = True
    similarity_threshold: float = 0.0  # Minimum similarity to get reward
    format_penalty: float = -1.0  # Penalty for invalid format
    use_parallel: bool = True  # Use multiprocessing for reward calculation
    num_workers: Optional[int] = None  # Auto-detect CPU cores


@dataclass
class SamplingConfig:
    """Dynamic sampling filter configuration."""
    # DISABLED by default - set to 0.01 to enable filtering
    variance_threshold: float = 0.0  # Keep ALL samples (was 0.01)
    warmup_steps: int = 10000  # Steps before reaching max threshold
    enabled: bool = False  # Whether to filter at all


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    backend: str = "console"  # "console", "tensorboard", or "wandb"
    log_dir: str = "./runs"  # TensorBoard log directory
    wandb_project: Optional[str] = "srl-training"  # WandB project name
    wandb_run_name: Optional[str] = None  # WandB run name (auto-generated if None)
    log_resources: bool = True  # Log CPU/GPU/VRAM/RAM usage
    verbose: bool = True  # Print to console even when using TB/WandB


@dataclass
class SRLConfig:
    """Complete SRL configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Paths
    train_data: str = "./data/srl_train.jsonl"
    val_data: str = "./data/srl_val.jsonl"
    output_dir: str = "./checkpoints"
    ref_policy_dir: str = "./_ref_policy"
    
    # Logging
    log_steps: int = 1
    save_steps: int = 100
    
    @classmethod
    def for_rtx_5060(cls) -> "SRLConfig":
        """Configuration optimized for RTX 5060 (8GB VRAM)."""
        config = cls()
        # Conservative settings for 8GB
        config.vllm.gpu_memory_utilization = 0.75
        config.training.batch_size = 1
        config.training.num_rollouts = 4
        config.model.max_seq_length = 2048  # Reduced for memory
        return config
    
    @classmethod
    def for_small_model(cls) -> "SRLConfig":
        """Configuration using smaller 3B model for testing with max utilization."""
        config = cls.for_rtx_5060()
        config.model.model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
        config.model.lora_rank = 16
        config.model.lora_alpha = 16
        
        # Disable format check for testing - model outputs won't match strict regex
        config.reward.format_check = False
        
        # MAXIMIZE UTILIZATION:
        # 1. Disable variance filter - keep ALL samples for training
        config.sampling.enabled = False
        config.sampling.variance_threshold = 0.0
        
        # 2. Increase batch size for better GPU utilization during training
        config.training.batch_size = 2  # 2x more work per step
        
        # 3. Increase rollouts for more reward computation parallelism
        config.training.num_rollouts = 8  # 8 rollouts per prompt = more CPU work
        
        return config


# Default configuration for RTX 5060
DEFAULT_CONFIG = SRLConfig.for_rtx_5060()
