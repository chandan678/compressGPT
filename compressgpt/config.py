"""
Training configuration classes for CompressGPT.

This module provides dataclasses for configuring LoRA, QLoRA, and general
training parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoraConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation) fine-tuning.
    
    Attributes:
        r: LoRA rank (dimensionality of the low-rank matrices)
        lora_alpha: LoRA scaling factor (typically 2*r)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Bias training strategy ("none", "all", or "lora_only")
        task_type: Task type for PEFT ("CAUSAL_LM" for text generation)
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.r <= 0:
            raise ValueError(f"r must be positive, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"bias must be 'none', 'all', or 'lora_only', got {self.bias}")


@dataclass
class QLoraConfig(LoraConfig):
    """
    Configuration for QLoRA (Quantized LoRA) fine-tuning.
    
    Extends LoraConfig with quantization parameters.
    
    Attributes:
        train_bits: Quantization bits for training (4 or 8)
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4")
        bnb_4bit_use_double_quant: Use double quantization for better quality
    """
    train_bits: int = 4
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()
        
        if self.train_bits not in [4, 8]:
            raise ValueError(
                f"train_bits must be 4 or 8 for QLoRA, got {self.train_bits}"
            )
        
        if self.train_bits == 4:
            if self.bnb_4bit_quant_type not in ["nf4", "fp4"]:
                raise ValueError(
                    f"bnb_4bit_quant_type must be 'nf4' or 'fp4', got {self.bnb_4bit_quant_type}"
                )


@dataclass
class TrainingConfig:
    """
    General training configuration for fine-tuning.
    
    Attributes:
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        warmup_ratio: Ratio of training steps for learning rate warmup
        lr_scheduler_type: Type of learning rate scheduler
        weight_decay: Weight decay for regularization
        max_seq_length: Maximum sequence length
        logging_steps: Log metrics every N steps
        eval_strategy: Evaluation strategy ("epoch", "steps", or "no")
        save_strategy: Save checkpoint strategy ("epoch", "steps", or "no")
        save_total_limit: Maximum number of checkpoints to keep
        load_best_model_at_end: Load best model at end of training
        metric_for_best_model: Metric to use for selecting best model
        greater_is_better: Whether higher metric value is better
        fp16: Use FP16 mixed precision
        bf16: Use BF16 mixed precision
        report_to: Where to report metrics ("wandb", "tensorboard", "none")
        run_name: Name for the training run
        early_stopping_patience: Patience for early stopping (0 = disabled)
        early_stopping_threshold: Minimum improvement threshold for early stopping
    """
    num_train_epochs: int = 6
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_seq_length: int = 256
    logging_steps: int = 20
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: Optional[int] = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True
    fp16: bool = True
    bf16: bool = False
    report_to: str = "none"
    run_name: Optional[str] = None
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.0
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_train_epochs <= 0:
            raise ValueError(f"num_train_epochs must be positive, got {self.num_train_epochs}")
        
        if self.per_device_train_batch_size <= 0:
            raise ValueError(f"per_device_train_batch_size must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        
        if self.eval_strategy not in ["epoch", "steps", "no"]:
            raise ValueError(f"eval_strategy must be 'epoch', 'steps', or 'no'")
        
        if self.save_strategy not in ["epoch", "steps", "no"]:
            raise ValueError(f"save_strategy must be 'epoch', 'steps', or 'no'")
        
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16 - choose one")
        
        if self.report_to not in ["wandb", "tensorboard", "none", "all"]:
            raise ValueError(f"report_to must be 'wandb', 'tensorboard', 'none', or 'all'")


@dataclass
class PipelineConfig:
    """
    Configuration for the complete training pipeline.
    
    Attributes:
        stages: List of stages to run ("ft", "qlora", "merge")
        skip_if_exists: Skip stage if output directory exists
        output_dir: Root output directory for all stages
        ft_output_dir: Directory for FT LoRA adapters
        qlora_output_dir: Directory for QLoRA adapters
        merged_output_dir: Directory for merged model
    """
    stages: List[str] = field(default_factory=lambda: ["ft", "qlora", "merge"])
    skip_if_exists: bool = False
    output_dir: str = "./output"
    ft_output_dir: Optional[str] = None
    qlora_output_dir: Optional[str] = None
    merged_output_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set default subdirectories."""
        valid_stages = ["ft", "qlora", "merge"]
        for stage in self.stages:
            if stage not in valid_stages:
                raise ValueError(
                    f"Invalid stage '{stage}'. Valid stages: {valid_stages}"
                )
        
        # Set default subdirectories if not provided
        if self.ft_output_dir is None:
            self.ft_output_dir = f"{self.output_dir}/ft_lora_adapter"
        
        if self.qlora_output_dir is None:
            self.qlora_output_dir = f"{self.output_dir}/qlora_adapter"
        
        if self.merged_output_dir is None:
            self.merged_output_dir = f"{self.output_dir}/merged_model"
