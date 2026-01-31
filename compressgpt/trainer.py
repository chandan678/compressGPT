"""
CompressGPT Trainer - Model Compression Pipeline

Orchestrates model compression workflow: FT -> Quantize -> Recovery -> Merge
with automatic metadata extraction from DatasetBuilder.
"""

import os
import logging
import warnings
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig as PeftLoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from .config import LoraConfig, QLoraConfig, TrainingConfig, QuantizationConfig, DeploymentConfig
from .compute_metrics import ComputeMetrics
from .label_space import LabelSpace
from .utils import (
    validate_response_template,
    setup_data_collator,
    clear_gpu_memory,
    save_metrics,
    format_metrics_table
)


logger = logging.getLogger(__name__)


class CompressTrainer:
    """
    CompressGPT Trainer - Model Compression Pipeline.
    
    Orchestrates compression-focused workflow:
    1. FT: Fine-tune on FP16 base (establish accuracy baseline)
    2. Compress: Atomic stage that Quantizes -> Trains Recovery -> Merges to FP16
    3. Deploy: Convert final model to production formats (GGUF, Quantized, etc.)
    
    Valid stage combinations:
    - ["ft", "deploy"]: Basic LoRA fine-tuning + deployment
    - ["ft", "compress_8bit", "deploy"]: Full 8-bit compression pipeline
    - ["ft", "compress_4bit", "deploy"]: Full 4-bit compression pipeline
    
    Example:
        from compressgpt import DatasetBuilder, CompressTrainer
        
        builder = DatasetBuilder(...).build()
        
        # Full compression pipeline
        trainer = CompressTrainer(
            model_id="meta-llama/Llama-3.2-1B",
            dataset_builder=builder,
            stages=["ft", "compress_4bit", "deploy"]
        )
        results = trainer.run()
    """
    
    def __init__(
        self,
        model_id: str,
        dataset_builder,  # DatasetBuilder instance
        *,
        stages: List[str],
        ft_config: Optional[LoraConfig] = None,
        recovery_config: Optional[QLoraConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        compress_training_config: Optional[TrainingConfig] = None,
        quant_config_8bit: Optional[QuantizationConfig] = None,
        quant_config_4bit: Optional[QuantizationConfig] = None,
        deployment_config: Optional[DeploymentConfig] = None,
        run_dir: str = "runs/default",
        resume: bool = True,
        hf_token: Optional[str] = None,
        train_test_split: float = 0.05,
        seed: int = 42,
    ):
        """
        Initialize CompressTrainer.
        
        Args:
            model_id: HuggingFace model ID or local path
            dataset_builder: DatasetBuilder instance (must have called build())
            stages: List of stages (e.g., ["ft", "compress_8bit", "merge", "deploy"])
            ft_config: LoRA configuration for FT stage
            recovery_config: Recovery configuration (includes train_bits for quantization)
            training_config: Training configuration for FT stage (also used for compress if compress_training_config not provided)
            compress_training_config: Optional separate training configuration for compress stages.
                If not provided, uses training_config. Useful for different epochs/LR for recovery training.
            quant_config_8bit: Quantization config for 8-bit compression (default: auto)
            quant_config_4bit: Quantization config for 4-bit compression (default: auto)
            deployment_config: Deployment output formats (default: merged FP16 only)
            run_dir: Base directory for outputs
            resume: If True, skip existing stages
            hf_token: HuggingFace token for gated models
            train_test_split: Test split ratio
            seed: Random seed for reproducibility
        
        Raises:
            RuntimeError: If dataset_builder.build() has not been called
            ValueError: If invalid stages provided
        """
        self.model_id = model_id
        self.dataset_builder = dataset_builder
        self.stages = stages
        self.run_dir = run_dir
        self.resume = resume
        self.hf_token = hf_token
        self.train_test_split = train_test_split
        self.seed = seed
        
        # Validate stages (support both old and new names)
        valid_stages = {
            "ft", 
            "compress_8bit", "compress_4bit",
            "merge", "deploy"
        }
        invalid = set(stages) - valid_stages
        if invalid:
            raise ValueError(f"Invalid stages: {invalid}. Must be from {valid_stages}")
        
        # Warn about deprecated stage names
        deprecated_stages = {"quantize_8bit", "quantize_4bit", "recovery"}
        if any(s in deprecated_stages for s in stages):
            logger.warning(
                "Using deprecated stage names. Please migrate to new atomic stages:\n"
                "  Old: ['ft', 'quantize_8bit', 'recovery', 'merge']\n"
                "  New: ['ft', 'compress_8bit', 'deploy']\n"
                "  The compress_* stages now include recovery and merge automatically."
            )
        
        logger.info("=" * 60)
        logger.info("CompressGPT Trainer - Initializing")
        logger.info("=" * 60)
        logger.info(f"Model: {model_id}")
        logger.info(f"Stages: {stages}")
        logger.info(f"Run directory: {run_dir}")
        
        # Extract metadata from dataset_builder
        try:
            self.metadata = dataset_builder.metadata
        except RuntimeError:
            raise RuntimeError(
                "DatasetBuilder has not been built yet. "
                "Call dataset_builder.build() before creating trainer."
            )
        
        # Extract dataset
        self.dataset = dataset_builder.dataset
        
        # Extract label_space from metadata
        label_space_dict = self.metadata.get("label_space")
        if label_space_dict is None:
            raise ValueError("Metadata missing 'label_space' field")
        
        # Load tokenizer (must match model_id)
        logger.info(f"Loading tokenizer from: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        # Set padding token if not present (required for batch training)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        # Reconstruct LabelSpace
        self.label_space = LabelSpace.from_dict(label_space_dict, self.tokenizer)
        
        # Extract response_trigger
        self.response_trigger = self.metadata.get("response_trigger")
        if not self.response_trigger:
            raise ValueError("Metadata missing 'response_trigger' field")
        
        logger.info(f"Response trigger: {self.response_trigger!r}")
        logger.info(f"LabelSpace: {len(self.label_space.labels)} labels")
        logger.info(f"  {self.label_space.labels}")
        
        # Initialize configs with defaults
        self.ft_config = ft_config or LoraConfig()
        self.recovery_config = recovery_config or QLoraConfig()
        self.training_config = training_config or TrainingConfig()
        # Compress training config defaults to training_config if not provided
        self.compress_training_config = compress_training_config or self.training_config
        
        # Quantization configs (create defaults if not provided)
        self.quant_config_8bit = quant_config_8bit or QuantizationConfig(
            bits=8,
            quant_type="int8",
            compute_dtype="float16"
        )
        self.quant_config_4bit = quant_config_4bit or QuantizationConfig(
            bits=4,
            quant_type="nf4",
            compute_dtype="float16",
            use_double_quant=True
        )
        
        # Deployment config
        self.deployment_config = deployment_config or DeploymentConfig()
        
        # Create output directories
        os.makedirs(run_dir, exist_ok=True)
        self.ft_output_dir = os.path.join(run_dir, "ft_adapter")
        self.quantized_8bit_dir = os.path.join(run_dir, "quantized_8bit")
        self.quantized_4bit_dir = os.path.join(run_dir, "quantized_4bit")
        self.recovery_output_dir = os.path.join(run_dir, "recovery_adapter")
        self.merged_output_dir = os.path.join(run_dir, "merged_model")
        
        # Detect device and warn about limitations
        self.device_type = self._detect_device()
        self._warn_device_limitations()
        
        # Setup metrics computer with label-restricted argmax
        self.metrics_computer = ComputeMetrics(
            labels=self.label_space.labels,
            valid_token_ids=self.label_space.valid_token_ids,
            id_to_label=self.label_space.id_to_label,
            tokenizer=self.tokenizer
        )
        
        # Split dataset
        self._split_dataset()
        
        # Store results
        self.results = {}
        
        logger.info("CompressTrainer initialized")
        logger.info("=" * 60 + "\n")
    
    def _detect_device(self) -> str:
        """Detect the device type (cuda/mps/cpu)."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_base_model_path(self, model_id: str) -> tuple[str, bool]:
        """
        Check if model_id is a LoRA adapter directory and extract base model path.
        
        Returns:
            (base_model_path, is_adapter): Path to base model and whether input was adapter
        """
        adapter_config_path = os.path.join(model_id, "adapter_config.json") if os.path.exists(model_id) else None
        
        if adapter_config_path and os.path.exists(adapter_config_path):
            # This is a LoRA adapter directory
            try:
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model = adapter_config.get("base_model_name_or_path", model_id)
                logger.info(f"Detected LoRA adapter. Base model: {base_model}")
                return base_model, True
            except Exception as e:
                logger.warning(f"Failed to read adapter_config.json: {e}")
                return model_id, False
        
        return model_id, False
    
    def _warn_device_limitations(self):
        """Warn about device-specific limitations."""
        if self.device_type == "mps":
            # Check if quantization/recovery stages are enabled
            if any(stage in self.stages for stage in ["quantize_8bit", "quantize_4bit", "recovery"]):
                warnings.warn(
                    "Apple Silicon (MPS) detected with quantization/recovery stages.\n"
                    "BitsAndBytes quantization is NOT supported on MPS.\n"
                    "Training will fail. Consider:\n"
                    "  1. Use only 'ft' stage: stages=['ft', 'merge']\n"
                    "  2. Train on a CUDA GPU\n"
                    "  3. Use CPU (slow): set PYTORCH_ENABLE_MPS_FALLBACK=1",
                    RuntimeWarning,
                    stacklevel=2
                )
            logger.info(f"Device: {self.device_type.upper()} (Apple Silicon)")
        elif self.device_type == "cpu":
            warnings.warn(
                "No GPU detected. Training will be extremely slow on CPU.",
                RuntimeWarning,
                stacklevel=2
            )
        else:
            logger.info(f"Device: {self.device_type.upper()}")
    
    def _split_dataset(self):
        """Split dataset into train and validation sets."""
        # Clean dataset - keep only 'text' column for SFTTrainer
        required_cols = ['text']
        extra_cols = [col for col in self.dataset.column_names if col not in required_cols]
        if extra_cols:
            logger.info(f"Removing extra columns from dataset: {extra_cols}")
            self.dataset = self.dataset.remove_columns(extra_cols)
        
        split = self.dataset.train_test_split(
            test_size=self.train_test_split,
            seed=self.seed
        )
        self.train_dataset = split["train"]
        self.eval_dataset = split["test"]
        
        logger.info(f"Dataset split - Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")
    
    def _apply_quantization(
        self,
        model_path: str,
        bits: int,
        for_training: bool = False
    ) -> AutoModelForCausalLM:
        """
        Apply quantization to a model using shared quantization configuration.
        
        This ensures quantization parameters match between training and deployment.
        
        Args:
            model_path: Path to model to quantize
            bits: Quantization bits (4 or 8)
            for_training: If True, use settings optimized for training (e.g., gradient checkpointing)
        
        Returns:
            Quantized model
        """
        logger.info(f"Applying {bits}-bit quantization to {model_path}")
        
        # Get quantization config
        quant_config = self.quant_config_8bit if bits == 8 else self.quant_config_4bit
        bnb_config = quant_config.to_bnb_config()
        
        # Load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.hf_token,
        )
        
        if for_training:
            # Enable gradient checkpointing for memory efficiency
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            logger.info("Gradient checkpointing enabled for training")
        
        logger.info(f"Model quantized to {bits}-bit")
        return model
    
    def _compress_atomic(self, bits: int) -> Dict:
        """
        Atomic compress stage: Quantize -> Train Recovery -> Merge to FP16.
        
        This replaces the separate quantize + recovery + merge stages.
        The result is always a merged FP16 model (canonical format).
        
        Args:
            bits: Quantization bits (4 or 8)
        
        Returns:
            Result dictionary with metrics and paths
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"Stage: Compress to {bits}-bit (Atomic)")
        logger.info("=" * 60)
        logger.info("Steps: 1) Quantize -> 2) Train Recovery -> 3) Merge to FP16")
        
        # Check device compatibility
        if self.device_type == "mps":
            raise RuntimeError(
                f"{bits}-bit quantization is not supported on Apple Silicon (MPS).\n"
                "BitsAndBytes quantization requires CUDA.\n\n"
                "Solutions:\n"
                "  1. Use 'ft' stage only\n"
                "  2. Train on a CUDA GPU"
            )
        
        # Output directories
        recovery_dir = os.path.join(self.run_dir, f"compress_{bits}bit_adapter")
        merged_dir = os.path.join(self.run_dir, f"compress_{bits}bit_merged")
        
        # Skip if merged output exists
        if self.resume and os.path.exists(merged_dir):
            logger.info(f"⏭️  Skipping compress_{bits}bit - output exists: {merged_dir}")
            return {
                "status": "skipped",
                "recovery_dir": recovery_dir,
                "merged_dir": merged_dir,
                "bits": bits
            }
        
        os.makedirs(recovery_dir, exist_ok=True)
        os.makedirs(merged_dir, exist_ok=True)
        
        # Determine base model path
        base_model_path, is_adapter = self._get_base_model_path(self.model_id)
        if is_adapter:
            logger.info(f"Using base model from adapter config: {base_model_path}")
        
        # Check if we should use FT checkpoint or base model
        if os.path.exists(self.ft_output_dir):
            logger.info(f"Using FT checkpoint as starting point")
            # Load FT adapter and merge to get FP16 base for quantization
            logger.info(f"Loading base model: {base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                token=self.hf_token
            )
            logger.info(f"Loading FT adapter: {self.ft_output_dir}")
            ft_model = PeftModel.from_pretrained(base_model, self.ft_output_dir)
            logger.info("Merging FT adapter to FP16...")
            merged_ft = ft_model.merge_and_unload()
            
            # Save merged FT temporarily for quantization
            temp_merged_path = os.path.join(self.run_dir, "temp_ft_merged")
            merged_ft.save_pretrained(temp_merged_path)
            self.tokenizer.save_pretrained(temp_merged_path)
            logger.info(f"FT merged model saved to {temp_merged_path}")
            
            # Clear memory
            del base_model, ft_model, merged_ft
            clear_gpu_memory()
            
            # Now quantize the merged FT model
            quantize_base_path = temp_merged_path
        else:
            logger.info(f"Using base model (no FT checkpoint found)")
            quantize_base_path = base_model_path
        
        # Load quantized model for recovery training
        logger.info(f"\n--- Train Recovery on {bits}-bit Quantized Model ---")
        model = self._apply_quantization(
            model_path=quantize_base_path,
            bits=bits,
            for_training=True
        )
        
        # Prepare model for PEFT
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA for recovery
        peft_config = PeftLoraConfig(
            r=self.recovery_config.r,
            lora_alpha=self.recovery_config.lora_alpha,
            lora_dropout=self.recovery_config.lora_dropout,
            target_modules=self.recovery_config.target_modules,
            bias=self.recovery_config.bias,
            task_type=self.recovery_config.task_type,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Setup data collator
        data_collator = setup_data_collator(
            self.tokenizer,
            self.response_trigger
        )
        
        # Setup metrics
        compute_metrics_fn = ComputeMetrics(
            labels=self.label_space.labels,
            valid_token_ids=self.label_space.valid_token_ids,
            id_to_label=self.label_space.id_to_label,
            tokenizer=self.tokenizer
        ).as_trainer_callback(log_first_n=5)
        
        # Training arguments - use compress_training_config for recovery
        run_name = f"compress_{bits}bit_recovery"
        training_args = SFTConfig(
            output_dir=recovery_dir,
            num_train_epochs=self.compress_training_config.num_train_epochs,
            per_device_train_batch_size=self.compress_training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.compress_training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.compress_training_config.gradient_accumulation_steps,
            learning_rate=self.compress_training_config.learning_rate,
            warmup_ratio=self.compress_training_config.warmup_ratio,
            lr_scheduler_type=self.compress_training_config.lr_scheduler_type,
            weight_decay=self.compress_training_config.weight_decay,
            max_length=self.compress_training_config.max_seq_length,
            logging_steps=self.compress_training_config.logging_steps,
            eval_strategy=self.compress_training_config.eval_strategy,
            save_strategy=self.compress_training_config.save_strategy,
            save_total_limit=self.compress_training_config.save_total_limit,
            load_best_model_at_end=self.compress_training_config.load_best_model_at_end,
            metric_for_best_model=self.compress_training_config.metric_for_best_model,
            greater_is_better=self.compress_training_config.greater_is_better,
            fp16=self.compress_training_config.fp16,
            bf16=self.compress_training_config.bf16,
            report_to=self.compress_training_config.report_to,
            run_name=run_name,
            dataset_text_field="text",
            eval_accumulation_steps=self.compress_training_config.eval_accumulation_steps,
        )
        
        # Train recovery adapter
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            preprocess_logits_for_metrics=self.metrics_computer.get_preprocess_logits(),
            callbacks=([EarlyStoppingCallback(
                early_stopping_patience=self.compress_training_config.early_stopping_patience,
                early_stopping_threshold=self.compress_training_config.early_stopping_threshold,
            )] if self.compress_training_config.early_stopping_patience > 0 else []),
        )
        
        logger.info("🚀 Starting recovery training...")
        trainer.train()
        
        # Evaluate
        logger.info("📊 Evaluating recovery adapter...")
        metrics = trainer.evaluate()
        logger.info(format_metrics_table(metrics, f"Compress {bits}-bit Recovery"))
        
        # Save recovery adapter
        trainer.model.save_pretrained(recovery_dir)
        self.tokenizer.save_pretrained(recovery_dir)
        logger.info(f"Recovery adapter saved to {recovery_dir}")
        
        # Merge recovery adapter back to FP16
        logger.info(f"\n--- Merge Recovery Adapter to FP16 ---")
        logger.info("Loading base model in FP16...")
        base_model = AutoModelForCausalLM.from_pretrained(
            quantize_base_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.hf_token
        )
        
        logger.info(f"Loading recovery adapter: {recovery_dir}")
        peft_model = PeftModel.from_pretrained(base_model, recovery_dir)
        
        logger.info("Merging adapter to FP16...")
        merged_model = peft_model.merge_and_unload()
        
        logger.info(f"Saving merged model to {merged_dir}")
        merged_model.save_pretrained(merged_dir)
        self.tokenizer.save_pretrained(merged_dir)
        
        logger.info(f"Compress {bits}-bit stage complete!")
        logger.info(f" Recovery adapter: {recovery_dir}")
        logger.info(f" Merged FP16 model: {merged_dir}")
        
        return {
            "status": "success",
            "recovery_dir": recovery_dir,
            "merged_dir": merged_dir,
            "bits": bits,
            "metrics": metrics
        }
    
    def run(self) -> Dict:
        """
        Run the complete training pipeline.
        
        Supports both old and new stage names:
        - Old: ["ft", "quantize_8bit", "recovery", "merge"]
        - New: ["ft", "compress_8bit", "merge", "deploy"]
        
        Returns:
            Dictionary with results from each stage
        """
        logger.info("=" * 60)
        logger.info("Starting CompressGPT Training Pipeline")
        logger.info("=" * 60)
        
        # FT Stage
        if "ft" in self.stages:
            self.results["ft"] = self._train_stage_ft()
            clear_gpu_memory()
        
        # New atomic compress stages (preferred)
        if "compress_8bit" in self.stages:
            self.results["compress_8bit"] = self._compress_atomic(bits=8)
            clear_gpu_memory()
        
        if "compress_4bit" in self.stages:
            self.results["compress_4bit"] = self._compress_atomic(bits=4)
            clear_gpu_memory()
        
        # Merge stage
        if "merge" in self.stages:
            self.results["merge"] = self._merge_and_save()
            clear_gpu_memory()
        
        # Deploy stage (new)
        if "deploy" in self.stages:
            self.results["deploy"] = self._deploy_model()
        
        # Save all metrics
        metrics_path = os.path.join(self.run_dir, "metrics.json")
        save_metrics(self.results, metrics_path)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _train_stage_ft(self) -> Dict:
        """Train FT stage: LoRA on fp16/bf16 base model."""
        logger.info("\n" + "=" * 60)
        logger.info("Stage 1: FT (LoRA on FP16/BF16 base)")
        logger.info("=" * 60)
        
        output_dir = self.ft_output_dir
        
        # Skip if exists
        if self.resume and os.path.exists(output_dir):
            logger.info(f"⏭️  Skipping FT stage - output exists: {output_dir}")
            return {"status": "skipped", "output_dir": output_dir}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base model path if model_id is a LoRA adapter
        base_model_path, is_adapter = self._get_base_model_path(self.model_id)
        if is_adapter:
            logger.info(f"Using base model from adapter config: {base_model_path}")
        
        # Load base model
        logger.info(f"Loading base model: {base_model_path}")
        dtype = torch.bfloat16 if self.training_config.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            token=self.hf_token,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Log actual device placement
        if hasattr(model, 'hf_device_map'):
            logger.info(f"Device map: {model.hf_device_map}")
        else:
            model_device = next(model.parameters()).device
            logger.info(f"Model loaded on: {model_device}")
        
        model.gradient_checkpointing_enable()
        
        # Log model size
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded {num_params/1e9:.2f}B parameters")
        if torch.cuda.is_available():
            logger.info(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            logger.info(f"MPS memory: {torch.mps.current_allocated_memory()/1e9:.2f} GB")
        
        # Setup PEFT
        peft_config = PeftLoraConfig(
            r=self.ft_config.r,
            lora_alpha=self.ft_config.lora_alpha,
            lora_dropout=self.ft_config.lora_dropout,
            target_modules=self.ft_config.target_modules,
            bias=self.ft_config.bias,
            task_type=self.ft_config.task_type
        )
        
        # Setup data collator
        data_collator = setup_data_collator(self.tokenizer, self.response_trigger)
        
        # Setup training args
        run_name = self.training_config.run_name or "compressgpt_ft"
        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            weight_decay=self.training_config.weight_decay,
            max_length=self.training_config.max_seq_length,
            logging_steps=self.training_config.logging_steps,
            eval_strategy="steps",  # Eval during training for progress visibility
            eval_steps=self.training_config.eval_steps or 500,  # Default 500 steps if not specified
            save_strategy="steps",  # Must match eval_strategy when load_best_model_at_end=True
            save_steps=self.training_config.save_steps or 500,  # Default 500 steps if not specified
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            report_to=self.training_config.report_to,
            run_name=run_name,
            dataset_text_field="text",
            eval_accumulation_steps=self.training_config.eval_accumulation_steps,
        )
        
        # Setup trainer
        callbacks = []
        if self.training_config.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.training_config.early_stopping_patience,
                early_stopping_threshold=self.training_config.early_stopping_threshold
            ))
        
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            peft_config=peft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.metrics_computer.as_trainer_callback(),
            preprocess_logits_for_metrics=self.metrics_computer.get_preprocess_logits(),
            callbacks=callbacks
        )
        
        # Train
        import time
        num_samples = len(self.train_dataset)
        eff_bs = self.training_config.per_device_train_batch_size * self.training_config.gradient_accumulation_steps
        logger.info(f"Training: {num_samples} samples, {self.training_config.num_train_epochs} epochs, batch_size={eff_bs}")
        
        start_time = time.time()
        trainer.train()
        duration = time.time() - start_time
        logger.info(f"Training completed in {duration/60:.1f} minutes")
        
        # Evaluate
        logger.info("Evaluating...")
        metrics = trainer.evaluate()
        
        # Clear memory after eval (critical for MPS)
        if self.device_type in ["mps", "cuda"]:
            clear_gpu_memory()
        
        # Save model
        logger.info(f"Saving FT adapter to {output_dir}")
        trainer.save_model(output_dir)
        
        # Print metrics
        print(format_metrics_table(metrics, "FT Stage"))
        
        return {
            "status": "completed",
            "output_dir": output_dir,
            "metrics": metrics
        }
    
    def _quantize_model(self, bits: int) -> Dict:
        """
        Quantize the last trained checkpoint to 8-bit or 4-bit.
        
        Args:
            bits: 8 or 4 for quantization bits
        
        Returns:
            Result dictionary with status and output path
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"Stage: Quantize to {bits}-bit")
        logger.info("=" * 60)
        
        # Check device compatibility
        if self.device_type == "mps":
            raise RuntimeError(
                f"{bits}-bit quantization is not supported on Apple Silicon (MPS).\n"
                "BitsAndBytes quantization requires CUDA.\n\n"
                "Solutions:\n"
                "  1. Skip quantization stages\n"
                "  2. Train on a CUDA GPU"
            )
        
        output_dir = self.quantized_8bit_dir if bits == 8 else self.quantized_4bit_dir
        
        # Skip if exists
        if self.resume and os.path.exists(output_dir):
            logger.info(f"⏭️  Skipping quantize_{bits}bit stage - output exists: {output_dir}")
            return {"status": "skipped", "output_dir": output_dir, "bits": bits}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base model path
        base_model_path, is_adapter = self._get_base_model_path(self.model_id)
        if is_adapter:
            logger.info(f"⚠️  Using base model from adapter config: {base_model_path}")
        
        # Determine which checkpoint to quantize (FT if available, else base model)
        if os.path.exists(self.ft_output_dir):
            logger.info(f"Quantizing FT adapter + base model to {bits}-bit")
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                token=self.hf_token
            )
            # Load and merge FT adapter
            model = PeftModel.from_pretrained(base_model, self.ft_output_dir)
            model = model.merge_and_unload()
        else:
            logger.info(f"Quantizing base model to {bits}-bit (no FT checkpoint found)")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                token=self.hf_token
            )
        
        # Log device placement
        if hasattr(model, 'hf_device_map'):
            logger.info(f"Device map: {model.hf_device_map}")
        else:
            model_device = next(model.parameters()).device
            logger.info(f"Model loaded on: {model_device}")
        
        # Setup quantization config
        logger.info(f"Applying {bits}-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=self.recovery_config.bnb_4bit_quant_type if bits == 4 else "fp4",
            bnb_4bit_use_double_quant=self.recovery_config.bnb_4bit_use_double_quant if bits == 4 else False,
        )
        
        # Reload with quantization (have to reload, can't quantize in-place)
        del model
        clear_gpu_memory()
        
        # Quantize by loading with quantization config
        quantized_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.hf_token
        )
        
        # If FT exists, load adapter on quantized base
        if os.path.exists(self.ft_output_dir):
            logger.info("Loading FT adapter on quantized base")
            quantized_model = PeftModel.from_pretrained(
                quantized_model,
                self.ft_output_dir,
                is_trainable=False
            )
            # Merge adapter
            quantized_model = quantized_model.merge_and_unload()
        
        # Save quantized model
        logger.info(f"Saving {bits}-bit quantized model to {output_dir}")
        quantized_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"{bits}-bit quantization complete")
        
        return {
            "status": "completed",
            "output_dir": output_dir,
            "bits": bits
        }
    
    def _train_stage_recovery(self) -> Dict:
        """
        Train Recovery stage: LoRA on quantized base to compensate quantization error.
        Uses full training epochs (same as FT) to recover accuracy lost during quantization.
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"Stage: Recovery Training ({self.recovery_config.train_bits}-bit)")
        logger.info("=" * 60)
        
        # Check device compatibility
        if self.device_type == "mps":
            raise RuntimeError(
                "Recovery training is not supported on Apple Silicon (MPS).\n"
                "BitsAndBytes quantization requires CUDA.\n\n"
                "Solutions:\n"
                "  1. Use only 'ft' stage: stages=['ft', 'merge']\n"
                "  2. Train on a CUDA GPU\n"
                "  3. Use CPU (slow): set PYTORCH_ENABLE_MPS_FALLBACK=1 and device_map='cpu'"
            )
        
        output_dir = self.recovery_output_dir
        
        # Skip if exists
        if self.resume and os.path.exists(output_dir):
            logger.info(f"⏭️  Skipping Recovery stage - output exists: {output_dir}")
            return {"status": "skipped", "output_dir": output_dir}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(self.recovery_config.train_bits == 4),
            load_in_8bit=(self.recovery_config.train_bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=self.recovery_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.recovery_config.bnb_4bit_use_double_quant,
        )
        
        # Determine which checkpoint to recover from
        # Priority: quantized checkpoint > FT checkpoint > base model
        if self.recovery_config.train_bits == 8 and os.path.exists(self.quantized_8bit_dir):
            model_path = self.quantized_8bit_dir
            logger.info(f"Recovering from 8-bit quantized checkpoint: {model_path}")
        elif self.recovery_config.train_bits == 4 and os.path.exists(self.quantized_4bit_dir):
            model_path = self.quantized_4bit_dir
            logger.info(f"Recovering from 4-bit quantized checkpoint: {model_path}")
        else:
            # Extract base model path if model_id is a LoRA adapter
            base_model_path, is_adapter = self._get_base_model_path(self.model_id)
            if is_adapter:
                logger.info(f"⚠️  Using base model from adapter config: {base_model_path}")
                logger.info(f"   Note: Passed adapter will be ignored, loading base model fresh for quantization")
            model_path = base_model_path
            logger.info(f"Recovering from base model (no quantized checkpoint): {model_path}")
        
        # Load quantized base model
        logger.info(f"Loading {self.recovery_config.train_bits}-bit quantized model...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                token=self.hf_token
            )
            
            # Log actual device placement
            if hasattr(base_model, 'hf_device_map'):
                logger.info(f"📍 Device map: {base_model.hf_device_map}")
            else:
                model_device = next(base_model.parameters()).device
                logger.info(f"📍 Quantized model loaded on: {model_device}")
                
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                raise RuntimeError(
                    f"Failed to load quantized model: {e}\n\n"
                    "BitsAndBytes quantization requires CUDA GPU.\n"
                    f"Current device: {self.device_type}\n\n"
                    "If on Apple Silicon, use stages=['ft', 'merge'] (skip recovery)"
                ) from e
            raise
        
        # Apply fresh LoRA config for recovery
        peft_config = PeftLoraConfig(
            r=self.recovery_config.r,
            lora_alpha=self.recovery_config.lora_alpha,
            lora_dropout=self.recovery_config.lora_dropout,
            target_modules=self.recovery_config.target_modules,
            bias=self.recovery_config.bias,
            task_type=self.recovery_config.task_type
        )
        model = get_peft_model(base_model, peft_config)
        
        # Setup data collator
        data_collator = setup_data_collator(self.tokenizer, self.response_trigger)
        
        # Use full training epochs (same as FT) to recover accuracy
        run_name = self.training_config.run_name or "compressgpt_recovery"
        
        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.training_config.num_train_epochs,  # Full epochs
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,  # Same LR as FT
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            weight_decay=self.training_config.weight_decay,
            max_length=self.training_config.max_seq_length,
            logging_steps=self.training_config.logging_steps,
            eval_strategy=self.training_config.eval_strategy,
            eval_steps=self.training_config.eval_steps if self.training_config.eval_strategy == "steps" else None,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps if self.training_config.save_strategy == "steps" else None,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            fp16=self.training_config.fp16,
            bf16=False,  # Recovery uses fp16 for compatibility with quantized base
            report_to=self.training_config.report_to,
            run_name=run_name,
            dataset_text_field="text",
            eval_accumulation_steps=self.training_config.eval_accumulation_steps,
        )
        
        # Setup trainer
        callbacks = []
        if self.training_config.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.training_config.early_stopping_patience,
                early_stopping_threshold=self.training_config.early_stopping_threshold
            ))
        
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.metrics_computer.as_trainer_callback(),
            preprocess_logits_for_metrics=self.metrics_computer.get_preprocess_logits(),
            callbacks=callbacks
        )
        
        # Train
        import time
        num_samples = len(self.train_dataset)
        eff_bs = self.training_config.per_device_train_batch_size * self.training_config.gradient_accumulation_steps
        logger.info(f"🚀 Recovery Training: {num_samples} samples, {self.training_config.num_train_epochs} epochs, batch_size={eff_bs}")
        
        start_time = time.time()
        trainer.train()
        duration = time.time() - start_time
        logger.info(f"✓ Recovery training completed in {duration/60:.1f} minutes")
        
        # Evaluate
        if self.eval_dataset:
            logger.info("📊 Evaluating...")
            metrics = trainer.evaluate()
        else:
            metrics = {}
        
        # Save model
        logger.info(f"💾 Saving Recovery adapter to {output_dir}")
        trainer.save_model(output_dir)
        
        # Print metrics
        if metrics:
            print(format_metrics_table(metrics, "Recovery Stage"))
        
        return {
            "status": "completed",
            "output_dir": output_dir,
            "metrics": metrics,
            "train_bits": self.recovery_config.train_bits
        }
    
    def _merge_and_save(self) -> Dict:
        """
        Merge LoRA adapters into base model and save as FP16.
        Automatically detects which adapter to merge (recovery, FT, or quantized).
        """
        logger.info("\n" + "=" * 60)
        logger.info("Stage: Merge and Save")
        logger.info("=" * 60)
        
        output_dir = self.merged_output_dir
        
        # Skip if exists
        if self.resume and os.path.exists(output_dir):
            logger.info(f"⏭️  Skipping merge stage - output exists: {output_dir}")
            return {"status": "skipped", "output_dir": output_dir}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which checkpoint to merge (priority: recovery > quantized > FT)
        adapter_path = None
        base_model_path = self.model_id
        merge_source = None
        
        if os.path.exists(self.recovery_output_dir):
            adapter_path = self.recovery_output_dir
            merge_source = "recovery"
            # Load quantized base for merging
            if self.recovery_config.train_bits == 8 and os.path.exists(self.quantized_8bit_dir):
                base_model_path = self.quantized_8bit_dir
            elif self.recovery_config.train_bits == 4 and os.path.exists(self.quantized_4bit_dir):
                base_model_path = self.quantized_4bit_dir
            logger.info(f"Merging Recovery adapter: {adapter_path}")
        elif os.path.exists(self.quantized_8bit_dir):
            # No adapter to merge, just convert quantized to FP16
            base_model_path = self.quantized_8bit_dir
            merge_source = "quantized_8bit"
            logger.info(f"Converting 8-bit quantized model to FP16: {base_model_path}")
        elif os.path.exists(self.quantized_4bit_dir):
            # No adapter to merge, just convert quantized to FP16
            base_model_path = self.quantized_4bit_dir
            merge_source = "quantized_4bit"
            logger.info(f"Converting 4-bit quantized model to FP16: {base_model_path}")
        elif os.path.exists(self.ft_output_dir):
            adapter_path = self.ft_output_dir
            merge_source = "ft"
            logger.info(f"Merging FT adapter: {adapter_path}")
        else:
            raise ValueError(
                "No checkpoint found to merge.\n"
                f"Checked:\n"
                f"  - Recovery: {self.recovery_output_dir}\n"
                f"  - Quantized 8-bit: {self.quantized_8bit_dir}\n"
                f"  - Quantized 4-bit: {self.quantized_4bit_dir}\n"
                f"  - FT: {self.ft_output_dir}"
            )
        
        # Extract base model path if model_id is an adapter
        extracted_base, is_adapter = self._get_base_model_path(base_model_path)
        if is_adapter:
            logger.info(f"⚠️  Extracted base model from adapter: {extracted_base}")
            base_model_path = extracted_base
        
        # Load base model (fp16 for merging)
        logger.info(f"Loading base model: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.hf_token
        )
        
        # Log actual device placement
        if hasattr(base_model, 'hf_device_map'):
            logger.info(f"Device map: {base_model.hf_device_map}")
        else:
            model_device = next(base_model.parameters()).device
            logger.info(f"Model loaded on: {model_device}")
        
        # Load and merge adapter if exists
        if adapter_path:
            logger.info(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            
            logger.info("Merging adapter into base model...")
            model = model.merge_and_unload()
        else:
            # No adapter, just use base model (quantized converted to FP16)
            model = base_model
        
        # Save merged model
        logger.info(f"Saving merged model to {output_dir}")
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Merge complete (source: {merge_source})")
        
        return {
            "status": "completed",
            "output_dir": output_dir,
            "merged_from": adapter_path if adapter_path else merge_source
        }
    
    def _deploy_model(self) -> Dict:
        """
        Deploy stage: Convert merged FP16 model to deployment formats.
        
        Supports multiple output formats based on deployment_config:
        - Merged FP16 (baseline)
        - Quantized 4-bit/8-bit (bitsandbytes)
        - GGUF formats (f16, q4_0, q4_1, q5_0, q5_1, q8_0)
        
        Returns:
            Result dictionary with output paths for each format
        """
        logger.info("\n" + "=" * 60)
        logger.info("Stage: Deploy Model")
        logger.info("=" * 60)
        
        if not self.deployment_config.has_any_output():
            logger.warning("No deployment formats enabled in deployment_config")
            return {"status": "skipped", "reason": "no_formats_enabled"}
        
        # Find source model (merged FP16 from previous stages)
        source_model_path = None
        
        # Priority: compress_*bit_merged > merged_model > ft_adapter + base
        for candidate in [
            os.path.join(self.run_dir, "compress_8bit_merged"),
            os.path.join(self.run_dir, "compress_4bit_merged"),
            self.merged_output_dir,
            self.ft_output_dir
        ]:
            if os.path.exists(candidate):
                source_model_path = candidate
                logger.info(f"Source model: {source_model_path}")
                break
        
        if not source_model_path:
            raise ValueError(
                "No merged model found for deployment.\n"
                "Run 'merge' stage or 'compress_*' stage first."
            )
        
        # If source is an adapter, merge it first
        if "adapter" in source_model_path.lower():
            logger.info("Source is an adapter - merging to FP16 first...")
            base_model_path, _ = self._get_base_model_path(self.model_id)
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                token=self.hf_token
            )
            peft_model = PeftModel.from_pretrained(base_model, source_model_path)
            merged_model = peft_model.merge_and_unload()
            
            # Save temporarily
            temp_merged_path = os.path.join(self.run_dir, "temp_deploy_merged")
            merged_model.save_pretrained(temp_merged_path)
            self.tokenizer.save_pretrained(temp_merged_path)
            source_model_path = temp_merged_path
            
            del base_model, peft_model, merged_model
            clear_gpu_memory()
        
        deploy_dir = os.path.join(self.run_dir, "deploy")
        os.makedirs(deploy_dir, exist_ok=True)
        
        results = {"status": "success", "formats": {}}
        
        # Format 1: Merged FP16 (copy source)
        if self.deployment_config.save_merged_fp16:
            logger.info("\nDeploying: Merged FP16")
            fp16_dir = os.path.join(deploy_dir, "merged_fp16")
            os.makedirs(fp16_dir, exist_ok=True)
            
            # Copy model
            model = AutoModelForCausalLM.from_pretrained(
                source_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                token=self.hf_token
            )
            model.save_pretrained(fp16_dir)
            self.tokenizer.save_pretrained(fp16_dir)
            
            logger.info(f"Merged FP16 saved to {fp16_dir}")
            results["formats"]["merged_fp16"] = fp16_dir
            
            del model
            clear_gpu_memory()
        
        # Format 2: Quantized 4-bit
        if self.deployment_config.save_quantized_4bit:
            logger.info("\nDeploying: Quantized 4-bit")
            quant4_dir = os.path.join(deploy_dir, "quantized_4bit")
            os.makedirs(quant4_dir, exist_ok=True)
            
            model = self._apply_quantization(
                model_path=source_model_path,
                bits=4,
                for_training=False
            )
            model.save_pretrained(quant4_dir)
            self.tokenizer.save_pretrained(quant4_dir)
            
            logger.info(f"Quantized 4-bit saved to {quant4_dir}")
            results["formats"]["quantized_4bit"] = quant4_dir
            
            del model
            clear_gpu_memory()
        
        # Format 3: Quantized 8-bit
        if self.deployment_config.save_quantized_8bit:
            logger.info("\nDeploying: Quantized 8-bit")
            quant8_dir = os.path.join(deploy_dir, "quantized_8bit")
            os.makedirs(quant8_dir, exist_ok=True)
            
            model = self._apply_quantization(
                model_path=source_model_path,
                bits=8,
                for_training=False
            )
            model.save_pretrained(quant8_dir)
            self.tokenizer.save_pretrained(quant8_dir)
            
            logger.info(f"Quantized 8-bit saved to {quant8_dir}")
            results["formats"]["quantized_8bit"] = quant8_dir
            
            del model
            clear_gpu_memory()
        
        # Format 4: GGUF formats
        gguf_formats = self.deployment_config.get_gguf_formats()
        if gguf_formats:
            logger.info(f"\nDeploying: GGUF formats {gguf_formats}")
            
            try:
                # Convert to GGUF using llama.cpp tools
                self._convert_to_gguf(
                    source_model_path=source_model_path,
                    output_dir=os.path.join(deploy_dir, "gguf"),
                    formats=gguf_formats
                )
                
                for fmt in gguf_formats:
                    gguf_path = os.path.join(deploy_dir, "gguf", f"model-{fmt}.gguf")
                    results["formats"][f"gguf_{fmt}"] = gguf_path
                    logger.info(f"GGUF {fmt} saved to {gguf_path}")
                    
            except Exception as e:
                logger.error(f"GGUF conversion failed: {e}")
                results["gguf_error"] = str(e)
        
        logger.info(f"\nDeploy stage complete!")
        logger.info(f"   Output directory: {deploy_dir}")
        logger.info(f"   Formats generated: {list(results['formats'].keys())}")
        
        return results
    
    def _convert_to_gguf(
        self,
        source_model_path: str,
        output_dir: str,
        formats: List[str]
    ):
        """
        Convert model to GGUF format using llama-cpp-python.
        
        Args:
            source_model_path: Path to source PyTorch model
            output_dir: Output directory for GGUF files
            formats: List of GGUF quantization formats (e.g., ["f16", "q4_0", "q8_0"])
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Try using llama-cpp-python's convert script
            import subprocess
            
            # First, convert to GGUF FP16 (base format)
            logger.info("Converting PyTorch model to GGUF FP16...")
            base_gguf = os.path.join(output_dir, "model-f16.gguf")
            
            # Use convert.py from llama.cpp if available
            convert_cmd = [
                "python", "-m", "llama_cpp.convert",
                source_model_path,
                "--outfile", base_gguf,
                "--outtype", "f16"
            ]
            
            result = subprocess.run(
                convert_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                # Fallback: Try using transformers export
                logger.warning("llama_cpp.convert not available, using manual conversion...")
                self._manual_gguf_conversion(source_model_path, base_gguf)
            
            logger.info(f"✓ Base GGUF created: {base_gguf}")
            
            # Now quantize to requested formats
            for fmt in formats:
                if fmt == "f16":
                    continue  # Already created
                
                logger.info(f"Quantizing to {fmt.upper()}...")
                output_file = os.path.join(output_dir, f"model-{fmt}.gguf")
                
                quantize_cmd = [
                    "python", "-m", "llama_cpp.quantize",
                    base_gguf,
                    output_file,
                    fmt.upper()
                ]
                
                result = subprocess.run(
                    quantize_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                logger.info(f"✓ Quantized to {fmt.upper()}: {output_file}")
                
        except Exception as e:
            logger.error(f"GGUF conversion error: {e}")
            raise RuntimeError(
                f"GGUF conversion failed: {e}\n\n"
                "Make sure llama-cpp-python is installed with:\n"
                "  pip install llama-cpp-python\n\n"
                "Or install llama.cpp tools manually."
            )
    
    def _manual_gguf_conversion(self, source_path: str, output_path: str):
        """
        Fallback manual GGUF conversion for when llama.cpp tools are not available.
        
        This is a simplified conversion - for production use, install llama.cpp tools.
        """
        logger.warning(
            "Manual GGUF conversion is experimental.\n"
            "For best results, install llama.cpp:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cd llama.cpp && make\n"
            "  python convert.py <model_path>"
        )
        
        # TODO: Implement basic GGUF writer
        # For now, just raise an error directing to proper tools
        raise NotImplementedError(
            "Manual GGUF conversion not yet implemented.\n"
            "Please install llama-cpp-python or llama.cpp tools."
        )
    
    def _print_summary(self):
        """Print summary of all stages."""
        print("\n" + "=" * 60)
        print("Training Pipeline Summary")
        print("=" * 60)
        
        for stage, result in self.results.items():
            print(f"\n{stage.upper()}:")
            print(f"  Status: {result.get('status', 'unknown')}")
            print(f"  Output: {result.get('output_dir', 'N/A')}")
            
            if "metrics" in result:
                metrics = result["metrics"]
                # Metrics from trainer.evaluate() have 'eval_' prefix
                accuracy = metrics.get('eval_accuracy', metrics.get('accuracy', 0))
                f1_macro = metrics.get('eval_f1_macro', metrics.get('f1_macro', 0))
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1 Macro: {f1_macro:.4f}")
        
        print("=" * 60)
