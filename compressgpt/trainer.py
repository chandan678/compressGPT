"""
CompressGPT Trainer - Orchestrates LoRA → QLoRA → Merge training pipeline.

This module provides the CompressTrainer class for managing the complete
fine-tuning pipeline with proper validation and memory management.
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

from .config import LoraConfig, QLoraConfig, TrainingConfig, PipelineConfig
from .compute_metrics import ComputeMetrics
from .utils import (
    validate_label_tokens,
    validate_response_template,
    setup_data_collator,
    clear_gpu_memory,
    save_metrics,
    format_metrics_table
)


logger = logging.getLogger(__name__)


class CompressTrainer:
    """
    Orchestrates the complete training pipeline: FT → QLoRA → Merge.
    
    Handles validation, memory management, and metric tracking across stages.
    
    Example:
        trainer = CompressTrainer(
            base_model_path="meta-llama/Llama-3.2-1B",
            dataset=dataset,
            metadata=metadata,
            pipeline_config=PipelineConfig(stages=["ft", "qlora", "merge"])
        )
        results = trainer.train()
    """
    
    def __init__(
        self,
        base_model_path: str,
        dataset: Dataset,
        metadata: Dict,
        tokenizer: Optional[AutoTokenizer] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        lora_config: Optional[LoraConfig] = None,
        qlora_config: Optional[QLoraConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        hf_token: Optional[str] = None,
        train_test_split: float = 0.05,
        seed: int = 42,
        check_compatibility: bool = True,
    ):
        """
        Initialize CompressTrainer.
        
        Args:
            base_model_path: Path or HF model ID for base model
            dataset: HuggingFace Dataset with 'text' column (prompt + response)
            metadata: Metadata dict from DatasetBuilder.get_metadata()
            tokenizer: Tokenizer for the base model. If None, will auto-load from base_model_path.
            pipeline_config: Pipeline configuration (stages, directories)
            lora_config: LoRA configuration for FT stage
            qlora_config: QLoRA configuration
            training_config: General training configuration
            hf_token: HuggingFace token for gated models
            train_test_split: Test split ratio
            seed: Random seed for reproducibility
            check_compatibility: Run model compatibility check before training
        """
        self.base_model_path = base_model_path
        self.hf_token = hf_token
        
        # Auto-load tokenizer if not provided
        if tokenizer is None:
            is_local = os.path.exists(base_model_path)
            if not is_local:
                logger.info(f"🔍 Model '{base_model_path}' not found locally. Will load from HuggingFace...")
                if hf_token is None:
                    import getpass
                    logger.info("⚠️  Model may require authentication.")
                    use_token = input("Do you have a HuggingFace token? (y/n): ").strip().lower()
                    if use_token == 'y':
                        self.hf_token = getpass.getpass("Enter your HuggingFace token: ")
                        hf_token = self.hf_token
                    else:
                        logger.info("Attempting without token (may fail for gated models)...")
            
            logger.info(f"📥 Loading tokenizer from: {base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                token=hf_token
            )
        else:
            self.tokenizer = tokenizer
        
        # Clean dataset - keep only 'text' column for SFTTrainer
        required_cols = ['text']
        extra_cols = [col for col in dataset.column_names if col not in required_cols]
        if extra_cols:
            logger.info(f"Removing extra columns from dataset: {extra_cols}")
            dataset = dataset.remove_columns(extra_cols)
        
        self.dataset = dataset
        self.metadata = metadata
        self.train_test_split = train_test_split
        self.seed = seed
        
        # Initialize configs with defaults
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.lora_config = lora_config or LoraConfig()
        self.qlora_config = qlora_config or QLoraConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Create output directories
        os.makedirs(self.pipeline_config.output_dir, exist_ok=True)
        
        # Detect device and warn about limitations
        self.device_type = self._detect_device()
        self._warn_device_limitations()
        
        # Check model compatibility
        if check_compatibility:
            self._check_and_display_compatibility()
        
        # Validate metadata and labels
        self._validate_metadata()
        
        # Setup metrics computer
        self.metrics_computer = ComputeMetrics(metadata, tokenizer)
        
        # Split dataset
        self._split_dataset()
        
        # Store results
        self.results = {}
        
        logger.info(f"CompressTrainer initialized with base model: {base_model_path}")
        logger.info(f"Output directory: {self.pipeline_config.output_dir}")
        logger.info(f"Stages to run: {self.pipeline_config.stages}")
    
    def _detect_device(self) -> str:
        """Detect the device type (cuda/mps/cpu)."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _warn_device_limitations(self):
        """Warn about device-specific limitations."""
        if self.device_type == "mps":
            # Check if QLoRA stage is enabled
            if "qlora" in self.pipeline_config.stages:
                warnings.warn(
                    "⚠️  Apple Silicon (MPS) detected with QLoRA stage enabled.\n"
                    "BitsAndBytes quantization is NOT supported on MPS.\n"
                    "QLoRA training will fail. Consider:\n"
                    "  1. Use only 'ft' stage (LoRA without quantization)\n"
                    "  2. Use CPU (very slow, set PYTORCH_ENABLE_MPS_FALLBACK=1)\n"
                    "  3. Use a CUDA GPU\n"
                    "To disable QLoRA stage, use: --stages ft,merge",
                    RuntimeWarning,
                    stacklevel=2
                )
            logger.info(f"Device: {self.device_type.upper()} (Apple Silicon)")
            logger.info("Note: Some operations may have limited support on MPS")
        elif self.device_type == "cpu":
            warnings.warn(
                "⚠️  No GPU detected. Training will be extremely slow on CPU.",
                RuntimeWarning,
                stacklevel=2
            )
        else:
            logger.info(f"Device: {self.device_type.upper()}")
    
    def _check_and_display_compatibility(self):
        """Check model compatibility and display recommendations."""
        from .model_check import check_model_compatibility
        
        try:
            logger.info("🔍 Checking model compatibility...")
            _, _, model_info, memory_reqs, compatibility = check_model_compatibility(
                model_id=self.base_model_path,
                lora_r=self.lora_config.r,
                safety_margin=0.8
            )
            
            # Display key info
            logger.info(f"Model: {model_info.estimated_parameters_b:.2f}B parameters")
            logger.info(f"Estimated memory - Training LoRA FP16: {memory_reqs.training_lora_fp16:.2f} GB")
            
            # Warn if recommended batch size differs
            if compatibility.recommended_training_batch_size < self.training_config.per_device_train_batch_size:
                logger.warning(
                    f"⚠️  Recommended batch size ({compatibility.recommended_training_batch_size}) "
                    f"< configured ({self.training_config.per_device_train_batch_size}). "
                    f"Consider reducing to avoid OOM."
                )
            
            # Estimate checkpoint disk usage
            checkpoint_gb = model_info.estimated_parameters_b * 2.0  # FP16 checkpoint
            logger.info(f"Estimated checkpoint size: {checkpoint_gb:.2f} GB")
            if checkpoint_gb > 10:
                logger.warning(
                    f"⚠️  Large checkpoint size ({checkpoint_gb:.2f} GB). "
                    f"Ensure sufficient disk space in {self.pipeline_config.output_dir}"
                )
            
            # Show warnings
            if compatibility.warnings:
                for warning in compatibility.warnings:
                    logger.warning(warning)
            
            self._compatibility_info = {
                "batch_size": compatibility.recommended_training_batch_size,
                "gradient_accumulation": compatibility.recommended_gradient_accumulation_steps,
            }
            
        except Exception as e:
            logger.warning(f"Could not check compatibility: {e}")
            self._compatibility_info = None
    
    def _validate_metadata(self):
        """Validate metadata has required fields and labels are valid."""
        required_fields = ["labels", "label_token_ids", "id_to_label", "response_template"]
        for field in required_fields:
            if field not in self.metadata:
                raise ValueError(f"metadata missing required field: {field}")
        
        # Validate response template
        response_template = self.metadata["response_template"]
        validate_response_template(response_template, allow_special_tokens=False)
        logger.info(f"✓ Response template validated: {response_template!r}")
        
        # Validate labels map to single tokens
        labels = self.metadata["labels"]
        label_token_ids = validate_label_tokens(self.tokenizer, labels)
        
        # Verify matches metadata
        if label_token_ids != self.metadata["label_token_ids"]:
            logger.warning(
                "Label token IDs from validation differ from metadata. "
                "Using validated IDs."
            )
            self.metadata["label_token_ids"] = label_token_ids
        
        logger.info(f"✓ Labels validated: {labels}")
        logger.info(f"✓ Label token IDs: {label_token_ids}")
    
    def _split_dataset(self):
        """Split dataset into train and validation sets."""
        split = self.dataset.train_test_split(
            test_size=self.train_test_split,
            seed=self.seed
        )
        self.train_dataset = split["train"]
        self.eval_dataset = split["test"]
        
        logger.info(f"Dataset split - Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")
    
    def train(self) -> Dict:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary with results from each stage
        """
        logger.info("=" * 60)
        logger.info("Starting CompressGPT Training Pipeline")
        logger.info("=" * 60)
        
        stages = self.pipeline_config.stages
        
        if "ft" in stages:
            self.results["ft"] = self._train_stage_ft()
            clear_gpu_memory()
        
        if "qlora" in stages:
            self.results["qlora"] = self._train_stage_qlora()
            clear_gpu_memory()
        
        if "merge" in stages:
            self.results["merge"] = self._merge_and_save()
        
        # Save all metrics
        metrics_path = os.path.join(self.pipeline_config.output_dir, "metrics.json")
        save_metrics(self.results, metrics_path)
        logger.info(f"✓ Metrics saved to {metrics_path}")
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _train_stage_ft(self) -> Dict:
        """Train FT stage: LoRA on fp16/bf16 base model."""
        logger.info("\n" + "=" * 60)
        logger.info("Stage 1: FT (LoRA on FP16/BF16 base)")
        logger.info("=" * 60)
        
        output_dir = self.pipeline_config.ft_output_dir
        
        # Skip if exists
        if self.pipeline_config.skip_if_exists and os.path.exists(output_dir):
            logger.info(f"⏭️  Skipping FT stage - output exists: {output_dir}")
            return {"status": "skipped", "output_dir": output_dir}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load base model
        logger.info(f"Loading base model: {self.base_model_path}")
        dtype = torch.bfloat16 if self.training_config.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            token=self.hf_token,
            dtype=dtype,
            device_map="auto"
        )
        model.gradient_checkpointing_enable()
        
        # Log model size
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Loaded {num_params/1e9:.2f}B parameters")
        if torch.cuda.is_available():
            logger.info(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            logger.info(f"MPS memory: {torch.mps.current_allocated_memory()/1e9:.2f} GB")
        
        # Setup PEFT
        peft_config = PeftLoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type
        )
        
        # Setup data collator
        response_template = self.metadata["response_template"]
        data_collator = setup_data_collator(self.tokenizer, response_template)
        
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
            eval_steps=500,  # Eval every 500 steps
            save_strategy="steps",  # Must match eval_strategy when load_best_model_at_end=True
            save_steps=500,  # Save every 500 steps
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            report_to=self.training_config.report_to,
            run_name=run_name,
            dataset_text_field="text",
            eval_accumulation_steps=1,  # Prevent OOM on MPS/small GPUs
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
            callbacks=callbacks
        )
        
        # Train
        import time
        num_samples = len(self.train_dataset)
        eff_bs = self.training_config.per_device_train_batch_size * self.training_config.gradient_accumulation_steps
        logger.info(f"🚀 Training: {num_samples} samples, {self.training_config.num_train_epochs} epochs, batch_size={eff_bs}")
        
        start_time = time.time()
        trainer.train()
        duration = time.time() - start_time
        logger.info(f"✓ Training completed in {duration/60:.1f} minutes")
        
        # Evaluate
        logger.info("📊 Evaluating...")
        metrics = trainer.evaluate()
        
        # Clear memory after eval (critical for MPS)
        if self.device_type in ["mps", "cuda"]:
            clear_gpu_memory()
        
        # Save model
        logger.info(f"💾 Saving FT adapter to {output_dir}")
        trainer.save_model(output_dir)
        
        # Print metrics
        print(format_metrics_table(metrics, "FT Stage"))
        
        return {
            "status": "completed",
            "output_dir": output_dir,
            "metrics": metrics
        }
    
    def _train_stage_qlora(self) -> Dict:
        """Train QLoRA stage: LoRA on quantized base + FT adapters."""
        logger.info("\n" + "=" * 60)
        logger.info(f"Stage 2: QLoRA ({self.qlora_config.train_bits}-bit)")
        logger.info("=" * 60)
        
        # Check device compatibility
        if self.device_type == "mps":
            raise RuntimeError(
                "QLoRA training is not supported on Apple Silicon (MPS).\n"
                "BitsAndBytes quantization requires CUDA.\n\n"
                "Solutions:\n"
                "  1. Use only 'ft' stage: --stages ft,merge\n"
                "  2. Train on a CUDA GPU\n"
                "  3. Use CPU (slow): set PYTORCH_ENABLE_MPS_FALLBACK=1 and device_map='cpu'"
            )
        
        output_dir = self.pipeline_config.qlora_output_dir
        
        # Skip if exists
        if self.pipeline_config.skip_if_exists and os.path.exists(output_dir):
            logger.info(f"⏭️  Skipping QLoRA stage - output exists: {output_dir}")
            return {"status": "skipped", "output_dir": output_dir}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(self.qlora_config.train_bits == 4),
            load_in_8bit=(self.qlora_config.train_bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.qlora_config.bnb_4bit_use_double_quant,
        )
        
        # Load quantized base model
        logger.info(f"Loading {self.qlora_config.train_bits}-bit quantized base model")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                token=self.hf_token
            )
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                raise RuntimeError(
                    f"Failed to load quantized model: {e}\n\n"
                    "BitsAndBytes quantization requires CUDA GPU.\n"
                    f"Current device: {self.device_type}\n\n"
                    "If on Apple Silicon, use --stages ft,merge (skip QLoRA)"
                ) from e
            raise
        
        # Load FT adapter if exists
        ft_adapter_path = self.pipeline_config.ft_output_dir
        if os.path.exists(ft_adapter_path):
            logger.info(f"Loading FT adapter from {ft_adapter_path}")
            model = PeftModel.from_pretrained(
                base_model,
                ft_adapter_path,
                is_trainable=True
            )
        else:
            logger.warning("FT adapter not found - training QLoRA from quantized base only")
            # Apply fresh LoRA config
            peft_config = PeftLoraConfig(
                r=self.qlora_config.r,
                lora_alpha=self.qlora_config.lora_alpha,
                lora_dropout=self.qlora_config.lora_dropout,
                target_modules=self.qlora_config.target_modules,
                bias=self.qlora_config.bias,
                task_type=self.qlora_config.task_type
            )
            model = get_peft_model(base_model, peft_config)
        
        # Setup data collator
        response_template = self.metadata["response_template"]
        data_collator = setup_data_collator(self.tokenizer, response_template)
        
        # Adjust training config for QLoRA (typically fewer epochs, higher LR)
        run_name = self.training_config.run_name or "compressgpt_qlora"
        qlora_epochs = max(1, self.training_config.num_train_epochs // 6)  # Typically 1 epoch
        qlora_lr = self.training_config.learning_rate * 5  # Higher LR for QLoRA
        
        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=qlora_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=qlora_lr,
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            weight_decay=self.training_config.weight_decay,
            max_length=self.training_config.max_seq_length,
            logging_steps=self.training_config.logging_steps,
            eval_strategy="steps",  # Eval during training for progress visibility
            eval_steps=500,  # Eval every 500 steps
            save_strategy="steps",  # Must match eval_strategy when load_best_model_at_end=True
            save_steps=500,  # Save every 500 steps
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            fp16=self.training_config.fp16,
            bf16=False,  # QLoRA uses fp16
            report_to=self.training_config.report_to,
            run_name=run_name,
            dataset_text_field="text",
            eval_accumulation_steps=1,  # Prevent OOM on MPS/small GPUs
        )
        
        # Setup trainer
        callbacks = []
        if self.training_config.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=1,  # Lower patience for QLoRA
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
            callbacks=callbacks
        )
        
        # Train
        import time
        num_samples = len(self.train_dataset)
        eff_bs = self.training_config.per_device_train_batch_size * self.training_config.gradient_accumulation_steps
        logger.info(f"🚀 QLoRA Training: {num_samples} samples, {qlora_epochs} epochs, batch_size={eff_bs}")
        
        start_time = time.time()
        trainer.train()
        duration = time.time() - start_time
        logger.info(f"✓ QLoRA training completed in {duration/60:.1f} minutes")
        
        # Evaluate
        logger.info("📊 Evaluating...")
        metrics = trainer.evaluate()
        
        # Save model
        logger.info(f"💾 Saving QLoRA adapter to {output_dir}")
        trainer.save_model(output_dir)
        
        # Print metrics
        print(format_metrics_table(metrics, "QLoRA Stage"))
        
        return {
            "status": "completed",
            "output_dir": output_dir,
            "metrics": metrics,
            "train_bits": self.qlora_config.train_bits
        }
    
    def _merge_and_save(self) -> Dict:
        """Merge QLoRA adapters into base model and save."""
        logger.info("\n" + "=" * 60)
        logger.info("Stage 3: Merge and Save")
        logger.info("=" * 60)
        
        output_dir = self.pipeline_config.merged_output_dir
        
        # Skip if exists
        if self.pipeline_config.skip_if_exists and os.path.exists(output_dir):
            logger.info(f"⏭️  Skipping merge stage - output exists: {output_dir}")
            return {"status": "skipped", "output_dir": output_dir}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which adapter to merge
        qlora_path = self.pipeline_config.qlora_output_dir
        if os.path.exists(qlora_path):
            adapter_path = qlora_path
            logger.info(f"Merging QLoRA adapter from {adapter_path}")
        else:
            ft_path = self.pipeline_config.ft_output_dir
            if os.path.exists(ft_path):
                adapter_path = ft_path
                logger.info(f"Merging FT adapter from {adapter_path}")
            else:
                raise ValueError("No adapter found to merge")
        
        # Load base model (fp16 for merging)
        logger.info(f"Loading base model: {self.base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            dtype=torch.float16,
            token=self.hf_token
        )
        
        # Load and merge adapter
        logger.info(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        logger.info("Merging adapter into base model...")
        model = model.merge_and_unload()
        
        # Save merged model
        logger.info(f"💾 Saving merged model to {output_dir}")
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("✓ Merge complete")
        
        return {
            "status": "completed",
            "output_dir": output_dir,
            "merged_from": adapter_path
        }
    
    def _print_summary(self):
        """Print summary of all stages."""
        print("\n" + "=" * 60)
        print("📋 Training Pipeline Summary")
        print("=" * 60)
        
        for stage, result in self.results.items():
            print(f"\n{stage.upper()}:")
            print(f"  Status: {result.get('status', 'unknown')}")
            print(f"  Output: {result.get('output_dir', 'N/A')}")
            
            if "metrics" in result:
                metrics = result["metrics"]
                print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"  F1 Macro: {metrics.get('f1_macro', 0):.4f}")
        
        print("=" * 60)
