"""
CompressGPT Trainer v2 - Simplified Model-Centric Training Pipeline

Orchestrates LoRA → QLoRA → Merge training pipeline with automatic metadata
extraction from DatasetBuilder.
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

from .config import LoraConfig, QLoraConfig, TrainingConfig
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
    Simplified trainer for CompressGPT - model-centric.
    
    Orchestrates the complete training pipeline: FT → QLoRA → Merge.
    Automatically extracts metadata from DatasetBuilder for validation and metrics.
    
    Example:
        from compressgpt import DatasetBuilder, CompressTrainer
        
        # Build dataset with model awareness
        builder = DatasetBuilder(
            data_path="data.csv",
            model_id="meta-llama/Llama-3.2-1B",
            prompt_template="...",
            input_column_map={...},
            label_column="label"
        )
        builder.build()
        
        # Train with simple API
        trainer = CompressTrainer(
            model_id="meta-llama/Llama-3.2-1B",
            dataset_builder=builder,
            stages=["ft", "merge"]
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
        qlora_config: Optional[QLoraConfig] = None,
        training_config: Optional[TrainingConfig] = None,
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
            stages: List of training stages (e.g., ["ft", "merge"] or ["ft", "qlora", "merge"])
            ft_config: LoRA configuration for FT stage
            qlora_config: QLoRA configuration (includes train_bits)
            training_config: General training configuration
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
        
        # Validate stages
        valid_stages = {"ft", "qlora", "merge"}
        invalid = set(stages) - valid_stages
        if invalid:
            raise ValueError(f"Invalid stages: {invalid}. Must be from {valid_stages}")
        
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
        logger.info(f"📥 Loading tokenizer from: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        # Set padding token if not present (required for batch training)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"✓ Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        # Reconstruct LabelSpace
        self.label_space = LabelSpace.from_dict(label_space_dict, self.tokenizer)
        
        # Extract response_trigger
        self.response_trigger = self.metadata.get("response_trigger")
        if not self.response_trigger:
            raise ValueError("Metadata missing 'response_trigger' field")
        
        logger.info(f"✓ Response trigger: {self.response_trigger!r}")
        logger.info(f"✓ LabelSpace: {len(self.label_space.labels)} labels")
        logger.info(f"  {self.label_space.labels}")
        
        # Initialize configs with defaults
        self.ft_config = ft_config or LoraConfig()
        self.qlora_config = qlora_config or QLoraConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Create output directories
        os.makedirs(run_dir, exist_ok=True)
        self.ft_output_dir = os.path.join(run_dir, "ft_adapter")
        self.qlora_output_dir = os.path.join(run_dir, "qlora_adapter")
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
        
        logger.info("✓ CompressTrainer initialized")
        logger.info("=" * 60 + "\n")
    
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
            if "qlora" in self.stages:
                warnings.warn(
                    "⚠️  Apple Silicon (MPS) detected with QLoRA stage enabled.\n"
                    "BitsAndBytes quantization is NOT supported on MPS.\n"
                    "QLoRA training will fail. Consider:\n"
                    "  1. Use only 'ft' stage: stages=['ft', 'merge']\n"
                    "  2. Train on a CUDA GPU\n"
                    "  3. Use CPU (slow): set PYTORCH_ENABLE_MPS_FALLBACK=1",
                    RuntimeWarning,
                    stacklevel=2
                )
            logger.info(f"Device: {self.device_type.upper()} (Apple Silicon)")
        elif self.device_type == "cpu":
            warnings.warn(
                "⚠️  No GPU detected. Training will be extremely slow on CPU.",
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
    
    def run(self) -> Dict:
        """
        Run the complete training pipeline (v2 API).
        
        Returns:
            Dictionary with results from each stage
        """
        logger.info("=" * 60)
        logger.info("Starting CompressGPT Training Pipeline v2")
        logger.info("=" * 60)
        
        if "ft" in self.stages:
            self.results["ft"] = self._train_stage_ft()
            clear_gpu_memory()
        
        if "qlora" in self.stages:
            self.results["qlora"] = self._train_stage_qlora()
            clear_gpu_memory()
        
        if "merge" in self.stages:
            self.results["merge"] = self._merge_and_save()
        
        # Save all metrics
        metrics_path = os.path.join(self.run_dir, "metrics.json")
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
        
        output_dir = self.ft_output_dir
        
        # Skip if exists
        if self.resume and os.path.exists(output_dir):
            logger.info(f"⏭️  Skipping FT stage - output exists: {output_dir}")
            return {"status": "skipped", "output_dir": output_dir}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load base model
        logger.info(f"Loading base model: {self.model_id}")
        dtype = torch.bfloat16 if self.training_config.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
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
                "  1. Use only 'ft' stage: stages=['ft', 'merge']\n"
                "  2. Train on a CUDA GPU\n"
                "  3. Use CPU (slow): set PYTORCH_ENABLE_MPS_FALLBACK=1 and device_map='cpu'"
            )
        
        output_dir = self.qlora_output_dir
        
        # Skip if exists
        if self.resume and os.path.exists(output_dir):
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
                self.model_id,
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
                    "If on Apple Silicon, use stages=['ft', 'merge'] (skip QLoRA)"
                ) from e
            raise
        
        # Load FT adapter if exists
        ft_adapter_path = self.ft_output_dir
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
        data_collator = setup_data_collator(self.tokenizer, self.response_trigger)
        
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
        
        output_dir = self.merged_output_dir
        
        # Skip if exists
        if self.resume and os.path.exists(output_dir):
            logger.info(f"⏭️  Skipping merge stage - output exists: {output_dir}")
            return {"status": "skipped", "output_dir": output_dir}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which adapter to merge
        qlora_path = self.qlora_output_dir
        if os.path.exists(qlora_path):
            adapter_path = qlora_path
            logger.info(f"Merging QLoRA adapter from {adapter_path}")
        else:
            ft_path = self.ft_output_dir
            if os.path.exists(ft_path):
                adapter_path = ft_path
                logger.info(f"Merging FT adapter from {adapter_path}")
            else:
                raise ValueError("No adapter found to merge")
        
        # Load base model (fp16 for merging)
        logger.info(f"Loading base model: {self.model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
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
