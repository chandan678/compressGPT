"""
CompressGPT Trainer v2 - Model Compression Pipeline

Orchestrates model compression workflow: FT → Quantize → Recovery → Merge
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
    CompressGPT Trainer - Model Compression Pipeline.
    
    Orchestrates compression-focused workflow:
    1. FT: Fine-tune on FP16 base (establish accuracy baseline)
    2. Quantize: Explicit 8-bit/4-bit quantization (compress model)
    3. Recovery: Train adapters on quantized base (compensate quantization error)
    4. Merge: Merge adapters back to FP16 (prepare for deployment quantization)
    
    Valid stage combinations:
    - ["ft", "merge"]: Basic LoRA fine-tuning
    - ["ft", "quantize_8bit", "recovery", "merge"]: Full 8-bit compression
    - ["ft", "quantize_4bit", "recovery", "merge"]: Full 4-bit compression
    
    Example:
        from compressgpt import DatasetBuilder, CompressTrainer
        
        builder = DatasetBuilder(...).build()
        
        # Full compression pipeline
        trainer = CompressTrainer(
            model_id="meta-llama/Llama-3.2-1B",
            dataset_builder=builder,
            stages=["ft", "quantize_8bit", "recovery", "merge"]
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
            stages: List of stages (e.g., ["ft", "quantize_8bit", "recovery", "merge"])
            ft_config: LoRA configuration for FT stage
            recovery_config: Recovery configuration (includes train_bits for quantization)
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
        valid_stages = {"ft", "quantize_8bit", "quantize_4bit", "recovery", "merge"}
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
        self.recovery_config = recovery_config or QLoraConfig()
        self.training_config = training_config or TrainingConfig()
        
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
                logger.info(f"⚠️  Detected LoRA adapter. Base model: {base_model}")
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
                    "⚠️  Apple Silicon (MPS) detected with quantization/recovery stages.\n"
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
        
        if "quantize_8bit" in self.stages:
            self.results["quantize_8bit"] = self._quantize_model(bits=8)
            clear_gpu_memory()
        
        if "quantize_4bit" in self.stages:
            self.results["quantize_4bit"] = self._quantize_model(bits=4)
            clear_gpu_memory()
        
        if "recovery" in self.stages:
            self.results["recovery"] = self._train_stage_recovery()
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
            logger.info(f"📍 Device map: {model.hf_device_map}")
        else:
            model_device = next(model.parameters()).device
            logger.info(f"📍 Model loaded on: {model_device}")
        
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
            logger.info(f"📍 Device map: {model.hf_device_map}")
        else:
            model_device = next(model.parameters()).device
            logger.info(f"📍 Model loaded on: {model_device}")
        
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
        logger.info(f"💾 Saving {bits}-bit quantized model to {output_dir}")
        quantized_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✓ {bits}-bit quantization complete")
        
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
            eval_accumulation_steps=1,  # Prevent OOM on small GPUs
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
            logger.info(f"📍 Device map: {base_model.hf_device_map}")
        else:
            model_device = next(base_model.parameters()).device
            logger.info(f"📍 Model loaded on: {model_device}")
        
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
        logger.info(f"💾 Saving merged model to {output_dir}")
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✓ Merge complete (source: {merge_source})")
        
        return {
            "status": "completed",
            "output_dir": output_dir,
            "merged_from": adapter_path if adapter_path else merge_source
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
