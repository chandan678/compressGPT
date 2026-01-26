"""
Model Runner for compressGPT Inference

This module provides the ModelRunner class for running inference on datasets
built with DatasetBuilder. Supports first-token prediction mode for classification.

Flow:
    1. DatasetBuilder preprocesses data, discovers labels
    2. ModelRunner runs model, cleans output, maps to token IDs
    3. ComputeMetrics compares pred vs actual token IDs

Example usage:
    from compressgpt import DatasetBuilder, ModelRunner, ComputeMetrics
    
    # Build dataset and get metadata
    builder = DatasetBuilder(...)
    dataset = builder.build()
    metadata = builder.get_metadata(tokenizer)
    
    # Run inference (returns token IDs for both pred and gold)
    runner = ModelRunner(model, tokenizer, metadata)
    predictions, gold_labels = runner.run(dataset)
    
    # Compute metrics
    metrics = ComputeMetrics(metadata, tokenizer)
    results = metrics.compute(predictions, gold_labels)
"""

import re
import os
import torch
from typing import Optional, Union
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelRunner:
    """
    Runs inference on datasets for classification tasks.
    
    The main `run()` method:
    1. Runs the model on each prompt
    2. Cleans/extracts the label from model output (handles " yes", "Yes!", etc.)
    3. Maps cleaned labels to token IDs
    4. Returns (pred_token_ids, gold_token_ids) for direct metric computation
    
    Attributes:
        model: The HuggingFace model for inference
        tokenizer: Tokenizer matching the model
        metadata: Dataset metadata from DatasetBuilder.get_metadata()
        device: Device to run inference on
    """
    
    # Token ID for unrecognized predictions
    UNKNOWN_TOKEN_ID = -1
    
    def __init__(
        self,
        model: Union[str, AutoModelForCausalLM],
        tokenizer: Union[str, AutoTokenizer, None] = None,
        metadata: dict = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the ModelRunner.
        
        Args:
            model: Either a loaded HuggingFace model OR a model path/ID string.
                   If string and not a local path, will load from HuggingFace.
            tokenizer: Either a loaded tokenizer, a tokenizer path/ID, or None.
                      If None and model is a string, will load matching tokenizer.
            metadata: Metadata dict from DatasetBuilder.get_metadata(tokenizer).
                     Required if tokenizer needs validation.
            device: Device for inference. If None, uses model's device or cuda/mps/cpu
            batch_size: Batch size for inference
            hf_token: HuggingFace token for gated/private models. Required if model
                     is a remote path that needs authentication.
        """
        # Load model if string path/ID provided
        if isinstance(model, str):
            model_path = model
            is_local = os.path.exists(model_path)
            
            if not is_local:
                print(f"🔍 Model '{model_path}' not found locally. Loading from HuggingFace...")
                if hf_token is None:
                    import getpass
                    print("⚠️  Model may require authentication.")
                    use_token = input("Do you have a HuggingFace token? (y/n): ").strip().lower()
                    if use_token == 'y':
                        hf_token = getpass.getpass("Enter your HuggingFace token: ")
                    else:
                        print("Attempting to load without token (may fail for gated models)...")
                        hf_token = None
            
            print(f"📥 Loading model from {'local path' if is_local else 'HuggingFace'}: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                token=hf_token,
                device_map="auto"
            )
            
            # Load tokenizer if not provided
            if tokenizer is None:
                print(f"📥 Loading tokenizer: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    token=hf_token
                )
            elif isinstance(tokenizer, str):
                print(f"📥 Loading tokenizer: {tokenizer}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer,
                    token=hf_token
                )
            else:
                self.tokenizer = tokenizer
        else:
            # Model already loaded
            self.model = model
            if tokenizer is None:
                raise ValueError("tokenizer must be provided when model is pre-loaded")
            self.tokenizer = tokenizer
        
        self.metadata = metadata
        self.batch_size = batch_size
        
        # Determine device
        if device is not None:
            self.device = device
        elif hasattr(self.model, 'device'):
            self.device = self.model.device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # Validate metadata has token IDs
        if metadata is not None and not metadata.get("label_token_ids"):
            raise ValueError(
                "metadata must contain label_token_ids. "
                "Call builder.get_metadata(tokenizer) with a tokenizer."
            )
        
        # Store mappings from metadata if provided
        if metadata is not None:
            # label_token_ids: {"yes": 3763, "no": 645} - tokens for " yes", " no" (with space)
            self.label_token_ids = metadata["label_token_ids"]
            self.id_to_label = metadata["id_to_label"]
            self.labels = metadata["labels"]
            self.valid_token_ids = set(self.label_token_ids.values())
            
            # Build reverse lookup: lowercase label -> canonical label
            # This handles case variations like "Yes" -> "yes"
            self._label_lookup = {label.lower(): label for label in self.labels}
            
            # Pre-compile regex for label extraction (case-insensitive)
            # Match any of the valid labels as whole words
            pattern = r'\b(' + '|'.join(re.escape(label) for label in self.labels) + r')\b'
            self._label_pattern = re.compile(pattern, re.IGNORECASE)
        else:
            # Metadata not provided - runner can still be used for raw inference
            self.label_token_ids = None
            self.id_to_label = None
            self.labels = None
            self.valid_token_ids = None
            self._label_lookup = None
            self._label_pattern = None
    
    def _extract_label_from_text(self, text: str) -> Optional[str]:
        """
        Extract a valid label from model-generated text.
        
        Handles cases like:
        - " yes" -> "yes"
        - "Yes!" -> "yes"
        - "The answer is no." -> "no"
        - "YES" -> "yes"
        
        Args:
            text: The decoded model output text
            
        Returns:
            Normalized label string, or None if no valid label found
        """
        # First try: direct match after stripping and lowercasing
        clean = text.strip().lower()
        if clean in self._label_lookup:
            return self._label_lookup[clean]
        
        # Second try: regex search for label as a word
        match = self._label_pattern.search(text)
        if match:
            found = match.group(1).lower()
            return self._label_lookup.get(found)
        
        return None
    
    def _token_id_to_normalized_label(self, token_id: int) -> Optional[str]:
        """
        Convert a predicted token ID to a normalized label.
        
        First checks if it's a known label token. If not, decodes and
        tries to extract a label from the text.
        
        Args:
            token_id: The predicted token ID
            
        Returns:
            Normalized label string, or None if not recognized
        """
        # Fast path: direct token ID match
        if token_id in self.id_to_label:
            return self.id_to_label[token_id]
        
        # Slow path: decode and try to extract
        decoded = self.tokenizer.decode([token_id])
        return self._extract_label_from_text(decoded)
    
    def _clean_and_map_to_token_id(self, raw_token_id: int) -> tuple[int, str, str]:
        """
        Clean model output and map to a valid label token ID.
        
        Args:
            raw_token_id: The raw predicted token ID from the model
            
        Returns:
            Tuple of (mapped_token_id, raw_decoded, cleaned_label)
        """
        raw_decoded = self.tokenizer.decode([raw_token_id])
        
        # Fast path: direct token ID match
        if raw_token_id in self.id_to_label:
            label = self.id_to_label[raw_token_id]
            return raw_token_id, raw_decoded, label
        
        # Slow path: decode, clean, and re-map
        cleaned_label = self._extract_label_from_text(raw_decoded)
        
        if cleaned_label is not None:
            mapped_token_id = self.label_token_ids[cleaned_label]
            return mapped_token_id, raw_decoded, cleaned_label
        
        return self.UNKNOWN_TOKEN_ID, raw_decoded, None
    
    def run(
        self,
        dataset,
        show_progress: bool = True,
        log_samples: int = 0,
    ) -> tuple[list[int], list[int]]:
        """
        Run inference on the dataset.
        
        Flow:
        1. Run model on each prompt to get predicted token
        2. Clean/extract label from model output (handles " yes", "Yes!", etc.)
        3. Map cleaned label to token ID
        4. Return (pred_token_ids, gold_token_ids) for metric computation
        
        Args:
            dataset: HuggingFace Dataset with 'prompt' and 'response' columns
            show_progress: Whether to show progress bar
            log_samples: Number of sample predictions to log (0 = none)
            
        Returns:
            Tuple of (predictions, gold_labels) as lists of token IDs.
            Unrecognized predictions get token ID -1 (UNKNOWN_TOKEN_ID).
        """
        if self.metadata is None:
            raise ValueError("metadata is required for run(). Provide metadata during initialization.")
        
        self.model.eval()
        predictions = []
        gold_labels = []
        sample_logs = []
        
        prompts = dataset["prompt"]
        responses = dataset["response"]
        
        iterator = range(0, len(prompts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Running inference")
        
        with torch.no_grad():
            for i in iterator:
                batch_prompts = prompts[i:i + self.batch_size]
                batch_responses = responses[i:i + self.batch_size]
                
                # Tokenize prompts
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                
                # Get model predictions (first token mode)
                outputs = self.model(**inputs)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                raw_pred_token_ids = next_token_logits.argmax(dim=-1).cpu().tolist()
                
                # Process each prediction
                for raw_token_id, response in zip(raw_pred_token_ids, batch_responses):
                    # Clean and map prediction to token ID
                    mapped_token_id, raw_decoded, cleaned_label = self._clean_and_map_to_token_id(raw_token_id)
                    predictions.append(mapped_token_id)
                    
                    # Get gold label token ID
                    gold_token_id = self.label_token_ids.get(response)
                    if gold_token_id is None:
                        raise ValueError(
                            f"Response '{response}' not found in label_token_ids. "
                            f"Available labels: {list(self.label_token_ids.keys())}"
                        )
                    gold_labels.append(gold_token_id)
                    
                    # Collect samples for logging
                    if log_samples > 0 and len(sample_logs) < log_samples:
                        sample_logs.append({
                            "raw_decoded": raw_decoded,
                            "cleaned_label": cleaned_label,
                            "pred_token_id": mapped_token_id,
                            "gold_label": response,
                            "gold_token_id": gold_token_id,
                            "match": mapped_token_id == gold_token_id,
                        })
        
        # Log samples if requested
        if sample_logs:
            self._log_samples(sample_logs)
        
        return predictions, gold_labels
    
    def _log_samples(self, samples: list[dict]):
        """Log sample predictions for debugging."""
        print(f"\n📋 Sample predictions (first {len(samples)}):")
        for i, s in enumerate(samples):
            match = "✓" if s["match"] else "✗"
            cleaned = s["cleaned_label"] if s["cleaned_label"] else "UNKNOWN"
            print(f"  {i+1}. raw='{s['raw_decoded'].strip()}' -> cleaned='{cleaned}' | gold='{s['gold_label']}' {match}")
        print()
    
    def run_single(self, prompt: str) -> tuple[int, str]:
        """
        Run inference on a single prompt.
        
        Args:
            prompt: The prompt string
            
        Returns:
            Tuple of (predicted_token_id, predicted_label_string)
            If label not recognized, returns (UNKNOWN_TOKEN_ID, raw_decoded)
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).to(self.device)
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            next_token_logits = logits[0, -1, :]
            raw_token_id = next_token_logits.argmax().item()
            
            # Clean and map to token ID
            mapped_token_id, raw_decoded, cleaned_label = self._clean_and_map_to_token_id(raw_token_id)
            
            if cleaned_label is not None:
                return mapped_token_id, cleaned_label
            else:
                return self.UNKNOWN_TOKEN_ID, raw_decoded.strip()
    
    def get_logits_for_labels(self, prompt: str) -> dict[str, float]:
        """
        Get the logit values for each valid label token.
        
        Useful for understanding model confidence across classes.
        
        Args:
            prompt: The prompt string
            
        Returns:
            Dict mapping label string to logit value
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).to(self.device)
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            next_token_logits = logits[0, -1, :]
            
            label_logits = {}
            for label, token_id in self.label_token_ids.items():
                label_logits[label] = next_token_logits[token_id].item()
        
        return label_logits
