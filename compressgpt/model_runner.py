"""
ModelRunner handles inference for classification-style prompts.
It maps next-token predictions to label IDs for metric computation.
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
    Returns predicted and gold label token IDs.
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
        """Initialize the ModelRunner."""
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
        """Extract a valid label from model-generated text."""
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
        """Convert a predicted token ID to a normalized label."""
        # Fast path: direct token ID match
        if token_id in self.id_to_label:
            return self.id_to_label[token_id]
        
        # Slow path: decode and try to extract
        decoded = self.tokenizer.decode([token_id])
        return self._extract_label_from_text(decoded)
    
    def _clean_and_map_to_token_id(self, raw_token_id: int) -> tuple[int, str, str]:
        """Clean model output and map to a valid label token ID."""
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
        """Run inference on a dataset and return prediction/gold token IDs."""
        if self.metadata is None:
            raise ValueError("metadata is required for run(). Provide metadata during initialization.")
        
        self.model.eval()
        predictions = []
        gold_labels = []
        sample_logs = []
        
        # Support both column naming conventions:
        # - DatasetBuilder creates: 'text' (full), 'gold_label', optionally 'prompt'/'response' with keep_fields=True
        # - Legacy format: 'prompt', 'response'
        # Also support dict-like objects for testing
        column_names = getattr(dataset, 'column_names', list(dataset.keys()) if isinstance(dataset, dict) else [])
        
        if "prompt" in column_names:
            prompts = dataset["prompt"]
        elif "text" in column_names:
            # Extract prompt from 'text' by removing the label part
            # For eval (is_train=False), 'text' is prompt-only
            prompts = dataset["text"]
        else:
            raise ValueError("Dataset must have 'prompt' or 'text' column")
        
        if "response" in column_names:
            responses = dataset["response"]
        elif "gold_label" in column_names:
            responses = dataset["gold_label"]
        else:
            raise ValueError("Dataset must have 'response' or 'gold_label' column")
        
        iterator = range(0, len(prompts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Running inference")
        
        with torch.no_grad():
            for i in iterator:
                batch_prompts = prompts[i:i + self.batch_size]
                batch_responses = responses[i:i + self.batch_size]
                
                # Tokenize prompts with LEFT padding for correct next-token prediction
                # Right padding would make logits[:, -1, :] look at pad tokens
                original_padding_side = getattr(self.tokenizer, 'padding_side', 'right')
                if hasattr(self.tokenizer, 'padding_side'):
                    self.tokenizer.padding_side = "left"
                
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                
                # Restore original padding side
                if hasattr(self.tokenizer, 'padding_side'):
                    self.tokenizer.padding_side = original_padding_side
                
                # Get model predictions (next token after prompt)
                outputs = self.model(**inputs)
                logits = outputs.logits
                # With left padding, the last position is always the actual last token
                next_token_logits = logits[:, -1, :]
                raw_pred_token_ids = next_token_logits.argmax(dim=-1).cpu().tolist()
                
                # Process each prediction
                for idx, (raw_token_id, response, prompt) in enumerate(zip(raw_pred_token_ids, batch_responses, batch_prompts)):
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
                            "prompt": prompt,
                            "raw_token_id": raw_token_id,
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
        """Log sample predictions for detailed debugging."""
        print(f"\n{'='*80}")
        print(f"📋 Sample Predictions (first {len(samples)})")
        print(f"{'='*80}")
        
        # Show label space info first
        print(f"\n📊 Label Space:")
        print(f"  Labels: {self.labels}")
        print(f"  Token IDs: {self.label_token_ids}")
        
        for i, s in enumerate(samples):
            match = "✓ CORRECT" if s["match"] else "✗ WRONG"
            cleaned = s["cleaned_label"] if s["cleaned_label"] else "UNKNOWN"
            
            print(f"\n{'─'*60}")
            print(f"Sample {i+1}: {match}")
            print(f"{'─'*60}")
            
            # Truncate prompt for display
            prompt = s.get("prompt", "N/A")
            if len(prompt) > 200:
                prompt = prompt[:200] + "..."
            print(f"Prompt: {prompt}")
            
            print(f"\nGold:  '{s['gold_label']}' (token_id={s['gold_token_id']})")
            print(f"Pred:  raw_token_id={s['raw_token_id']} -> decoded='{s['raw_decoded']}' -> cleaned='{cleaned}' (mapped_id={s['pred_token_id']})")
        
        print(f"\n{'='*80}\n")
        print()
    
    def run_single(self, prompt: str) -> tuple[int, str]:
        """Run inference on a single prompt."""
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
        """Return logits for each valid label token."""
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

    def run_detailed(
        self,
        dataset,
        show_progress: bool = True,
    ) -> list[dict]:
        """Run inference and return detailed per-sample results."""
        if self.metadata is None:
            raise ValueError("metadata is required for run_detailed(). Provide metadata during initialization.")
        
        self.model.eval()
        results = []
        
        # Get columns
        column_names = getattr(dataset, 'column_names', list(dataset.keys()) if isinstance(dataset, dict) else [])
        
        if "prompt" in column_names:
            prompts = dataset["prompt"]
        elif "text" in column_names:
            prompts = dataset["text"]
        else:
            raise ValueError("Dataset must have 'prompt' or 'text' column")
        
        if "response" in column_names:
            responses = dataset["response"]
        elif "gold_label" in column_names:
            responses = dataset["gold_label"]
        else:
            raise ValueError("Dataset must have 'response' or 'gold_label' column")
        
        iterator = range(0, len(prompts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Running inference")
        
        with torch.no_grad():
            for i in iterator:
                batch_prompts = prompts[i:i + self.batch_size]
                batch_responses = responses[i:i + self.batch_size]
                
                # Use left padding for correct next-token prediction
                original_padding_side = getattr(self.tokenizer, 'padding_side', 'right')
                if hasattr(self.tokenizer, 'padding_side'):
                    self.tokenizer.padding_side = "left"
                
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                
                if hasattr(self.tokenizer, 'padding_side'):
                    self.tokenizer.padding_side = original_padding_side
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                raw_pred_token_ids = next_token_logits.argmax(dim=-1).cpu().tolist()
                
                for raw_token_id, response, prompt in zip(raw_pred_token_ids, batch_responses, batch_prompts):
                    mapped_token_id, raw_decoded, cleaned_label = self._clean_and_map_to_token_id(raw_token_id)
                    
                    gold_token_id = self.label_token_ids.get(response)
                    if gold_token_id is None:
                        raise ValueError(
                            f"Response '{response}' not found in label_token_ids. "
                            f"Available labels: {list(self.label_token_ids.keys())}"
                        )
                    
                    results.append({
                        "prompt": prompt,
                        "gold_label": response,
                        "pred_label": cleaned_label if cleaned_label else "UNKNOWN",
                        "gold_token_id": gold_token_id,
                        "pred_token_id": mapped_token_id,
                        "raw_decoded": raw_decoded.strip(),
                        "correct": mapped_token_id == gold_token_id,
                    })
        
        # Print summary
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        print(f"\n📊 Inference Summary: {correct}/{total} correct ({100*correct/total:.1f}% accuracy)")
        
        return results
