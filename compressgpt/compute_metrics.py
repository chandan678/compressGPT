"""
This module provides the ComputeMetrics class with label-restricted argmax
for computing accuracy metrics on model predictions.

Example usage:
    from compressgpt import DatasetBuilder, ComputeMetrics
    
    builder = DatasetBuilder(...)
    builder.build()
    
    # Create metrics with label restriction
    metrics = ComputeMetrics(
        labels=builder.label_space.labels,
        valid_token_ids=builder.label_space.valid_token_ids,
        id_to_label=builder.label_space.id_to_label,
        tokenizer=tokenizer
    )
    
    # Use as trainer callback
    trainer = SFTTrainer(..., compute_metrics=metrics.as_trainer_callback())
"""

from typing import Optional, List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import torch


class ComputeMetrics:
    """
    Compute classification metrics with label-restricted argmax.
    
    Key feature: Restricts predictions to valid label token IDs only,
    preventing vocabulary leakage and special token predictions.
    
    Attributes:
        labels: List of label strings
        valid_token_ids: List of valid label token IDs (for logits indexing)
        id_to_label: Dict mapping token ID to label string
        tokenizer: Tokenizer for decoding (optional)
    """
    
    def __init__(
        self,
        labels: List[str],
        valid_token_ids: List[int],
        id_to_label: Dict[int, str],
        tokenizer=None
    ):
        """
        Initialize ComputeMetrics with label-restricted vocabulary.
        
        Args:
            labels: List of label strings (e.g., ["yes", "no"])
            valid_token_ids: List of valid label token IDs (e.g., [3763, 912])
            id_to_label: Dict mapping token ID to label string
            tokenizer: Optional tokenizer for logging
        """
        self.labels = labels
        self.valid_token_ids = np.array(valid_token_ids)  # Convert to numpy for indexing
        self.id_to_label = id_to_label
        self.tokenizer = tokenizer
        # Create reverse mapping for per-class F1 computation (filter to valid labels only)
        self.label_token_ids = {lbl: tid for tid, lbl in self.id_to_label.items() if lbl in set(self.labels)}
    
    def _detect_input_type(self, values: list) -> str:
        """Detect whether input is token IDs or label strings."""
        if not values:
            return "empty"
        first = values[0]
        if isinstance(first, int):
            return "token_ids"
        elif isinstance(first, str):
            return "labels"
        else:
            return "unknown"
    
    def _labels_to_token_ids(self, labels: list[str]) -> list[int]:
        """Convert label strings to token IDs for metric computation."""
        token_ids = []
        for label in labels:
            if label in self.label_token_ids:
                token_ids.append(self.label_token_ids[label])
            else:
                # Unknown label - use -1 as placeholder
                token_ids.append(-1)
        return token_ids
    
    def compute(
        self,
        predictions: list,
        gold_labels: list,
        log_samples: int = 0,
    ) -> dict:
        """
        Compute classification metrics.
        
        Accepts either token IDs or label strings for both predictions and gold_labels.
        Automatically detects the input type and handles conversion.
        
        Args:
            predictions: List of predicted token IDs OR label strings
            gold_labels: List of gold token IDs OR label strings
            log_samples: Number of sample predictions to print (0 = none)
            
        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy
                - f1_macro: Macro-averaged F1 score
                - f1_{label}: Per-class F1 for each label
                - precision_macro: Macro-averaged precision
                - recall_macro: Macro-averaged recall
        """
        if len(predictions) != len(gold_labels):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(gold_labels)} gold labels"
            )
        
        # Detect input types
        pred_type = self._detect_input_type(predictions)
        gold_type = self._detect_input_type(gold_labels)
        
        # Convert to token IDs if needed for sklearn metrics
        if pred_type == "labels":
            pred_ids = self._labels_to_token_ids(predictions)
        else:
            pred_ids = predictions
            
        if gold_type == "labels":
            gold_ids = self._labels_to_token_ids(gold_labels)
        else:
            gold_ids = gold_labels
        
        # Log sample predictions if requested
        if log_samples > 0:
            self._log_samples(pred_ids, gold_ids, log_samples, predictions, gold_labels)
        
        # Compute metrics
        results = {
            "accuracy": accuracy_score(gold_ids, pred_ids),
            "f1_macro": f1_score(
                gold_ids, pred_ids,
                labels=self.valid_token_ids,
                average="macro",
                zero_division=0
            ),
            "precision_macro": precision_score(
                gold_ids, pred_ids,
                labels=self.valid_token_ids,
                average="macro",
                zero_division=0
            ),
            "recall_macro": recall_score(
                gold_ids, pred_ids,
                labels=self.valid_token_ids,
                average="macro",
                zero_division=0
            ),
        }
        
        # Per-class F1 scores
        for label in self.labels:
            token_id = self.label_token_ids[label]
            f1 = f1_score(
                gold_ids, pred_ids,
                labels=[token_id],
                average="macro",
                zero_division=0
            )
            results[f"f1_{label}"] = f1
        
        return results
    
    def _log_samples(
        self,
        pred_ids: list[int],
        gold_ids: list[int],
        n: int,
        orig_preds: list = None,
        orig_gold: list = None,
    ):
        """Log sample predictions for debugging."""
        print(f"\n📋 Sample predictions (first {n}):")
        for i in range(min(n, len(pred_ids))):
            pred_id = pred_ids[i]
            gold_id = gold_ids[i]
            
            # Get display values - prefer original strings if available
            if orig_preds is not None and isinstance(orig_preds[i], str):
                pred_display = orig_preds[i]
            elif self.tokenizer is not None and pred_id >= 0:
                pred_display = self.tokenizer.decode([pred_id]).strip()
            else:
                pred_display = str(pred_id)
                
            if orig_gold is not None and isinstance(orig_gold[i], str):
                gold_display = orig_gold[i]
            elif self.tokenizer is not None and gold_id >= 0:
                gold_display = self.tokenizer.decode([gold_id]).strip()
            else:
                gold_display = str(gold_id)
            
            match = "✓" if pred_id == gold_id else "✗"
            print(f"  {i+1}. pred='{pred_display}' | gold='{gold_display}' {match}")
        print()
    
    def get_confusion_matrix(
        self,
        predictions: list,
        gold_labels: list,
    ) -> dict:
        """
        Get confusion matrix for the predictions.
        
        Args:
            predictions: List of predicted token IDs OR label strings
            gold_labels: List of gold token IDs OR label strings
            
        Returns:
            Dictionary containing:
                - matrix: 2D numpy array of the confusion matrix
                - labels: List of label strings in matrix order
        """
        # Convert to token IDs if needed
        pred_type = self._detect_input_type(predictions)
        gold_type = self._detect_input_type(gold_labels)
        
        pred_ids = self._labels_to_token_ids(predictions) if pred_type == "labels" else predictions
        gold_ids = self._labels_to_token_ids(gold_labels) if gold_type == "labels" else gold_labels
        
        cm = confusion_matrix(
            gold_ids, pred_ids,
            labels=self.valid_token_ids
        )
        return {
            "matrix": cm,
            "labels": self.labels,
        }
    
    def print_report(self, predictions: list, gold_labels: list):
        """
        Print a formatted classification report.
        
        Args:
            predictions: List of predicted token IDs OR label strings
            gold_labels: List of gold token IDs OR label strings
        """
        results = self.compute(predictions, gold_labels)
        cm_data = self.get_confusion_matrix(predictions, gold_labels)
        
        print("\n" + "=" * 60)
        print("📊 Classification Report")
        print("=" * 60)
        print(f"Total samples: {len(predictions)}")
        print("-" * 60)
        print(f"Accuracy:        {results['accuracy']:.4f}")
        print(f"F1 (macro):      {results['f1_macro']:.4f}")
        print(f"Precision (macro): {results['precision_macro']:.4f}")
        print(f"Recall (macro):  {results['recall_macro']:.4f}")
        print("-" * 60)
        print("Per-class F1 scores:")
        for label in self.labels:
            print(f"  {label:12s}: {results[f'f1_{label}']:.4f}")
        print("-" * 60)
        print("Confusion Matrix:")
        print(f"  Labels: {self.labels}")
        print(cm_data["matrix"])
        print("=" * 60 + "\n")
    
    def as_trainer_callback(self, log_first_n: int = 5):
        """
        Create a compute_metrics function for HuggingFace Trainer.
        
        This returns a closure compatible with Trainer's compute_metrics
        parameter, handling the logits/labels format from eval_preds.
        
        CRITICAL: Uses label-restricted argmax (not full vocab) for classification.
        Finds first non-masked position and extracts prediction at that position.
        
        Args:
            log_first_n: Number of samples to log on first evaluation
            
        Returns:
            A compute_metrics function for Trainer
        
        Example:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metrics.as_trainer_callback(),
            )
        """
        seen = False
        valid_token_ids = self.valid_token_ids  # Already numpy array
        labels_list = self.labels
        tokenizer = self.tokenizer
        
        def compute_metrics(eval_preds):
            nonlocal seen
            logits, labels = eval_preds  # logits: [B, T, V] or [B, T, num_labels], labels: [B, T]
            
            gold, pred = [], []
            unknown_gold_count = 0
            
            for i in range(labels.shape[0]):
                l_row = labels[i]
                
                # Find first non-masked position (where label != -100)
                # CRITICAL: This extracts the FIRST token after the response trigger,
                # not "the" from "the answer is yes". The DataCollatorForCompletionOnlyLM
                # masks everything before response_trigger, so first unmasked token IS the label.
                idxs = np.where(l_row != -100)[0]
                if idxs.size == 0:
                    continue
                
                pos = int(idxs[0])
                gold_id = int(l_row[pos])
                
                # Validate gold label is in our valid set (use numpy's isin for array membership)
                if not np.isin(gold_id, valid_token_ids):
                    unknown_gold_count += 1
                    continue
                
                # Extract logits at the answer position: [V] or [num_labels]
                step_logits = logits[i, pos, :]
                
                # Check if logits are already filtered (from preprocess_logits_for_metrics)
                if step_logits.shape[0] == len(valid_token_ids):
                    # Already filtered to label tokens only by preprocess_logits_for_metrics
                    # Shape is [num_labels], indices 0..num_labels-1 correspond to valid_token_ids
                    label_logits = step_logits
                else:
                    # Full vocabulary - need to filter
                    # CRITICAL: Restrict to label tokens only (not full vocab)
                    # This prevents the model from predicting arbitrary tokens like "the", "answer", etc.
                    # and forces classification behavior over valid labels only.
                    label_logits = step_logits[valid_token_ids]  # [num_labels]
                
                # Argmax among label tokens: returns index 0..num_labels-1
                best_label_idx = int(label_logits.argmax())
                
                # CRITICAL: Map index back to actual token ID
                # After preprocessing, best_label_idx is in range [0, num_labels-1]
                # We need to map it to the actual token ID using valid_token_ids
                pred_id = int(valid_token_ids[best_label_idx])
                
                gold.append(gold_id)
                pred.append(pred_id)
            
            # Log samples on first call
            if not seen and len(gold) >= log_first_n and tokenizer is not None:
                samples = [
                    (tokenizer.decode([p]).strip(), tokenizer.decode([g]).strip())
                    for p, g in zip(pred[:log_first_n], gold[:log_first_n])
                ]
                print(f"📋 Sample pred↔gold (label-restricted): {samples}")
                seen = True
            
            # Warn if gold labels outside valid set
            if unknown_gold_count > 0:
                print(f"⚠️  {unknown_gold_count} samples had gold labels outside valid set")
            
            # Compute metrics
            results = {
                "accuracy": accuracy_score(gold, pred),
                "f1_macro": f1_score(
                    gold, pred,
                    labels=valid_token_ids.tolist(),
                    average="macro",
                    zero_division=0
                ),
            }
            
            # Per-class F1
            for label in labels_list:
                token_id = self.label_token_ids[label]
                results[f"f1_{label}"] = f1_score(
                    gold, pred,
                    labels=[token_id],
                    average="macro",
                    zero_division=0
                )
            
            return results
        
        return compute_metrics

    def get_preprocess_logits(self):
        """
        Create a preprocess_logits_for_metrics function for HuggingFace Trainer.
        
        CRITICAL FOR MEMORY: Filters logits to only label tokens BEFORE accumulation.
        This reduces memory from ~128k vocab to ~2-10 label tokens (massive reduction!).
        
        Without this, eval uses: [batch_size, seq_len, 128k] * 4 bytes = huge memory
        With this, eval uses: [batch_size, seq_len, num_labels] * 4 bytes = tiny memory
        
        Returns:
            A preprocess function that filters logits to label tokens only
            
        Example:
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                compute_metrics=metrics.as_trainer_callback(),
                preprocess_logits_for_metrics=metrics.get_preprocess_logits(),  # ADD THIS
            )
        """
        valid_token_ids = self.valid_token_ids
        
        def preprocess_logits(logits, labels):
            """
            Filter logits to only label tokens before storing.
            
            Args:
                logits: Tensor [B, T, V] or tuple/list containing tensor
                labels: Tensor [B, T] with label IDs
                
            Returns:
                Filtered logits [B, T, num_labels] - only label token logits
            """
            import torch
            
            # Handle tuple/list wrapping (some models return (logits,))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            
            # Create device-safe tensor for indexing (critical for performance)
            # Must be same device as logits and dtype=long for indexing
            ids = torch.tensor(valid_token_ids, device=logits.device, dtype=torch.long)
            
            # Use index_select for efficient filtering: [B, T, V] -> [B, T, num_labels]
            # This is faster than fancy indexing and works on all devices
            return torch.index_select(logits, dim=-1, index=ids)
        
        return preprocess_logits
