"""
Compute Metrics for compressGPT

This module provides the ComputeMetrics class for computing accuracy metrics
on model predictions. Supports dynamic label-based metrics without hardcoding.

Example usage:
    from compressgpt import DatasetBuilder, ModelRunner, ComputeMetrics
    
    # Build dataset and get metadata
    builder = DatasetBuilder(...)
    dataset = builder.build()
    metadata = builder.get_metadata(tokenizer)
    
    # Run inference
    runner = ModelRunner(model, tokenizer, metadata)
    predictions, gold_labels = runner.run(dataset)
    
    # Compute metrics
    metrics = ComputeMetrics(metadata, tokenizer)
    results = metrics.compute(predictions, gold_labels)
    # → {"accuracy": 0.92, "f1_macro": 0.89, "f1_yes": 0.91, "f1_no": 0.87}
"""

from typing import Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np


class ComputeMetrics:
    """
    Compute classification metrics dynamically based on dataset labels.
    
    This class computes accuracy, F1 scores (macro and per-class), and
    other metrics based on the labels discovered from the dataset.
    No hardcoded labels - everything is derived from metadata.
    
    Attributes:
        metadata: Dataset metadata from DatasetBuilder.get_metadata()
        tokenizer: Tokenizer for decoding predictions (optional, for logging)
        labels: List of label strings
        label_token_ids: Dict mapping label string to token ID
        id_to_label: Dict mapping token ID to label string
    """
    
    def __init__(self, metadata: dict, tokenizer=None):
        """
        Initialize ComputeMetrics with dataset metadata.
        
        Args:
            metadata: Metadata dict from DatasetBuilder.get_metadata(tokenizer)
            tokenizer: Optional tokenizer for decoding predictions in logs
        
        Raises:
            ValueError: If metadata is missing required fields
        """
        self.metadata = metadata
        self.tokenizer = tokenizer
        
        # Validate metadata
        required_fields = ["labels", "label_token_ids", "id_to_label"]
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                raise ValueError(
                    f"metadata must contain '{field}'. "
                    "Call builder.get_metadata(tokenizer) with a tokenizer."
                )
        
        self.labels = metadata["labels"]
        self.label_token_ids = metadata["label_token_ids"]
        self.id_to_label = metadata["id_to_label"]
        self.valid_token_ids = sorted(self.label_token_ids.values())
    
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
        label_token_ids = self.label_token_ids
        valid_token_ids = np.array(self.valid_token_ids)  # Convert to numpy for indexing
        labels_list = self.labels
        tokenizer = self.tokenizer
        
        def compute_metrics(eval_preds):
            nonlocal seen
            logits, labels = eval_preds  # logits: [B, T, V], labels: [B, T]
            
            gold, pred = [], []
            unknown_gold_count = 0
            
            for i in range(labels.shape[0]):
                l_row = labels[i]
                
                # Find first non-masked position (where label != -100)
                idxs = np.where(l_row != -100)[0]
                if idxs.size == 0:
                    continue
                
                pos = int(idxs[0])
                gold_id = int(l_row[pos])
                
                # Validate gold label is in our valid set
                if gold_id not in valid_token_ids:
                    unknown_gold_count += 1
                    continue
                
                # Extract logits at the answer position: [V]
                step_logits = logits[i, pos, :]
                
                # CRITICAL: Restrict to label tokens only (not full vocab)
                label_logits = step_logits[valid_token_ids]  # [num_labels]
                
                # Argmax among label tokens
                best_label_idx = int(label_logits.argmax())
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
                token_id = label_token_ids[label]
                results[f"f1_{label}"] = f1_score(
                    gold, pred,
                    labels=[token_id],
                    average="macro",
                    zero_division=0
                )
            
            return results
        
        return compute_metrics
