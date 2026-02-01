"""
Utility functions for CompressGPT training and validation.

This module provides helper functions for label validation, response template
checking, data collator setup, and memory management.
"""

import gc
import json
import torch
from typing import Dict, List, Optional


def validate_label_tokens(tokenizer, labels: List[str]) -> Dict[str, int]:
    """
    Validate that each label maps to exactly one token ID.
    
    This is critical for first-token classification. Multi-token labels
    would require different scoring logic.
    
    Args:
        tokenizer: HuggingFace tokenizer
        labels: List of label strings (e.g., ["yes", "no", "partial"])
        
    Returns:
        Dict mapping label string to token ID
        
    Raises:
        ValueError: If any label maps to multiple tokens
        
    Example:
        >>> validate_label_tokens(tokenizer, ["yes", "no"])
        {"yes": 3763, "no": 912}
    """
    label_token_ids = {}
    errors = []
    
    for label in labels:
        # Encode with leading space (common for classification tasks)
        token_ids = tokenizer.encode(f" {label}", add_special_tokens=False)
        
        if len(token_ids) != 1:
            errors.append(
                f"Label '{label}' tokenizes to {len(token_ids)} tokens: {token_ids}. "
                f"Decoded: {[tokenizer.decode([tid]) for tid in token_ids]}"
            )
        else:
            label_token_ids[label] = token_ids[0]
    
    if errors:
        error_msg = "❌ Label validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        error_msg += (
            "\n\n💡 Solutions:"
            "\n  1. Use simpler labels that map to single tokens (e.g., 'yes'/'no' instead of 'affirmative')"
            "\n  2. Adjust tokenizer vocabulary if possible"
            "\n  3. Enable multi-token scoring (not yet implemented)"
        )
        raise ValueError(error_msg)
    
    return label_token_ids


def validate_response_template(template: str, allow_special_tokens: bool = False) -> None:
    """
    Validate that response template is a simple trigger string.
    
    DataCollatorForCompletionOnlyLM expects a human-readable trigger like
    "Answer:" not special tokens like "<|start_header_id|>".
    
    Args:
        template: The response template string to validate
        allow_special_tokens: If False, raises error on special tokens
        
    Raises:
        ValueError: If template contains special tokens and not allowed
        
    Example:
        >>> validate_response_template("Answer:")  # OK
        >>> validate_response_template("<|start_header_id|>")  # Error
    """
    if not template or not template.strip():
        raise ValueError("Response template cannot be empty")
    
    # Check for common special token patterns
    special_token_patterns = ["<|", "|>", "<eos>", "<bos>", "<pad>", "<unk>"]
    
    if not allow_special_tokens:
        for pattern in special_token_patterns:
            if pattern in template:
                raise ValueError(
                    f"❌ Response template contains special token pattern '{pattern}': {template!r}\n"
                    f"💡 DataCollatorForCompletionOnlyLM requires a simple trigger like 'Answer:'\n"
                    f"   If using chat templates, ensure the template ends with a human-readable trigger.\n"
                    f"   Set allow_special_tokens=True to bypass this check (not recommended)."
                )


def setup_data_collator(
    tokenizer, 
    response_template: str, 
    *, 
    model_mode: str = "base",
    allow_fallback: bool = False
):
    """
    Create a completion-only collator that masks prompt tokens so loss is computed
    only on response tokens (classification label tokens in your case).
    
    CRITICAL: For instruct models with chat templates, we use the assistant header
    as the response template instead of the user-provided trigger. This is because
    tokenization is context-sensitive: "Answer:" tokenizes differently when preceded
    by a space vs. standalone. The assistant header is always tokenized consistently.
    
    Args:
        tokenizer: The tokenizer
        response_template: The response trigger (e.g., "Answer:") - used for base models
        model_mode: "base" or "instruct" - determines which template to use
        allow_fallback: If True, fall back to full-sequence loss if trl not available
    
    Returns:
        DataCollator for completion-only loss
    """
    try:
        from trl import DataCollatorForCompletionOnlyLM
        
        if model_mode == "instruct":
            # For instruct models, use the assistant header as response template
            # This is more reliable than user triggers due to context-sensitive tokenization
            #
            # Common assistant headers:
            # - Llama 3: <|start_header_id|>assistant<|end_header_id|>\n\n
            # - Mistral/ChatML: <|im_start|>assistant\n
            # - Llama 2: [/INST]
            
            # Try to detect the assistant header from the tokenizer's chat template
            assistant_header = None
            
            # Check for Llama 3 style
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                if '<|start_header_id|>' in tokenizer.chat_template:
                    assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                elif '<|im_start|>' in tokenizer.chat_template:
                    assistant_header = "<|im_start|>assistant\n"
                elif '[/INST]' in tokenizer.chat_template:
                    assistant_header = "[/INST]"
            
            if assistant_header:
                # Use assistant header for reliable masking
                return DataCollatorForCompletionOnlyLM(
                    tokenizer=tokenizer,
                    response_template=assistant_header,
                )
            else:
                # Fallback: try stripped response_template
                import warnings
                warnings.warn(
                    f"Could not detect assistant header for instruct model. "
                    f"Falling back to '{response_template.strip()}'. "
                    f"This may cause masking issues due to context-sensitive tokenization."
                )
                return DataCollatorForCompletionOnlyLM(
                    tokenizer=tokenizer,
                    response_template=response_template.strip(),
                )
        else:
            # Base model: use the response template directly
            response_template = response_template.strip()
            return DataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                response_template=response_template,
            )
            
    except ImportError as e:
        if not allow_fallback:
            raise ImportError(
                "trl is required for DataCollatorForCompletionOnlyLM. "
                "Install `trl` or call with allow_fallback=True (not recommended for label-only training)."
            ) from e

        from transformers import DataCollatorForLanguageModeling
        import warnings
        warnings.warn(
            "Falling back to DataCollatorForLanguageModeling (full-sequence loss). "
            "This changes the training objective and may hurt label-only classification prompting."
        )
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



def clear_gpu_memory():
    """Clear GPU/MPS memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def save_metrics(metrics: dict, path: str):
    """Save metrics dictionary to JSON file."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def format_metrics_table(metrics: dict, stage_name: str = "") -> str:
    """Format metrics as a readable table string."
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    if stage_name:
        lines.append(f"📊 Metrics: {stage_name}")
    else:
        lines.append("📊 Metrics")
    lines.append("=" * 60)
    
    # Sort keys: accuracy first, then f1_macro, then per-class
    main_keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    other_keys = sorted([k for k in metrics.keys() if k not in main_keys])
    
    for key in main_keys + other_keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (int, float)):
                lines.append(f"  {key:20s}: {value:.4f}")
            else:
                lines.append(f"  {key:20s}: {value}")
    
    lines.append("=" * 60)
    return "\n".join(lines)
