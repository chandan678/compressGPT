"""
LabelSpace - Manages label tokenization and validation for CompressGPT.

Ensures all labels map to single tokens and provides consistent label handling
across dataset building, training, and inference.
"""

from typing import Optional
import logging


logger = logging.getLogger(__name__)


class LabelSpace:
    """
    Manages label vocabulary and tokenization for classification tasks.
    
    Validates that all labels tokenize to single tokens (required for
    completion-only training) and provides mappings between labels and token IDs.
    
    Attributes:
        labels: Sorted list of canonical label strings (e.g., ["no", "partial", "yes"])
        label_prefix: Prefix added before labels for tokenization (default " ")
        label_token_ids: Dict mapping label string to token ID
        id_to_label: Dict mapping token ID to label string
        valid_token_ids: Sorted list of valid label token IDs (for logits indexing)
        single_token_labels: Always True (multi-token labels not yet supported)
    
    Example:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        label_space = LabelSpace(
            tokenizer=tokenizer,
            labels=["yes", "no"],
            label_prefix=" "  # Leading space for proper tokenization
        )
        
        # Use in training
        valid_ids = label_space.valid_token_ids  # [3763, 912]
        
        # Decode predictions
        label = label_space.id_to_label[predicted_token_id]
    """
    
    def __init__(
        self,
        tokenizer,
        labels: list[str],
        label_prefix: str = " ",
    ):
        """
        Initialize LabelSpace with tokenizer and label list.
        
        Args:
            tokenizer: HuggingFace tokenizer for the model
            labels: List of label strings (will be sorted and normalized)
            label_prefix: Prefix added before labels during tokenization (default " ")
                         Important: " yes" tokenizes differently than "yes"
        
        Raises:
            ValueError: If any label tokenizes to multiple tokens
            ValueError: If duplicate labels exist after normalization
        """
        self.tokenizer = tokenizer
        self.label_prefix = label_prefix
        
        # Normalize and sort labels (lowercase, strip whitespace)
        normalized = sorted(set(label.lower().strip() for label in labels))
        
        if len(normalized) != len(labels):
            original_count = len(labels)
            unique_count = len(normalized)
            logger.warning(
                f"Label list contained duplicates or case variations. "
                f"Original: {original_count}, Unique: {unique_count}"
            )
        
        self.labels = normalized
        
        # Build label <-> token ID mappings
        self.label_token_ids = {}
        self.id_to_label = {}
        
        for label in self.labels:
            # Tokenize with prefix (e.g., " yes" not "yes")
            token_ids = tokenizer.encode(
                f"{label_prefix}{label}",
                add_special_tokens=False
            )
            
            # Validate single-token constraint
            if len(token_ids) != 1:
                raise ValueError(
                    f"Label '{label}' with prefix '{label_prefix}' tokenizes to "
                    f"{len(token_ids)} tokens: {token_ids}. "
                    f"All labels must map to exactly 1 token.\n"
                    f"Solutions:\n"
                    f"  1. Use different label text (e.g., 'yes'/'no' instead of 'correct'/'incorrect')\n"
                    f"  2. Adjust label_prefix (try '', ' ', or experiment)\n"
                    f"  3. Use a different tokenizer with appropriate vocabulary"
                )
            
            token_id = token_ids[0]
            
            # Check for collisions (different labels -> same token)
            if token_id in self.id_to_label:
                raise ValueError(
                    f"Token collision: labels '{self.id_to_label[token_id]}' and '{label}' "
                    f"both tokenize to token ID {token_id}. "
                    f"Use distinct label text."
                )
            
            self.label_token_ids[label] = token_id
            self.id_to_label[token_id] = label
        
        # Sorted list of valid token IDs (for indexing into logits)
        self.valid_token_ids = sorted(self.label_token_ids.values())
        
        # Multi-token labels not yet supported
        self.single_token_labels = True
        
        logger.info(f"LabelSpace initialized with {len(self.labels)} labels")
        logger.info(f"Labels: {self.labels}")
        logger.info(f"Token IDs: {self.label_token_ids}")
    
    def to_dict(self) -> dict:
        """
        Serialize LabelSpace to dictionary for storage/transmission.
        
        Returns:
            Dictionary containing all LabelSpace fields (no tokenizer)
        """
        return {
            "labels": self.labels,
            "label_prefix": self.label_prefix,
            "label_token_ids": self.label_token_ids,
            "id_to_label": self.id_to_label,
            "valid_token_ids": self.valid_token_ids,
            "single_token_labels": self.single_token_labels,
        }
    
    @classmethod
    def from_dict(cls, data: dict, tokenizer) -> "LabelSpace":
        """
        Deserialize LabelSpace from dictionary.
        
        Args:
            data: Dictionary from to_dict()
            tokenizer: HuggingFace tokenizer
        
        Returns:
            Reconstructed LabelSpace instance
        """
        # Create instance using normal init (validates everything)
        instance = cls(
            tokenizer=tokenizer,
            labels=data["labels"],
            label_prefix=data["label_prefix"],
        )
        
        # Verify consistency with stored data
        if instance.label_token_ids != data["label_token_ids"]:
            logger.warning(
                "Label token IDs from stored data differ from current tokenizer. "
                "Using current tokenizer mappings."
            )
        
        return instance
    
    def validate_token_id(self, token_id: int) -> bool:
        """
        Check if a token ID is a valid label.
        
        Args:
            token_id: Token ID to validate
        
        Returns:
            True if token_id is in valid_token_ids
        """
        return token_id in self.valid_token_ids
    
    def decode_token_id(self, token_id: int) -> Optional[str]:
        """
        Decode a token ID to its label string.
        
        Args:
            token_id: Token ID to decode
        
        Returns:
            Label string if valid, None otherwise
        """
        return self.id_to_label.get(token_id)
    
    def __repr__(self) -> str:
        return (
            f"LabelSpace(labels={self.labels}, "
            f"label_prefix={self.label_prefix!r}, "
            f"n_labels={len(self.labels)})"
        )
