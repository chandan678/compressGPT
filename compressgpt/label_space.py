"""
LabelSpace manages label tokenization and validation.
Ensures labels map to single tokens and provides stable mappings.
"""

from typing import Optional
import logging


logger = logging.getLogger(__name__)


class LabelSpace:
    """
    Manages label vocabulary and tokenization for classification tasks.
    Validates single-token labels and exposes label/token mappings.
    """
    
    def __init__(
        self,
        tokenizer,
        labels: list[str],
        label_prefix: str = " ",
    ):
        """Initialize LabelSpace with a tokenizer and label list."""
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
        """Serialize LabelSpace to a dictionary (no tokenizer)."""
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
        """Deserialize LabelSpace from a dictionary."""
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
        """Return True if token_id is a valid label."""
        return token_id in self.valid_token_ids
    
    def decode_token_id(self, token_id: int) -> Optional[str]:
        """Decode a token ID to its label string."""
        return self.id_to_label.get(token_id)
    
    def __repr__(self) -> str:
        return (
            f"LabelSpace(labels={self.labels}, "
            f"label_prefix={self.label_prefix!r}, "
            f"n_labels={len(self.labels)})"
        )
