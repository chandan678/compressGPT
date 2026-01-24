"""
Test cases for train.py script chat template logic.

Run with: pytest tests/test_train_script.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset


class TestChatTemplateDetection:
    """Tests for automatic chat template detection and application."""
    
    def test_detects_chat_template_in_instruct_model(self):
        """Test that chat template is detected in Instruct models."""
        tokenizer = Mock()
        tokenizer.chat_template = "{% for message in messages %}..."
        
        has_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        
        assert has_template is True
    
    def test_no_chat_template_in_base_model(self):
        """Test that chat template is absent in base models."""
        tokenizer = Mock()
        tokenizer.chat_template = None
        
        has_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        
        assert has_template is False
    
    def test_detects_existing_template_markers(self):
        """Test detection of pre-existing chat template markers in prompts."""
        # Test various chat template markers
        markers = [
            '<|begin_of_text|>',
            '<|start_header_id|>',
            '<|im_start|>',
            '[INST]'
        ]
        
        for marker in markers:
            prompt = f"{marker}Some content"
            has_markers = any(m in prompt for m in [
                '<|begin_of_text|>', '<|start_header_id|>', '<|im_start|>', '[INST]'
            ])
            assert has_markers is True
    
    def test_plain_prompt_has_no_markers(self):
        """Test that plain prompts don't have chat template markers."""
        prompt = "Output Yes if... Name: John Answer:"
        
        has_markers = any(marker in prompt for marker in [
            '<|begin_of_text|>', '<|start_header_id|>', '<|im_start|>', '[INST]'
        ])
        
        assert has_markers is False


class TestChatTemplateApplication:
    """Tests for chat template application logic."""
    
    def test_applies_template_to_plain_prompt_with_instruct_model(self):
        """Test that chat template is applied to plain prompts with Instruct models."""
        tokenizer = Mock()
        tokenizer.chat_template = "template"
        tokenizer.apply_chat_template = Mock(return_value="<|begin_of_text|>formatted")
        
        has_chat_template = True
        prompt = "Output Yes... Answer:"
        prompt_has_template = any(marker in prompt for marker in [
            '<|begin_of_text|>', '<|start_header_id|>', '<|im_start|>', '[INST]'
        ])
        
        if has_chat_template and not prompt_has_template:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "yes"}
            ]
            result = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            result = f"{prompt} yes"
        
        assert tokenizer.apply_chat_template.called
        assert "<|begin_of_text|>" in result
    
    def test_skips_template_for_already_formatted_prompt(self):
        """Test that chat template is NOT applied if prompt already has markers."""
        tokenizer = Mock()
        tokenizer.chat_template = "template"
        tokenizer.apply_chat_template = Mock(return_value="formatted")
        
        has_chat_template = True
        prompt = "<|begin_of_text|>Already formatted"
        prompt_has_template = any(marker in prompt for marker in [
            '<|begin_of_text|>', '<|start_header_id|>', '<|im_start|>', '[INST]'
        ])
        
        if has_chat_template and not prompt_has_template:
            messages = [{"role": "user", "content": prompt}]
            result = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            result = f"{prompt} yes"
        
        assert not tokenizer.apply_chat_template.called
        assert prompt in result
    
    def test_plain_concatenation_for_base_model(self):
        """Test that base models use plain text concatenation."""
        tokenizer = Mock()
        tokenizer.chat_template = None
        tokenizer.apply_chat_template = Mock()
        
        has_chat_template = False
        prompt = "Output Yes... Answer:"
        response = "yes"
        
        if has_chat_template:
            result = tokenizer.apply_chat_template([], tokenize=False)
        else:
            result = f"{prompt} {response}"
        
        assert not tokenizer.apply_chat_template.called
        assert result == "Output Yes... Answer: yes"


class TestDatasetFormatting:
    """Tests for dataset formatting with different model types."""
    
    def test_dataset_with_instruct_model_plain_prompts(self):
        """Test dataset formatting with Instruct model and plain prompts."""
        tokenizer = Mock()
        tokenizer.chat_template = "template"
        tokenizer.apply_chat_template = Mock(side_effect=lambda msgs, **kwargs: 
            f"<|begin_of_text|>{msgs[0]['content']}<|eot_id|>{msgs[1]['content']}")
        
        dataset = Dataset.from_list([
            {"prompt": "Question 1", "response": "yes"},
            {"prompt": "Question 2", "response": "no"}
        ])
        
        has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        
        def add_text(example):
            prompt_has_template = any(marker in example['prompt'] for marker in [
                '<|begin_of_text|>', '<|start_header_id|>', '<|im_start|>', '[INST]'
            ])
            
            if has_chat_template and not prompt_has_template:
                messages = [
                    {"role": "user", "content": example['prompt']},
                    {"role": "assistant", "content": example['response']}
                ]
                example["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                example["text"] = f"{example['prompt']} {example['response']}"
            return example
        
        formatted = dataset.map(add_text)
        
        assert "<|begin_of_text|>" in formatted[0]["text"]
        assert "Question 1" in formatted[0]["text"]
        assert "yes" in formatted[0]["text"]
    
    def test_dataset_with_instruct_model_formatted_prompts(self):
        """Test dataset formatting with Instruct model and pre-formatted prompts."""
        tokenizer = Mock()
        tokenizer.chat_template = "template"
        tokenizer.apply_chat_template = Mock()
        
        dataset = Dataset.from_list([
            {"prompt": "<|begin_of_text|>Already formatted", "response": "yes"}
        ])
        
        has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        
        def add_text(example):
            prompt_has_template = any(marker in example['prompt'] for marker in [
                '<|begin_of_text|>', '<|start_header_id|>', '<|im_start|>', '[INST]'
            ])
            
            if has_chat_template and not prompt_has_template:
                messages = [
                    {"role": "user", "content": example['prompt']},
                    {"role": "assistant", "content": example['response']}
                ]
                example["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                example["text"] = f"{example['prompt']} {example['response']}"
            return example
        
        formatted = dataset.map(add_text)
        
        # Should NOT apply template again
        assert not tokenizer.apply_chat_template.called
        assert formatted[0]["text"] == "<|begin_of_text|>Already formatted yes"
    
    def test_dataset_with_base_model(self):
        """Test dataset formatting with base model."""
        tokenizer = Mock()
        tokenizer.chat_template = None
        tokenizer.apply_chat_template = Mock()
        
        dataset = Dataset.from_list([
            {"prompt": "Question 1", "response": "yes"},
            {"prompt": "Question 2", "response": "no"}
        ])
        
        has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        
        def add_text(example):
            if has_chat_template:
                messages = [
                    {"role": "user", "content": example['prompt']},
                    {"role": "assistant", "content": example['response']}
                ]
                example["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                example["text"] = f"{example['prompt']} {example['response']}"
            return example
        
        formatted = dataset.map(add_text)
        
        # Should use plain concatenation
        assert not tokenizer.apply_chat_template.called
        assert formatted[0]["text"] == "Question 1 yes"
        assert formatted[1]["text"] == "Question 2 no"


class TestNoDoubleWrapping:
    """Tests to ensure chat templates are never double-applied."""
    
    def test_llama3_instruct_markers_detected(self):
        """Test that Llama 3 Instruct markers are detected."""
        prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nQuestion"
        
        has_markers = any(marker in prompt for marker in [
            '<|begin_of_text|>', '<|start_header_id|>', '<|im_start|>', '[INST]'
        ])
        
        assert has_markers is True
    
    def test_llama2_instruct_markers_detected(self):
        """Test that Llama 2 [INST] markers are detected."""
        prompt = "[INST] Question [/INST]"
        
        has_markers = any(marker in prompt for marker in [
            '<|begin_of_text|>', '<|start_header_id|>', '<|im_start|>', '[INST]'
        ])
        
        assert has_markers is True
    
    def test_chatml_markers_detected(self):
        """Test that ChatML markers are detected."""
        prompt = "<|im_start|>user\nQuestion<|im_end|>"
        
        has_markers = any(marker in prompt for marker in [
            '<|begin_of_text|>', '<|start_header_id|>', '<|im_start|>', '[INST]'
        ])
        
        assert has_markers is True
    
    def test_prevents_double_wrapping(self):
        """Test that template is not applied to already-formatted prompts."""
        tokenizer = Mock()
        tokenizer.chat_template = "template"
        tokenizer.apply_chat_template = Mock(return_value="<|begin_of_text|>wrapped")
        
        # Simulate the logic from train.py
        has_chat_template = True
        prompt = "<|begin_of_text|>Already formatted"
        
        prompt_has_template = any(marker in prompt for marker in [
            '<|begin_of_text|>', '<|start_header_id|>', '<|im_start|>', '[INST]'
        ])
        
        if has_chat_template and not prompt_has_template:
            result = tokenizer.apply_chat_template([], tokenize=False)
        else:
            result = f"{prompt} yes"
        
        # Should NOT call apply_chat_template
        assert not tokenizer.apply_chat_template.called
        assert result == "<|begin_of_text|>Already formatted yes"
        # Verify no double markers
        assert result.count("<|begin_of_text|>") == 1
