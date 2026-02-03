"""
Test cases for config.py module.

These tests verify that configuration validation catches common issues,
especially around default value mismatches and type-specific requirements.

Run with: pytest tests/test_config.py -v
"""

import importlib.util
import pytest
from compressgpt.config import (
    LoraConfig,
    QLoraConfig,
    TrainingConfig,
    QuantizationConfig,
    DeploymentConfig
)


class TestLoraConfig:
    """Tests for LoraConfig validation."""
    
    def test_default_config_valid(self):
        """Test that default LoraConfig is valid."""
        config = LoraConfig()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.task_type == "CAUSAL_LM"
    
    def test_invalid_r_raises(self):
        """Test that invalid r values raise ValueError."""
        with pytest.raises(ValueError, match="r must be positive"):
            LoraConfig(r=0)
        
        with pytest.raises(ValueError, match="r must be positive"):
            LoraConfig(r=-1)
    
    def test_invalid_lora_alpha_raises(self):
        """Test that invalid lora_alpha raises ValueError."""
        with pytest.raises(ValueError, match="lora_alpha must be positive"):
            LoraConfig(lora_alpha=0)
        
        with pytest.raises(ValueError, match="lora_alpha must be positive"):
            LoraConfig(lora_alpha=-10)
    
    def test_invalid_dropout_raises(self):
        """Test that invalid dropout values raise ValueError."""
        with pytest.raises(ValueError, match="lora_dropout must be in"):
            LoraConfig(lora_dropout=-0.1)
        
        with pytest.raises(ValueError, match="lora_dropout must be in"):
            LoraConfig(lora_dropout=1.0)
        
        with pytest.raises(ValueError, match="lora_dropout must be in"):
            LoraConfig(lora_dropout=1.5)
    
    def test_invalid_bias_raises(self):
        """Test that invalid bias values raise ValueError."""
        with pytest.raises(ValueError, match="bias must be"):
            LoraConfig(bias="invalid")
    
    def test_custom_values_valid(self):
        """Test that custom valid values work."""
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="all"
        )
        assert config.r == 8
        assert config.lora_alpha == 16


class TestQLoraConfig:
    """Tests for QLoraConfig validation."""
    
    def test_default_config_valid(self):
        """Test that default QLoraConfig is valid."""
        config = QLoraConfig()
        assert config.train_bits == 4
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.r == 16  # Inherited from LoraConfig
    
    def test_invalid_train_bits_raises(self):
        """Test that invalid train_bits raises ValueError."""
        with pytest.raises(ValueError, match="train_bits must be 4 or 8"):
            QLoraConfig(train_bits=2)
        
        with pytest.raises(ValueError, match="train_bits must be 4 or 8"):
            QLoraConfig(train_bits=16)
    
    def test_4bit_invalid_quant_type_raises(self):
        """Test that invalid 4-bit quant_type raises ValueError."""
        with pytest.raises(ValueError, match="bnb_4bit_quant_type must be"):
            QLoraConfig(train_bits=4, bnb_4bit_quant_type="int8")
    
    def test_valid_4bit_quant_types(self):
        """Test that nf4 and fp4 are valid for 4-bit."""
        config_nf4 = QLoraConfig(train_bits=4, bnb_4bit_quant_type="nf4")
        assert config_nf4.bnb_4bit_quant_type == "nf4"
        
        config_fp4 = QLoraConfig(train_bits=4, bnb_4bit_quant_type="fp4")
        assert config_fp4.bnb_4bit_quant_type == "fp4"
    
    def test_8bit_config_valid(self):
        """Test that 8-bit config is valid."""
        config = QLoraConfig(train_bits=8)
        assert config.train_bits == 8


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""
    
    def test_default_config_valid(self):
        """Test that default TrainingConfig is valid."""
        config = TrainingConfig()
        assert config.num_train_epochs == 6  # Current default
        assert config.learning_rate > 0
    
    def test_invalid_epochs_raises(self):
        """Test that invalid epochs raise ValueError."""
        with pytest.raises(ValueError, match="num_train_epochs must be positive"):
            TrainingConfig(num_train_epochs=0)
        
        with pytest.raises(ValueError, match="num_train_epochs must be positive"):
            TrainingConfig(num_train_epochs=-1)
    
    def test_invalid_batch_size_raises(self):
        """Test that invalid batch size raises ValueError."""
        with pytest.raises(ValueError, match="per_device_train_batch_size must be positive"):
            TrainingConfig(per_device_train_batch_size=0)
    
    def test_invalid_learning_rate_raises(self):
        """Test that invalid learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=0)
        
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-0.001)
    
    def test_invalid_warmup_ratio_raises(self):
        """Test that invalid warmup_ratio raises ValueError."""
        with pytest.raises(ValueError, match="warmup_ratio must be in"):
            TrainingConfig(warmup_ratio=-0.1)
        
        with pytest.raises(ValueError, match="warmup_ratio must be in"):
            TrainingConfig(warmup_ratio=1.5)
    
    def test_invalid_eval_strategy_raises(self):
        """Test that invalid eval_strategy raises ValueError."""
        with pytest.raises(ValueError, match="eval_strategy must be"):
            TrainingConfig(eval_strategy="invalid")


class TestQuantizationConfig:
    """
    Tests for QuantizationConfig validation.
    
    This is critical - tests the bug we found where 8-bit configs were
    created with default quant_type="nf4" instead of "int8".
    """
    
    def test_default_config_is_4bit_nf4(self):
        """Test that default config is 4-bit with nf4."""
        config = QuantizationConfig()
        assert config.bits == 4
        assert config.quant_type == "nf4"
        assert config.use_double_quant == True
    
    def test_8bit_requires_int8_quant_type(self):
        """
        CRITICAL: Test that 8-bit quantization requires int8 quant_type.
        
        This is the bug we found - creating QuantizationConfig(bits=8)
        with default quant_type="nf4" should raise ValueError.
        """
        with pytest.raises(ValueError, match="quant_type for 8-bit must be 'int8'"):
            QuantizationConfig(bits=8)  # Missing quant_type, defaults to "nf4"
        
        with pytest.raises(ValueError, match="quant_type for 8-bit must be 'int8'"):
            QuantizationConfig(bits=8, quant_type="nf4")
        
        with pytest.raises(ValueError, match="quant_type for 8-bit must be 'int8'"):
            QuantizationConfig(bits=8, quant_type="fp4")
    
    def test_8bit_valid_with_int8(self):
        """Test that 8-bit with int8 quant_type is valid."""
        config = QuantizationConfig(bits=8, quant_type="int8")
        assert config.bits == 8
        assert config.quant_type == "int8"
    
    def test_4bit_requires_nf4_or_fp4(self):
        """Test that 4-bit quantization requires nf4 or fp4."""
        with pytest.raises(ValueError, match="quant_type for 4-bit must be"):
            QuantizationConfig(bits=4, quant_type="int8")
    
    def test_4bit_valid_quant_types(self):
        """Test that nf4 and fp4 are valid for 4-bit."""
        config_nf4 = QuantizationConfig(bits=4, quant_type="nf4")
        assert config_nf4.quant_type == "nf4"
        
        config_fp4 = QuantizationConfig(bits=4, quant_type="fp4")
        assert config_fp4.quant_type == "fp4"
    
    def test_invalid_bits_raises(self):
        """Test that invalid bits values raise ValueError."""
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            QuantizationConfig(bits=2, quant_type="nf4")
        
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            QuantizationConfig(bits=16, quant_type="nf4")
    
    def test_invalid_compute_dtype_raises(self):
        """Test that invalid compute_dtype raises ValueError."""
        with pytest.raises(ValueError, match="compute_dtype must be"):
            QuantizationConfig(bits=4, compute_dtype="invalid")
    
    def test_valid_compute_dtypes(self):
        """Test that float16, bfloat16, float32 are valid."""
        for dtype in ["float16", "bfloat16", "float32"]:
            config = QuantizationConfig(bits=4, compute_dtype=dtype)
            assert config.compute_dtype == dtype
    
    @pytest.mark.skipif(
        importlib.util.find_spec("bitsandbytes") is None,
        reason="bitsandbytes not installed"
    )
    def test_to_bnb_config_4bit(self):
        """Test conversion to BitsAndBytesConfig for 4-bit."""
        config = QuantizationConfig(bits=4, quant_type="nf4")
        bnb_config = config.to_bnb_config()
        
        assert bnb_config.load_in_4bit == True
        assert bnb_config.bnb_4bit_quant_type == "nf4"
    
    def test_to_bnb_config_8bit(self):
        """Test conversion to BitsAndBytesConfig for 8-bit."""
        config = QuantizationConfig(bits=8, quant_type="int8")
        bnb_config = config.to_bnb_config()
        
        assert bnb_config.load_in_8bit == True


class TestDeploymentConfig:
    """
    Tests for DeploymentConfig validation.
    
    DeploymentConfig now only supports GGUF formats (f16, bf16, q8_0) which
    use the bundled llama.cpp conversion code.
    """
    
    def test_default_config_valid(self):
        """Test that default DeploymentConfig is valid."""
        config = DeploymentConfig()
        assert config.save_merged_fp16 == True
        assert config.save_gguf_q8_0 == False  # Not enabled by default
    
    def test_gguf_formats_supported(self):
        """Test that supported GGUF formats can be enabled."""
        config = DeploymentConfig(
            save_gguf_f16=True,
            save_gguf_bf16=True,
            save_gguf_q8_0=True
        )
        
        assert config.save_gguf_f16 == True
        assert config.save_gguf_bf16 == True
        assert config.save_gguf_q8_0 == True
    
    def test_has_any_output(self):
        """Test has_any_output() method."""
        # Default has merged_fp16
        config = DeploymentConfig()
        assert config.has_any_output() == True
        
        # All disabled
        config_empty = DeploymentConfig(save_merged_fp16=False)
        assert config_empty.has_any_output() == False
        
        # GGUF only
        config_gguf = DeploymentConfig(
            save_merged_fp16=False,
            save_gguf_q8_0=True
        )
        assert config_gguf.has_any_output() == True
    
    def test_get_gguf_formats(self):
        """Test get_gguf_formats() method."""
        config = DeploymentConfig(
            save_gguf_f16=True,
            save_gguf_bf16=True,
            save_gguf_q8_0=True
        )
        
        formats = config.get_gguf_formats()
        assert formats == ["f16", "bf16", "q8_0"]
    
    def test_no_gguf_formats(self):
        """Test get_gguf_formats() returns empty list when none enabled."""
        config = DeploymentConfig()
        assert config.get_gguf_formats() == []
    
    def test_multiple_formats_simultaneously(self):
        """Test enabling multiple output formats at once."""
        config = DeploymentConfig(
            save_merged_fp16=True,
            save_gguf_f16=True,
            save_gguf_q8_0=True
        )
        
        assert config.has_any_output() == True
        assert len(config.get_gguf_formats()) == 2


class TestConfigIntegration:
    """
    Integration tests for config usage patterns.
    
    Tests realistic usage patterns to catch edge cases.
    """
    
    def test_trainer_default_quant_configs_valid(self):
        """
        Test that default quantization configs used in trainer are valid.
        
        Mimics the pattern from trainer.py lines 188-199.
        """
        # 8-bit config (as used in trainer)
        quant_config_8bit = QuantizationConfig(
            bits=8,
            quant_type="int8",
            compute_dtype="float16"
        )
        assert quant_config_8bit.bits == 8
        
        # 4-bit config (as used in trainer)
        quant_config_4bit = QuantizationConfig(
            bits=4,
            quant_type="nf4",
            compute_dtype="float16",
            use_double_quant=True
        )
        assert quant_config_4bit.bits == 4
    
    def test_deployment_config_all_formats(self):
        """Test deploying to all GGUF formats simultaneously."""
        config = DeploymentConfig(
            save_merged_fp16=True,
            save_gguf_f16=True,
            save_gguf_bf16=True,
            save_gguf_q8_0=True,
        )
        
        assert config.has_any_output() == True
        assert len(config.get_gguf_formats()) == 3
    
    def test_minimal_ft_config(self):
        """Test minimal config for FT-only pipeline."""
        lora_config = LoraConfig()
        training_config = TrainingConfig()
        
        # Should be valid
        assert lora_config.r > 0
        assert training_config.num_train_epochs > 0
    
    def test_full_compression_pipeline_configs(self):
        """Test configs for full compression pipeline."""
        ft_config = LoraConfig(r=16)
        recovery_config = QLoraConfig(train_bits=4)
        training_config = TrainingConfig(num_train_epochs=3)
        deployment_config = DeploymentConfig(
            save_merged_fp16=True,
            save_gguf_q8_0=True,
        )
        
        # All configs should be valid
        assert ft_config.r == 16
        assert recovery_config.train_bits == 4
        assert deployment_config.save_gguf_q8_0 == True


class TestConfigEdgeCases:
    """Tests for edge cases and potential gotchas."""
    
    def test_quantization_config_bit_quant_type_mismatch_cases(self):
        """
        Comprehensive test of all bit/quant_type mismatch combinations.
        
        This ensures we catch the bug in all possible forms.
        """
        # Valid combinations
        valid_combos = [
            (4, "nf4"),
            (4, "fp4"),
            (8, "int8"),
        ]
        
        for bits, quant_type in valid_combos:
            config = QuantizationConfig(bits=bits, quant_type=quant_type)
            assert config.bits == bits
            assert config.quant_type == quant_type
        
        # Invalid combinations that should raise
        invalid_combos = [
            (4, "int8", "4-bit must be"),
            (8, "nf4", "8-bit must be 'int8'"),
            (8, "fp4", "8-bit must be 'int8'"),
        ]
        
        for bits, quant_type, error_msg in invalid_combos:
            with pytest.raises(ValueError, match=error_msg):
                QuantizationConfig(bits=bits, quant_type=quant_type)
    
    def test_deployment_config_gguf_only(self):
        """
        Test deployment config with only GGUF outputs.
        
        Users who only need GGUF can disable merged_fp16 to save disk space.
        """
        config = DeploymentConfig(
            save_merged_fp16=False,
            save_gguf_q8_0=True
        )
        
        assert config.save_merged_fp16 == False
        assert config.save_gguf_q8_0 == True
        assert config.has_any_output() == True
    
    def test_zero_and_negative_values_caught(self):
        """Test that zero and negative values are caught across configs."""
        invalid_configs = [
            (LoraConfig, {"r": 0}),
            (LoraConfig, {"r": -1}),
            (LoraConfig, {"lora_alpha": 0}),
            (TrainingConfig, {"num_train_epochs": 0}),
            (TrainingConfig, {"learning_rate": 0}),
            (TrainingConfig, {"learning_rate": -0.001}),
        ]
        
        for config_class, kwargs in invalid_configs:
            with pytest.raises(ValueError):
                config_class(**kwargs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
