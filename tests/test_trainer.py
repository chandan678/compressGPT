"""
Test cases for CompressTrainer class.

These tests mock peft/trl at the module level to avoid import errors,
then test the actual CompressTrainer implementation.

Run with: pytest tests/test_trainer.py -v
"""

import sys
from importlib.machinery import ModuleSpec
import pytest
from unittest.mock import Mock, MagicMock, PropertyMock, patch

# Create proper mock modules with __spec__ for transformers compatibility
# transformers checks importlib.util.find_spec() which needs __spec__ set properly
mock_peft = MagicMock()
mock_peft.__spec__ = ModuleSpec("peft", None)
mock_peft.LoraConfig = MagicMock()
mock_peft.PeftModel = MagicMock()
mock_peft.get_peft_model = MagicMock()
mock_peft.prepare_model_for_kbit_training = MagicMock()
sys.modules['peft'] = mock_peft

mock_trl = MagicMock()
mock_trl.__spec__ = ModuleSpec("trl", None)
mock_trl.SFTTrainer = MagicMock()
mock_trl.SFTConfig = MagicMock()
mock_trl.DataCollatorForCompletionOnlyLM = MagicMock()
sys.modules['trl'] = mock_trl

# Now we can import trainer
from compressgpt.trainer import CompressTrainer
from compressgpt.config import LoraConfig, QLoraConfig, TrainingConfig, QuantizationConfig, DeploymentConfig


@pytest.fixture
def mock_dataset_builder():
    """Create a mock DatasetBuilder with metadata."""
    builder = Mock()
    
    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.column_names = ["text", "gold_label"]
    mock_dataset.__len__ = Mock(return_value=100)
    mock_dataset.remove_columns = Mock(return_value=mock_dataset)
    
    # Mock train_test_split to return dict-like object
    train_mock = Mock()
    train_mock.__len__ = Mock(return_value=95)
    test_mock = Mock()
    test_mock.__len__ = Mock(return_value=5)
    mock_dataset.train_test_split = Mock(return_value={
        "train": train_mock,
        "test": test_mock
    })
    
    builder.dataset = mock_dataset
    
    # Mock metadata
    builder.metadata = {
        "model_id": "mock-model",
        "model_mode": "base",
        "response_trigger": "Answer:",
        "label_space": {
            "labels": ["yes", "no"],
            "label_prefix": " ",
            "label_token_ids": {"yes": 100, "no": 200},
            "valid_token_ids": [100, 200],
            "id_to_label": {100: "yes", 200: "no"},
        },
        "label_counts": {"yes": 50, "no": 50},
        "num_samples": 100,
        "is_train": True
    }
    
    return builder


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens=True: 
        [100] if " yes" in text else [200])
    return tokenizer


class TestCompressTrainerInit:
    """Tests for CompressTrainer initialization."""
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_basic_init(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test basic initialization."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft"]
        )
        
        assert trainer.model_id == "mock-model"
        assert trainer.stages == ["ft"]
        assert trainer.response_trigger == "Answer:"
        assert len(trainer.label_space.labels) == 2
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_init_without_build_raises(self, mock_auto_tokenizer, mock_tokenizer):
        """Test that initialization without built dataset raises error."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        builder = Mock()
        # metadata property raises RuntimeError when accessing
        type(builder).metadata = PropertyMock(side_effect=RuntimeError("not built yet"))
        
        with pytest.raises(RuntimeError, match="has not been built yet"):
            CompressTrainer(
                model_id="mock-model",
                dataset_builder=builder,
                stages=["ft"]
            )
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_invalid_stages_raises(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that invalid stages raise ValueError."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        with pytest.raises(ValueError, match="Invalid stages"):
            CompressTrainer(
                model_id="mock-model",
                dataset_builder=mock_dataset_builder,
                stages=["ft", "invalid_stage"]
            )
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_pad_token_set_on_init(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that pad_token is set during initialization."""
        assert mock_tokenizer.pad_token is None
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft"]
        )
        
        # pad_token should be set to eos_token
        assert mock_tokenizer.pad_token == "<eos>"
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_output_directories_created(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that output directory paths are set correctly."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft", "compress_8bit", "merge"],
            run_dir="output/my_run"
        )
        
        assert trainer.ft_output_dir == "output/my_run/ft_adapter"
        assert trainer.recovery_output_dir == "output/my_run/recovery_adapter"
        assert trainer.merged_output_dir == "output/my_run/merged_model"
        assert trainer.quantized_8bit_dir == "output/my_run/quantized_8bit"
        assert trainer.quantized_4bit_dir == "output/my_run/quantized_4bit"
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_metrics_computer_initialized(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that ComputeMetrics is initialized correctly."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft"]
        )
        
        assert trainer.metrics_computer is not None
        assert hasattr(trainer.metrics_computer, 'as_trainer_callback')


class TestCompressTrainerConfig:
    """Tests for trainer configuration."""
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_default_configs_created(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that default configs are created if not provided."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft"]
        )
        
        assert isinstance(trainer.ft_config, LoraConfig)
        assert isinstance(trainer.recovery_config, QLoraConfig)
        assert isinstance(trainer.training_config, TrainingConfig)
        assert isinstance(trainer.quant_config_8bit, QuantizationConfig)
        assert isinstance(trainer.quant_config_4bit, QuantizationConfig)
        assert isinstance(trainer.deployment_config, DeploymentConfig)
        
        # Verify default quantization configs
        assert trainer.quant_config_8bit.bits == 8
        assert trainer.quant_config_4bit.bits == 4
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_custom_configs_used(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that custom configs are used when provided."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        custom_ft = LoraConfig(r=32, lora_alpha=64)
        custom_training = TrainingConfig(num_train_epochs=10)
        custom_quant_4bit = QuantizationConfig(bits=4, quant_type="fp4")
        custom_deploy = DeploymentConfig(save_gguf_q8_0=True)
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft"],
            ft_config=custom_ft,
            training_config=custom_training,
            quant_config_4bit=custom_quant_4bit,
            deployment_config=custom_deploy
        )
        
        assert trainer.ft_config.r == 32
        assert trainer.ft_config.lora_alpha == 64
        assert trainer.training_config.num_train_epochs == 10
        assert trainer.quant_config_4bit.quant_type == "fp4"
        assert trainer.deployment_config.save_gguf_q8_0 == True


class TestCompressTrainerDatasetSplit:
    """Tests for dataset splitting."""
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_dataset_split_performed(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that dataset is split into train/eval."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        # Mock dataset with train_test_split
        train_mock = Mock()
        train_mock.__len__ = Mock(return_value=95)
        test_mock = Mock()
        test_mock.__len__ = Mock(return_value=5)
        
        mock_dataset = Mock()
        mock_dataset.column_names = ["text", "gold_label"]
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.remove_columns = Mock(return_value=mock_dataset)
        mock_dataset.train_test_split = Mock(return_value={
            "train": train_mock,
            "test": test_mock
        })
        mock_dataset_builder.dataset = mock_dataset
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft"],
            train_test_split=0.05
        )
        
        # train_test_split should have been called
        mock_dataset.train_test_split.assert_called_once()


class TestCompressTrainerDeviceDetection:
    """Tests for device detection."""
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    @patch('torch.cuda.is_available')
    def test_cuda_device_detected(self, mock_cuda, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test CUDA device detection."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        mock_cuda.return_value = True
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft"]
        )
        
        assert trainer.device_type == "cuda"
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_mps_device_detected(self, mock_mps, mock_cuda, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test MPS device detection."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        mock_cuda.return_value = False
        mock_mps.return_value = True
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft"]
        )
        
        assert trainer.device_type == "mps"
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_cpu_fallback(self, mock_mps, mock_cuda, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test CPU fallback when no GPU available."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        mock_cuda.return_value = False
        mock_mps.return_value = False
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft"]
        )
        
        assert trainer.device_type == "cpu"


class TestCompressTrainerMetadata:
    """Tests for metadata handling."""
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_metadata_missing_label_space_raises(self, mock_auto_tokenizer, mock_tokenizer):
        """Test that missing label_space in metadata raises error."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        builder = Mock()
        builder.metadata = {"model_id": "mock-model", "response_trigger": "Answer:"}
        builder.dataset = Mock()
        builder.dataset.column_names = ["text"]
        builder.dataset.__len__ = Mock(return_value=100)
        builder.dataset.remove_columns = Mock(return_value=builder.dataset)
        builder.dataset.train_test_split = Mock(return_value={"train": Mock(), "test": Mock()})
        
        with pytest.raises(ValueError, match="missing 'label_space'"):
            CompressTrainer(
                model_id="mock-model",
                dataset_builder=builder,
                stages=["ft"]
            )
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_metadata_missing_response_trigger_raises(self, mock_auto_tokenizer, mock_tokenizer):
        """Test that missing response_trigger in metadata raises error."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        builder = Mock()
        builder.metadata = {
            "model_id": "mock-model",
            "label_space": {
                "labels": ["yes", "no"],
                "label_prefix": " ",
                "label_token_ids": {"yes": 100, "no": 200},
                "valid_token_ids": [100, 200],
                "id_to_label": {100: "yes", 200: "no"},
            }
        }  # Missing response_trigger
        builder.dataset = Mock()
        builder.dataset.column_names = ["text"]
        builder.dataset.__len__ = Mock(return_value=100)
        builder.dataset.remove_columns = Mock(return_value=builder.dataset)
        builder.dataset.train_test_split = Mock(return_value={"train": Mock(), "test": Mock()})
        
        with pytest.raises(ValueError, match="missing 'response_trigger'"):
            CompressTrainer(
                model_id="mock-model",
                dataset_builder=builder,
                stages=["ft"]
            )


class TestCompressTrainerStageValidation:
    """Tests for training stage validation."""
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_valid_stage_ft(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that 'ft' is a valid stage."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft"]
        )
        
        assert "ft" in trainer.stages
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_valid_stage_compress_8bit(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that 'compress_8bit' is a valid stage."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["compress_8bit"]
        )
        
        assert "compress_8bit" in trainer.stages
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_valid_stage_compress_4bit(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that 'compress_4bit' is a valid stage."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["compress_4bit"]
        )
        
        assert "compress_4bit" in trainer.stages
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_valid_stage_deploy(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that 'deploy' is a valid stage."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["deploy"]
        )
        
        assert "deploy" in trainer.stages
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_valid_stage_merge(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that 'merge' is a valid stage."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["merge"]
        )
        
        assert "merge" in trainer.stages
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_multiple_stages_new(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test multiple new atomic stages can be specified."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        trainer = CompressTrainer(
            model_id="mock-model",
            dataset_builder=mock_dataset_builder,
            stages=["ft", "compress_8bit", "merge", "deploy"]
        )
        
        assert trainer.stages == ["ft", "compress_8bit", "merge", "deploy"]
    
    @patch('compressgpt.trainer.AutoTokenizer.from_pretrained')
    def test_invalid_deprecated_stages_raise(self, mock_auto_tokenizer, mock_dataset_builder, mock_tokenizer):
        """Test that old deprecated stage names now raise ValueError."""
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        # Old stage names should now raise errors
        with pytest.raises(ValueError, match="Invalid stages"):
            CompressTrainer(
                model_id="mock-model",
                dataset_builder=mock_dataset_builder,
                stages=["ft", "quantize_8bit", "recovery", "merge"]
            )
