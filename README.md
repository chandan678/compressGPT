# compressGPT

**compressGPT** is a flexible, modular training pipeline designed to bridge the gap between large foundation models and efficient edge-ready deployment.

It orchestrates the full lifecycle of Large Language Model (LLM) optimization — from supervised fine-tuning, through post-quantization recovery, to production-ready artifact generation — with a single, composable API.

Unlike rigid training scripts, compressGPT allows developers to define **custom compression workflows** by composing high-level stages such as `ft`, `compress_4bit`, and `deploy`. Whether you need a high-accuracy FP16 model for server inference or a highly compressed GGUF model for CPU-only deployment, compressGPT automates tokenization, adapter training, memory-efficient evaluation, and artifact generation to deliver the **smallest runnable model that preserves task-level accuracy**.

---

## 🚀 Quick Start

Below is a complete example that transforms a CSV dataset into a compressed, deployment-ready 4-bit Llama-3 model.

```python
from compressgpt import (
    CompressTrainer,
    DatasetBuilder,
    TrainingConfig,
    DeploymentConfig,
)

prompt_template = (
    'Classify this notification as "Important" or "Ignore".\n'
    'Important: Security alerts, direct messages, payment confirmations.\n'
    'Ignore: Marketing promos, news digests, social media likes.\n\n'
    'Notification: {text}\n'
    'Answer:'
)

MODEL_ID = "meta-llama/Llama-3.2-1B"

# Build dataset
builder = DatasetBuilder(
    data_path="notifications.csv",
    model_id=MODEL_ID,
    prompt_template=prompt_template,
    input_column_map={"text": "message_body"},
    label_column="label",
).build()

# Run compression pipeline
trainer = CompressTrainer(
    model_id=MODEL_ID,
    dataset_builder=builder,
    stages=["ft", "compress_4bit", "deploy"],
    training_config=TrainingConfig(
        num_train_epochs=1,
        eval_strategy="epoch",
        save_strategy="epoch",
    ),
    deployment_config=DeploymentConfig(
        save_merged_fp16=True,     # Canonical dense model
        save_quantized_4bit=True,  # BitsAndBytes 4-bit
        save_gguf_q4_0=True,       # GGUF for llama.cpp
    ),
)

results = trainer.run()

print("Training complete!")
print(results)
