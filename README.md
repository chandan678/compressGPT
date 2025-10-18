# compressGPT

**compressGPT** is an open-source project that automates **LLM compression and optimization**.  
Given any dataset and base model, it builds a complete pipeline to produce the **smallest runnable model** that preserves target accuracy and performance.

## 🚀 Goal
To make large language models practical for real-world and CPU-based deployments by:
- Automatically fine-tuning and compressing models (quantization, pruning, distillation)
- Tracking trade-offs between accuracy, size, and latency
- Outputting an optimized model ready for inference

## 🧠 Core Idea
```text
Model + Dataset  ➜  compressGPT  ➜  Smallest Accurate Model
is 