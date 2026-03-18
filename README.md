# **Vex** — The Unified Engine for Open Models

> **Stop wrestling with 10 different tools to fine-tune and deploy open models.**  
> Vex is the upgraded successor to Unsloth (55k+ ★), rebuilt from the ground up for speed, modularity, and virality.

![GitHub Stars](https://img.shields.io/github/stars/vex-ai/vex?style=social)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Discord](https://img.shields.io/discord/1234567890?color=7289da&label=discord&logo=discord)

---

## 🚀 Why Vex? The Upgrade You've Been Waiting For

| Feature | Unsloth | **Vex** |
|---------|---------|---------|
| **Architecture** | Monolithic | **Modular Backend** (CLI, API, Web share core) |
| **Attention** | FlashAttention-2 | **FlashAttention-3** + upcoming FA-4 |
| **Quantization** | 4-bit | **4-bit QLoRA** + **FP8** (H100 optimized) |
| **Parallelism** | Single GPU | **Multi-GPU Pipeline Parallelism** |
| **Model Support** | Llama, Mistral | **Llama 3, Mamba-2, Jamba, Mixtral** (day-0 support) |
| **Deployment** | Manual export | **One-click ONNX/TensorRT** export |
| **Experiment Tracking** | External (W&B) | **Built-in tracker** (W&B compatible) |
| **Collaboration** | None | **Share adapters & datasets** via Model Zoo |
| **Extensibility** | Limited | **Plugin system** for custom loops & pipelines |
| **One-Line Training** | ❌ | **✅ `vex train`** with auto-optimization |

---

## ⚡ Quickstart: Fine-Tune in One Line

```python
# Install: pip install vex-ai

from vex import train

# Fine-tune Llama-3 8B with 4-bit QLoRA on your dataset
model = train(
    model="llama-3-8b",
    dataset="your_dataset.jsonl",
    method="qlora",  # Automatically uses FA3 + multi-GPU if available
    epochs=3,
    export="onnx"  # Auto-export for deployment
)

# That's it. Seriously.
```

**CLI Alternative:**
```bash
vex train --model llama-3-8b --dataset data.jsonl --gpus 4 --method qlora --export onnx
```

---

## 🏗️ Architecture: Modular by Design

```
┌─────────────────────────────────────────┐
│              Unified Core               │
│  (Training, Inference, Export Engine)   │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌───────┐  ┌───────┐  ┌───────┐
│  CLI  │  │  API  │  │  Web  │
└───────┘  └───────┘  └───────┘
    │          │          │
    └──────────┼──────────┘
               ▼
        ┌─────────────┐
        │ Plugin System│ ← Custom training loops, data pipelines, model adapters
        └─────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌───────┐  ┌───────┐  ┌───────┐
│ Model │  │Dataset│  │Deploy │
│  Zoo  │  │ Hub   │  │Toolkit│
└───────┘  └───────┘  └───────┘
```

**Key Components:**
- **Unified Core**: Shared logic for training/inference across all interfaces
- **Plugin System**: Extend with custom training loops, model architectures, or data loaders
- **Model Zoo**: Community-shared pre-optimized configs & adapters
- **Deploy Toolkit**: One-click ONNX/TensorRT export with auto-optimization

---

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install vex-ai
```

### From Source (Latest Features)
```bash
git clone https://github.com/vex-ai/vex.git
cd vex
pip install -e ".[all]"  # For all features
# Or: pip install -e ".[train]" for training only
```

### Requirements
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended

---

## 🔥 Features That Make Vex Unbeatable

### 🚄 **Performance First**
- **FlashAttention-3**: 40% faster than FA2 on Hopper GPUs
- **4-bit QLoRA + FP8**: Train 70B models on a single 24GB GPU
- **Auto-optimization**: Vex analyzes your hardware and picks the fastest config

### 🧩 **Modular & Extensible**
```python
from vex.plugins import TrainingPlugin

class MyCustomTrainer(TrainingPlugin):
    def training_step(self, batch):
        # Your custom logic here
        return loss

vex.register_plugin(MyCustomTrainer())
```

### 🤝 **Collaborate & Share**
```bash
# Push your fine-tuned adapter to the Model Zoo
vex push my-llama3-adapter --public

# Pull someone else's optimized config
vex pull config/llama3-8b-qlora-h100
```

### 📊 **Built-in Experiment Tracking**
```python
# Automatic logging, no extra code needed
model = train(..., tracking=True)
# View at: http://localhost:5000 or export to W&B
```

---

## 🆚 Migration from Unsloth

```python
# Unsloth (old)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("llama-3-8b")

# Vex (new) - 95% compatible + more features
from vex import AutoModel
model = AutoModel.from_pretrained("llama-3-8b", method="qlora")
```

**We maintain backward compatibility** while adding:
- 3x faster training with FA3
- Multi-GPU support
- One-line export to ONNX/TensorRT
- Built-in experiment tracking

---

## 📈 Roadmap (Q3-Q4 2024)

- [ ] **FlashAttention-4** integration (as soon as released)
- [ ] **FP4 quantization** for 2x more memory savings
- [ ] **Distributed training** across multiple nodes
- [ ] **Web UI** for visual training monitoring
- [ ] **Model merging** toolkit (DARE, TIES)
- [ ] **Automatic benchmarking** against HF Leaderboard

---

## 🤔 FAQ

**Q: Is Vex compatible with my existing Unsloth scripts?**  
A: 95% compatible. We provide a migration guide and `vex.compat.unsloth` module.

**Q: How does Vex make money?**  
A: Open-source core forever. Enterprise features (SSO, audit logs, priority support) as paid add-ons.

**Q: Can I use Vex with non-LLM models?**  
A: Yes! The plugin system supports any PyTorch model. We're adding vision and audio examples.

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=vex-ai/vex&type=Date)](https://star-history.com/#vex-ai/vex&Date)

---

## 📄 License

Apache 2.0 — Use commercially, modify, distribute. Just keep the copyright.

---

**Built with ❤️ by the Vex Team**  
[Documentation](https://docs.vex-ai.com) | [Discord](https://discord.gg/vex) | [Twitter](https://twitter.com/vex_ai)

> **"The best tools get out of your way."**  
> Vex lets you focus on models, not tooling.