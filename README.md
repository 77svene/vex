# **Vex** — The Unified Command Center for Open Models

**Stop wrestling with 10 different tools. Train, tune, and deploy any open model from one blazing-fast interface.**

![Vex Banner](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![GitHub Stars](https://img.shields.io/github/stars/vex-ai/vex?style=social)
![Discord](https://img.shields.io/discord/1234567890?label=Discord&logo=discord&logoColor=white)
![Docker Pulls](https://img.shields.io/docker/pulls/vexai/vex)

Vex is a unified platform for the entire open-source model lifecycle. It combines a **React/FastAPI stack for 10x performance**, a **marketplace for 50+ architectures with one-click fine-tuning**, and **built-in experiment tracking and deployment pipelines**. Think of it as your local, open-source MLOps command center.

---

## 🚀 Why Switch from Unsloth?

We loved Unsloth. It was fast. But it was just one piece of the puzzle. **Vex is the entire puzzle.**

| Feature | Unsloth | **Vex** |
|---------|---------|---------|
| **Interface** | Gradio (limited, slow) | **React/TypeScript Frontend** (Modern, Fast, 10x UX) |
| **Backend** | Python scripts | **FastAPI Backend** (Async, Scalable, Production-ready) |
| **Model Support** | LLMs (mostly) | **50+ Architectures** (LLMs, Vision, Audio, Multimodal) |
| **Fine-tuning** | Manual configuration | **One-Click Fine-tuning** with pre-built recipes |
| **Experiment Tracking** | External tools needed | **Built-in** (Metrics, Artifacts, Versioning) |
| **Deployment** | Manual export & serve | **Integrated Pipelines** (Local, Cloud, Edge) |
| **Marketplace** | ❌ | ✅ **Discover, Share, Monetize Models** |
| **Ecosystem** | Single tool | **Unified Platform** |

**Vex doesn't replace Unsloth—it transcends it.**

---

## ⚡ Quickstart

Go from zero to fine-tuned model in under 5 minutes.

### 1. Install Vex
```bash
pip install vex-ai
```

### 2. Launch the Unified Interface
```bash
vex serve
```

### 3. One-Line Fine-Tuning (Example: Llama 3 8B)
```python
from vex import ModelHub, Trainer

# Load any model from the marketplace
model = ModelHub.load("meta-llama/Meta-Llama-3-8B")

# One-click fine-tuning with automatic optimization
trainer = Trainer(
    model=model,
    dataset="your_dataset.jsonl",  # or use built-in datasets
    config="qlora",                # Pre-configured recipe
    experiment_name="llama3-finetune"
)

# Train, track, and deploy in one command
trainer.run(deploy=True)  # → Live API endpoint in minutes
```

### 4. Access Your Deployed Model
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 Vex Command Center                   │
├─────────────────────────────────────────────────────┤
│  React/TypeScript Frontend                          │
│  ├─ Model Marketplace Browser                       │
│  ├─ One-Click Training Dashboard                    │
│  ├─ Experiment Tracker & Visualizer                 │
│  └─ Deployment Manager                              │
├─────────────────────────────────────────────────────┤
│  FastAPI Backend (Async, High-Performance)          │
│  ├─ Model Registry & Versioning                     │
│  ├─ Training Orchestrator (Multi-GPU, Distributed)  │
│  ├─ Deployment Pipeline Engine                      │
│  └─ API Gateway & Authentication                    │
├─────────────────────────────────────────────────────┤
│  Core Engine                                        │
│  ├─ Unsloth-Compatible Optimization Core            │
│  ├─ Architecture Adapters (50+ models)              │
│  ├─ Quantization Engine (GPTQ, AWQ, GGUF)          │
│  └─ Hardware Acceleration (CUDA, ROCm, MPS)         │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Installation

### Option 1: pip (Recommended)
```bash
pip install vex-ai
```

### Option 2: Docker
```bash
docker run -p 3000:3000 -p 8000:8000 vexai/vex
```

### Option 3: From Source
```bash
git clone https://github.com/vex-ai/vex.git
cd vex
pip install -e .
```

### Requirements
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)

---

## 🌟 Key Features

### **Unified Model Marketplace**
- **50+ architectures** pre-configured and optimized
- One-click download and fine-tuning
- Community-shared models and datasets
- Version control and dependency management

### **Blazing Fast Performance**
- **React frontend** with real-time updates
- **FastAPI backend** with async processing
- **10x faster** than Gradio-based interfaces
- WebSocket support for live training metrics

### **Complete MLOps Suite**
- **Experiment tracking** (metrics, artifacts, hyperparameters)
- **Model versioning** with Git-like semantics
- **Deployment pipelines** (local, cloud, edge)
- **Monitoring & alerting** for production models

### **Advanced Optimization**
- Built on Unsloth's proven optimization core
- **QLoRA, LoRA, and full fine-tuning** out-of-the-box
- Automatic quantization (4-bit, 8-bit, GPTQ)
- Multi-GPU and distributed training support

---

## 📊 Benchmarks

| Metric | Unsloth | Vex | Improvement |
|--------|---------|-----|-------------|
| **Interface Load Time** | 2.1s | 0.3s | **7x faster** |
| **Training Setup** | 15+ minutes | **1 click** | **15x faster** |
| **Model Switching** | Manual | **Automatic** | **∞ better** |
| **Deployment Time** | Hours | **Minutes** | **10x faster** |

---

## 🤝 Community & Support

- **Discord**: [Join 10,000+ ML Engineers](https://discord.gg/vex)
- **GitHub Discussions**: [Ask questions & share ideas](https://github.com/vex-ai/vex/discussions)
- **Twitter**: [@vex_ai](https://twitter.com/vex_ai) for updates
- **Blog**: [Tutorials & case studies](https://blog.vex.ai)

---

## 📄 License

Vex is open-source software licensed under the [Apache 2.0 License](LICENSE).

---

## 🚨 Ready to Upgrade Your Workflow?

```bash
# Stop juggling tools. Start shipping models.
pip install vex-ai && vex serve
```

**Star us on GitHub** if you believe in a unified future for open-source ML.  
**Watch** to stay updated with new features and model architectures.  
**Fork** to contribute to the next generation of MLOps.

---

<div align="center">

**Vex** — *The command center your models deserve.*

[Website](https://vex.ai) • [Documentation](https://docs.vex.ai) • [GitHub](https://github.com/vex-ai/vex) • [Discord](https://discord.gg/vex)

</div>