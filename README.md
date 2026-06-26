# PaliGemma Implementation

A from-scratch PyTorch implementation of Google's **PaliGemma** vision-language model. This repository reimplements the complete architecture — SigLIP vision encoder, Gemma language decoder, and multimodal fusion — without relying on high-level libraries such as Hugging Face `transformers`.

## Architecture

```
Input Image               Input Text
(224×224×3 RGB)           (tokenized IDs)
        │                        │
        ▼                        ▼
┌─────────────────┐   ┌─────────────────────┐
│  SigLIP Vision  │   │  Gemma Text Embed   │
│  Model (ViT)    │   │  (nn.Embedding)     │
│  • Patch Embed  │   └──────────┬──────────┘
│  • 12× Encoder  │              │
│    Layers       │              │
│  • LayerNorm    │              │
└────────┬────────┘              │
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│  Multimodal Projection                  │
│  (nn.Linear: 768 → 2048)               │
│  + Merge image features into text       │
│    token sequence (scatter with mask)   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Gemma Language Model                   │
│  • Embedding scaling (×√d)             │
│  • N× GemmaDecoderLayer:               │
│      - Pre/Post-Attn RMSNorm           │
│      - Grouped-Query Attention + RoPE  │
│      - Pre/Post-MLP RMSNorm            │
│      - GELU-tanh MLP (gate/up/down)    │
│  • Final RMSNorm                       │
│  • LM Head (tied weights)              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
           Output Logits
           (over vocabulary)
```

### Components

| Module | File | Description |
|---|---|---|
| **SigLIP Vision Model** | `src/components_vision_model/` | Vision Transformer with patch embeddings, 12 encoder layers, LayerNorm |
| **Gemma Language Model** | `src/gemma/gemma_model.py` | Decoder with GQA, RoPE, RMSNorm, GELU-tanh MLP |
| **PaliGemma Model** | `src/gemma/pali_gemma_modeling.py` | Multimodal fusion — vision projection + masked scatter merge + language model |
| **Config** | `src/config/` | Dataclasses for vision, language, and multimodal configs |
| **Preprocessing** | `src/pre_processing/` | Image resize/normalize, prompt construction with image tokens |

### Key Features

- **SigLIP Vision Encoder** — Conv2d patch embedding (patch_size=16, 196 patches for 224×224) with learned position embeddings and 12 Transformer encoder layers using GELU-tanh activation
- **Gemma Decoder** — Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE), pre/post-attention and pre/post-MLP RMSNorm, GELU-tanh gated MLP
- **Multimodal Fusion** — Linear projection (768 → 2048) followed by masked scatter to interleave image features with text token embeddings
- **KV Cache** — Custom `KVCache` for efficient autoregressive generation
- **Special Tokens** — 1024 location tokens (`<loc_0000>`–`<loc_1023>`), 128 segmentation tokens (`<seg_000>`–`<seg_127>`), and `<image>` token
- **Weight Tying** — LM head shares weights with the embedding layer
- **Zero external dependencies** — No Hugging Face `transformers`, pure PyTorch + NumPy + PIL

## Getting Started

### Prerequisites

- Python 3.12+
- PyTorch 2.12+

### Installation

```bash
# Clone the repository
git clone https://github.com/NawalSaw/Pali-Gemma-Implementation.git
cd Pali-Gemma-Implementation

# Create and activate virtual environment
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pillow  # (required but not listed in requirements.txt)
```

### Usage

```python
from src.config.pali_gemma_config import PaliGemmaConfig
from src.gemma.pali_gemma_modeling import PaliGemmaModel

config = PaliGemmaConfig(...)
model = PaliGemmaModel(config)
model.tie_weights()

# Forward pass
logits = model(input_ids, pixel_values, attention_mask)
```

## Repository Structure

```
├── src/
│   ├── config/
│   │   ├── vision_config.py         # SigLIP vision encoder config
│   │   ├── gemma_config.py          # Gemma language model config
│   │   └── pali_gemma_config.py     # Multimodal model config
│   ├── components_vision_model/
│   │   ├── embedding.py             # Patch + position embeddings
│   │   ├── attention.py             # SigLIP multi-head attention
│   │   ├── mlp.py                   # GELU-tanh MLP
│   │   ├── encoder.py               # 12-layer Transformer encoder
│   │   └── vision_model.py          # SigLIP vision model assembly
│   ├── gemma/
│   │   ├── gemma_model.py           # Gemma decoder (GQA, RoPE, MLP, RMSNorm)
│   │   └── pali_gemma_modeling.py   # PaliGemma multimodal model + KVCache
│   ├── pre_processing/
│   │   └── paligemma_preproccessing.py  # Image preprocessing & tokenization
│   ├── data_analysis.ipynb          # Exploration notebook
│   └── train.py                     # Training script (placeholder)
├── architecture/
│   └── architecture.webp            # Architecture diagram
├── requirements.txt                 # Python dependencies
└── README.md
```

## Status

The model architecture and forward pass are implemented. The training loop and inference pipeline have **not yet been built** — `src/train.py` is currently a placeholder.

## License

This project is open source. No license has been explicitly declared.

## References

- [PaliGemma: A Versatile Vision-Language Model](https://arxiv.org/abs/2407.07726) — Google DeepMind
- [SigLIP: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- [Gemma: Open Models Based on Gemini Research](https://ai.google.dev/gemma)
