# PaliGemma Implementation

A from-scratch PyTorch implementation of Google's **PaliGemma** vision-language model. This repository reimplements the complete architecture — SigLIP vision encoder, Gemma language decoder, and multimodal fusion — without relying on high-level libraries such as Hugging Face `transformers` for model components, while using Hugging Face libraries only for tokenization and weight loading.

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
│  • N× GemmaLayer:                      │
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
| **Weight Loader** | `src/utils.py` | Load pretrained weights from Hugging Face safetensors files |
| **Inference** | `src/inference.py` | Autoregressive generation with top-p sampling and KV cache |
| **Config** | `src/config/` | Dataclasses for vision, language, and multimodal configs |
| **Preprocessing** | `src/pre_processing/` | Image resize/normalize, prompt construction with image tokens |

### Key Features

- **SigLIP Vision Encoder** — Conv2d patch embedding (patch_size=16, 196 patches for 224×224) with learned position embeddings and 12 Transformer encoder layers using GELU-tanh activation
- **Gemma Decoder** — Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE), pre/post-attention and pre/post-MLP RMSNorm, GELU-tanh gated MLP
- **Multimodal Fusion** — Linear projection (768 → 2048) followed by masked scatter to interleave image features with text token embeddings
- **KV Cache** — Custom `KVCache` for efficient autoregressive generation
- **Inference Pipeline** — End-to-end generation with top-p (nucleus) sampling and temperature control
- **Weight Loading** — Load pretrained PaliGemma weights from Hugging Face safetensors format
- **Special Tokens** — 1024 location tokens (`<loc_0000>`–`<loc_1023>`), 128 segmentation tokens (`<seg_000>`–`<seg_127>`), and `<image>` token
- **Weight Tying** — LM head shares weights with the embedding layer

## Getting Started

### Prerequisites

- Python 3.12+
- PyTorch 2.12+

### Installation

```bash
git clone https://github.com/NawalSaw/Pali-Gemma-Implementation.git
cd Pali-Gemma-Implementation

python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### Pretrained Weights

Download a pretrained PaliGemma model from [Hugging Face](https://huggingface.co/google) (e.g., `google/paligemma-3b-mix-224`) and provide the local path to the inference script.

### Inference

```bash
python src/inference.py \
    --model_path /path/to/paligemma-weights \
    --image_path /path/to/image.jpg \
    --prompt "What is shown in this image?" \
    --max_tokens 100 \
    --temperature 1.0 \
    --top_p 0.9 \
    --do_sample True
```

### Programmatic Usage

```python
from src.config.pali_gemma_config import PaliGemmaConfig
from src.gemma.pali_gemma_modeling import PaliGemmaModel, KVCache
from src.pre_processing.paligemma_preproccessing import PaligemmaPreprocessing
from src.utils import load_hf_model
from transformers import AutoTokenizer

# Load pretrained model and tokenizer
model, tokenizer = load_hf_model("/path/to/paligemma-weights", device="cuda")

# Preprocess
preprocessor = PaligemmaPreprocessing(tokenizer, num_image_tokens=196, image_size=224)
input_ids, pixel_values, attention_mask = get_model_inputs(preprocessor, "cuda", "Describe this image.", "image.jpg")

# Forward pass
logits = model(input_ids, pixel_values, attention_mask)
```

## Repository Structure

```
├── src/
│   ├── config/
│   │   ├── vision_config.py            # SigLIP vision encoder config
│   │   ├── gemma_config.py             # Gemma language model config
│   │   └── pali_gemma_config.py        # Multimodal model config
│   ├── components_vision_model/
│   │   ├── embedding.py                # Patch + position embeddings
│   │   ├── attention.py                # SigLIP multi-head attention
│   │   ├── mlp.py                      # GELU-tanh MLP
│   │   ├── encoder.py                  # 12-layer Transformer encoder
│   │   └── vision_model.py             # SigLIP vision model assembly
│   ├── gemma/
│   │   ├── gemma_model.py              # Gemma decoder components
│   │   └── pali_gemma_modeling.py      # PaliGemma multimodal model
│   ├── pre_processing/
│   │   └── paligemma_preproccessing.py # Image preprocessing & tokenization
│   ├── inference.py                    # Autoregressive generation pipeline
│   ├── utils.py                        # Hugging Face weight loader
│   └── data_analysis.ipynb             # PyTorch scratchpad (not model analysis)
├── architecture/
│   └── architecture.webp               # Architecture diagram
├── requirements.txt                    # Python dependencies
└── README.md
```

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.12+ | Deep learning framework |
| `numpy` | 2.4+ | Numerical operations |
| `pillow` | 11.1+ | Image loading and processing |
| `transformers` | 4.50+ | Tokenizer and weight config loading |
| `safetensors` | 0.6+ | Safe tensor weight loading |
| `fire` | 0.7+ | CLI argument parsing |
| `pandas` | 2.3+ | *(unused)* |

## API Overview

### `PaliGemmaModel` (`src/gemma/pali_gemma_modeling.py`)
The main multimodal model. Composed of `SiglipVisionModel`, `PaliGemmaMultiModalProjector`, and `GemmaForCausalLM`.

```python
model = PaliGemmaModel(config)
model.tie_weights()                        # Bind LM head to embedding weights
logits = model(input_ids, pixel_values, attention_mask, kv_cache)
```

### `KVCache`
Key-value cache for autoregressive decoding.

```python
cache = KVCache()
cache.update(key_states, value_states, layer_idx)  # → (keys, values)
cache.get_seq_length()                              # → cached sequence length
```

### `PaligemmaPreprocessing` (`src/pre_processing/paligemma_preproccessing.py`)
Prepares images and text prompts for the model. Resizes images to 224×224, normalizes with mean=0.5 and std=0.5, and prepends `<image>` tokens to the prompt.

```python
preprocessor = PaligemmaPreprocessing(tokenizer, num_image_tokens=196, image_size=224)
data = preprocessor([image], ["Describe this image."])
# Returns: pixel_values, input_ids, attention_mask
```

### `load_hf_model` (`src/utils.py`)
Loads a pretrained PaliGemma model from a directory of Hugging Face safetensors files.

```python
model, tokenizer = load_hf_model("/path/to/model", device="cuda")
```

### Inference (`src/inference.py`)
End-to-end generation with top-p sampling, temperature scaling, and KV cache.

```bash
python src/inference.py --model_path <path> --image_path <path> --prompt "<prompt>"
```

## Status

The model architecture, weight loading, and inference pipeline are implemented and functional. There are no `__init__.py` files, so the package must be used from the repository root directory.

## References
- [Umar's PaliGemma Implementation](https://github.com/hkproj/pytorch-paligemma)
- [PaliGemma: A Versatile Vision-Language Model](https://arxiv.org/abs/2407.07726) — Google DeepMind
- [SigLIP: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- [Gemma: Open Models Based on Gemini Research](https://ai.google.dev/gemma)
