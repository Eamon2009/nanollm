<h1 align="center"> nanollm</h1>
<img width="1604" height="328" alt="image" src="https://github.com/user-attachments/assets/aa40fd1b-e288-4f82-9a0d-d0c45b724aba" />
# llm.cpp

llm.cpp is the simplest setup for training small language models and image classifiers from scratch. It runs on a single GPU, a multi-GPU node, or even CPU. The code is minimal and hackable -- no frameworks, no config hell, just PyTorch and your data.

You can train a GPT-2 style tokenizer-based LLM on your own text shards and talk to it over CLI. Or flip a switch and train an image classifier on a folder of images. Both share the same training bones: cosine LR, warmup, gradient clipping, mixed precision, and checkpointing.

The whole thing fits in a few files. Read them, change them, break them.

## What's here

| Path | What it does |
|------|-------------|
| `nanollm/cpu/main.py` | Single-GPU LLM training + interactive chat |
| `nanollm/distributed/train.py` | Multi-GPU DDP LLM training |
| `nanollm/cpu/inference.py` | Standalone generation |
| `nanollm/distributed/infer.py` | Distributed inference |
| `data/data_set.py` | Tokenize your text into binary shards |
| `model/train-model.py` | Image classification training |

## quick start

```bash
# 1. Install deps
bash setup.sh

# 2. Put your text in data/input.txt, then tokenize
bash prepare_data.sh

# 3. Train on GPU and chat
bash train_gpu.sh
bash chat.sh
```

Or do it all at once:

```bash
bash run_all.sh
```

## Shell scripts

### setup.sh

Installs torch, torchvision, numpy, Pillow, tiktoken, tqdm, tensorboard. Run this once.

```bash
bash setup.sh
```

### prepare_data.sh

Reads `data/input.txt` and runs `python data/data_set.py` to produce binary shards in `nanollm/data/shards/`. Run this before training.

```bash
bash prepare_data.sh
```

### train_cpu.sh

Trains the LLM on CPU using `nanollm/cpu/main.py`. Slow but works on any machine.

```bash
bash train_cpu.sh
```

### train_gpu.sh

Trains the LLM on a single GPU. Auto-detects CUDA. If no GPU is found, it falls back to CPU.

```bash
bash train_gpu.sh
```

### train_ddp.sh

Trains the LLM on all available GPUs using `torchrun`. Auto-detects GPU count via `nvidia-smi`. If no GPUs are found, it exits with an error.

```bash
bash train_ddp.sh
```

### chat.sh

Loads `nanollm/cpu/llm.pt` and starts an interactive terminal. Type `exit` or `q` to quit.

```bash
bash chat.sh
```

Example:

```
user > Why is the sky blue?
Model > The sky appears blue due to Rayleigh scattering of sunlight...
```

### infer.sh

Run one-off inference without entering the interactive loop.

```bash
bash infer.sh "Hello world" 100
```

First arg is the prompt, second arg is max tokens (optional, defaults to 100).

### train_image.sh

Train an image classifier. Pass data directory, number of classes, and model name.

```bash
bash train_image.sh ./data/images 10 resnet50
```

Defaults: `./data/images`, `10` classes, `resnet50`. Organize your photos like this:

```
data/images/
├── train/
│   ├── cats/
│   └── dogs/
└── val/
    ├── cats/
    └── dogs/
```

### run_all.sh

Runs the full pipeline: `prepare_data.sh` -> `train_gpu.sh` -> `chat.sh`. Good for a first run.

```bash
bash run_all.sh
```

## File structure

```
.
├── setup.sh
├── prepare_data.sh
├── train_cpu.sh
├── train_gpu.sh
├── train_ddp.sh
├── chat.sh
├── infer.sh
├── train_image.sh
├── run_all.sh
├── data/
│   ├── data_set.py          # Tokenizer + shard writer
│   ├── dataset.py           # Image dataset helpers
│   ├── input.txt            # Your raw text
│   └── shards/              # Binary token shards
├── nanollm/
│   ├── cpu/
│   │   ├── main.py          # Single-GPU train + chat
│   │   ├── inference.py     # Standalone generation
│   │   ├── safe.py          # Safetensors export
│   │   └── logs/            # Training logs
│   ├── data/
│   │   └── shards/          # Binary shards (output)
│   └── distributed/
│       ├── train.py         # DDP training
│       └── infer.py         # DDP inference
├── model/
│   ├── config.yml           # Optional image model config
│   └── train-model.py       # Image classification trainer
├── assets/                  # Plots, images, etc.
├── bench/                   # Benchmarks
├── scripts/                 # Misc scripts
└── test/                    # Tests
```

## How it works

**LLM training** uses a custom GPT-style transformer (`MiniQuadtrix`) with:
- Tiktoken GPT-2 tokenizer (50,257 vocab)
- Causal self-attention with optional Flash Attention
- Pre-LayerNorm, residual connections, ReLU FFN
- Cosine learning rate with linear warmup
- Gradient norm logging and clipping
- Binary shard memmap for zero-overhead data loading

**Image training** uses torchvision backbones (ResNet, EfficientNet) with:
- Transfer learning (ImageNet pretrained backbone)
- Progressive unfreezing (head first, backbone at epoch 5)
- MixUp / CutMix, AutoAugment, RandomErasing
- Label smoothing, mixed precision, gradient accumulation

Both use the same training loop bones: AdamW, AMP, checkpointing on best metric, and early stopping.

## Notes

- The LLM scripts expect `nanollm/data/shards/shard_*.bin` to exist. Run `prepare_data.sh` first.
- `main.py` saves checkpoints as `llm.pt` in the working directory. `train.py` (distributed) does the same.
- Image classifier outputs go to `./outputs/` by default.
- All scripts are self-contained. No pip install of a custom package -- just `bash setup.sh` and go.

## License

GPL-3.0 (see `main.py` header)
