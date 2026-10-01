#!/usr/bin/env python3
"""
bench/benchmark.py

Small benchmark harness for nanollm.
Measures training throughput and inference latency.
Run standalone or import the functions.

Usage:
    python bench/benchmark.py --mode train --iters 100
    python bench/benchmark.py --mode infer --tokens 500
"""

import time
import sys
import os
import argparse
from pathlib import Path
import torch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "nanollm" / "cpu"))
try:
    from main import MiniQuadtrix, get_miniq_tokenizer, get_batch, block_size, batch_size, device
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


def benchmark_train(iters: int = 100):
    """Benchmark LLM training throughput."""
    if not HAS_LLM:
        print("Could not import LLM modules. Run from repo root.")
        return

    print(f"Benchmarking training for {iters} iterations...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Block size: {block_size}")

    tokenizer, vocab_size = get_miniq_tokenizer("gpt2")
    model = MiniQuadtrix().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup
    for _ in range(3):
        x, y = get_batch("train")
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for i in range(iters):
        x, y = get_batch("train")
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = time.time() - t0
    tokens = iters * batch_size * block_size
    tok_per_sec = tokens / dt

    print(f"Time: {dt:.2f}s")
    print(f"Tokens processed: {tokens:,}")
    print(f"Throughput: {tok_per_sec:,.0f} tok/sec")
    print(f"ms per step: {dt / iters * 1000:.2f}")

    return tok_per_sec


def benchmark_infer(tokens: int = 500, prompt: str = "Hello world"):
    """Benchmark LLM inference latency."""
    if not HAS_LLM:
        print("Could not import LLM modules. Run from repo root.")
        return

    print(f"Benchmarking inference: generating {tokens} tokens...")
    print(f"Device: {device}")

    tokenizer, vocab_size = get_miniq_tokenizer("gpt2")
    model = MiniQuadtrix().to(device)
    model.eval()

    encoded = tokenizer.encode_ordinary(prompt)
    context = torch.tensor([encoded], dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(context, max_new_tokens=10)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        _ = model.generate(context, max_new_tokens=tokens)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = time.time() - t0
    tok_per_sec = tokens / dt

    print(f"Time: {dt:.2f}s")
    print(f"Tokens generated: {tokens}")
    print(f"Throughput: {tok_per_sec:,.0f} tok/sec")
    print(f"ms per token: {dt / tokens * 1000:.2f}")

    return tok_per_sec


def benchmark_image(batch_size: int = 32, iters: int = 50, model_name: str = "resnet50"):
    """Benchmark image classifier throughput."""
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    try:
        from train_image_classifier import ImageClassifier
    except ImportError:
        print("Could not import ImageClassifier.")
        return

    print(f"Benchmarking image classifier: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Iters: {iters}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier(model_name, num_classes=10,
                            pretrained=False).to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    dummy_images = torch.randn(batch_size, 3, 224, 224, device=dev)
    dummy_labels = torch.randint(0, 10, (batch_size,), device=dev)

    # Warmup
    for _ in range(3):
        out = model(dummy_images)
        loss = criterion(out, dummy_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        out = model(dummy_images)
        loss = criterion(out, dummy_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = time.time() - t0
    images = iters * batch_size
    img_per_sec = images / dt

    print(f"Time: {dt:.2f}s")
    print(f"Images processed: {images:,}")
    print(f"Throughput: {img_per_sec:,.0f} img/sec")
    print(f"ms per batch: {dt / iters * 1000:.2f}")

    return img_per_sec


def main():
    parser = argparse.ArgumentParser(description="llm.cpp benchmark")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "infer", "image"])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="resnet50")
    args = parser.parse_args()

    print("llm.cpp benchmark")

    if args.mode == "train":
        benchmark_train(args.iters)
    elif args.mode == "infer":
        benchmark_infer(args.tokens)
    elif args.mode == "image":
        benchmark_image(args.batch_size, args.iters, args.model_name)


if __name__ == "__main__":
    main()
