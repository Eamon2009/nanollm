"""
test/test_llm.py

Pytest tests for the LLM components.
Run with: pytest test/test_llm.py -v
"""

from main import MiniQuadtrix, MiniQuadtrixBlock, MiniQuadtrixHead, get_miniq_tokenizer, vocab_size, n_embd, n_head, n_layer, block_size
import sys
import os
from pathlib import Path

import pytest
import torch
import numpy as np
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "nanollm" / "cpu"))


class TestMiniQuadtrixHead:
    def test_output_shape(self):
        head = MiniQuadtrixHead(head_size=32)
        x = torch.randn(2, 10, n_embd)
        out = head(x)
        assert out.shape == (2, 10, 32)

    def test_causal_mask(self):
        head = MiniQuadtrixHead(head_size=32)
        x = torch.randn(1, 5, n_embd)
        out = head(x)
        # Causal attention: output should not be NaN (masked positions handled)
        assert not torch.isnan(out).any()

    def test_flash_attention_fallback(self):
        head = MiniQuadtrixHead(head_size=32)
        x = torch.randn(1, block_size, n_embd)
        out = head(x)
        assert out.shape == (1, block_size, 32)


class TestMiniQuadtrixBlock:
    def test_residual_connection(self):
        block = MiniQuadtrixBlock(n_embd, n_head)
        x = torch.randn(2, 10, n_embd)
        out = block(x)
        # Output should be different from input (transform happened)
        assert not torch.allclose(out, x)
        # But shape preserved
        assert out.shape == x.shape

    def test_shape_preservation(self):
        block = MiniQuadtrixBlock(n_embd, n_head)
        x = torch.randn(4, block_size, n_embd)
        out = block(x)
        assert out.shape == (4, block_size, n_embd)


class TestMiniQuadtrix:
    def test_forward_with_targets(self):
        model = MiniQuadtrix()
        idx = torch.randint(0, vocab_size, (2, block_size))
        targets = torch.randint(0, vocab_size, (2, block_size))
        logits, loss = model(idx, targets)
        assert logits.shape == (2 * block_size, vocab_size)
        assert loss.item() > 0

    def test_forward_without_targets(self):
        model = MiniQuadtrix()
        idx = torch.randint(0, vocab_size, (1, block_size))
        logits, loss = model(idx)
        assert logits.shape == (1, block_size, vocab_size)
        assert loss is None

    def test_generate(self):
        model = MiniQuadtrix()
        model.eval()
        idx = torch.randint(0, vocab_size, (1, 5))
        out = model.generate(idx, max_new_tokens=10)
        assert out.shape == (1, 15)
        assert out.shape[1] == 5 + 10

    def test_gradient_flow(self):
        model = MiniQuadtrix()
        idx = torch.randint(0, vocab_size, (1, block_size))
        targets = torch.randint(0, vocab_size, (1, block_size))
        _, loss = model(idx, targets)
        loss.backward()
        # Check that embeddings got gradients
        assert model.token_embedding_table.weight.grad is not None

    def test_parameter_count(self):
        model = MiniQuadtrix()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        # With default settings it should be in the ~100K-1M range
        assert n_params < 10_000_000

    def test_init_weights(self):
        model = MiniQuadtrix()
        for name, p in model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN in {name}"


class TestTokenizer:
    def test_encode_decode_roundtrip(self):
        tokenizer, _ = get_miniq_tokenizer("gpt2")
        text = "Hello world"
        tokens = tokenizer.encode_ordinary(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_vocab_size(self):
        tokenizer, vocab = get_miniq_tokenizer("gpt2")
        assert vocab == 50257
        assert tokenizer.n_vocab == 50257

    def test_empty_string(self):
        tokenizer, _ = get_miniq_tokenizer("gpt2")
        tokens = tokenizer.encode_ordinary("")
        assert tokens == []
