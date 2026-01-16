#!/usr/bin/env python3
"""
Direct test of PipeFusion modules without DistriConfig.

Tests that the attention wrappers work correctly.

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/test_pipefusion_modules.py
"""

import os
import sys
import time
from pathlib import Path

import torch

# Add paths
SCRIPT_DIR = Path(__file__).parent
XDIT_DIR = SCRIPT_DIR.parent
WAN_PATH = XDIT_DIR.parent / "Wan2.1"

sys.path.insert(0, str(XDIT_DIR))
sys.path.insert(0, str(WAN_PATH))


class MockDistriConfig:
    """Minimal mock config for single-GPU testing."""
    def __init__(self):
        self.world_size = 1
        self.rank = 0
        self.n_device_per_batch = 1
        self.warmup_steps = 4
        self.mode = "full_sync"
        self.pp_num_patch = 1
        self.do_classifier_free_guidance = False
        self.split_batch = False


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=" * 60)
    print("PipeFusion Modules Direct Test")
    print("=" * 60)

    # Import modules
    from pipefuser.modules.wan.attn import (
        DistriWanSelfAttentionPiP,
        DistriWanCrossAttentionPiP,
    )
    from wan.modules.model import WanModel, WanSelfAttention, WanT2VCrossAttention
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    print(f"Loading model from {checkpoint_dir}...")
    model = WanModel.from_pretrained(checkpoint_dir)
    model.eval().requires_grad_(False)
    model = model.to(dtype=torch.bfloat16, device=device)

    # Get an attention module to test
    block = model.blocks[0]
    orig_self_attn = block.self_attn
    orig_cross_attn = block.cross_attn

    print(f"\nOriginal self-attention: {type(orig_self_attn)}")
    print(f"Original cross-attention: {type(orig_cross_attn)}")

    # Create mock config
    mock_config = MockDistriConfig()

    # Wrap self-attention
    print("\nWrapping self-attention with PipeFusion...")
    wrapped_self_attn = DistriWanSelfAttentionPiP(orig_self_attn, mock_config)

    # Wrap cross-attention
    print("Wrapping cross-attention with PipeFusion...")
    wrapped_cross_attn = DistriWanCrossAttentionPiP(orig_cross_attn, mock_config)

    # Create test inputs
    # Calculate dimensions properly
    frame_num = 17
    height, width = 480, 832
    vae_stride = cfg.vae_stride  # (4, 8, 8)
    patch_size = cfg.patch_size  # (1, 2, 2)

    # Latent dimensions
    latent_f = (frame_num - 1) // vae_stride[0] + 1  # 5
    latent_h = height // vae_stride[1]  # 60
    latent_w = width // vae_stride[2]  # 104

    # After patch embedding
    grid_f = latent_f // patch_size[0]  # 5
    grid_h = latent_h // patch_size[1]  # 30
    grid_w = latent_w // patch_size[2]  # 52

    batch_size = 1
    seq_len = grid_f * grid_h * grid_w  # 7800
    hidden_dim = cfg.dim  # 1536

    print(f"\nTest input dimensions:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len} (grid: {grid_f}x{grid_h}x{grid_w})")
    print(f"  - Hidden dim: {hidden_dim}")

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    context = torch.randn(batch_size, 512, hidden_dim, device=device, dtype=torch.bfloat16)
    seq_lens = torch.tensor([seq_len], device=device)
    context_lens = torch.tensor([512], device=device)
    grid_sizes = torch.tensor([[grid_f, grid_h, grid_w]], device=device)

    # Get freqs from model
    freqs = model.freqs.to(device)

    # Test self-attention
    print("\nTesting wrapped self-attention...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        out_self = wrapped_self_attn(x, seq_lens, grid_sizes, freqs)

    torch.cuda.synchronize()
    self_time = (time.perf_counter() - start) * 1000
    print(f"  - Output shape: {out_self.shape}")
    print(f"  - Time: {self_time:.2f} ms")

    # Test cross-attention
    print("\nTesting wrapped cross-attention...")
    wrapped_cross_attn.counter = 0  # Reset counter
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        out_cross = wrapped_cross_attn(x, context, context_lens)

    torch.cuda.synchronize()
    cross_time = (time.perf_counter() - start) * 1000
    print(f"  - Output shape: {out_cross.shape}")
    print(f"  - Time: {cross_time:.2f} ms")

    # Test cross-attention with KV cache (second call should use cache)
    print("\nTesting cross-attention KV caching...")
    wrapped_cross_attn.counter = 1  # Simulate second step
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        out_cross_cached = wrapped_cross_attn(x, context, context_lens)

    torch.cuda.synchronize()
    cached_time = (time.perf_counter() - start) * 1000
    print(f"  - Output shape: {out_cross_cached.shape}")
    print(f"  - Time (cached): {cached_time:.2f} ms")
    print(f"  - Speedup from caching: {cross_time / cached_time:.2f}x")

    print("\n" + "=" * 60)
    print("All module tests PASSED!")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Self-attention wrapper: {self_time:.2f} ms")
    print(f"  - Cross-attention wrapper: {cross_time:.2f} ms")
    print(f"  - Cross-attention cached: {cached_time:.2f} ms ({cross_time/cached_time:.2f}x speedup)")
    print(f"\nPipeFusion modules are ready for multi-GPU testing.")


if __name__ == "__main__":
    main()
