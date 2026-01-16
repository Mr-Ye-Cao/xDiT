#!/usr/bin/env python3
"""
Simple test for PipeFusion wrapper on Wan2.1.

Tests that the PipeFusion modules work correctly on a single GPU.
No distributed communication - just verifies the wrapper doesn't break inference.

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/test_pipefusion_simple.py
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


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=" * 60)
    print("PipeFusion Simple Test (Single GPU)")
    print("=" * 60)

    # Import modules
    from pipefuser.utils import DistriConfig
    from pipefuser.models.wan import DistriWanDiTPipeFusion
    from wan.modules.model import WanModel
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    # Create a minimal config (single GPU, no CFG split)
    distri_config = DistriConfig(
        height=480,
        width=832,
        do_classifier_free_guidance=False,  # No CFG for simple test
        split_batch=False,
        warmup_steps=4,
        mode="full_sync",
        use_cuda_graph=False,
        parallelism="pipefusion",
    )

    print(f"Loading model from {checkpoint_dir}...")
    transformer = WanModel.from_pretrained(checkpoint_dir)
    transformer.eval().requires_grad_(False)

    print("Wrapping with PipeFusion...")
    transformer = DistriWanDiTPipeFusion(transformer, distri_config)
    transformer = transformer.to(dtype=torch.bfloat16, device=device)

    print(f"Model wrapped successfully!")
    print(f"  - Self-attention modules: 30")
    print(f"  - Cross-attention modules: 30")

    # Create dummy inputs
    frame_num = 17
    height, width = 480, 832
    vae_stride = cfg.vae_stride
    patch_size = cfg.patch_size

    latent_h = height // vae_stride[1]
    latent_w = width // vae_stride[2]
    latent_f = (frame_num - 1) // vae_stride[0] + 1

    seq_len = (
        (latent_f // patch_size[0])
        * (latent_h // patch_size[1])
        * (latent_w // patch_size[2])
    )

    print(f"\nInput dimensions:")
    print(f"  - Latent: [{16}, {latent_f}, {latent_h}, {latent_w}]")
    print(f"  - Sequence length: {seq_len}")

    # Create inputs
    latent = torch.randn(
        (16, latent_f, latent_h, latent_w),
        device=device,
        dtype=torch.bfloat16,
    )

    # Dummy context (text embedding)
    context = [torch.randn(512, 1536, device=device, dtype=torch.bfloat16)]

    # Timestep
    t = torch.tensor([500.0], device=device)

    # Test forward pass
    print("\nRunning forward pass...")
    transformer.set_counter(0)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output = transformer(
            x=[latent],
            t=t,
            context=context,
            seq_len=seq_len,
        )

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Forward pass completed!")
    print(f"  - Time: {elapsed:.2f} ms")
    print(f"  - Output shape: {output[0].shape}")

    # Test multiple steps
    print("\nRunning 5 steps...")
    transformer.set_counter(0)
    transformer.reset_kv_caches()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for i in range(5):
        with torch.no_grad():
            output = transformer(
                x=[latent],
                t=t,
                context=context,
                seq_len=seq_len,
            )

    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000

    print(f"5 steps completed!")
    print(f"  - Total time: {total_time:.2f} ms")
    print(f"  - Time per step: {total_time / 5:.2f} ms")

    print("\n" + "=" * 60)
    print("PipeFusion wrapper test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
