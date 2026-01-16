#!/usr/bin/env python3
"""
Minimal multi-GPU forward pass test without T5.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 scripts/debug_minimal.py
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# Add paths
SCRIPT_DIR = Path(__file__).parent
XDIT_DIR = SCRIPT_DIR.parent
WAN_PATH = XDIT_DIR.parent / "Wan2.1"

sys.path.insert(0, str(XDIT_DIR))
sys.path.insert(0, str(WAN_PATH))


def log(rank, msg):
    print(f"[Rank {rank}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    log(rank, f"Started on device {device}")

    # Import
    from wan.modules.model import WanModel
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    # Load model
    log(rank, "Loading model...")
    transformer = WanModel.from_pretrained(checkpoint_dir)
    transformer.eval().requires_grad_(False)
    # Convert entire model to bfloat16 for consistency
    transformer = transformer.to(dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    log(rank, "Model loaded and on GPU")

    dist.barrier()
    log(rank, "All ranks have model loaded")

    # Create fake context - WanModel expects list of 2D tensors [text_len, 4096]
    # T5 outputs 4096-dim embeddings, text_embedding projects to model dim
    # Use bfloat16 to match model weight dtype
    t5_dim = 4096  # T5-XXL output dimension
    context_len = 12  # Match typical prompt length
    context = [torch.randn(context_len, t5_dim, device=device, dtype=torch.bfloat16)]
    log(rank, f"Context shape: {context[0].shape}, dtype: {context[0].dtype}")

    # Create latent
    frame_num = 17
    height, width = 480, 832
    vae_stride = cfg.vae_stride
    patch_size = cfg.patch_size

    latent_f = (frame_num - 1) // vae_stride[0] + 1
    latent_h = height // vae_stride[1]
    latent_w = width // vae_stride[2]

    seq_len = (latent_f // patch_size[0]) * (latent_h // patch_size[1]) * (latent_w // patch_size[2])

    # Use same seed for reproducibility, bfloat16 to match model weights
    generator = torch.Generator(device=device).manual_seed(42 + rank)  # Different per rank
    latent = torch.randn((16, latent_f, latent_h, latent_w), generator=generator, device=device, dtype=torch.bfloat16)
    t = torch.tensor([500.0], device=device)

    log(rank, f"Latent shape: {latent.shape}, seq_len={seq_len}")

    dist.barrier()
    log(rank, "Starting forward pass...")

    torch.cuda.synchronize()
    start = time.perf_counter()

    # WanModel is designed to run with autocast (bfloat16)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = transformer(
            x=[latent],
            t=t,
            context=context,
            seq_len=seq_len,
        )

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    log(rank, f"Forward pass done in {elapsed:.1f}ms, output shape: {output[0].shape}")

    dist.barrier()
    log(rank, "All ranks completed forward pass")

    # All-gather
    gathered = [torch.zeros_like(output[0]) for _ in range(world_size)]
    dist.all_gather(gathered, output[0])
    log(rank, f"All-gather done, got {len(gathered)} outputs")

    # CFG combination (for testing)
    combined = gathered[0] + 7.5 * (gathered[1] - gathered[0])
    log(rank, f"CFG done, shape: {combined.shape}")

    dist.barrier()
    log(rank, "ALL TESTS PASSED!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
