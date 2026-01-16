#!/usr/bin/env python3
"""
Test CFG communication pattern without model forward.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 scripts/test_cfg_communication.py
"""

import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] Starting on device {device}")

    # Create dummy latent
    latent_shape = (16, 5, 60, 104)
    latent = torch.randn(latent_shape, device=device, dtype=torch.bfloat16)

    # Allocate buffer for all-gather
    gathered = [torch.zeros_like(latent) for _ in range(world_size)]

    print(f"[Rank {rank}] Latent shape: {latent.shape}")

    num_steps = 5

    dist.barrier()
    print(f"[Rank {rank}] Starting {num_steps} steps...")

    torch.cuda.synchronize()
    start = time.perf_counter()

    for step in range(num_steps):
        # Simulate forward pass with sleep
        my_output = latent * (1.0 + step * 0.1)

        print(f"[Rank {rank}] Step {step}: computed output, starting all_gather")

        # All-gather
        dist.all_gather(gathered, my_output)

        print(f"[Rank {rank}] Step {step}: all_gather done")

        # CFG combination
        uncond = gathered[0]
        cond = gathered[1]
        combined = uncond + 7.5 * (cond - uncond)

        latent = combined

        print(f"[Rank {rank}] Step {step}: CFG combination done")

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = (time.perf_counter() - start) * 1000

    if rank == 0:
        print(f"\n[Rank 0] {num_steps} steps completed in {elapsed:.2f} ms")
        print(f"[Rank 0] Time per step: {elapsed/num_steps:.2f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
