#!/usr/bin/env python3
"""
PipeFusion benchmark for Wan2.1 with proper CFG combination.

Key approach:
- Rank 0: Unconditional pass
- Rank 1: Conditional pass
- After each step: all-gather outputs, combine with CFG formula

Usage:
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 scripts/benchmark_wan_pipefusion_cfg.py
"""

import argparse
import gc
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=2)
    return parser.parse_args()


def log(rank, msg):
    print(f"[Rank {rank}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    args = parse_args()

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    assert world_size == 2, "This benchmark requires exactly 2 GPUs for CFG splitting"

    if rank == 0:
        print("=" * 60)
        print(f"PipeFusion CFG Benchmark (2 GPUs)")
        print("=" * 60)
        print(f"Frames: {args.frame_num}, Size: {args.size}, Steps: {args.num_steps}")
        print(f"CFG Scale: {args.cfg_scale}")
        print(f"\nRank 0: Unconditional")
        print(f"Rank 1: Conditional")

    # Import modules
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    # Parse video size
    height, width = map(int, args.size.split("*"))

    if rank == 0:
        print(f"\nLoading model from {checkpoint_dir}...")

    # Load transformer (each rank loads its own copy)
    transformer = WanModel.from_pretrained(checkpoint_dir)
    transformer.eval().requires_grad_(False)
    # Convert to bfloat16 for consistency
    transformer = transformer.to(dtype=torch.bfloat16, device=device)

    dist.barrier()

    # Text encoder on rank 0 only
    if rank == 0:
        print("Loading T5 encoder...")
        text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, cfg.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, cfg.t5_tokenizer),
        )

    dist.barrier()

    # Encode prompts (rank 0 only)
    prompt = "A cat walks on the grass, realistic style."
    negative_prompt = ""

    if rank == 0:
        print("Encoding prompts...")
        with torch.no_grad():
            pos_embeds_list = text_encoder([prompt], torch.device('cpu'))
            neg_embeds_list = text_encoder([negative_prompt], torch.device('cpu'))
            pos_embeds = pos_embeds_list[0].to(device)
            neg_embeds = neg_embeds_list[0].to(device)
        print(f"Embeddings: pos={pos_embeds.shape}, neg={neg_embeds.shape}")

    dist.barrier()

    # Broadcast text embeddings - shapes may differ!
    if rank == 0:
        pos_shape = torch.tensor(pos_embeds.shape, device=device)
        neg_shape = torch.tensor(neg_embeds.shape, device=device)
    else:
        pos_shape = torch.zeros(2, dtype=torch.long, device=device)
        neg_shape = torch.zeros(2, dtype=torch.long, device=device)

    dist.broadcast(pos_shape, src=0)
    dist.broadcast(neg_shape, src=0)

    if rank != 0:
        pos_embeds = torch.zeros(pos_shape.tolist(), dtype=cfg.t5_dtype, device=device)
        neg_embeds = torch.zeros(neg_shape.tolist(), dtype=cfg.t5_dtype, device=device)

    dist.broadcast(pos_embeds, src=0)
    dist.broadcast(neg_embeds, src=0)

    # Each rank gets its context
    if rank == 0:
        context = [neg_embeds]  # Unconditional
    else:
        context = [pos_embeds]  # Conditional

    # Calculate dimensions
    vae_stride = cfg.vae_stride
    patch_size = cfg.patch_size

    latent_h = height // vae_stride[1]
    latent_w = width // vae_stride[2]
    latent_f = (args.frame_num - 1) // vae_stride[0] + 1

    seq_len = (
        (latent_f // patch_size[0])
        * (latent_h // patch_size[1])
        * (latent_w // patch_size[2])
    )

    if rank == 0:
        print(f"\nLatent shape: [{16}, {latent_f}, {latent_h}, {latent_w}]")
        print(f"Sequence length: {seq_len}")

    # Create initial noise (same on both ranks for proper CFG)
    generator = torch.Generator(device=device).manual_seed(42)
    latent = torch.randn(
        (16, latent_f, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    # Scheduler timesteps (simplified linear schedule for benchmarking)
    timesteps = torch.linspace(1000, 0, args.num_steps + 1, device=device)[:-1]

    # Allocate buffer for all-gather
    gathered_outputs = [torch.zeros_like(latent) for _ in range(world_size)]

    dist.barrier()

    # Warmup
    if rank == 0:
        print(f"\nWarmup runs: {args.warmup_runs}")

    for warmup_idx in range(args.warmup_runs):
        latent_input = latent.clone()

        for step_idx, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0)

            # Each rank computes its output with autocast
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=context,
                    seq_len=seq_len,
                )
            # Convert to bfloat16 for all-gather (model returns float32)
            my_output = output[0].to(torch.bfloat16)

            # All-gather outputs from both ranks
            dist.all_gather(gathered_outputs, my_output)

            # CFG combination: output = uncond + cfg_scale * (cond - uncond)
            uncond_output = gathered_outputs[0]
            cond_output = gathered_outputs[1]
            combined = uncond_output + args.cfg_scale * (cond_output - uncond_output)

            # Simple Euler step (for benchmark purposes)
            latent_input = combined

        torch.cuda.synchronize()
        dist.barrier()

        if rank == 0:
            print(f"  Warmup {warmup_idx + 1}/{args.warmup_runs} done")

    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark
    if rank == 0:
        print(f"\nBenchmark runs: {args.benchmark_runs}")

    times = []

    for run_idx in range(args.benchmark_runs):
        latent_input = latent.clone()

        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()

        for step_idx, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0)

            # Each rank computes its output with autocast
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=context,
                    seq_len=seq_len,
                )
            # Convert to bfloat16 for all-gather (model returns float32)
            my_output = output[0].to(torch.bfloat16)

            # All-gather outputs from both ranks
            dist.all_gather(gathered_outputs, my_output)

            # CFG combination
            uncond_output = gathered_outputs[0]
            cond_output = gathered_outputs[1]
            combined = uncond_output + args.cfg_scale * (cond_output - uncond_output)

            latent_input = combined

        torch.cuda.synchronize()
        dist.barrier()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        if rank == 0:
            print(f"  Run {run_idx + 1}/{args.benchmark_runs}: {elapsed:.2f} ms ({elapsed/args.num_steps:.2f} ms/step)")

    mean_time = sum(times) / len(times)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"PipeFusion CFG Results (2 GPUs):")
        print(f"  Mean inference time: {mean_time:.2f} ms")
        print(f"  Time per step: {mean_time / args.num_steps:.2f} ms")
        print(f"\nCommunication per step: 1 all-gather (latent size)")
        print(f"vs TP: 90 all-reduces per step")
        print(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
