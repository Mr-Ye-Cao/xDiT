#!/usr/bin/env python3
"""
Benchmark PipeFusion for Wan2.1 video generation.

Usage:
    # Single GPU baseline (GPU 5)
    CUDA_VISIBLE_DEVICES=5 python scripts/benchmark_wan_pipefusion.py --mode single

    # 2 GPUs with PipeFusion (GPUs 5,6)
    CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 scripts/benchmark_wan_pipefusion.py --mode multi

Key Differences from TP:
- PipeFusion: CFG batch split, 0 all-reduces per forward
- TP: Head sharding, 90 all-reduces per forward
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

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Wan2.1 PipeFusion")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="single",
                        help="Run mode: single (1 GPU) or multi (distributed)")
    parser.add_argument("--frame_num", type=int, default=17, help="Number of frames")
    parser.add_argument("--num_steps", type=int, default=10, help="Inference steps")
    parser.add_argument("--size", type=str, default="480*832", help="Video size")
    parser.add_argument("--warmup_runs", type=int, default=1, help="Warmup runs")
    parser.add_argument("--benchmark_runs", type=int, default=2, help="Benchmark runs")
    return parser.parse_args()


def benchmark_single_gpu(args):
    """Benchmark single GPU baseline."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=" * 60)
    print("Single GPU Baseline Benchmark")
    print("=" * 60)
    print(f"Frames: {args.frame_num}, Size: {args.size}, Steps: {args.num_steps}")

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    print(f"\nLoading model from {checkpoint_dir}...")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )

    prompt = "A cat walks on the grass, realistic style."

    # Warmup
    print(f"\nWarmup runs: {args.warmup_runs}")
    for i in range(args.warmup_runs):
        with torch.no_grad():
            _ = wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                sampling_steps=args.num_steps,
                seed=42,
                offload_model=False
            )
        torch.cuda.synchronize()
        print(f"  Warmup {i+1}/{args.warmup_runs} done")

    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark
    print(f"\nBenchmark runs: {args.benchmark_runs}")
    times = []

    for i in range(args.benchmark_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                sampling_steps=args.num_steps,
                seed=42 + i,
                offload_model=False
            )

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}/{args.benchmark_runs}: {elapsed:.2f} ms ({elapsed/args.num_steps:.2f} ms/step)")

    mean_time = sum(times) / len(times)
    print(f"\n{'='*60}")
    print(f"Single GPU Results:")
    print(f"  Mean inference time: {mean_time:.2f} ms")
    print(f"  Time per step: {mean_time / args.num_steps:.2f} ms")
    print(f"{'='*60}")

    return mean_time


def benchmark_multi_gpu_pipefusion(args):
    """Benchmark PipeFusion with multiple GPUs."""
    # Import PipeFusion components FIRST (before any dist init)
    from pipefuser.utils import DistriConfig
    from pipefuser.models.wan import DistriWanDiTPipeFusion

    # Parse video size
    height, width = map(int, args.size.split("*"))

    # Create distributed config - this initializes the process group internally
    distri_config = DistriConfig(
        height=height,
        width=width,
        do_classifier_free_guidance=True,
        split_batch=True,
        warmup_steps=4,
        mode="corrected_async_gn",
        use_cuda_graph=False,
        parallelism="pipefusion",
    )

    # Use original rank for text encoding (distri_config.rank is modified by CFG split)
    original_rank = dist.get_rank()
    rank = distri_config.rank  # This is modified for CFG batch splitting
    world_size = distri_config.world_size
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = distri_config.device

    if original_rank == 0:
        print("=" * 60)
        print(f"PipeFusion Benchmark ({world_size} GPUs)")
        print("=" * 60)
        print(f"Frames: {args.frame_num}, Size: {args.size}, Steps: {args.num_steps}")

    if original_rank == 0:
        print(f"\nWorld size: {world_size}")
        print(f"Devices per batch: {distri_config.n_device_per_batch}")

    # Load Wan2.1 model
    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    if original_rank == 0:
        print(f"Loading model from {checkpoint_dir}...")

    # Load components
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE

    # Load transformer and wrap with PipeFusion
    transformer = WanModel.from_pretrained(checkpoint_dir)
    transformer.eval().requires_grad_(False)

    # Wrap with PipeFusion
    transformer = DistriWanDiTPipeFusion(transformer, distri_config)
    # Convert to bfloat16 and move to device
    transformer = transformer.to(dtype=torch.bfloat16, device=device)

    # Text encoder on original_rank 0 only (use original_rank, not CFG-modified rank)
    if original_rank == 0:
        text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, cfg.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, cfg.t5_tokenizer),
        )

    dist.barrier()

    # Encode prompts (original_rank 0 only)
    prompt = "A cat walks on the grass, realistic style."
    negative_prompt = ""

    if original_rank == 0:
        with torch.no_grad():
            # T5 is on CPU, so encode on CPU then move to GPU
            pos_embeds_list = text_encoder([prompt], torch.device('cpu'))
            neg_embeds_list = text_encoder([negative_prompt], torch.device('cpu'))
            # Stack and move to GPU
            pos_embeds = torch.stack([e.to(device).to(torch.bfloat16) for e in pos_embeds_list])
            neg_embeds = torch.stack([e.to(device).to(torch.bfloat16) for e in neg_embeds_list])

    dist.barrier()

    # Broadcast embeddings using original ranks
    if original_rank == 0:
        embeds_shape = torch.tensor(pos_embeds.shape, device=device)
    else:
        embeds_shape = torch.zeros(3, dtype=torch.long, device=device)

    dist.broadcast(embeds_shape, src=0)

    if original_rank != 0:
        pos_embeds = torch.zeros(embeds_shape.tolist(), dtype=torch.bfloat16, device=device)
        neg_embeds = torch.zeros(embeds_shape.tolist(), dtype=torch.bfloat16, device=device)

    dist.broadcast(pos_embeds, src=0)
    dist.broadcast(neg_embeds, src=0)

    # Determine if this rank processes uncond or cond
    # With split_batch=True and 2 GPUs: original_rank 0 -> Uncond, original_rank 1 -> Cond
    n_device_per_batch = distri_config.n_device_per_batch
    batch_idx = original_rank // n_device_per_batch if world_size >= 2 else 0

    if batch_idx == 0:
        context = [neg_embeds[0]]  # Unconditional
        if original_rank == 0:
            print(f"\nRank 0: Processing UNCONDITIONAL")
            print(f"Rank 1: Processing CONDITIONAL")
    else:
        context = [pos_embeds[0]]  # Conditional

    # Prepare latent dimensions
    latent_channels = 16
    patch_size = cfg.patch_size
    vae_stride = cfg.vae_stride

    # Calculate latent size
    latent_h = height // vae_stride[1]
    latent_w = width // vae_stride[2]
    latent_f = (args.frame_num - 1) // vae_stride[0] + 1

    # Calculate sequence length
    seq_len = (
        (latent_f // patch_size[0])
        * (latent_h // patch_size[1])
        * (latent_w // patch_size[2])
    )

    # Create initial noise
    generator = torch.Generator(device=device).manual_seed(42)
    latent = torch.randn(
        (latent_channels, latent_f, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    # Setup scheduler with default values
    from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        shift=3.0,  # Default shift value for Wan2.1
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(args.num_steps, device=device)
    timesteps = scheduler.timesteps

    dist.barrier()

    # Warmup
    if original_rank == 0:
        print(f"\nWarmup runs: {args.warmup_runs}")

    for i in range(args.warmup_runs):
        transformer.set_counter(0)
        transformer.reset_kv_caches()

        latent_input = latent.clone()
        for step_idx, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0).to(device)
            with torch.no_grad():
                output = transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=context,
                    seq_len=seq_len,
                )
            # Simple Euler step (for benchmark purposes)
            latent_input = output[0]

        torch.cuda.synchronize()
        dist.barrier()

        if original_rank == 0:
            print(f"  Warmup {i+1}/{args.warmup_runs} done")

    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark
    if original_rank == 0:
        print(f"\nBenchmark runs: {args.benchmark_runs}")

    times = []

    for i in range(args.benchmark_runs):
        transformer.set_counter(0)
        transformer.reset_kv_caches()

        latent_input = latent.clone()

        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()

        for step_idx, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0).to(device)
            with torch.no_grad():
                output = transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=context,
                    seq_len=seq_len,
                )
            latent_input = output[0]

        torch.cuda.synchronize()
        dist.barrier()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        if original_rank == 0:
            print(f"  Run {i+1}/{args.benchmark_runs}: {elapsed:.2f} ms ({elapsed/args.num_steps:.2f} ms/step)")

    mean_time = sum(times) / len(times)

    if original_rank == 0:
        print(f"\n{'='*60}")
        print(f"PipeFusion Results ({world_size} GPUs):")
        print(f"  Mean inference time: {mean_time:.2f} ms")
        print(f"  Time per step: {mean_time / args.num_steps:.2f} ms")
        print(f"  NOTE: Each GPU processes half the CFG batch (Uncond OR Cond)")
        print(f"  NOTE: No all-reduce communication (vs 90 in TP)")
        print(f"{'='*60}")

    dist.destroy_process_group()
    return mean_time


def main():
    args = parse_args()

    print(f"Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Frames: {args.frame_num}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Video size: {args.size}")

    if args.mode == "single":
        benchmark_single_gpu(args)
    else:
        benchmark_multi_gpu_pipefusion(args)


if __name__ == "__main__":
    main()
