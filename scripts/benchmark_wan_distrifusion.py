#!/usr/bin/env python3
"""
DistriFusion benchmark for Wan2.1.

DistriFusion splits the sequence across GPUs and uses stale KV from
previous timestep to overlap communication with computation.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 scripts/benchmark_wan_distrifusion.py
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
    parser.add_argument("--warmup_steps", type=int, default=2,
                        help="Number of warmup steps for DistriFusion (sync all-gather)")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=2)
    parser.add_argument("--mode", type=str, default="default",
                        choices=["default", "full_sync", "no_sync"],
                        help="DistriFusion mode: default (async after warmup), full_sync, no_sync")
    return parser.parse_args()


def log(rank, msg):
    print(f"[Rank {rank}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    args = parse_args()

    # Parse video size first (needed for DistriConfig)
    height, width = map(int, args.size.split("*"))

    # Import modules
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.configs import WAN_CONFIGS

    from pipefuser.utils import DistriConfig
    from pipefuser.modules.wan import wrap_wan_model_distrifusion

    # Create DistriConfig - this initializes distributed internally
    # For DistriFusion, we don't split batch for CFG, each GPU processes the same batch
    distri_config = DistriConfig(
        height=height,
        width=width,
        do_classifier_free_guidance=False,  # We handle CFG separately
        split_batch=False,  # All GPUs work on same sequence
        warmup_steps=args.warmup_steps,
        mode=args.mode,
        parallelism="patch",  # DistriFusion uses patch/sequence parallelism
        use_cuda_graph=False,  # Disable for now
    )

    rank = distri_config.rank
    world_size = distri_config.world_size
    device = distri_config.device

    if rank == 0:
        print("=" * 60)
        print(f"DistriFusion Benchmark for Wan2.1")
        print("=" * 60)
        print(f"Frames: {args.frame_num}, Size: {args.size}, Steps: {args.num_steps}")
        print(f"CFG Scale: {args.cfg_scale}")
        print(f"World size: {world_size}")
        print(f"DistriFusion warmup steps: {args.warmup_steps}")
        print(f"Mode: {args.mode}")

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    if rank == 0:
        print(f"\nLoading model from {checkpoint_dir}...")

    # Load transformer
    transformer = WanModel.from_pretrained(checkpoint_dir)
    transformer.eval().requires_grad_(False)
    transformer = transformer.to(dtype=torch.bfloat16, device=device)

    dist.barrier()

    # Wrap model with DistriFusion
    if rank == 0:
        print("Wrapping model with DistriFusion...")

    wrapped_transformer = wrap_wan_model_distrifusion(transformer, distri_config)

    if rank == 0:
        print("DistriFusion wrapper ready")

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

    # Broadcast embeddings
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
        print(f"Tokens per GPU: {seq_len // world_size}")

    # Create initial noise
    generator = torch.Generator(device=device).manual_seed(42)
    latent = torch.randn(
        (16, latent_f, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    # Scheduler timesteps
    timesteps = torch.linspace(1000, 0, args.num_steps + 1, device=device)[:-1]

    dist.barrier()

    # Warmup
    if rank == 0:
        print(f"\nWarmup runs: {args.warmup_runs}")

    for warmup_idx in range(args.warmup_runs):
        latent_input = latent.clone()
        wrapped_transformer.reset()

        for step_idx, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0)

            # CFG: Run both unconditional and conditional
            # Note: For DistriFusion with CFG, we need to run both passes
            # Each GPU processes its sequence portion for each pass

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Unconditional
                uncond_output = wrapped_transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=[neg_embeds],
                    seq_len=seq_len,
                )[0]

                # Conditional
                cond_output = wrapped_transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=[pos_embeds],
                    seq_len=seq_len,
                )[0]

            # CFG combination
            combined = uncond_output + args.cfg_scale * (cond_output - uncond_output)
            latent_input = combined.to(torch.bfloat16)

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
        wrapped_transformer.reset()

        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()

        for step_idx, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Unconditional
                uncond_output = wrapped_transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=[neg_embeds],
                    seq_len=seq_len,
                )[0]

                # Conditional
                cond_output = wrapped_transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=[pos_embeds],
                    seq_len=seq_len,
                )[0]

            # CFG combination
            combined = uncond_output + args.cfg_scale * (cond_output - uncond_output)
            latent_input = combined.to(torch.bfloat16)

        torch.cuda.synchronize()
        dist.barrier()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        if rank == 0:
            print(f"  Run {run_idx + 1}/{args.benchmark_runs}: {elapsed:.2f} ms ({elapsed/args.num_steps:.2f} ms/step)")

    mean_time = sum(times) / len(times)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DistriFusion Results ({world_size} GPUs):")
        print(f"  Mean inference time: {mean_time:.2f} ms")
        print(f"  Time per step: {mean_time / args.num_steps:.2f} ms")
        print(f"  Warmup steps: {args.warmup_steps}")
        print(f"  Mode: {args.mode}")
        print(f"\nCommunication:")
        print(f"  - Self-attn: all-gather KV ({seq_len // world_size} tokens/GPU)")
        print(f"  - After warmup: uses stale KV + async update")
        print(f"  - Cross-attn: text KV cached (no comm)")
        print(f"  - Final: all-gather output")
        print(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
