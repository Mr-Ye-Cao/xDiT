#!/usr/bin/env python3
"""
Single GPU baseline benchmark for Wan2.1 with CFG.

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_wan_single_gpu.py
"""

import argparse
import gc
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=" * 60)
    print(f"Single GPU Baseline with CFG")
    print("=" * 60)
    print(f"Frames: {args.frame_num}, Size: {args.size}, Steps: {args.num_steps}")
    print(f"CFG Scale: {args.cfg_scale}")

    # Import modules
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    # Parse video size
    height, width = map(int, args.size.split("*"))

    print(f"\nLoading model from {checkpoint_dir}...")

    # Load transformer
    transformer = WanModel.from_pretrained(checkpoint_dir)
    transformer.eval().requires_grad_(False)
    transformer = transformer.to(dtype=torch.bfloat16, device=device)

    # Load text encoder
    print("Loading T5 encoder...")
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(checkpoint_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint_dir, cfg.t5_tokenizer),
    )

    # Encode prompts
    prompt = "A cat walks on the grass, realistic style."
    negative_prompt = ""

    print("Encoding prompts...")
    with torch.no_grad():
        pos_embeds_list = text_encoder([prompt], torch.device('cpu'))
        neg_embeds_list = text_encoder([negative_prompt], torch.device('cpu'))
        pos_embeds = pos_embeds_list[0].to(device)
        neg_embeds = neg_embeds_list[0].to(device)

    print(f"Embeddings: pos={pos_embeds.shape}, neg={neg_embeds.shape}")

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

    print(f"\nLatent shape: [{16}, {latent_f}, {latent_h}, {latent_w}]")
    print(f"Sequence length: {seq_len}")

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

    # Warmup
    print(f"\nWarmup runs: {args.warmup_runs}")

    for warmup_idx in range(args.warmup_runs):
        latent_input = latent.clone()

        for step_idx, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0)

            # CFG: run both unconditional and conditional forward passes
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Unconditional
                uncond_output = transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=[neg_embeds],
                    seq_len=seq_len,
                )[0]

                # Conditional
                cond_output = transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=[pos_embeds],
                    seq_len=seq_len,
                )[0]

            # CFG combination
            combined = uncond_output + args.cfg_scale * (cond_output - uncond_output)
            latent_input = combined.to(torch.bfloat16)

        torch.cuda.synchronize()
        print(f"  Warmup {warmup_idx + 1}/{args.warmup_runs} done")

    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark
    print(f"\nBenchmark runs: {args.benchmark_runs}")

    times = []

    for run_idx in range(args.benchmark_runs):
        latent_input = latent.clone()

        torch.cuda.synchronize()
        start = time.perf_counter()

        for step_idx, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0)

            # CFG: run both unconditional and conditional forward passes
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Unconditional
                uncond_output = transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=[neg_embeds],
                    seq_len=seq_len,
                )[0]

                # Conditional
                cond_output = transformer(
                    x=[latent_input],
                    t=t_batch,
                    context=[pos_embeds],
                    seq_len=seq_len,
                )[0]

            # CFG combination
            combined = uncond_output + args.cfg_scale * (cond_output - uncond_output)
            latent_input = combined.to(torch.bfloat16)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        print(f"  Run {run_idx + 1}/{args.benchmark_runs}: {elapsed:.2f} ms ({elapsed/args.num_steps:.2f} ms/step)")

    mean_time = sum(times) / len(times)

    print(f"\n{'='*60}")
    print(f"Single GPU Baseline Results:")
    print(f"  Mean inference time: {mean_time:.2f} ms")
    print(f"  Time per step: {mean_time / args.num_steps:.2f} ms")
    print(f"\nCFG requires 2 forward passes per step")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
