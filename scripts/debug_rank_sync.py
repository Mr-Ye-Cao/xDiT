#!/usr/bin/env python3
"""
Debug script to identify where ranks get out of sync.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 scripts/debug_rank_sync.py
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
    """Log with rank and timestamp."""
    print(f"[Rank {rank}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    log(rank, f"Started on device {device}, world_size={world_size}")

    # Step 1: Import modules
    log(rank, "Importing Wan modules...")
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.configs import WAN_CONFIGS
    log(rank, "Imports done")

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    # Step 2: Load transformer
    log(rank, f"Loading WanModel from {checkpoint_dir}...")
    load_start = time.perf_counter()
    transformer = WanModel.from_pretrained(checkpoint_dir)
    load_time = time.perf_counter() - load_start
    log(rank, f"WanModel loaded in {load_time:.1f}s")

    transformer.eval().requires_grad_(False)
    log(rank, "Moving model to GPU...")
    move_start = time.perf_counter()
    # Keep model in float32 - WanModel's autocast expects float32
    transformer = transformer.to(device=device)
    move_time = time.perf_counter() - move_start
    log(rank, f"Model moved to GPU in {move_time:.1f}s")

    # Synchronize
    log(rank, "Barrier 1: after model load")
    torch.cuda.synchronize()
    dist.barrier()
    log(rank, "Barrier 1: passed")

    # Step 3: Text encoder (rank 0 only)
    if rank == 0:
        log(rank, "Loading T5 encoder...")
        text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, cfg.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, cfg.t5_tokenizer),
        )
        log(rank, "T5 encoder loaded")
    else:
        log(rank, "Waiting for rank 0 to load T5...")

    log(rank, "Barrier 2: after T5 load")
    dist.barrier()
    log(rank, "Barrier 2: passed")

    # Step 4: Encode prompts
    prompt = "A cat walks on the grass, realistic style."
    negative_prompt = ""

    if rank == 0:
        log(rank, "Encoding prompts...")
        with torch.no_grad():
            pos_embeds_list = text_encoder([prompt], torch.device('cpu'))
            neg_embeds_list = text_encoder([negative_prompt], torch.device('cpu'))
            # Keep in float32 to match model dtype
            pos_embeds = pos_embeds_list[0].to(device)
            neg_embeds = neg_embeds_list[0].to(device)
        log(rank, f"Prompts encoded, shapes: pos={pos_embeds.shape}, neg={neg_embeds.shape}")
    else:
        log(rank, "Waiting for prompt encoding...")

    log(rank, "Barrier 3: after encoding")
    dist.barrier()
    log(rank, "Barrier 3: passed")

    # Step 5: Broadcast embeddings - FIXED: broadcast shapes separately
    if rank == 0:
        pos_shape = torch.tensor(pos_embeds.shape, device=device)
        neg_shape = torch.tensor(neg_embeds.shape, device=device)
    else:
        pos_shape = torch.zeros(2, dtype=torch.long, device=device)
        neg_shape = torch.zeros(2, dtype=torch.long, device=device)

    log(rank, f"Broadcasting pos_shape...")
    dist.broadcast(pos_shape, src=0)
    torch.cuda.synchronize()
    log(rank, f"pos_shape = {pos_shape.tolist()}")

    log(rank, f"Broadcasting neg_shape...")
    dist.broadcast(neg_shape, src=0)
    torch.cuda.synchronize()
    log(rank, f"neg_shape = {neg_shape.tolist()}")

    if rank != 0:
        pos_embeds = torch.zeros(pos_shape.tolist(), dtype=torch.float32, device=device)
        neg_embeds = torch.zeros(neg_shape.tolist(), dtype=torch.float32, device=device)

    log(rank, f"Broadcasting pos_embeds... (shape={pos_embeds.shape}, dtype={pos_embeds.dtype})")
    dist.broadcast(pos_embeds, src=0)
    torch.cuda.synchronize()
    log(rank, "pos_embeds received")

    log(rank, f"Broadcasting neg_embeds... (shape={neg_embeds.shape}, dtype={neg_embeds.dtype})")
    dist.broadcast(neg_embeds, src=0)
    torch.cuda.synchronize()
    log(rank, "neg_embeds received")

    # Each rank gets its context
    if rank == 0:
        context = [neg_embeds]  # Unconditional
    else:
        context = [pos_embeds]  # Conditional

    log(rank, "Barrier 4: after broadcast")
    dist.barrier()
    log(rank, "Barrier 4: passed")

    # Step 6: Create inputs
    height, width = 480, 832
    frame_num = 17
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

    log(rank, f"Latent: [{16}, {latent_f}, {latent_h}, {latent_w}], seq_len={seq_len}")

    generator = torch.Generator(device=device).manual_seed(42)
    latent = torch.randn(
        (16, latent_f, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    t = torch.tensor([500.0], device=device)

    log(rank, "Barrier 5: before forward")
    torch.cuda.synchronize()
    dist.barrier()
    log(rank, "Barrier 5: passed")

    # Step 7: Single forward pass
    log(rank, "Starting forward pass...")
    torch.cuda.synchronize()
    fwd_start = time.perf_counter()

    with torch.no_grad():
        output = transformer(
            x=[latent],
            t=t,
            context=context,
            seq_len=seq_len,
        )

    torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - fwd_start) * 1000
    log(rank, f"Forward pass done in {fwd_time:.1f}ms, output shape: {output[0].shape}")

    log(rank, "Barrier 6: after forward")
    dist.barrier()
    log(rank, "Barrier 6: passed")

    # Step 8: All-gather test
    gathered = [torch.zeros_like(output[0]) for _ in range(world_size)]
    log(rank, "Starting all-gather...")
    dist.all_gather(gathered, output[0])
    log(rank, f"All-gather done, got {len(gathered)} outputs")

    # CFG combination
    uncond = gathered[0]
    cond = gathered[1]
    combined = uncond + 7.5 * (cond - uncond)
    log(rank, f"CFG combination done, shape: {combined.shape}")

    log(rank, "Barrier 7: final")
    dist.barrier()
    log(rank, "Barrier 7: passed - ALL TESTS PASSED!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
