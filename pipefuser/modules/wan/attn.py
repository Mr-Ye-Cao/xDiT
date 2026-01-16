# Copyright 2024 xDiT Authors. Adapted for Wan2.1 video generation.
"""
PipeFusion (Patch Parallel) for Wan2.1 attention.

Key differences from Tensor Parallelism:
- TP: Shard heads, all-reduce every layer (90 all-reduces/forward)
- PipeFusion: Split batch for CFG (half GPUs for Uncond, half for Cond)

CFG Handling:
- xDiT splits batch across GPUs (half for Uncond, half for Cond)
- No cache mixing issue like TP approach
- No all-reduce needed (unlike TP which requires 90 all-reduces per forward)

KV Cache Strategy:
- Self-attention: Compute fresh each time (3D RoPE complicates caching)
- Cross-attention: Cache text KV (doesn't change within inference, no RoPE)
"""

import torch
from torch import nn
from torch.nn import functional as F

from pipefuser.modules.base_module import BaseModule
from pipefuser.utils import DistriConfig
from pipefuser.logger import init_logger

logger = init_logger(__name__)


class DistriWanSelfAttentionPiP(BaseModule):
    """
    PipeFusion wrapper for WanSelfAttention.

    For video with 3D RoPE, we compute fresh Q, K, V each time
    (caching KV with RoPE positions is complex).
    The speedup comes from CFG batch splitting (no all-reduce).
    """

    def __init__(self, module: nn.Module, distri_config: DistriConfig):
        super().__init__(module, distri_config)

        # Store module attributes
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.dim = module.dim
        self.window_size = module.window_size

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        Forward pass - delegates to original module.

        PipeFusion benefit: CFG batch splitting means this GPU only processes
        either Uncond or Cond, not both. No all-reduce needed.

        Args:
            x: Input tensor [B, L, C]
            seq_lens: Sequence lengths [B]
            grid_sizes: Grid sizes [B, 3] containing (F, H, W)
            freqs: RoPE frequencies
        """
        # Just call the original module - no KV caching for self-attention
        # The benefit is that this GPU only processes half the batch (Uncond or Cond)
        output = self.module(x, seq_lens, grid_sizes, freqs)
        self.counter += 1
        return output


class DistriWanCrossAttentionPiP(BaseModule):
    """
    PipeFusion wrapper for WanT2VCrossAttention.

    Cross-attention KV comes from text embeddings which don't change,
    so we cache KV after first computation. No RoPE for cross-attention.
    """

    def __init__(self, module: nn.Module, distri_config: DistriConfig):
        super().__init__(module, distri_config)

        # Combine K and V projections for efficiency
        to_k = module.k
        to_v = module.v

        in_size, out_size = to_k.in_features, to_k.out_features

        to_kv = nn.Linear(
            in_size, out_size * 2,
            bias=to_k.bias is not None,
            device=to_k.weight.device,
            dtype=to_k.weight.dtype,
        )
        to_kv.weight.data[:out_size].copy_(to_k.weight.data)
        to_kv.weight.data[out_size:].copy_(to_v.weight.data)

        if to_k.bias is not None:
            to_kv.bias.data[:out_size].copy_(to_k.bias.data)
            to_kv.bias.data[out_size:].copy_(to_v.bias.data)

        self.to_kv = to_kv
        self.kv_cache = None  # Cache for text KV

        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.dim = module.dim

    def forward(self, x, context, context_lens):
        """
        Forward with text KV caching.

        Args:
            x: Input tensor [B, L1, C]
            context: Text context [B, L2, C]
            context_lens: Context lengths [B]
        """
        module = self.module

        b, n, d = x.shape[0], self.num_heads, self.head_dim

        # Compute Q (always fresh - depends on x)
        q = module.norm_q(module.q(x)).view(b, -1, n, d)

        # Compute or retrieve cached KV
        # Text context doesn't change within a generation, so cache after first step
        recompute_kv = self.counter == 0 or self.kv_cache is None

        if recompute_kv:
            kv = self.to_kv(context)
            self.kv_cache = kv
        else:
            kv = self.kv_cache

        # Split and reshape KV
        k_raw, v = torch.split(kv, kv.shape[-1] // 2, dim=-1)
        k = module.norm_k(k_raw).view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        # Attention (no RoPE for cross-attention)
        from wan.modules.attention import attention as flash_attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # Output
        x = x.flatten(2)
        x = module.o(x)

        self.counter += 1

        return x

    def reset_kv_cache(self):
        """Reset KV cache at start of new generation."""
        self.kv_cache = None
        self.counter = 0
