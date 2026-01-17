"""
DistriFusion attention modules for Wan2.1.

DistriFusion splits the sequence across GPUs and uses stale KV from previous timestep
to reduce communication overhead.

Key mechanism:
1. Split sequence: each GPU processes seq_len/N tokens
2. All-gather KV during warmup steps
3. After warmup: use stale KV from previous timestep + async update

For Wan2.1's 3D RoPE:
- Sequence is ordered as F*H*W (frame, height, width)
- Position p maps to (p//(H*W), (p//W)%H, p%W)
- Each GPU applies RoPE with correct global positions for its local portion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from typing import Optional, List
import torch.cuda.amp as amp

from pipefuser.models.base_model import BaseModule
from pipefuser.utils import DistriConfig


def rope_params(max_seq_len, dim, theta=10000):
    """Compute RoPE frequency parameters."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply_local(x, start_idx, grid_sizes, freqs):
    """
    Apply 3D RoPE to a local portion of the sequence.

    Args:
        x: [B, local_len, num_heads, head_dim] - local Q or K
        start_idx: global start position for this GPU's portion
        grid_sizes: [B, 3] containing (F, H, W)
        freqs: precomputed rope frequencies

    Returns:
        x with RoPE applied, maintaining global positional consistency
    """
    b, local_len, n, d = x.shape
    c = d // 2

    # Split freqs into F, H, W components
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        full_seq_len = f * h * w

        # Handle case where local portion extends beyond sequence
        end_idx = min(start_idx + local_len, full_seq_len)
        actual_local_len = end_idx - start_idx

        if actual_local_len <= 0:
            # This GPU has no valid tokens for this sample
            output.append(x[i])
            continue

        x_i = x[i, :actual_local_len]  # [local_len, n, d]

        # Convert to complex for RoPE
        x_i = torch.view_as_complex(
            x_i.to(torch.float64).reshape(actual_local_len, n, -1, 2)
        )

        # Build positional encoding for our local positions
        # Position p in full sequence maps to:
        #   frame = p // (h * w)
        #   height = (p // w) % h
        #   width = p % w
        pos_indices = torch.arange(start_idx, end_idx, device=x.device)
        frame_indices = pos_indices // (h * w)
        height_indices = (pos_indices // w) % h
        width_indices = pos_indices % w

        # Gather frequencies for each position
        freqs_f = freqs[0][frame_indices.clamp(max=freqs[0].size(0) - 1)]  # [local_len, c_f]
        freqs_h = freqs[1][height_indices.clamp(max=freqs[1].size(0) - 1)]  # [local_len, c_h]
        freqs_w = freqs[2][width_indices.clamp(max=freqs[2].size(0) - 1)]  # [local_len, c_w]

        # Concatenate and reshape
        freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)  # [local_len, c]
        freqs_i = freqs_i.unsqueeze(1)  # [local_len, 1, c]

        # Apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        # Pad back to local_len if needed
        if actual_local_len < local_len:
            padding = x[i, actual_local_len:]
            x_i = torch.cat([x_i, padding], dim=0)

        output.append(x_i)

    return torch.stack(output).float()


class DistriWanSelfAttentionDF(BaseModule):
    """
    DistriFusion self-attention for Wan2.1.

    Each GPU processes a portion of the sequence. KV is gathered from all GPUs
    for full attention computation. After warmup, uses stale KV from previous step.
    """

    def __init__(self, module: nn.Module, distri_config: DistriConfig):
        super().__init__(module, distri_config)

        # The original Wan self-attention module
        self.self_attn = module

        # KV cache for stale activations (DistriFusion technique)
        self.kv_buffer_list: Optional[List[torch.Tensor]] = None
        self.async_handle: Optional[dist.Work] = None

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with DistriFusion.

        For Wan2.1:
        - x: [B, seq_len, dim] where seq_len = F*H*W grid
        - We split seq_len across GPUs
        """
        distri_config = self.distri_config

        if distri_config.world_size == 1:
            # Single GPU - no distribution needed
            return self.self_attn(x, seq_lens, grid_sizes, freqs)

        b, full_seq_len, dim = x.shape
        n_gpus = distri_config.n_device_per_batch
        rank = distri_config.rank

        # Calculate local sequence range
        local_seq_len = full_seq_len // n_gpus
        start_idx = rank * local_seq_len
        end_idx = start_idx + local_seq_len if rank < n_gpus - 1 else full_seq_len
        actual_local_len = end_idx - start_idx

        # Get local portion of x
        x_local = x[:, start_idx:end_idx, :]

        # Compute local Q, K, V using the original attention's projections
        attn = self.self_attn
        num_heads = attn.num_heads
        head_dim = dim // num_heads

        # Get Q, K, V projections
        q_local = attn.q(x_local)
        k_local = attn.k(x_local)
        v_local = attn.v(x_local)

        # Apply QK normalization if present
        if hasattr(attn, 'norm_q') and not isinstance(attn.norm_q, nn.Identity):
            q_local = attn.norm_q(q_local)
        if hasattr(attn, 'norm_k') and not isinstance(attn.norm_k, nn.Identity):
            k_local = attn.norm_k(k_local)

        # Reshape for RoPE: [B, local_len, num_heads, head_dim]
        q_local = q_local.view(b, actual_local_len, num_heads, head_dim)
        k_local = k_local.view(b, actual_local_len, num_heads, head_dim)
        v_local = v_local.view(b, actual_local_len, num_heads, head_dim)

        # Apply 3D RoPE with correct global positions
        q_local = rope_apply_local(q_local, start_idx, grid_sizes, freqs)
        k_local = rope_apply_local(k_local, start_idx, grid_sizes, freqs)

        # Stack K and V for efficient all-gather [B, local_seq, num_heads, head_dim*2]
        kv_local = torch.cat([k_local, v_local], dim=-1)

        # Initialize buffer list if needed
        if self.kv_buffer_list is None:
            self.kv_buffer_list = [
                torch.zeros_like(kv_local) for _ in range(n_gpus)
            ]

        # Determine if we need fresh KV or can use stale
        use_stale = (
            self.counter > distri_config.warmup_steps
            and distri_config.mode != "full_sync"
        )

        if not use_stale:
            # Warmup: synchronous all-gather
            dist.all_gather(
                self.kv_buffer_list,
                kv_local,
                group=distri_config.batch_parallel_group,
                async_op=False,
            )
            full_kv = torch.cat(self.kv_buffer_list, dim=1)
        else:
            # DistriFusion: use stale KV from previous step
            # Wait for previous async communication if any
            if self.async_handle is not None:
                self.async_handle.wait()
                self.async_handle = None

            # Update our local portion in the buffer
            self.kv_buffer_list[rank] = kv_local
            full_kv = torch.cat(self.kv_buffer_list, dim=1)

            # Start async communication for next step
            if distri_config.mode != "no_sync":
                self.async_handle = dist.all_gather(
                    self.kv_buffer_list,
                    kv_local,
                    group=distri_config.batch_parallel_group,
                    async_op=True,
                )

        # Split full_kv back to K and V [B, full_seq, num_heads, head_dim]
        k_full, v_full = full_kv.split(full_kv.shape[-1] // 2, dim=-1)

        # Reshape for attention: [B, num_heads, seq_len, head_dim]
        q_local = q_local.transpose(1, 2)  # [B, num_heads, local_len, head_dim]
        k_full = k_full.transpose(1, 2)  # [B, num_heads, full_len, head_dim]
        v_full = v_full.transpose(1, 2)  # [B, num_heads, full_len, head_dim]

        # Compute attention: local Q attends to full K, V
        hidden_states = F.scaled_dot_product_attention(
            q_local, k_full, v_full, dropout_p=0.0, is_causal=False
        )

        # Reshape output: [B, local_len, dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(b, actual_local_len, dim)

        # Output projection
        hidden_states = attn.o(hidden_states)

        self.counter += 1

        return hidden_states


class DistriWanCrossAttentionDF(BaseModule):
    """
    DistriFusion cross-attention for Wan2.1.

    For cross-attention:
    - Q comes from visual features (split across GPUs)
    - K, V come from text context (same for all GPUs, can be cached)
    """

    def __init__(self, module: nn.Module, distri_config: DistriConfig):
        super().__init__(module, distri_config)
        self.cross_attn = module
        self.kv_cache = None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention forward with local Q and full context KV.

        Args:
            x: [B, local_seq_len, dim] - local visual features for this GPU
            context: [B, context_len, dim] - text context (same on all GPUs)
            context_lens: optional lengths
        """
        distri_config = self.distri_config

        # Single GPU case
        if distri_config.world_size == 1:
            return self.cross_attn(x, context, context_lens)

        # Recompute KV on first step only (text doesn't change)
        recompute_kv = self.counter == 0 or self.kv_cache is None

        attn = self.cross_attn
        b, seq_len, dim = x.shape
        num_heads = attn.num_heads
        head_dim = dim // num_heads
        context_len = context.shape[1]

        # Query from local visual features
        q = attn.q(x)
        if hasattr(attn, 'norm_q') and not isinstance(attn.norm_q, nn.Identity):
            q = attn.norm_q(q)

        # KV from text context (cached after first step)
        if recompute_kv:
            k = attn.k(context)
            v = attn.v(context)
            if hasattr(attn, 'norm_k') and not isinstance(attn.norm_k, nn.Identity):
                k = attn.norm_k(k)
            self.kv_cache = (k, v)
        else:
            k, v = self.kv_cache

        # Reshape for attention
        q = q.view(b, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(b, context_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(b, context_len, num_heads, head_dim).transpose(1, 2)

        # Compute attention
        hidden_states = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )

        # Reshape and project output
        hidden_states = hidden_states.transpose(1, 2).reshape(b, seq_len, dim)
        hidden_states = attn.o(hidden_states)

        self.counter += 1

        return hidden_states


class DistriWanFFNDF(BaseModule):
    """
    DistriFusion FFN wrapper for Wan2.1.

    FFN operates independently on each token, so no communication needed.
    Just processes the local portion of the sequence.
    """

    def __init__(self, module: nn.Module, distri_config: DistriConfig):
        super().__init__(module, distri_config)
        self.ffn = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN forward on local tokens.

        Args:
            x: [B, local_seq_len, dim] - local features for this GPU
        """
        # FFN is token-wise, no communication needed
        return self.ffn(x)
