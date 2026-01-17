"""
DistriFusion model wrapper for Wan2.1.

DistriFusion key technique:
1. Split sequence across GPUs: each GPU processes seq_len/n_gpus tokens
2. Self-attention: all-gather KV from all GPUs, compute attention locally
3. After warmup: use stale KV from previous timestep + async update
4. Cross-attention: cache text KV (doesn't change across timesteps)
5. FFN: token-wise, no communication needed

This is different from PipeFusion (CFG batch splitting) which splits the
batch between uncond/cond processing.
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch import distributed as dist
from typing import Optional, List, Tuple

from pipefuser.models.base_model import BaseModel
from pipefuser.utils import DistriConfig
from pipefuser.logger import init_logger

logger = init_logger(__name__)


class DistriWanBlockDF(nn.Module):
    """
    DistriFusion wrapper for a single WanAttentionBlock.

    Handles sequence splitting at the block level:
    - Self-attention: all-gathers KV, outputs local portion
    - Cross-attention: caches text KV, outputs local portion
    - FFN: processes local portion
    """

    def __init__(self, block: nn.Module, distri_config: DistriConfig, block_idx: int):
        super().__init__()
        self.block = block
        self.distri_config = distri_config
        self.block_idx = block_idx

        # Get model dimensions
        self.dim = block.dim
        self.num_heads = block.num_heads
        self.head_dim = self.dim // self.num_heads

        # KV buffers for self-attention DistriFusion
        self.kv_buffer_list: Optional[List[torch.Tensor]] = None
        self.async_handle: Optional[dist.Work] = None

        # KV cache for cross-attention (text doesn't change)
        self.cross_kv_cache = None

        # Counter for warmup
        self.counter = 0

    def reset(self):
        """Reset caches and counters for new generation."""
        self.kv_buffer_list = None
        self.async_handle = None
        self.cross_kv_cache = None
        self.counter = 0

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        start_idx: int,
        end_idx: int,
        full_seq_len: int,
    ) -> torch.Tensor:
        """
        Forward pass for a single block with DistriFusion.

        Args:
            x: Local hidden states [B, local_len, dim]
            e: Modulation embedding [B, 6, dim]
            seq_lens: Full sequence lengths [B]
            grid_sizes: Grid sizes [B, 3] containing (F, H, W)
            freqs: RoPE frequencies
            context: Text context [B, context_len, dim]
            context_lens: Context lengths [B]
            start_idx: Start index of local portion in full sequence
            end_idx: End index of local portion in full sequence
            full_seq_len: Full sequence length

        Returns:
            x: Updated local hidden states [B, local_len, dim]
        """
        config = self.distri_config
        block = self.block
        b = x.shape[0]
        local_len = end_idx - start_idx

        # Unpack modulation
        with amp.autocast(dtype=torch.float32):
            mod = (block.modulation + e).chunk(6, dim=1)

        # === Self-attention with DistriFusion ===
        # Compute Q, K, V for local portion
        x_normed = block.norm1(x).float() * (1 + mod[1]) + mod[0]

        attn = block.self_attn
        q_local = attn.norm_q(attn.q(x_normed)).view(b, local_len, self.num_heads, self.head_dim)
        k_local = attn.norm_k(attn.k(x_normed)).view(b, local_len, self.num_heads, self.head_dim)
        v_local = attn.v(x_normed).view(b, local_len, self.num_heads, self.head_dim)

        # Apply 3D RoPE with correct global positions
        q_local = self._rope_apply_local(q_local, start_idx, grid_sizes, freqs)
        k_local = self._rope_apply_local(k_local, start_idx, grid_sizes, freqs)

        # Flatten for all-gather: [B, local_len, num_heads * head_dim * 2]
        kv_local = torch.cat([k_local.flatten(2), v_local.flatten(2)], dim=-1)

        # DistriFusion: gather KV from all GPUs
        if config.world_size > 1:
            n_gpus = config.n_device_per_batch
            rank = config.rank

            # Initialize buffer if needed
            if self.kv_buffer_list is None:
                self.kv_buffer_list = [torch.zeros_like(kv_local) for _ in range(n_gpus)]

            use_stale = (
                self.counter > config.warmup_steps
                and config.mode != "full_sync"
            )

            if not use_stale:
                # Warmup: synchronous all-gather
                dist.all_gather(
                    self.kv_buffer_list,
                    kv_local,
                    group=config.local_batch_parallel_group,
                    async_op=False,
                )
                full_kv = torch.cat(self.kv_buffer_list, dim=1)
            else:
                # DistriFusion: use stale KV
                if self.async_handle is not None:
                    self.async_handle.wait()
                    self.async_handle = None

                self.kv_buffer_list[rank] = kv_local
                full_kv = torch.cat(self.kv_buffer_list, dim=1)

                if config.mode != "no_sync":
                    self.async_handle = dist.all_gather(
                        self.kv_buffer_list,
                        kv_local,
                        group=config.local_batch_parallel_group,
                        async_op=True,
                    )

            # Split back to K, V
            kv_dim = self.num_heads * self.head_dim
            k_full = full_kv[..., :kv_dim].view(b, full_seq_len, self.num_heads, self.head_dim)
            v_full = full_kv[..., kv_dim:].view(b, full_seq_len, self.num_heads, self.head_dim)
        else:
            k_full = k_local
            v_full = v_local

        # Compute attention: local Q attends to full K, V
        # Use Wan's flash attention
        from wan.modules.attention import attention as flash_attention
        y_self = flash_attention(q_local, k_full, v_full, k_lens=seq_lens)

        # Output projection and residual
        y_self = y_self.flatten(2)
        y_self = attn.o(y_self)

        with amp.autocast(dtype=torch.float32):
            x = x + y_self * mod[2]

        # === Cross-attention with text KV caching ===
        cross_attn = block.cross_attn

        # Query from local visual features
        x_normed3 = block.norm3(x)
        q_cross = cross_attn.norm_q(cross_attn.q(x_normed3)).view(b, local_len, self.num_heads, self.head_dim)

        # KV from text (cached after first step)
        if self.counter == 0 or self.cross_kv_cache is None:
            k_cross = cross_attn.norm_k(cross_attn.k(context)).view(b, -1, self.num_heads, self.head_dim)
            v_cross = cross_attn.v(context).view(b, -1, self.num_heads, self.head_dim)
            self.cross_kv_cache = (k_cross, v_cross)
        else:
            k_cross, v_cross = self.cross_kv_cache

        y_cross = flash_attention(q_cross, k_cross, v_cross, k_lens=context_lens)
        y_cross = y_cross.flatten(2)
        y_cross = cross_attn.o(y_cross)

        x = x + y_cross

        # === FFN (token-wise, no communication) ===
        y_ffn = block.ffn(block.norm2(x).float() * (1 + mod[4]) + mod[3])
        with amp.autocast(dtype=torch.float32):
            x = x + y_ffn * mod[5]

        self.counter += 1

        return x

    @amp.autocast(enabled=False)
    def _rope_apply_local(self, x, start_idx, grid_sizes, freqs):
        """Apply 3D RoPE to local portion with correct global positions."""
        b, local_len, n, d = x.shape
        c = d // 2

        # Split freqs into F, H, W components
        freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            full_seq_len = f * h * w
            end_idx = min(start_idx + local_len, full_seq_len)
            actual_local_len = max(0, end_idx - start_idx)

            if actual_local_len <= 0:
                output.append(x[i])
                continue

            x_i = x[i, :actual_local_len]

            # Convert to complex for RoPE
            x_i = torch.view_as_complex(
                x_i.to(torch.float64).reshape(actual_local_len, n, -1, 2)
            )

            # Compute global positions for local indices
            pos_indices = torch.arange(start_idx, end_idx, device=x.device)
            frame_idx = pos_indices // (h * w)
            height_idx = (pos_indices // w) % h
            width_idx = pos_indices % w

            # Gather frequencies
            freqs_f = freqs_split[0][frame_idx.clamp(max=freqs_split[0].size(0) - 1)]
            freqs_h = freqs_split[1][height_idx.clamp(max=freqs_split[1].size(0) - 1)]
            freqs_w = freqs_split[2][width_idx.clamp(max=freqs_split[2].size(0) - 1)]

            freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).unsqueeze(1)

            # Apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

            # Pad if needed
            if actual_local_len < local_len:
                x_i = torch.cat([x_i, x[i, actual_local_len:]], dim=0)

            output.append(x_i)

        return torch.stack(output).float()


class DistriWanModelDF(BaseModel):
    """
    DistriFusion wrapper for Wan2.1 model.

    Splits sequence across GPUs, each GPU processes its portion with
    all-gathered KV for self-attention and cached KV for cross-attention.
    """

    def __init__(self, model: nn.Module, distri_config: DistriConfig):
        # Call parent init first
        super().__init__(model, distri_config)

        # Wrap blocks with DistriFusion
        self.df_blocks = nn.ModuleList([
            DistriWanBlockDF(block, distri_config, i)
            for i, block in enumerate(model.blocks)
        ])

        logger.info(
            f"DistriFusion wrapped {len(self.df_blocks)} blocks, "
            f"world_size: {distri_config.world_size}, "
            f"warmup_steps: {distri_config.warmup_steps}"
        )

    def reset(self):
        """Reset all caches and counters for new generation."""
        for block in self.df_blocks:
            block.reset()
        self.counter = 0

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass with DistriFusion sequence splitting.

        Each GPU processes seq_len/n_gpus tokens through all layers,
        with all-gathered KV for self-attention.
        """
        model = self.model
        config = self.distri_config

        # Import sinusoidal_embedding_1d from Wan module
        from wan.modules.model import sinusoidal_embedding_1d

        # Get device
        device = model.patch_embedding.weight.device
        if model.freqs.device != device:
            model.freqs = model.freqs.to(device)

        # Process inputs following WanModel.forward pattern
        # Patchify each input
        x_patches = [model.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x_patches]
        )
        x_patches = [u.flatten(2).transpose(1, 2) for u in x_patches]  # [1, seq, dim]
        seq_lens = torch.tensor([u.size(1) for u in x_patches], dtype=torch.long, device=device)

        # Pad to max seq_len and concatenate
        x_emb = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x_patches
        ])  # [B, seq_len, dim]

        b = x_emb.shape[0]
        full_seq_len = seq_lens.max().item()

        # Calculate local sequence range
        n_gpus = config.n_device_per_batch
        rank = config.rank
        local_seq_len = full_seq_len // n_gpus
        start_idx = rank * local_seq_len
        end_idx = start_idx + local_seq_len if rank < n_gpus - 1 else full_seq_len

        # Split to local portion
        x_local = x_emb[:, start_idx:end_idx, :].contiguous()

        # Get precomputed RoPE freqs
        freqs = model.freqs

        # Time embedding
        with amp.autocast(dtype=torch.float32):
            e = model.time_embedding(
                sinusoidal_embedding_1d(model.freq_dim, t).float()
            )
            e0 = model.time_projection(e).unflatten(1, (6, model.dim))

        # Prepare context (following WanModel pattern)
        context_emb = model.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )
        context_lens = None  # WanModel uses None for context_lens

        # Process through DistriFusion blocks
        for df_block in self.df_blocks:
            x_local = df_block(
                x_local, e0, seq_lens, grid_sizes, freqs,
                context_emb, context_lens,
                start_idx, end_idx, full_seq_len
            )

        # Final head on local portion
        x_local = model.head(x_local, e)

        # All-gather to reconstruct full sequence
        if n_gpus > 1:
            gathered = [torch.zeros_like(x_local) for _ in range(n_gpus)]
            dist.all_gather(gathered, x_local, group=config.local_batch_parallel_group)
            x_emb = torch.cat(gathered, dim=1)
        else:
            x_emb = x_local

        # Unpatchify
        out = model.unpatchify(x_emb, grid_sizes)

        self.counter += 1

        return [u.float() for u in out]


def wrap_wan_model_distrifusion(
    model: nn.Module,
    distri_config: DistriConfig,
) -> DistriWanModelDF:
    """
    Wrap a WanModel with DistriFusion for distributed inference.

    Args:
        model: WanModel instance
        distri_config: Distribution configuration

    Returns:
        DistriFusion-wrapped model
    """
    return DistriWanModelDF(model, distri_config)
