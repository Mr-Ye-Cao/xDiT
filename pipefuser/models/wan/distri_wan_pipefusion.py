# Copyright 2024 xDiT Authors. Adapted for Wan2.1 video generation.
"""
PipeFusion wrapper for Wan2.1 DiT model.

Key Benefits over Tensor Parallelism:
1. CFG Batch Splitting:
   - Half GPUs process Unconditional, half process Conditional
   - No all-reduce needed (TP requires 90 all-reduces per forward)

2. Text KV Caching:
   - Cross-attention KV is cached (text doesn't change during generation)
   - Saves compute on repeated text projection

Communication Pattern:
- TP: 90 all-reduces per forward pass (30 layers × 3 ops)
- PipeFusion: 0 all-reduces per forward pass (just final output combine)
"""

import torch
from torch import nn
from diffusers import ModelMixin, ConfigMixin

from pipefuser.models.base_model import BaseModel
from pipefuser.modules.base_module import BaseModule
from pipefuser.modules.wan.attn import (
    DistriWanSelfAttentionPiP,
    DistriWanCrossAttentionPiP,
)
from pipefuser.utils import DistriConfig
from pipefuser.logger import init_logger

logger = init_logger(__name__)


class DistriWanDiTPipeFusion(BaseModel):
    """
    PipeFusion wrapper for Wan2.1 DiT model.

    Wraps WanModel's attention blocks with PipeFusion variants that:
    - Split CFG batch across GPUs (half for Uncond, half for Cond)
    - Cache text cross-attention KV
    - Avoid all-reduce communication
    """

    def __init__(self, model: nn.Module, distri_config: DistriConfig):
        """
        Initialize PipeFusion wrapper for Wan2.1.

        Args:
            model: WanModel instance
            distri_config: Distributed configuration
        """
        # Import here to avoid circular imports
        from wan.modules.model import WanSelfAttention, WanT2VCrossAttention

        # Wrap attention modules in each block
        num_self_attn = 0
        num_cross_attn = 0

        for block in model.blocks:
            # Wrap self-attention
            if hasattr(block, 'self_attn') and isinstance(block.self_attn, WanSelfAttention):
                block.self_attn = DistriWanSelfAttentionPiP(
                    block.self_attn, distri_config
                )
                num_self_attn += 1

            # Wrap cross-attention
            if hasattr(block, 'cross_attn') and isinstance(block.cross_attn, WanT2VCrossAttention):
                block.cross_attn = DistriWanCrossAttentionPiP(
                    block.cross_attn, distri_config
                )
                num_cross_attn += 1

        logger.info(
            f"PipeFusion wrapped {num_self_attn} self-attention and "
            f"{num_cross_attn} cross-attention modules"
        )
        logger.info(
            f"Using PipeFusion parallelism, world_size: {distri_config.world_size}, "
            f"n_device_per_batch: {distri_config.n_device_per_batch}"
        )

        super().__init__(model, distri_config)

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        """
        Forward pass through the wrapped Wan model.

        This is the same interface as WanModel.forward().
        The PipeFusion benefit is that this GPU only processes
        its portion of the CFG batch (either Uncond or Cond).

        Args:
            x: List of input video tensors [C_in, F, H, W]
            t: Diffusion timesteps [B]
            context: List of text embeddings [L, C]
            seq_len: Maximum sequence length
            clip_fea: CLIP features for I2V (optional)
            y: Conditional video for I2V (optional)

        Returns:
            List of denoised video tensors
        """
        output = self.model(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y,
        )
        self.counter += 1
        return output

    def reset_kv_caches(self):
        """Reset all KV caches at start of new generation."""
        for module in self.model.modules():
            if isinstance(module, DistriWanCrossAttentionPiP):
                module.reset_kv_cache()

    @property
    def config(self):
        """Return the model config."""
        return self.model.config


def wrap_wan_model_pipefusion(model: nn.Module, distri_config: DistriConfig):
    """
    Wrap a WanModel with PipeFusion parallelism.

    Args:
        model: WanModel instance
        distri_config: Distributed configuration

    Returns:
        DistriWanDiTPipeFusion: Wrapped model
    """
    return DistriWanDiTPipeFusion(model, distri_config)
