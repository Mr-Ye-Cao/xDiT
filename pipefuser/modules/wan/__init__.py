# PipeFusion modules for Wan2.1 video generation
from .attn import (
    DistriWanSelfAttentionPiP,
    DistriWanCrossAttentionPiP,
)

__all__ = [
    "DistriWanSelfAttentionPiP",
    "DistriWanCrossAttentionPiP",
]
