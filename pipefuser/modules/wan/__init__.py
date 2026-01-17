# PipeFusion modules for Wan2.1 video generation
from .attn import (
    DistriWanSelfAttentionPiP,
    DistriWanCrossAttentionPiP,
)

# DistriFusion modules for Wan2.1
from .distrifusion_model import (
    DistriWanBlockDF,
    DistriWanModelDF,
    wrap_wan_model_distrifusion,
)

__all__ = [
    # PipeFusion (CFG batch splitting)
    "DistriWanSelfAttentionPiP",
    "DistriWanCrossAttentionPiP",
    # DistriFusion (sequence splitting with stale KV)
    "DistriWanBlockDF",
    "DistriWanModelDF",
    "wrap_wan_model_distrifusion",
]
