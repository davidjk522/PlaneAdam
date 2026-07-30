# Extract 2D features from each plane using DINO model
# Backbone Frozen unless finetune is specified
import torch
from torch import nn

from models.hub.backbones import _make_dinov3_vit_model_arch
from models.layers import Mlp, SwiGLUFFN


class DinoBackboneExtractor:
    """
    Wraps a DINOv3 ViT backbone for 2D feature extraction from each plane.
    """

    # (embed_dim, depth, num_heads, ffn_type) -> compact_arch_name, as used by
    # models/hub/backbones.py to build the dinov3_* factory/checkpoint names.
    ARCH_CONFIGS = {
        (384, 12, 6, Mlp): "vits",
        (384, 12, 6, SwiGLUFFN): "vitsplus",
        (768, 12, 12, Mlp): "vitb",
        (1024, 24, 16, Mlp): "vitl",
        (1024, 24, 16, SwiGLUFFN): "vitlplus",
        (1280, 32, 20, SwiGLUFFN): "vithplus",
        (4096, 40, 32, SwiGLUFFN): "vit7b",
    }

    def __init__(self, backbone: nn.Module):
        self.device = self.get_device()
        self.backbone = backbone.to(self.device)
        self.arch_name = self.find_backbone(self.backbone)

    @staticmethod
    def detect_cuda() -> bool:
        """
        Detect if CUDA is available.
        """
        return torch.cuda.is_available()

    @classmethod
    def get_device(cls) -> torch.device:
        """
        Get the device to use for computation.
        """
        return torch.device("cuda" if cls.detect_cuda() else "cpu")

    @classmethod
    def find_backbone(cls, backbone: nn.Module) -> str:
        """
        Autodetect which DINOv3 ViT backbone architecture is loaded, by matching
        its embed_dim/depth/num_heads/ffn-type against the known dinov3_* configs.
        Returns a name like "dinov3_vitb16", or "unknown" if it doesn't match.
        """
        embed_dim = getattr(backbone, "embed_dim", None)
        depth = getattr(backbone, "n_blocks", None)
        num_heads = getattr(backbone, "num_heads", None)
        patch_size = getattr(backbone, "patch_size", None)
        blocks = getattr(backbone, "blocks", None)
        if embed_dim is None or depth is None or num_heads is None or patch_size is None or not blocks:
            return "unknown"

        ffn_type = type(blocks[0].mlp)
        compact_arch_name = cls.ARCH_CONFIGS.get((embed_dim, depth, num_heads, ffn_type))
        if compact_arch_name is None:
            return "unknown"

        model_arch = _make_dinov3_vit_model_arch(patch_size=patch_size, compact_arch_name=compact_arch_name)
        return f"dinov3_{model_arch}"
    
