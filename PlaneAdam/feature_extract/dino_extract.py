# Extract 2D features from each plane using DINO model
# Backbone Frozen unless finetune is specified
import torch
from torch import nn

from models.hub.backbones import _make_dinov3_vit_model_arch
from models.layers import Mlp, SwiGLUFFN
from planecycle.converters.converter import PlaneCycleConverter


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

    def __init__(self, backbone: nn.Module, plane_chunk_size: int | None = None):
        self.device = self.get_device()
        self.backbone = backbone.to(self.device)
        self.arch_name = self.find_backbone(self.backbone)
        self.planecycle = PlaneCycleConverter(self.backbone, plane_chunk_size=plane_chunk_size)

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

    def extract_feature_planecycle(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Extract the 3D spatial feature map from a volume via PlaneCycle.

        Args:
            volume: Input volume (B, C, D, H, W).
        Returns:
            xf: Spatial feature map (B, D, H, W, C) — used for registration.
            (The per-plane global feature vector xcls is also produced internally
            by PlaneCycle but isn't needed for registration, so it's discarded here.)
        """
        self.backbone.eval()
        with torch.no_grad():
            # bf16 autocast: inference-only forward, halves activation memory with negligible
            # accuracy impact — bf16 shares fp32's exponent range, so no overflow risk, unlike
            # fp16. Added after 896x896 + dinov3_vitb16 combined to OOM a 24GB A5000. Cast back to
            # fp32 immediately since downstream PCA/correlate/Adam-refinement code expects it.
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                xf, _xcls = self.planecycle(volume.to(self.device))
            return xf.float()