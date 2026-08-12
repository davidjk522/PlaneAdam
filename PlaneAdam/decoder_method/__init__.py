"""
decoder_method holds Method 2 from the project README: a VoxelMorph-style CNN decoder
that predicts a deformation field from PlaneCycle/DINO feature maps, as an alternative to
the ConvexAdam optimization in convex_optimization/.

This directory was originally copied from a separate, larger registration-model repo. Most
of its files (loaders/, utils/getters.py's getModel dispatcher, ~20 model variants) still
reference that other repo's `models` package and dataset loaders, neither of which exist
here — they're left in place as reference/dead code rather than deleted, but importing them
will raise ImportError.

Only FMIR (this file's `getModel`-equivalent) is wired up to work standalone in PlaneAdam:
its missing `models.backbones.*` dependency has been reimplemented in `backbones.py`, and
its channel count is configurable to match PCA-reduced DINOv3 features instead of the
original 768/256-dim setup. See PlaneAdam/scripts/smoke_test_fmir_decoder.py for a working
end-to-end example on OASIS.
"""
from .FMIR import FMIR

__all__ = ["FMIR"]
