# PlaneAdam

Use PlaneCycle to lift 2D DINO feature maps into a proper 3D feature map for volumetric image registration.

## Approach

- **Method 1:** Use ConvexAdam to optimize the deformation field directly on PlaneCycle features.
- **Method 2:** Use a VoxelMorph-style CNN decoder to predict the deformation field.
  - Open question: can we naively concatenate two large DINO-extracted feature maps?
  - If not, the fix is likely the approach used in FMIR — PCA for dimensionality reduction, then convolution on top.

## Development stages

1. Extract two feature maps (fixed + moving) using DINOv2/DINOv3.
2. Process the feature map tensors.
3. Use the ConvexAdam method to optimize the deformation field.
    - Add PCA to lower the calculation burden
    - Could be better improved if we could use GLIDE-REG-style PCA, but it's not open-sourced yet.
    - What else?
4. Check the deformation quality via L2R.

## Usage

```bash
python train.py --model:<name> --dataset:<dataset directory> --epochs:<epoch number> --batch:<batch>

python PlaneAdam/convex_optimization/convex_run_paired_dino.py <gpu> <config.json>

# with PCA channel reduction (recommended for large backbones)
python PlaneAdam/convex_optimization/convex_run_paired_dino.py <gpu> <config.json> --pca 64
```

Example:

```bash
python PlaneAdam/convex_optimization/convex_run_paired_dino.py 0 OASIS_config.json --pca 64
```

## Technical notes

### Native ViT grid vs. image grid resolution

The pipeline can run its correlation search at two different spatial resolutions for the same volume:

- **Image grid** — the raw voxel resolution: `H, W, D = 160, 224, 192`. This is the grid the nib-loaded image, the segmentation labels, and Dice evaluation all live on; every voxel is one real anatomical sample point.
- **Native ViT patch grid** — the resolution DINOv3's feature map actually comes out at: `Hf, Wf, Df = 10, 14, 192`.

The native grid is *not* a uniform downsample of the image grid — it's anisotropic, because of how PlaneCycle drives the 2D ViT backbone over a 3D volume:

- `H` and `W` are divided by the ViT's `patch_size=16` (each 16×16 pixel block becomes one token): `160/16=10`, `224/16=14`.
- `D` is the axis PlaneCycle cycles the 2D backbone over — the volume is processed slice-by-slice along `D`, so `D` is never patchified: it stays `192`.

So one cell of the native ViT grid covers a 16×16×1 voxel block of the image grid — 16 real voxels in `H`, 16 in `W`, but only 1 in `D`.

**Why this matters:** when the correlation search runs directly on the native grid, a candidate displacement of "1 cell" means 16 voxels in `H`/`W` but only 1 voxel in `D` — the search can't propose anything finer than a 16-voxel jump in-plane, while real registration needs corrections of just a few voxels. This is what causes `H`/`W` displacement to collapse to zero.

The alternative (not currently taken) is to upsample features back to the image grid: trilinearly interpolate the DINO feature map from `(10, 14, 192)` up to `(160, 224, 192)` before the `grid_sp` pooling and correlation search — the same resolution the old MIND descriptor's features were already at. This makes one search-cell equal `grid_sp` voxels uniformly on every axis (no 16x asymmetry), letting the search represent small in-plane shifts — at the cost of running the correlation loop (already `disp_hw³` candidates) over much larger tensors.

### PlaneCycle axis-cycling correctness

The axis-cycling itself is entirely internal to `PlaneCycleConverter` ([planecycle/converters/converter.py](planecycle/converters/converter.py)), which the rest of the pipeline calls but never modifies:

```python
def __init__(self, backbone, cycle_order=("HW", "DW", "DH", "HW"), ...):
    ...
    self.backbone.blocks = nn.ModuleList([
        PlaneCycleOp(..., plane=cycle_order[i % len(cycle_order)], ...)
        for i, blk in enumerate(self.backbone.blocks)
    ])
```

Each transformer block gets assigned a plane label round-robin from `("HW", "DW", "DH", "HW")` — block 0 processes axial (H×W) slices stacked along `D`, block 1 processes coronal (D×W) slices stacked along `H`, block 2 sagittal (D×H) stacked along `W`, block 3 back to axial, etc. This is what lets a 2D backbone build 3D context: different blocks see different cross-sectional orientations of the same volume.

[PlaneAdam/feature_extract/dino_extract.py](PlaneAdam/feature_extract/dino_extract.py) builds this once in `DinoBackboneExtractor.__init__` (`self.planecycle = PlaneCycleConverter(self.backbone)`), and `extract_feature_planecycle` just forwards through it — `cycle_order`, `PlaneCycleOp`, and the block assignment are never touched downstream.

What *is* ours to get right is the axis labeling of the volume fed in, since PlaneCycle's `(B, C, D, H, W)` convention only means what it's supposed to if `D, H, W` actually correspond to the real depth/height/width axes. That's `to_volume_tensor` in [PlaneAdam/Dataset/load_dataset_OASIS.py](PlaneAdam/Dataset/load_dataset_OASIS.py):

```python
def to_volume_tensor(img: torch.Tensor) -> torch.Tensor:
    x = img.permute(2, 0, 1).unsqueeze(0)  # (H,W,D) -> (1, D, H, W)
    return x.repeat(3, 1, 1, 1)  # (3, D, H, W)
```

`nib` loads volumes as `(H, W, D)`; this permutes to `(D, H, W)` before the channel-repeat, matching `PlaneCycleConverter.forward`'s expected input (`x: (B, C, D, H, W)`). Validated by a self-comparison cosine similarity of ~1.0 in the smoke test.

### PCA channel reduction

`--pca <k>` reduces the DINO feature channels via a real linear-subspace projection ([PlaneAdam/convex_optimization/convex_run_paired_dino.py](PlaneAdam/convex_optimization/convex_run_paired_dino.py), `reduce_channels_pca`), not random dropping or a subset of existing channels:

```python
X = torch.cat([flat_fix, flat_mov], dim=0).float()
mean = X.mean(0, keepdim=True)
_, _, V = torch.pca_lowrank(X - mean, q=k, niter=4)

def project(flat):
    return ((flat.float() - mean) @ V).to(features_fix.dtype)
```

1. **Pool the voxels:** stack every fixed and moving native-grid voxel from this pair into one `(N, 384)` matrix, fit jointly (not separately) so both sides land in the same reduced space — otherwise their distance in `correlate()` would be meaningless.
2. **Center:** subtract the mean feature vector.
3. **`torch.pca_lowrank(..., q=k)`:** a randomized SVD (Halko et al. — random projection + `niter=4` power iterations) finds the top-`k` right singular vectors `V` of the centered data — the `k` orthogonal directions in the original 384-dim space capturing the most variance. This is top-`k` *directions*, not top-`k` original channels.
4. **Project:** every voxel's 384-dim vector is multiplied by `V`, producing a `k`-dim vector that's a linear combination of all 384 original channels. No original channel is kept or dropped wholesale.

Two caveats:
- The basis is refit fresh for every (setting × pair) combination — fitting once per pair instead of per-run would be more efficient if this becomes a bottleneck.
- It's an approximation (randomized SVD, not exact), so the basis isn't bit-identical between calls for the same input, though `niter=4` keeps it close to the true top-k SVD for well-conditioned data like this.

### Input resolution upsampling before extraction

The native ViT grid is coarse (`10×14` in-plane, see above) because `patch_size=16` is fixed for every DINOv3 variant — there's no finer-patch checkpoint to switch to. [MedDINOv3](https://arxiv.org/abs/2509.02379) validated a different lever for this: keep `patch_size` fixed, but increase the *input* resolution before patch embedding (their ablation: `640×640 → 896×896`, `+2.06%` DSC on AMOS22). We apply the same idea in [PlaneAdam/convex_optimization/convex_run_paired_dino.py](PlaneAdam/convex_optimization/convex_run_paired_dino.py) — trilinearly upsample the volume's in-plane (`H, W`) resolution right before `extract_feature_planecycle`, leaving `D` untouched (it was never patchified to begin with):

```python
fixed_volume = F.interpolate(fixed_volume, size=(D, 320, 320), mode='trilinear', align_corners=False)
```

This is isolated to the extraction input only — everything downstream (`correlate`, `coupled_convex`, the final displacement-field upsample, Dice evaluation) still targets the real image resolution `(160, 224, 192)`, unchanged.

**The literal MedDINOv3 number doesn't fit here.** Tested empirically on a 12GB RTX 5070:
- `896×896` — OOMs on all 32 swept settings, during extraction itself (never reaches `correlate()`).
- `512×512` — also OOMs on all 32 settings.
- `320×320` — works cleanly, zero OOMs across all 32 settings (~140s for a 1-pair sanity sweep). Gives a `20×20` native token grid, up from `10×14`.

Note: our native `H, W = 160, 224` isn't square like MedDINOv3's CT slices, so resizing to `320×320` stretches `H` (`160→320`, 2x) more than `W` (`224→320`, 1.4x) — a real aspect-ratio distortion, currently accepted rather than fixed (an aspect-ratio-preserving resize, e.g. `320×448`, is the alternative if this turns out to matter).

**Dev Notes**
0731
    - ConvexAdam may not work well with the 3d feature map extracted through PlaneCycle
    - DINOv3 is fixed at 16x16 size patch for all model options
    - The "fixed-size" patch may be too coarse for registration task
    - It's more likely an issue with the resolution, as low resolution can't provide enough information in a fixed patch
    
    - Is it the problem in Dataset?
    - OASIS vs NLST
    - Is it preprocessed properly?

0731 (cont'd)
    - OASIS dataset confirmed already preprocessed: identity affine, 1mm isotropic, uniform 160x224x192 shape,
      intensity normalized to [0, ~0.82], skull-stripped (72% zero voxels, zeroed background corners) — this
      isn't a dataset-prep issue on our end, matches the HyperMorph/L2R OASIS release as documented.
    - Compared against DINO-Reg (MICCAI 2024, DINOv2 ViT-L/14) and MedDINOv3 (arXiv 2509.02379) for what
      actually differs from our pipeline:
        - DINO-Reg: two-stage optimization — SSD discrete cost volume for a coarse init, THEN iterative
          Adam/gradient-descent refinement with a Local Cross-Correlation loss. We only have the discrete
          stage (ConvexAdam's correlate()/coupled_convex()); there's no continuous refinement stage.
        - This likely explains the empirical pattern seen in our own sweeps better than resolution alone:
          finer grid_sp made results *worse*, not better, and DSC30 stayed flat (~0.28-0.30) across all 32
          settings regardless of grid_sp/disp_hw — consistent with a discrete-search ceiling, not something
          more hyperparameter tuning fixes.
        - Backbone size also differs: DINO-Reg uses ViT-L/14 (1024-dim), we use DINOv3 ViT-S/16 (384-dim).
    - Takeaway: resolution upsampling (above) addresses the coarse-quantization symptom, but the deeper gap
      is likely the missing continuous refinement stage. Full 19-pair validation of the 320x320 resolution
      change is still pending as of this note.
    