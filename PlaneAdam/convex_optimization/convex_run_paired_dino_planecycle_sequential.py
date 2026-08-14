# EXPERIMENTAL VARIANT of convex_run_paired_dino_planecycle_chunked.py, changing how the Adam
# refinement stage combines the DINO feature-SSD term with the local (image-space NCC / MIND-SSC)
# terms. The chunked script sums all active terms into one loss every iteration
# (loss = feature_ssd [+ lambda_ncc*ncc] [+ lambda_mind*mind]) and takes one gradient step against
# that combined objective throughout. This script instead runs the SAME displacement field (same
# net/optimizer state, not restarted) through two back-to-back phases:
#
#   phase 1 ("dino"):  --adam N iterations minimizing feature_ssd + reg_loss only.
#   phase 2 ("local"): --adam-local-niter M iterations continuing from phase 1's displacement,
#                       minimizing (lambda_ncc*ncc + lambda_mind*mind) + reg_loss only - the DINO
#                       feature term is dropped entirely for this phase.
#
# Rationale: with the combined-loss approach, the DINO term and the local term are pulling the
# same gradient step every iteration, so a strong pull from the local term can fight or dilute
# whatever the DINO term found (and vice versa) - the two never get a "clean" turn each. Doing it
# sequentially instead lets DINO's semantic/coarse correspondence establish the field first, then
# lets the local, native-resolution term refine detail the pooled DINO grid can't represent, without
# the DINO term still actively resisting that refinement at the same time. This is a genuinely
# different optimization trajectory, not merely a reweighting - worth comparing against the
# combined-loss version's results, not assumed better.
#
# Phase 2 requires at least one of --lambda-ncc / --lambda-mind > 0 (same requirement the chunked
# script has for the combined loss) and --adam-local-niter > 0; --adam-local-niter is otherwise
# optional (omit it to fall back to phase-1-only, i.e. DINO-only Adam refinement, same as
# convex_run_paired_dino_planecycle_chunked.py's --adam with no --lambda-ncc/--lambda-mind).
#
# Everything else (PlaneCycle chunking, PCA, cycle_order, 896x896 aspect-preserving resize) is
# unchanged from convex_run_paired_dino_planecycle_chunked.py - see that file's own header comments
# for the reasoning behind those.
import argparse
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
import json
import os

# convex_optimization/ has no __init__.py and this script is invoked directly
# (python convex_run_paired_dino.py <gpu> <config>), so the repo root needs to
# be put on sys.path explicitly for the PlaneAdam.*/models.* imports below.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import models.hub.backbones as backbones_module
from convexAdam_hyper_util import (correlate, coupled_convex, dice_coeff,
                                   inverse_consistency, jacobian_determinant_3d,
                                   sort_rank)
from PlaneAdam.Dataset.load_dataset_OASIS import (REPO_ROOT, get_data_train,
                                                   to_volume_tensor)
from PlaneAdam.decoder_method.utils.loss import NccLoss
from PlaneAdam.decoder_method.utils.mind import MINDSSC
from PlaneAdam.feature_extract.dino_extract import DinoBackboneExtractor
from tqdm.auto import trange

# ConvexAdam Adam-refinement-stage defaults, matching the original repo
# (convexAdam/src/convexAdam/convex_adam_MIND.py: grid_sp_adam=2, lambda_weight=1.25).
GRID_SP_ADAM = 2
LAMBDA_WEIGHT = 1.25

DINO_TARGET_LONG_EDGE = 896  # in-plane (H,W) resize budget before DINO feature extraction - see the
# resize comment further down for how this is applied without distorting aspect ratio.
DINO_PATCH_SIZE = 16


def _aspect_preserving_hw(H, W, target_long_edge=DINO_TARGET_LONG_EDGE, patch_size=DINO_PATCH_SIZE):
    """
    Pick a resize target for the in-plane (H,W) axes that scales both axes by the *same* factor
    (preserving native aspect ratio, unlike resizing straight to a fixed square) while keeping the
    longer axis within target_long_edge and both output axes patch_size-aligned for the ViT.
    """
    s = target_long_edge / max(H, W)
    H_out = max(patch_size, round(s * H / patch_size) * patch_size)
    W_out = max(patch_size, round(s * W / patch_size) * patch_size)
    return H_out, W_out


PCA_FIT_MAX_SAMPLES = 100_000  # cap on voxels used to *fit* the PCA basis (see reduce_channels_pca
# docstring) - fitting a k-dim basis doesn't need every native-grid voxel, only enough to estimate
# it reliably. At 896x896/vitb16 the full concatenated fix+mov matrix is Hn*Wn*Dn*2*768*4 bytes ~=
# 3.7GB (the exact OOM size hit repeatedly on a 24GB A5000, independent of --plane-chunk-size or
# NCC/MIND flags - confirmed by matching this formula against the OOM error's byte count), plus
# further transients from mean-subtraction and pca_lowrank's internal SVD workspace on top of that.
# Subsampling the *fit* only (projection below still runs on the full voxel set, so output
# resolution is unaffected) cuts this to a fixed, small cost regardless of resolution/backbone.

def reduce_channels_pca(features_fix, features_mov, k):
    """
    Fit a PCA basis jointly on a pair's fixed+moving native-grid voxels and project both onto
    it. Fitting jointly (not separately) is required for correctness: correlate()'s SSD distance
    only means something if both sides live in the same reduced basis.
    """
    B, C, Hn, Wn, Dn = features_fix.shape
    flat_fix = features_fix.permute(0, 2, 3, 4, 1).reshape(-1, C)
    flat_mov = features_mov.permute(0, 2, 3, 4, 1).reshape(-1, C)
    X_full = torch.cat([flat_fix, flat_mov], dim=0).float()
    if X_full.shape[0] > PCA_FIT_MAX_SAMPLES:
        idx = torch.randperm(X_full.shape[0], device=X_full.device)[:PCA_FIT_MAX_SAMPLES]
        X = X_full[idx]
    else:
        X = X_full
    mean = X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(X - mean, q=k, niter=4)

    def project(flat):
        return ((flat.float() - mean) @ V).to(features_fix.dtype)

    reduced_fix = project(flat_fix).reshape(B, Hn, Wn, Dn, k).permute(0, 4, 1, 2, 3).contiguous()
    reduced_mov = project(flat_mov).reshape(B, Hn, Wn, Dn, k).permute(0, 4, 1, 2, 3).contiguous()
    return reduced_fix, reduced_mov


def adam_refine_displacement_sequential(disp_init, features_fix_native, features_mov_native, H, W, D,
                                         grid_sp_adam, dino_niters, local_niters, lambda_weight, device,
                                         img_fixed=None, img_moving=None, lambda_ncc=0.0, ncc_win=9,
                                         lambda_mind=0.0, mind_r=1, mind_d=2):
    """
    Two-phase variant of adam_refine_displacement (convex_run_paired_dino_planecycle_chunked.py):
    instead of summing the DINO feature-SSD term and the local (NCC/MIND) terms into one loss for
    every iteration, this runs `dino_niters` iterations against the DINO feature-SSD term alone,
    then continues the SAME net/optimizer state (i.e. picks up from wherever phase 1 left the
    displacement field, not a fresh restart) for `local_niters` further iterations against the
    local terms alone (feature_ssd is dropped from the loss for phase 2). See module docstring for
    the rationale.

    Setup below (patch_features_fix/mov pooling, MINDSSC precompute, NCC loss fn, grid machinery)
    is unchanged from adam_refine_displacement - the only difference is which term(s) `loss` is
    built from inside the iteration loop, and that this now runs in two passes instead of one.
    """
    pooled_size = (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam)
    with torch.no_grad():
        patch_features_fix = F.interpolate(features_fix_native, size=pooled_size, mode='trilinear', align_corners=False).float()
        patch_features_mov = F.interpolate(features_mov_native, size=pooled_size, mode='trilinear', align_corners=False).float()
    n_ch = patch_features_fix.shape[1]

    disp_lr = F.interpolate(disp_init, size=pooled_size, mode='trilinear', align_corners=False)

    net = nn.Sequential(nn.Conv3d(3, 1, pooled_size, bias=False))
    net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1)

    grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), (1, 1) + pooled_size, align_corners=False)
    scale = torch.tensor([(pooled_size[0] - 1) / 2, (pooled_size[1] - 1) / 2, (pooled_size[2] - 1) / 2]).to(device).unsqueeze(0)

    use_ncc = lambda_ncc > 0
    use_mind = lambda_mind > 0
    if use_ncc or use_mind:
        assert img_fixed is not None and img_moving is not None, \
            "lambda_ncc > 0 / lambda_mind > 0 requires img_fixed/img_moving"
        img_fixed_native = img_fixed.view(1, 1, H, W, D).float()
        img_moving_native = img_moving.view(1, 1, H, W, D).float()
        # Same (grid0, flip(-1), div(scale)) sampling convention main() already uses to warp the
        # native-resolution segmentation for Dice - reused here so the native-resolution warp is
        # known-correct, not re-derived.
        grid0_native = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), (1, 1, H, W, D), align_corners=False)
        scale1_native = torch.tensor([D - 1, W - 1, H - 1], device=device).float() / 2
    if use_ncc:
        ncc_loss_fn = NccLoss([ncc_win, ncc_win, ncc_win])
    if use_mind:
        # Computed once, not per-iteration - see adam_refine_displacement's docstring for why this
        # differs from the NCC term's "recompute on the warped image" approach.
        with torch.no_grad():
            mind_fixed_native = MINDSSC(img_fixed_native, radius=mind_r, dilation=mind_d)
            mind_moving_native = MINDSSC(img_moving_native, radius=mind_r, dilation=mind_d)
        mind_ch = mind_fixed_native.shape[1]

    # phase: (n_iters, use_dino_term, use_local_terms) - phase 2 only runs if local_niters > 0 and
    # at least one local term is actually enabled (checked by the caller/CLI, but guarded here too
    # so this function is safe to call directly).
    phases = [(dino_niters, True, False)]
    if local_niters > 0 and (use_ncc or use_mind):
        phases.append((local_niters, False, True))

    # main() calls this from inside a `with torch.no_grad():` block; without explicitly
    # re-enabling grad here, net[0].weight's ops wouldn't be recorded and .backward() below
    # would either error or silently no-op.
    with torch.enable_grad():
        for n_iters, use_dino_term, use_local_terms in phases:
            for _ in range(n_iters):
                optimizer.zero_grad()

                # triple 3x3x3 box-filter smoothing of the raw learnable parameter approximates a
                # B-spline smoothing kernel, keeping the optimized field naturally smooth
                disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1, padding=1).permute(0, 2, 3, 4, 1)

                # diffusion regularizer: penalize voxel-to-voxel displacement changes along each axis.
                # Kept active in both phases - it's a smoothness prior on the field itself, not tied
                # to either similarity term, so there's no reason to drop it in phase 2.
                reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                           lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                           lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

                grid_disp = grid0.view(-1, 3).float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

                loss = torch.zeros((), device=device)

                if use_dino_term:
                    patch_mov_sampled = F.grid_sample(patch_features_mov, grid_disp.view((1,) + pooled_size + (3,)), align_corners=False, mode='bilinear')
                    sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * n_ch
                    loss = loss + sampled_cost.mean()

                if use_local_terms:
                    # Upsample the pooled displacement field to native resolution (same
                    # "* grid_sp_adam, then interpolate" convention the function's own return value
                    # uses) - shared by both native-resolution terms below.
                    disp_native = F.interpolate(disp_sample.permute(0, 4, 1, 2, 3) * grid_sp_adam,
                                                 size=(H, W, D), mode='trilinear', align_corners=False)
                    warp_grid_native = grid0_native + disp_native.permute(0, 2, 3, 4, 1).flip(-1).div(scale1_native)

                    # NOTE: previously wrapped in bf16 autocast to save memory, but reverted - NccLoss
                    # computes cc = cross^2 / (I_var*J_var + 1e-5), a division by a potentially-small
                    # variance term, which is numerically unstable under bf16's coarser mantissa and
                    # was observed to blow up Adam's optimization (see adam_refine_displacement's own
                    # comment) - kept fp32 for correctness here as well.
                    if use_ncc:
                        # Warp the raw moving image with it - differentiably, so the NCC term's
                        # gradient reaches net[0].weight the same way the feature term's does.
                        warped_img_native = F.grid_sample(img_moving_native, warp_grid_native, align_corners=False, mode='bilinear')
                        loss = loss + lambda_ncc * ncc_loss_fn(img_fixed_native, warped_img_native)

                    if use_mind:
                        # Warp the precomputed MIND-SSC descriptor map the same way the DINO feature
                        # term warps patch_features_mov - same grid_sample-then-SSD pattern, different
                        # features.
                        mind_mov_warped_native = F.grid_sample(mind_moving_native, warp_grid_native, align_corners=False, mode='bilinear')
                        mind_cost = (mind_fixed_native - mind_mov_warped_native).pow(2).mean(1) * mind_ch
                        loss = loss + lambda_mind * mind_cost.mean()

                (loss + reg_loss).backward()
                optimizer.step()

    fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
    return F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)


def main(gpunum, configfile, pca_dim=None, adam_niter=None, adam_local_niter=0, plane_chunk_size=None,
         lambda_ncc=0.0, ncc_win=9, lambda_mind=0.0, mind_r=1, mind_d=2,
         cycle_order=("HW", "DW", "DH", "HW")):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str((gpunum))
    print(torch.cuda.get_device_name())

    with open(configfile, 'r') as f:
        config = json.load(f)
    pairs = config['pairs']
    num_labels = config['num_labels'] - 1

    print('using', len(pairs), 'registration pairs')
    imgs_fixed, segs_fixed, masks_fixed, imgs_moving, segs_moving, masks_moving = get_data_train(pairs, config['HWD'])

    backbone_fn = getattr(backbones_module, config['backbone'])
    checkpoint_path = os.path.join(REPO_ROOT, config['checkpoint'])
    backbone = backbone_fn(pretrained=True, weights=checkpoint_path)
    extractor = DinoBackboneExtractor(backbone, plane_chunk_size=plane_chunk_size, cycle_order=cycle_order)
    print(f'using backbone {extractor.arch_name} on {extractor.device}')
    print(f'PCA channel reduction: {pca_dim}' if pca_dim else 'PCA channel reduction: off (using all channels)')
    print(f'Adam refinement: {adam_niter} DINO-phase iterations' if adam_niter else 'Adam refinement: off (discrete stage only)')
    if adam_local_niter > 0:
        print(f'Adam refinement: +{adam_local_niter} local-phase iterations (DINO-only, then local-only, sequential)')
    print(f'PlaneCycle plane_chunk_size: {plane_chunk_size}' if plane_chunk_size else
          'PlaneCycle plane_chunk_size: unchunked (original behavior)')
    print(f'PlaneCycle cycle_order: {"->".join(cycle_order)}')
    if lambda_ncc > 0:
        print(f'Image-space NCC term: lambda={lambda_ncc}, window={ncc_win} (native resolution, local phase only)')
    if lambda_mind > 0:
        print(f'MIND-SSC term: lambda={lambda_mind}, radius={mind_r}, dilation={mind_d} (native resolution, local phase only)')

    robust30 = []
    for i in range(len(pairs)):
        dice0 = dice_coeff(segs_fixed[i].cuda(), segs_moving[i].cuda(), num_labels + 1)
        robust30.append(dice0.topk(max(1, int(config['num_labels'] * .3)), largest=False).indices)

    # Once Adam refinement runs, it converges to nearly the same result regardless of which
    # grid_sp/disp_hw the discrete stage started it from (confirmed empirically: a full sweep
    # with --adam active came back with every one of 32 settings within 0.002 of each other) — so
    # with --adam active, fix the discrete stage at a cheap, representative setting and sweep the
    # Adam stage's own hyperparameters (grid_sp_adam, lambda_weight) instead, since those are what
    # actually still moves the result. Without --adam, keep sweeping grid_sp/disp_hw as before
    # (full-image-resolution range reused from the original MIND-based pipeline).
    FIXED_GRID_SP, FIXED_DISP_HW = 6, 4
    if adam_niter is not None:
        param0_name, param1_name = 'grid_sp_adam', 'lambda_weight'
        combos = [(g, l) for g in (2, 3, 4) for l in (0.5, 1.25, 2.5, 5.0)]
    else:
        param0_name, param1_name = 'grid_sp', 'disp_hw'
        combos = []
        for grid_sp in range(2, 7):
            max_disp_hw = 5 if grid_sp == 2 else 8
            for disp_hw in range(2, max_disp_hw + 1):
                combos.append((grid_sp, disp_hw))
    settings = torch.tensor(combos, dtype=torch.float32)
    num_settings = len(combos)

    print(f'{num_settings} ({param0_name}, {param1_name}) settings:')
    print(settings.min(0).values, settings.max(0).values)

    t_mind = torch.zeros(num_settings)
    t_convex = torch.zeros(num_settings)
    dice = torch.zeros(num_settings, 2)
    jstd = torch.zeros(num_settings, 2)
    dice_max = 0
    for s in range(num_settings):
        if adam_niter is not None:
            grid_sp, disp_hw = FIXED_GRID_SP, FIXED_DISP_HW
            grid_sp_adam, lambda_weight = int(settings[s, 0]), settings[s, 1].item()
            setting_desc = (f'grid_sp_adam={grid_sp_adam}, lambda_weight={lambda_weight:.2f} '
                             f'(discrete stage fixed at grid_sp={grid_sp}, disp_hw={disp_hw})')
        else:
            grid_sp, disp_hw = int(settings[s, 0]), int(settings[s, 1])
            grid_sp_adam, lambda_weight = GRID_SP_ADAM, LAMBDA_WEIGHT
            setting_desc = f'grid_sp={grid_sp}, disp_hw={disp_hw}'

        print('starting full run ', s, ' out of', num_settings)
        print('setting', setting_desc)
        try:
            for i in trange(len(pairs)):

                t0 = time.time()

                img_fixed = imgs_fixed[i].cuda()
                seg_fixed = segs_fixed[i].cuda()

                img_moving = imgs_moving[i].cuda()
                seg_moving = segs_moving[i].cuda()

                H, W, D = img_moving.shape
                grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, H, W, D), align_corners=False)
                torch.cuda.synchronize()
                t0 = time.time()

                with torch.no_grad():

                    # get features from the DINOV3 backbone, using the PlaneCycle feature extractor
                    fixed_volume = to_volume_tensor(imgs_fixed[i]).unsqueeze(0).to(extractor.device)
                    moving_volume = to_volume_tensor(imgs_moving[i]).unsqueeze(0).to(extractor.device)

                    # Upsample the in-plane (H,W) resolution before extraction (patch_size unchanged at
                    # 16) — targeting a long-edge budget of 896 (MedDINOv3's second documented
                    # baseline step) — giving up to a 56-token long edge. D is left untouched (it was
                    # never patchified to begin with). Was previously downgraded to 512x512 to fit a
                    # 12GB GPU (see convex_run_paired_dino_planecycle_chunked.py's module-level
                    # comment), then briefly restored to 640, then pushed to this 896 budget now that
                    # we're on an A5000 (24GB) — watch memory here; consider --plane-chunk-size if this
                    # OOMs.
                    #
                    # One uniform scale factor (pinned to the longer axis, W, so the 896 budget is
                    # still respected) is applied to both axes, each then rounded to a patch_size=16
                    # multiple so the ViT still tokenizes cleanly - preserves native aspect ratio
                    # instead of forcing a distorting 896x896 square.
                    H_resized, W_resized = _aspect_preserving_hw(H, W)
                    fixed_volume = F.interpolate(fixed_volume, size=(D, H_resized, W_resized), mode='trilinear', align_corners=False)
                    moving_volume = F.interpolate(moving_volume, size=(D, H_resized, W_resized), mode='trilinear', align_corners=False)

                    # extract_feature_planecycle returns (B, D, H, W, C) on the *native ViT patch grid*
                    # (D at full native resolution, H/W collapsed to patch_size steps) — permute to
                    # (B, C, H, W, D) so the spatial axis order matches nib's (H, W, D) convention.
                    features_fix_native = extractor.extract_feature_planecycle(fixed_volume).permute(0, 4, 2, 3, 1).contiguous()
                    features_mov_native = extractor.extract_feature_planecycle(moving_volume).permute(0, 4, 2, 3, 1).contiguous()

                    if pca_dim is not None:
                        features_fix_native, features_mov_native = reduce_channels_pca(features_fix_native, features_mov_native, pca_dim)

                    # Resample straight from the native ViT grid to the *pooled full-image* resolution
                    # (H//grid_sp, W//grid_sp, D//grid_sp) in one step — this both undoes the native
                    # grid's anisotropy (H,W patchified 16x by the ViT, D untouched, so registering
                    # directly on the native grid made every H/W displacement step 16x coarser than a
                    # D step, collapsing in-plane displacement to zero) and avoids ever materializing
                    # the full (H,W,D)-resolution feature tensor (~10GB/tensor in fp32), which doesn't
                    # fit in GPU memory. One grid_sp cell now means grid_sp real voxels on every axis.
                    pooled_size = (H // grid_sp, W // grid_sp, D // grid_sp)
                    features_fix_smooth = F.interpolate(features_fix_native, size=pooled_size, mode='trilinear', align_corners=False)
                    features_mov_smooth = F.interpolate(features_mov_native, size=pooled_size, mode='trilinear', align_corners=False)
                    if adam_niter is None:
                        # the Adam refinement stage (below) needs the pre-pooled native features to
                        # re-pool at its own, finer grid_sp_adam resolution — keep them if it's active
                        del features_fix_native, features_mov_native

                    n_ch = features_fix_smooth.shape[1]
                    t1 = time.time()
                    torch.cuda.empty_cache()
                    ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)
                    # use correalation to get a soft displacement field -> convexAdam_hyper_util.py:148-174
                    disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0), (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1), align_corners=True).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)

                    disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

                    scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1, 1).cuda().half() / 2
                    torch.cuda.empty_cache()
                    # calculate the inverse consistency of the displacement field -> convexAdam_hyper_util.py:176-190
                    # ssd vs ssd_ / ssd_argmin vs ssd_argmin_ are the two directions of the correlation
                    ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

                    # disp_soft vs disp_soft_ are the two directions of the displacement field
                    disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
                    disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

                    disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear', align_corners=False)

                    if adam_niter is not None:
                        # Two-phase refinement: DINO-only iterations first, then (if enabled)
                        # local-only iterations continuing from that displacement - see the module
                        # docstring for why this differs from the chunked script's combined-loss
                        # approach.
                        disp_hr = adam_refine_displacement_sequential(
                            disp_hr, features_fix_native, features_mov_native, H, W, D,
                            grid_sp_adam=grid_sp_adam, dino_niters=adam_niter, local_niters=adam_local_niter,
                            lambda_weight=lambda_weight, device=extractor.device,
                            img_fixed=img_fixed, img_moving=img_moving,
                            lambda_ncc=lambda_ncc, ncc_win=ncc_win,
                            lambda_mind=lambda_mind, mind_r=mind_r, mind_d=mind_d,
                        )

                    t2 = time.time()

                    scale1 = torch.tensor([D - 1, W - 1, H - 1]).cuda() / 2
                    jac_det = jacobian_determinant_3d(disp_hr.float(), False)
                    torch.cuda.empty_cache()

                seg_warped = F.grid_sample(seg_moving.view(1, 1, H, W, D), grid0 + disp_hr.permute(0, 2, 3, 4, 1).flip(-1).div(scale1), mode='nearest').squeeze()
                DICE1 = dice_coeff(seg_fixed, seg_warped, num_labels + 1)

                t_mind[s] += t1 - t0
                t_convex[s] += t2 - t1
                dice[s, 0] += 1 / len(pairs) * DICE1.mean()
                dice[s, 1] += 1 / len(pairs) * DICE1[robust30[i]].mean()
                jac_det_log = jac_det.add(3).clamp_(0.000000001, 1000000000).log()
                jstd[s, 0] += 1 / len(pairs) * (jac_det_log).std().cpu()
                jstd[s, 1] += 1 / len(pairs) * ((jac_det < 0).float().mean()).cpu()
        except torch.cuda.OutOfMemoryError:
            print(f'setting {s} ({setting_desc}) ran out of GPU memory - skipping')
            torch.cuda.empty_cache()
            dice[s, :] = 0.0
            jstd[s, :] = 1e6
            continue

        # crash-recovery checkpoint: overwritten after every setting so a killed/interrupted
        # sweep still leaves usable partial results at the plain path from the config.
        torch.save([dice, jstd, t_convex], config['output'])

        if (dice[s, 0] > dice_max):
            print(f'  new best so far: setting {s} ({setting_desc}) '
                  f'dice={dice[s, 0].item():.3f} (robust={dice[s, 1].item():.3f}) jstd={jstd[s, 0].item():.4f}')
            dice_max = dice[s, 0]

    rank1 = sort_rank(-dice[:, 0])
    rank1 *= sort_rank(-dice[:, 1])
    rank1 *= sort_rank(jstd[:, 0])
    rank1 = rank1.pow(1 / 3)

    best_idx = int(rank1.argmax())
    best_param0 = int(settings[best_idx, 0])
    best_param1 = settings[best_idx, 1].item()
    best_param1_str = f'{best_param1:.2f}' if adam_niter is not None else str(int(best_param1))
    best_dice, best_dice_robust = dice[best_idx, 0].item(), dice[best_idx, 1].item()
    best_jstd, best_foldover = jstd[best_idx, 0].item(), jstd[best_idx, 1].item()
    best_time = t_convex[best_idx].item()

    # Full per-setting comparison so the winner is auditable, not just asserted.
    table_header = f'{"#":>3}  {param0_name:>11}  {param1_name:>11}  {"dice":>6}  {"robust":>6}  {"jstd":>7}  {"time(s)":>8}'
    table_rows = [table_header, '-' * len(table_header)]
    for s in range(num_settings):
        marker = '*' if s == best_idx else ' '
        val1_str = f'{settings[s, 1].item():>11.2f}' if adam_niter is not None else f'{int(settings[s, 1]):>11}'
        table_rows.append(
            f'{marker}{s:>2}  {int(settings[s, 0]):>11}  {val1_str}  '
            f'{dice[s, 0].item():>6.3f}  {dice[s, 1].item():>6.3f}  {jstd[s, 0].item():>7.4f}  {t_convex[s].item():>8.2f}'
        )
    table_str = '\n'.join(table_rows)

    summary_lines = [
        f'All {num_settings} settings (* = best):',
        table_str,
        '',
        f'Best setting: #{best_idx} of {num_settings} ({param0_name}={best_param0}, {param1_name}={best_param1_str})',
        f'Dice (overall):        {best_dice:.3f}',
        f'Dice (robust-30):      {best_dice_robust:.3f}',
        f'Jacobian std:          {best_jstd:.4f}',
        f'Foldover fraction:     {best_foldover:.4f}',
        f'Compute time (s/pair): {best_time:.2f}',
    ]
    print('\n===== Sweep complete =====')
    print('\n'.join(summary_lines))

    # Final results go in their own folder named after the winning setting, so re-running the
    # sweep (e.g. with different data or a different backbone) never clobbers a previous run's
    # results and the folder name alone tells you how the run went.
    pca_suffix = f'_pca{pca_dim}' if pca_dim else ''
    adam_suffix = f'_adam{adam_niter}' if adam_niter else ''
    local_suffix = f'+{adam_local_niter}local' if adam_local_niter else ''
    chunk_suffix = f'_planechunk{plane_chunk_size}' if plane_chunk_size else ''
    cycle_suffix = (f'_cycle{"".join(cycle_order)}'
                     if cycle_order != ("HW", "DW", "DH", "HW") else '')
    ncc_suffix = f'_ncc{lambda_ncc:g}w{ncc_win}' if lambda_ncc > 0 else ''
    mind_suffix = f'_mind{lambda_mind:g}r{mind_r}d{mind_d}' if lambda_mind > 0 else ''
    param_suffix = (f'_gridspadam{best_param0}_lambda{best_param1_str}' if adam_niter is not None
                     else f'_gridsp{best_param0}_disphw{best_param1_str}')
    run_name = f'dice{best_dice:.3f}{param_suffix}{pca_suffix}{adam_suffix}{local_suffix}{chunk_suffix}{cycle_suffix}{ncc_suffix}{mind_suffix}_planecycle896_sequential_{time.strftime("%Y%m%d_%H%M%S")}'
    output_dir = os.path.join(os.path.dirname(config['output']) or '.', run_name)
    os.makedirs(output_dir, exist_ok=True)

    torch.save([rank1, dice, jstd, t_convex, settings], os.path.join(output_dir, 'convex_search_results.pth'))
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')

    print(f'\nResults written to {output_dir}/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    parser.add_argument("configfile", type=str)
    parser.add_argument("--pca", type=int, default=None,
                         help="Reduce DINO feature channels to this many dims via PCA before "
                              "correlate(). Omit to use the full channel count (no reduction).")
    parser.add_argument("--adam", type=int, default=None,
                         help="Run N iterations of ConvexAdam's Adam instance-optimization "
                              "refinement stage (DINO feature-SSD phase) after the discrete stage. "
                              "Omit to skip Adam refinement entirely (discrete-only). Required if "
                              "--adam-local-niter is given.")
    parser.add_argument("--adam-local-niter", type=int, default=0,
                         help="Run this many further Adam iterations AFTER the --adam DINO-phase "
                              "iterations finish, continuing the same displacement field but now "
                              "minimizing only the local (NCC/MIND) terms - the DINO feature-SSD "
                              "term is dropped for this phase. 0 (default) skips this phase "
                              "entirely (DINO-only Adam refinement). Requires --adam and at least "
                              "one of --lambda-ncc / --lambda-mind > 0.")
    parser.add_argument("--plane-chunk-size", type=int, default=None,
                         help="Sub-batch PlaneCycle's DW/DH blocks over this many plane-rows at a "
                              "time instead of all of them at once (mathematically identical "
                              "output, lower peak memory, more forward-pass calls). Omit to "
                              "reproduce the original unchunked behavior, which OOMs on most "
                              "settings at this resolution.")
    parser.add_argument("--lambda-ncc", type=float, default=0.0,
                         help="Weight of an image-space local-NCC term used in the local phase "
                              "(--adam-local-niter), computed at native (H,W,D) resolution. 0 "
                              "(default) disables it entirely.")
    parser.add_argument("--ncc-win", type=int, default=9,
                         help="Local NCC window size (cubic), only used when --lambda-ncc > 0.")
    parser.add_argument("--lambda-mind", type=float, default=0.0,
                         help="Weight of a MIND-SSC term used in the local phase "
                              "(--adam-local-niter), computed at native (H,W,D) resolution using "
                              "the local MINDSSC descriptor (decoder_method/utils/mind.py) instead "
                              "of the DINO backbone. MIND-SSC is structurally modality-invariant by "
                              "construction, so this is aimed at cross-modality datasets (e.g. "
                              "AbdomenMRCT's MR-CT pairs). 0 (default) disables it entirely. Can be "
                              "combined with --lambda-ncc.")
    parser.add_argument("--mind-r", type=int, default=1,
                         help="MIND-SSC patch radius, only used when --lambda-mind > 0. Matches "
                              "the original ConvexAdam repo's default (convex_adam_MIND.py).")
    parser.add_argument("--mind-d", type=int, default=2,
                         help="MIND-SSC dilation, only used when --lambda-mind > 0. Matches the "
                              "original ConvexAdam repo's default (convex_adam_MIND.py).")
    parser.add_argument("--cycle-order", type=str, default="HW,DW,DH,HW",
                         help="Comma-separated sequence of planes ('HW','DW','DH') that "
                              "PlaneCycleOp cycles round-robin across backbone blocks "
                              "(plane assigned to block i is cycle_order[i %% len(cycle_order)]). "
                              "Default 'HW,DW,DH,HW' matches the original PlaneCycle behavior.")
    args = parser.parse_args()
    if args.adam_local_niter > 0 and args.adam is None:
        parser.error("--adam-local-niter > 0 requires --adam (it's the phase that runs after the "
                      "DINO-phase Adam iterations)")
    if args.adam_local_niter > 0 and args.lambda_ncc <= 0 and args.lambda_mind <= 0:
        parser.error("--adam-local-niter > 0 requires at least one of --lambda-ncc / --lambda-mind "
                      "> 0 (otherwise the local phase's loss would be empty)")
    if args.lambda_ncc > 0 and args.adam is None:
        parser.error("--lambda-ncc > 0 requires --adam")
    if args.lambda_mind > 0 and args.adam is None:
        parser.error("--lambda-mind > 0 requires --adam")
    cycle_order = tuple(p.strip() for p in args.cycle_order.split(","))
    if not cycle_order or any(p not in ("HW", "DW", "DH") for p in cycle_order):
        parser.error(f"--cycle-order must be a comma-separated list of 'HW'/'DW'/'DH', got {args.cycle_order!r}")
    main(args.gpu, args.configfile, args.pca, args.adam, args.adam_local_niter, args.plane_chunk_size,
         args.lambda_ncc, args.ncc_win, args.lambda_mind, args.mind_r, args.mind_d,
         cycle_order)
