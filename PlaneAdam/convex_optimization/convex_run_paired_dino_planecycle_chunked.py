# EXPERIMENTAL VARIANT of convex_run_paired_dino_640.py (itself a 512x512-resolution PlaneCycle
# run, despite the "640" in its filename - see that file's own comments), adding --plane-chunk-size.
# This file's own resolution has since been pushed to 896x896 (see the resize comment below),
# MedDINOv3's second documented baseline step, now that it's running on an A5000 (24GB) instead
# of the original 12GB card.
#
# Why this exists: convex_run_paired_dino_640.py's PlaneCycle sweep OOMs on ~31/32 settings at
# 512x512 on the original 12GB GPU. The cause isn't attention itself being memory-inefficient (verified:
# PyTorch's scaled_dot_product_attention already picks an efficient kernel here, even in fp32).
# It's that PlaneCycleOp's DW/DH blocks fold the *other* spatial token axis into the batch
# dimension (see planecycle_op.py's own comment: "expand the plane dimension into the batch
# dimension ... to make each slice as a separate batch") - and for DW/DH blocks specifically,
# that batch axis (`P`, e.g. H_tok or W_tok) grows with resolution just like the sequence length
# does. Memory for one block call is roughly `batch(P) x sequence(L) x embed_dim`, and since both
# P and L scale linearly with resolution for these blocks, full-batch memory scales as
# resolution^2 - not because attention itself is quadratic, but because the batch dimension is
# quietly growing right alongside the sequence.
#
# `P` is a genuine, fully independent batch dimension here though - nothing in
# PlaneCycleOp._forward_vit lets attention cross different P-rows, and `rope` doesn't depend on P
# at all (only on the D/perpendicular-axis shapes that make up the sequence). So chunking it -
# running the shared 2D block over sub-batches of P instead of all P rows at once - is
# mathematically identical to the unchunked version (NOT an approximation, not a smaller
# attention window per row: every row still attends over the *full* D-axis and full perpendicular
# axis it always did). It only trades peak memory for more sequential forward-pass calls, exactly
# the trade SLICE_CHUNK_SIZE already makes for Slice2D's D-axis - just applied to PlaneCycle's
# batch axis instead, which is the one axis here safe to chunk without changing any attention
# computation's scope. See planecycle_op.py's plane_chunk_size docstring for the implementation.
#
# New flag: --plane-chunk-size N. Omit (default: unchunked, identical to convex_run_paired_dino_640.py)
# to reproduce the original OOM-prone behavior as a control/baseline.
#
# Also adds --lambda-ncc (ported from convex_run_paired_dino_ncc.py): an optional image-space
# local-NCC term added to the Adam-refinement stage's loss, computed at native (H,W,D) resolution
# rather than the coarse grid_sp_adam-pooled grid the feature term uses - since the raw image
# never suffers the patch_size=16 tokenization the features do. Independent of --plane-chunk-size
# (one affects feature extraction, the other affects the downstream Adam stage), but both are
# controlled by CLI flags here so this file covers "chunked PlaneCycle, optionally with NCC" in
# one place. --lambda-ncc > 0 requires --adam, same as in convex_run_paired_dino_ncc.py.
#
# Also adds --lambda-mind: an optional MIND-SSC term added alongside (not instead of) the DINO
# feature-SSD and NCC terms, using the local MINDSSC implementation (decoder_method/utils/mind.py)
# rather than the DINO backbone at all. Rationale: MIND-SSC was purpose-built for cross-modal
# registration (Heinrich et al., MICCAI 2013) - it encodes each voxel's local self-similarity
# *pattern* rather than raw intensity, so it's structurally modality-invariant by construction,
# unlike DINOv3 (pretrained on natural images, no exposure to any medical modality or cross-modal
# correspondence task) or raw-intensity NCC (which assumes fixed/moving share an intensity
# relationship - true-ish for MR-MR, not for MR-CT). Most useful on datasets like AbdomenMRCT
# where the fixed/moving pair are genuinely different modalities. Computed once per pair at native
# (H,W,D) resolution (same native grid the NCC term uses) and warped like the DINO feature term
# rather than recomputed after warping - MINDSSC's descriptor doesn't need to be regenerated every
# iteration, just resampled, exactly how features_fix_native/features_mov_native are already
# treated. Independent of --lambda-ncc (both can be on at once, added into the same loss with
# their own weights). --lambda-mind > 0 requires --adam, same reasoning as --lambda-ncc.
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


def reduce_channels_pca(features_fix, features_mov, k):
    """
    Fit a PCA basis jointly on a pair's fixed+moving native-grid voxels and project both onto
    it. Fitting jointly (not separately) is required for correctness: correlate()'s SSD distance
    only means something if both sides live in the same reduced basis.
    """
    B, C, Hn, Wn, Dn = features_fix.shape
    flat_fix = features_fix.permute(0, 2, 3, 4, 1).reshape(-1, C)
    flat_mov = features_mov.permute(0, 2, 3, 4, 1).reshape(-1, C)
    X = torch.cat([flat_fix, flat_mov], dim=0).float()
    mean = X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(X - mean, q=k, niter=4)

    def project(flat):
        return ((flat.float() - mean) @ V).to(features_fix.dtype)

    reduced_fix = project(flat_fix).reshape(B, Hn, Wn, Dn, k).permute(0, 4, 1, 2, 3).contiguous()
    reduced_mov = project(flat_mov).reshape(B, Hn, Wn, Dn, k).permute(0, 4, 1, 2, 3).contiguous()
    return reduced_fix, reduced_mov


def adam_refine_displacement(disp_init, features_fix_native, features_mov_native, H, W, D,
                              grid_sp_adam, n_iters, lambda_weight, device,
                              img_fixed=None, img_moving=None, lambda_ncc=0.0, ncc_win=9,
                              lambda_mind=0.0, mind_r=1, mind_d=2):
    """
    ConvexAdam's second stage: Adam-based continuous instance optimization. Ported from the
    original ConvexAdam repo's convex_adam_pt() (convexAdam/src/convexAdam/convex_adam_MIND.py,
    lines ~146-191) — same recipe (B-spline-style triple-box-filter smoothing of a learnable
    displacement parameter, diffusion regularization, direct feature-SSD loss, Adam), adapted to
    our variable-channel DINO features (the original hardcodes `* 12` for MIND's fixed 12
    channels; we use the actual channel count instead).

    Unlike the discrete stage (correlate/coupled_convex), this runs with gradients enabled and
    directly minimizes the warped-vs-fixed feature error, so it can represent any real-valued
    displacement rather than only the discrete correlate() candidate set.

    If lambda_ncc > 0 (img_fixed/img_moving required), an extra `lambda_ncc * NCC(fixed_image,
    warped_moving_image)` term is added, computed at *native* (H, W, D) resolution — deliberately
    finer than the grid_sp_adam-pooled grid the feature term uses, since the whole point is to
    give the optimizer access to detail the patchified/pooled features don't carry.

    If lambda_mind > 0 (img_fixed/img_moving required), an extra `lambda_mind * SSD(MINDSSC(fixed),
    warped MINDSSC(moving))` term is added, also at native resolution. Unlike the NCC term (which
    recomputes on the *warped image* each iteration), MINDSSC descriptors for both images are
    computed once, up front, under no_grad — then the moving side is resampled by the current
    displacement each iteration, exactly like features_fix_native/features_mov_native already are.
    That's deliberate: MIND-SSC is a per-voxel structural descriptor, not a scalar similarity
    metric like NCC, so it fits the same "compute once, warp per-iteration" pattern the DINO
    feature-SSD term uses, rather than NCC's "recompute after warping the raw image" pattern.
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
        # Computed once, not per-iteration - see the docstring above for why this differs from
        # the NCC term's "recompute on the warped image" approach.
        with torch.no_grad():
            mind_fixed_native = MINDSSC(img_fixed_native, radius=mind_r, dilation=mind_d)
            mind_moving_native = MINDSSC(img_moving_native, radius=mind_r, dilation=mind_d)
        mind_ch = mind_fixed_native.shape[1]

    # main() calls this from inside a `with torch.no_grad():` block; without explicitly
    # re-enabling grad here, net[0].weight's ops wouldn't be recorded and .backward() below
    # would either error or silently no-op.
    with torch.enable_grad():
        for _ in range(n_iters):
            optimizer.zero_grad()

            # triple 3x3x3 box-filter smoothing of the raw learnable parameter approximates a
            # B-spline smoothing kernel, keeping the optimized field naturally smooth
            disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1, padding=1).permute(0, 2, 3, 4, 1)

            # diffusion regularizer: penalize voxel-to-voxel displacement changes along each axis
            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

            grid_disp = grid0.view(-1, 3).float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
            patch_mov_sampled = F.grid_sample(patch_features_mov, grid_disp.view((1,) + pooled_size + (3,)), align_corners=False, mode='bilinear')

            sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * n_ch
            loss = sampled_cost.mean()

            if use_ncc or use_mind:
                # Upsample the pooled displacement field to native resolution (same
                # "* grid_sp_adam, then interpolate" convention the function's own return value
                # uses) - shared by both native-resolution terms below, computed once per
                # iteration regardless of how many of them are active.
                disp_native = F.interpolate(disp_sample.permute(0, 4, 1, 2, 3) * grid_sp_adam,
                                             size=(H, W, D), mode='trilinear', align_corners=False)
                warp_grid_native = grid0_native + disp_native.permute(0, 2, 3, 4, 1).flip(-1).div(scale1_native)

            # bf16 autocast: these two terms are the dominant memory cost in this loop (native
            # H,W,D-resolution conv3d for NCC's box filter and MIND-SSC's patch-SSD), and were the
            # source of repeated OOMs even with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on
            # a 24GB A5000 at 896x896/vitb16. Safe here because this is still a full forward+backward
            # pass (gradients flow to net[0].weight normally, same as standard AMP training) - not a
            # no_grad inference shortcut. bf16 shares fp32's exponent range so there's no overflow
            # risk. loss/reg_loss stay fp32; PyTorch's bf16+fp32 promotion rules upcast automatically
            # when these terms are added below.
            if use_ncc:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Warp the raw moving image with it - differentiably, so the NCC term's gradient
                    # reaches net[0].weight the same way the feature term's does.
                    warped_img_native = F.grid_sample(img_moving_native, warp_grid_native, align_corners=False, mode='bilinear')
                    ncc_term = ncc_loss_fn(img_fixed_native, warped_img_native)
                loss = loss + lambda_ncc * ncc_term

            if use_mind:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Warp the precomputed MIND-SSC descriptor map the same way the DINO feature term
                    # warps patch_features_mov - same grid_sample-then-SSD pattern, different features.
                    mind_mov_warped_native = F.grid_sample(mind_moving_native, warp_grid_native, align_corners=False, mode='bilinear')
                    mind_cost = (mind_fixed_native - mind_mov_warped_native).pow(2).mean(1) * mind_ch
                loss = loss + lambda_mind * mind_cost.mean()

            (loss + reg_loss).backward()
            optimizer.step()

    fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
    return F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)


def main(gpunum, configfile, pca_dim=None, adam_niter=None, plane_chunk_size=None,
         lambda_ncc=0.0, ncc_win=9, lambda_mind=0.0, mind_r=1, mind_d=2):

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
    extractor = DinoBackboneExtractor(backbone, plane_chunk_size=plane_chunk_size)
    print(f'using backbone {extractor.arch_name} on {extractor.device}')
    print(f'PCA channel reduction: {pca_dim}' if pca_dim else 'PCA channel reduction: off (using all channels)')
    print(f'Adam refinement: {adam_niter} iterations' if adam_niter else 'Adam refinement: off (discrete stage only)')
    print(f'PlaneCycle plane_chunk_size: {plane_chunk_size}' if plane_chunk_size else
          'PlaneCycle plane_chunk_size: unchunked (original behavior)')
    if lambda_ncc > 0:
        print(f'Image-space NCC term: lambda={lambda_ncc}, window={ncc_win} (native resolution)')
    if lambda_mind > 0:
        print(f'MIND-SSC term: lambda={lambda_mind}, radius={mind_r}, dilation={mind_d} (native resolution)')

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
                    # 16) — 896x896, MedDINOv3's second documented baseline step (after 640x640, before
                    # their 896x896 being the largest resolution they report scaling to), giving a
                    # 56x56 native token grid. D is left untouched (it was never patchified to begin
                    # with). Was previously downgraded to 512x512 to fit a 12GB GPU (see the
                    # module-level comment above), then briefly restored to 640, then pushed to this
                    # 896 step now that we're on an A5000 (24GB) — watch memory here; consider
                    # --plane-chunk-size if this OOMs.
                    # Note: our native H,W (160,224) isn't square like MedDINOv3's CT slices, so this
                    # resize stretches H ~5.6x vs W ~4x — a real aspect-ratio distortion, flagged
                    # rather than silently applied.
                    fixed_volume = F.interpolate(fixed_volume, size=(D, 896, 896), mode='trilinear', align_corners=False)
                    moving_volume = F.interpolate(moving_volume, size=(D, 896, 896), mode='trilinear', align_corners=False)

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
                        # ConvexAdam's second stage: continuous refinement of the discrete stage's
                        # output, using this setting's swept grid_sp_adam/lambda_weight, plus the
                        # native-resolution image NCC term when lambda_ncc > 0.
                        disp_hr = adam_refine_displacement(
                            disp_hr, features_fix_native, features_mov_native, H, W, D,
                            grid_sp_adam=grid_sp_adam, n_iters=adam_niter,
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
    chunk_suffix = f'_planechunk{plane_chunk_size}' if plane_chunk_size else ''
    ncc_suffix = f'_ncc{lambda_ncc:g}w{ncc_win}' if lambda_ncc > 0 else ''
    mind_suffix = f'_mind{lambda_mind:g}r{mind_r}d{mind_d}' if lambda_mind > 0 else ''
    param_suffix = (f'_gridspadam{best_param0}_lambda{best_param1_str}' if adam_niter is not None
                     else f'_gridsp{best_param0}_disphw{best_param1_str}')
    run_name = f'dice{best_dice:.3f}{param_suffix}{pca_suffix}{adam_suffix}{chunk_suffix}{ncc_suffix}{mind_suffix}_planecycle896_{time.strftime("%Y%m%d_%H%M%S")}'
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
                              "refinement stage after the discrete stage. Omit to skip "
                              "(discrete-only, current default behavior).")
    parser.add_argument("--plane-chunk-size", type=int, default=None,
                         help="Sub-batch PlaneCycle's DW/DH blocks over this many plane-rows at a "
                              "time instead of all of them at once (mathematically identical "
                              "output, lower peak memory, more forward-pass calls - see the "
                              "module docstring). Omit to reproduce the original unchunked "
                              "behavior (convex_run_paired_dino_640.py), which OOMs on most "
                              "settings at this resolution.")
    parser.add_argument("--lambda-ncc", type=float, default=0.0,
                         help="Weight of an image-space local-NCC term added to the Adam stage's loss, "
                              "computed at native (H,W,D) resolution. 0 (default) disables it entirely. "
                              "Requires --adam.")
    parser.add_argument("--ncc-win", type=int, default=9,
                         help="Local NCC window size (cubic), only used when --lambda-ncc > 0.")
    parser.add_argument("--lambda-mind", type=float, default=0.0,
                         help="Weight of a MIND-SSC term added to the Adam stage's loss, computed "
                              "at native (H,W,D) resolution using the local MINDSSC descriptor "
                              "(decoder_method/utils/mind.py) instead of the DINO backbone. "
                              "MIND-SSC is structurally modality-invariant by construction, so "
                              "this is aimed at cross-modality datasets (e.g. AbdomenMRCT's "
                              "MR-CT pairs) where raw-intensity NCC and DINO's natural-image "
                              "features both have to bridge a real appearance gap. 0 (default) "
                              "disables it entirely. Can be combined with --lambda-ncc. Requires "
                              "--adam.")
    parser.add_argument("--mind-r", type=int, default=1,
                         help="MIND-SSC patch radius, only used when --lambda-mind > 0. Matches "
                              "the original ConvexAdam repo's default (convex_adam_MIND.py).")
    parser.add_argument("--mind-d", type=int, default=2,
                         help="MIND-SSC dilation, only used when --lambda-mind > 0. Matches the "
                              "original ConvexAdam repo's default (convex_adam_MIND.py).")
    args = parser.parse_args()
    if args.lambda_ncc > 0 and args.adam is None:
        parser.error("--lambda-ncc > 0 requires --adam (it's only wired into the Adam refinement stage)")
    if args.lambda_mind > 0 and args.adam is None:
        parser.error("--lambda-mind > 0 requires --adam (it's only wired into the Adam refinement stage)")
    main(args.gpu, args.configfile, args.pca, args.adam, args.plane_chunk_size,
         args.lambda_ncc, args.ncc_win, args.lambda_mind, args.mind_r, args.mind_d)
