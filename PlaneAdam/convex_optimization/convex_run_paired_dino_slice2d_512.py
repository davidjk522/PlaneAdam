# same as convex_run_paired_dino.py but with extract_feature_slice2d() instead of extract_feature_planecycle() — the PlaneCycle alternative under test, which runs each D slice independently through the ViT with no cross-slice attention at all. This is the 512x512 version (the 320x320 version is convex_run_paired_dino_slice2d.py).
# 512x 512 -> 32 x 32 patch tokens
#
# Also adds --lambda-ncc and --lambda-mind (ported from convex_run_paired_dino_planecycle_chunked.py,
# same implementation, same defaults - kept identical across both so slice2d vs. PlaneCycle stays an
# apples-to-apples comparison of the feature-extraction method, not a comparison of different loss
# terms too). Both are optional extra terms added to the Adam-refinement stage's loss on top of the
# existing DINO feature-SSD term:
#   --lambda-ncc:  image-space local-NCC between fixed and warped-moving, at native (H,W,D)
#                  resolution - finer than the grid_sp_adam-pooled grid the feature term uses.
#   --lambda-mind: MIND-SSC descriptor SSD (decoder_method/utils/mind.py), also native resolution.
#                  MIND-SSC is structurally modality-invariant by construction (built for cross-
#                  modal registration), unlike DINOv3 (natural-image pretraining only) or raw-
#                  intensity NCC - most useful on cross-modality datasets like AbdomenMRCT.
# Both require --adam (only wired into the Adam stage); can be combined with each other.

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
from tqdm.auto import trange

# ConvexAdam Adam-refinement-stage defaults, matching the original repo
# (convexAdam/src/convexAdam/convex_adam_MIND.py: grid_sp_adam=2, lambda_weight=1.25).
GRID_SP_ADAM = 2
LAMBDA_WEIGHT = 1.25


SLICE_CHUNK_SIZE = 32  # of the 192 D-slices processed through the ViT per forward call; lower this
                        # further if 512x512 still OOMs, raise it (up to 192, i.e. unchunked) if
                        # there's headroom to spare — it only trades peak memory for wall-clock time.


def extract_feature_slice2d(backbone, volume, device, chunk_size=SLICE_CHUNK_SIZE):
    """
    Plain per-slice 2D feature extraction — the PlaneCycle alternative under test. Each of the D
    axial slices in `volume` is run through the vanilla ViT completely independently, with no
    cross-slice or cross-plane attention mixing at all (unlike extract_feature_planecycle, which
    cycles blocks across HW/DW/DH planes to build 3D context via self-attention).

    Requires `backbone` to have been constructed fresh with block_type="Slice2D" and NEVER passed
    through PlaneCycleConverter — PlaneCycleConverter.__init__ mutates backbone.blocks in place,
    permanently wrapping each block in PlaneCycleOp, which expects a different calling convention
    incompatible with this plain per-slice forward pass. That's why main() below constructs its own
    separate backbone instance for this variant rather than reusing DinoBackboneExtractor's.

    block_type="Slice2D" is a real, built-in option on DinoVisionTransformer (see
    models/vision_transformer.py: block_type in {"PlaneCycle", "Slice2D", "Flatten3D"}) — when set,
    prepare_tokens_with_masks() flattens (B, D) together as a plain batch dimension and
    forward_features_list() calls each block as blk(x, rope_sincos) (the standard 2D signature),
    so every one of the B*D slices is processed independently through ordinary self-attention
    within its own tokens only.

    Unlike the 320x320 version, this processes D in chunks of `chunk_size` slices at a time instead
    of one (B*D, tokens, C) batch — at 512x512 (1024 tokens/slice, 2.56x the 320x320 case) pushing all
    192 slices through in one batch risks the same OOM that hit 512x512/896x896 earlier this
    session. Peak memory scales with chunk_size x tokens-per-slice^2 (attention), not with the full
    D=192 — chunking trades more forward-pass calls (more wall-clock time) for lower peak memory,
    it does not reduce the total compute. The final concatenated feature tensor itself is small
    (~288MB at 32x32 tokens x 192 x 384 channels x fp32) — chunking only bounds the *transient*
    per-call activation memory, so chunks are accumulated on `device`, not offloaded to CPU.

    Returns (B, D, H_tok, W_tok, C) — the same convention extract_feature_planecycle uses, so
    everything downstream (permute, PCA, resample, correlate, Adam) is unaffected.
    """
    backbone.eval()
    with torch.no_grad():
        B, _C, D, H, W = volume.shape
        H_tok = H // backbone.patch_size
        W_tok = W // backbone.patch_size
        chunks = []
        for start in range(0, D, chunk_size):
            end = min(start + chunk_size, D)
            vol_chunk = volume[:, :, start:end, :, :].to(device)
            patch_tokens = backbone.forward_features(vol_chunk)['x_norm_patchtokens']  # (B*chunk_d, H_tok*W_tok, C)
            C = patch_tokens.shape[-1]
            chunks.append(patch_tokens.view(B, end - start, H_tok, W_tok, C))
        return torch.cat(chunks, dim=1)


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

            if use_ncc:
                # Warp the raw moving image with it - differentiably, so the NCC term's gradient
                # reaches net[0].weight the same way the feature term's does.
                warped_img_native = F.grid_sample(img_moving_native, warp_grid_native, align_corners=False, mode='bilinear')
                loss = loss + lambda_ncc * ncc_loss_fn(img_fixed_native, warped_img_native)

            if use_mind:
                # Warp the precomputed MIND-SSC descriptor map the same way the DINO feature term
                # warps patch_features_mov - same grid_sample-then-SSD pattern, different features.
                mind_mov_warped_native = F.grid_sample(mind_moving_native, warp_grid_native, align_corners=False, mode='bilinear')
                mind_cost = (mind_fixed_native - mind_mov_warped_native).pow(2).mean(1) * mind_ch
                loss = loss + lambda_mind * mind_cost.mean()

            (loss + reg_loss).backward()
            optimizer.step()

    fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
    return F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)


def main(gpunum, configfile, pca_dim=None, adam_niter=None,
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
    # block_type="Slice2D" here, not going through DinoBackboneExtractor/PlaneCycleConverter at
    # all — see extract_feature_slice2d()'s docstring for why this has to be a fresh instance.
    backbone = backbone_fn(pretrained=True, weights=checkpoint_path, block_type="Slice2D")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    print(f'using backbone {config["backbone"]} (block_type=Slice2D) on {device}')
    print(f'PCA channel reduction: {pca_dim}' if pca_dim else 'PCA channel reduction: off (using all channels)')
    print(f'Adam refinement: {adam_niter} iterations' if adam_niter else 'Adam refinement: off (discrete stage only)')
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

                    # get features from the DINOV3 backbone, plain per-slice (no PlaneCycle)
                    fixed_volume = to_volume_tensor(imgs_fixed[i]).unsqueeze(0).to(device)
                    moving_volume = to_volume_tensor(imgs_moving[i]).unsqueeze(0).to(device)

                    # Upsample the in-plane (H,W) resolution before extraction (patch_size unchanged at
                    # 16) — 512x512 (below MedDINOv3's own *baseline* resolution of 640x640, before
                    # their 896x896 step), giving a 32x32 native token grid, 1.6x the 20x20 the
                    # 320x320 version used (2.56x the tokens). D is left untouched (it was never
                    # patchified to begin with). Combined with chunked extraction (see
                    # extract_feature_slice2d/SLICE_CHUNK_SIZE above) to fit in 12GB — the unchunked
                    # 320x320 version doesn't chunk because it didn't need to.
                    # Note: our native H,W (160,224) isn't square like MedDINOv3's CT slices, so this
                    # resize stretches H ~3.2x vs W ~2.3x — a real aspect-ratio distortion, flagged
                    # rather than silently applied.
                    fixed_volume = F.interpolate(fixed_volume, size=(D, 512, 512), mode='trilinear', align_corners=False)
                    moving_volume = F.interpolate(moving_volume, size=(D, 512, 512), mode='trilinear', align_corners=False)

                    # extract_feature_slice2d returns (B, D, H, W, C) on the *native ViT patch grid*
                    # (D at full native resolution, H/W collapsed to patch_size steps) — permute to
                    # (B, C, H, W, D) so the spatial axis order matches nib's (H, W, D) convention.
                    features_fix_native = extract_feature_slice2d(backbone, fixed_volume, device).permute(0, 4, 2, 3, 1).contiguous()
                    features_mov_native = extract_feature_slice2d(backbone, moving_volume, device).permute(0, 4, 2, 3, 1).contiguous()

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
                        # output, using this setting's swept grid_sp_adam/lambda_weight.
                        disp_hr = adam_refine_displacement(
                            disp_hr, features_fix_native, features_mov_native, H, W, D,
                            grid_sp_adam=grid_sp_adam, n_iters=adam_niter,
                            lambda_weight=lambda_weight, device=device,
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
    ncc_suffix = f'_ncc{lambda_ncc:g}w{ncc_win}' if lambda_ncc > 0 else ''
    mind_suffix = f'_mind{lambda_mind:g}r{mind_r}d{mind_d}' if lambda_mind > 0 else ''
    param_suffix = (f'_gridspadam{best_param0}_lambda{best_param1_str}' if adam_niter is not None
                     else f'_gridsp{best_param0}_disphw{best_param1_str}')
    run_name = f'dice{best_dice:.3f}{param_suffix}{pca_suffix}{adam_suffix}{ncc_suffix}{mind_suffix}_slice2d512_{time.strftime("%Y%m%d_%H%M%S")}'
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
                              "MR-CT pairs). 0 (default) disables it entirely. Can be combined "
                              "with --lambda-ncc. Requires --adam.")
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
    main(args.gpu, args.configfile, args.pca, args.adam,
         args.lambda_ncc, args.ncc_win, args.lambda_mind, args.mind_r, args.mind_d)
