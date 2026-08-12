# DINO-ABLATION CONTROL: structurally identical to convex_run_paired_dino_planecycle_chunked.py
# (same discrete stage, same Adam-refinement recipe, same --lambda-ncc aux term, same output/
# sweep format) except the DINO backbone is never loaded and never touches the pipeline. The
# feature that feeds BOTH the discrete correlate() stage AND the Adam stage's primary loss term
# is classic MIND-SSC (Heinrich et al., MICCAI 2013 - decoder_method/utils/mind.py::MINDSSC()),
# computed once per pair directly on the native image, exactly like the original (pre-DINO)
# ConvexAdam repo's default configuration.
#
# Why this file exists: every convex_run_paired_dino*.py script in this directory always has
# DINO in the loop somewhere - MIND has so far only ever been added as a native-resolution
# *auxiliary* term (--lambda-mind) on top of DINO features, never as DINO's replacement. That
# means no run in OASIS_output/ can answer "are the DINO features actually contributing
# anything, or is classic MIND-SSC alone doing the real work?" This script is the one variable
# swap needed to answer that: same sweep, same Adam recipe, same aux-loss machinery, DINO
# subtracted out entirely.
#
# No --pca flag: MIND-SSC is a fixed 12-channel descriptor (self-similarity across 6 neighbor
# offsets, order picked pairwise), already far smaller than DINO's raw channel count - reducing
# a 12-dim descriptor further via PCA isn't the same kind of lever it was for DINO's much wider
# feature space, and the original ConvexAdam repo never PCA-reduces MIND either.
#
# No --plane-chunk-size: that flag existed purely to fix a DINOv3 PlaneCycle-specific memory
# blowup (see convex_run_paired_dino_planecycle_chunked.py's module docstring) - MIND-SSC's
# conv3d-based descriptor computation has no such batch-dimension blowup, nothing to chunk.
#
# No --lambda-mind: MIND is now the primary feature everywhere, so a second "MIND auxiliary
# term" would just be comparing the same descriptor to itself. --lambda-ncc is kept, unchanged.
# from convex_run_paired_dino_ncc.py's recipe, since NCC is a genuinely different signal (raw
# native-image intensity correlation) and stacking classic-MIND-primary + NCC-aux is a fair
# question in its own right.
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
# (python convex_run_paired_mind_only.py <gpu> <config>), so the repo root needs to
# be put on sys.path explicitly for the PlaneAdam.*/models.* imports below.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from convexAdam_hyper_util import (correlate, coupled_convex, dice_coeff,
                                   inverse_consistency, jacobian_determinant_3d,
                                   sort_rank)
from PlaneAdam.Dataset.load_dataset_OASIS import REPO_ROOT, get_data_train
from PlaneAdam.decoder_method.utils.loss import NccLoss
from PlaneAdam.decoder_method.utils.mind import MINDSSC
from tqdm.auto import trange

# ConvexAdam Adam-refinement-stage defaults, matching the original repo
# (convexAdam/src/convexAdam/convex_adam_MIND.py: grid_sp_adam=2, lambda_weight=1.25).
GRID_SP_ADAM = 2
LAMBDA_WEIGHT = 1.25


def adam_refine_displacement(disp_init, features_fix_native, features_mov_native, H, W, D,
                              grid_sp_adam, n_iters, lambda_weight, device,
                              img_fixed=None, img_moving=None, lambda_ncc=0.0, ncc_win=9):
    """
    ConvexAdam's second stage: Adam-based continuous instance optimization. Identical recipe to
    convex_run_paired_dino_planecycle_chunked.py's function of the same name - see that file for
    the full derivation - with `features_fix_native`/`features_mov_native` now MIND-SSC (12
    channels, fixed) instead of DINO features (variable channel count via --pca).
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
    if use_ncc:
        assert img_fixed is not None and img_moving is not None, "lambda_ncc > 0 requires img_fixed/img_moving"
        ncc_loss_fn = NccLoss([ncc_win, ncc_win, ncc_win])
        img_fixed_native = img_fixed.view(1, 1, H, W, D).float()
        img_moving_native = img_moving.view(1, 1, H, W, D).float()
        grid0_native = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), (1, 1, H, W, D), align_corners=False)
        scale1_native = torch.tensor([D - 1, W - 1, H - 1], device=device).float() / 2

    with torch.enable_grad():
        for _ in range(n_iters):
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1, padding=1).permute(0, 2, 3, 4, 1)

            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

            grid_disp = grid0.view(-1, 3).float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
            patch_mov_sampled = F.grid_sample(patch_features_mov, grid_disp.view((1,) + pooled_size + (3,)), align_corners=False, mode='bilinear')

            sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * n_ch
            loss = sampled_cost.mean()

            if use_ncc:
                disp_native = F.interpolate(disp_sample.permute(0, 4, 1, 2, 3) * grid_sp_adam,
                                             size=(H, W, D), mode='trilinear', align_corners=False)
                warp_grid_native = grid0_native + disp_native.permute(0, 2, 3, 4, 1).flip(-1).div(scale1_native)
                warped_img_native = F.grid_sample(img_moving_native, warp_grid_native, align_corners=False, mode='bilinear')
                loss = loss + lambda_ncc * ncc_loss_fn(img_fixed_native, warped_img_native)

            (loss + reg_loss).backward()
            optimizer.step()

    fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
    return F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)


def main(gpunum, configfile, adam_niter=None, lambda_ncc=0.0, ncc_win=9, mind_r=2, mind_d=2):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str((gpunum))
    print(torch.cuda.get_device_name())
    device = torch.device('cuda')

    with open(configfile, 'r') as f:
        config = json.load(f)
    pairs = config['pairs']
    num_labels = config['num_labels'] - 1

    print('using', len(pairs), 'registration pairs')
    imgs_fixed, segs_fixed, masks_fixed, imgs_moving, segs_moving, masks_moving = get_data_train(pairs, config['HWD'])

    print('feature extractor: MIND-SSC only (no DINO backbone) - DINO-ablation control run')
    print(f'MIND-SSC radius={mind_r}, dilation={mind_d}')
    print(f'Adam refinement: {adam_niter} iterations' if adam_niter else 'Adam refinement: off (discrete stage only)')
    if lambda_ncc > 0:
        print(f'Image-space NCC term: lambda={lambda_ncc}, window={ncc_win} (native resolution)')

    robust30 = []
    for i in range(len(pairs)):
        dice0 = dice_coeff(segs_fixed[i].cuda(), segs_moving[i].cuda(), num_labels + 1)
        robust30.append(dice0.topk(max(1, int(config['num_labels'] * .3)), largest=False).indices)

    # Same rationale as convex_run_paired_dino_planecycle_chunked.py: once Adam refinement runs,
    # the discrete stage's starting point barely matters, so with --adam active fix it at a
    # cheap, representative setting and sweep the Adam stage's own hyperparameters instead.
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

    t_extract = torch.zeros(num_settings)
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

                img_fixed = imgs_fixed[i].cuda()
                seg_fixed = segs_fixed[i].cuda()

                img_moving = imgs_moving[i].cuda()
                seg_moving = segs_moving[i].cuda()

                H, W, D = img_moving.shape
                grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, H, W, D), align_corners=False)
                torch.cuda.synchronize()
                t0 = time.time()

                with torch.no_grad():
                    # MIND-SSC computed once, directly on the native-resolution image - no
                    # backbone, no upsampling/aspect-ratio distortion, no PCA. This IS the
                    # feature that both the discrete stage and the Adam stage's primary loss
                    # term use, unlike every convex_run_paired_dino*.py script where this same
                    # descriptor (when present at all) only ever supplements DINO features.
                    features_fix_native = MINDSSC(img_fixed.view(1, 1, H, W, D), radius=mind_r, dilation=mind_d)
                    features_mov_native = MINDSSC(img_moving.view(1, 1, H, W, D), radius=mind_r, dilation=mind_d)

                    pooled_size = (H // grid_sp, W // grid_sp, D // grid_sp)
                    features_fix_smooth = F.interpolate(features_fix_native, size=pooled_size, mode='trilinear', align_corners=False)
                    features_mov_smooth = F.interpolate(features_mov_native, size=pooled_size, mode='trilinear', align_corners=False)
                    if adam_niter is None:
                        del features_fix_native, features_mov_native

                    n_ch = features_fix_smooth.shape[1]
                    t1 = time.time()
                    torch.cuda.empty_cache()
                    ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)
                    disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0), (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1), align_corners=True).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)

                    disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

                    scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1, 1).cuda().half() / 2
                    torch.cuda.empty_cache()
                    ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

                    disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
                    disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

                    disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear', align_corners=False)

                    if adam_niter is not None:
                        disp_hr = adam_refine_displacement(
                            disp_hr, features_fix_native, features_mov_native, H, W, D,
                            grid_sp_adam=grid_sp_adam, n_iters=adam_niter,
                            lambda_weight=lambda_weight, device=device,
                            img_fixed=img_fixed, img_moving=img_moving,
                            lambda_ncc=lambda_ncc, ncc_win=ncc_win,
                        )

                    t2 = time.time()

                    scale1 = torch.tensor([D - 1, W - 1, H - 1]).cuda() / 2
                    jac_det = jacobian_determinant_3d(disp_hr.float(), False)
                    torch.cuda.empty_cache()

                seg_warped = F.grid_sample(seg_moving.view(1, 1, H, W, D), grid0 + disp_hr.permute(0, 2, 3, 4, 1).flip(-1).div(scale1), mode='nearest').squeeze()
                DICE1 = dice_coeff(seg_fixed, seg_warped, num_labels + 1)

                t_extract[s] += t1 - t0
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
        'DINO-ABLATION CONTROL: MIND-SSC only, no DINO backbone at any stage.',
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

    adam_suffix = f'_adam{adam_niter}' if adam_niter else ''
    ncc_suffix = f'_ncc{lambda_ncc:g}w{ncc_win}' if lambda_ncc > 0 else ''
    param_suffix = (f'_gridspadam{best_param0}_lambda{best_param1_str}' if adam_niter is not None
                     else f'_gridsp{best_param0}_disphw{best_param1_str}')
    run_name = f'dice{best_dice:.3f}{param_suffix}{adam_suffix}{ncc_suffix}_mindonly_r{mind_r}d{mind_d}_{time.strftime("%Y%m%d_%H%M%S")}'
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
    parser.add_argument("--adam", type=int, default=None,
                         help="Run N iterations of ConvexAdam's Adam instance-optimization "
                              "refinement stage after the discrete stage. Omit to skip "
                              "(discrete-only).")
    parser.add_argument("--lambda-ncc", type=float, default=0.0,
                         help="Weight of an image-space local-NCC term added to the Adam stage's loss, "
                              "computed at native (H,W,D) resolution. 0 (default) disables it entirely. "
                              "Requires --adam.")
    parser.add_argument("--ncc-win", type=int, default=9,
                         help="Local NCC window size (cubic), only used when --lambda-ncc > 0.")
    parser.add_argument("--mind-r", type=int, default=2,
                         help="MIND-SSC patch radius for the PRIMARY feature (used by both the "
                              "discrete and Adam stages). Default 2 matches this repo's "
                              "convexAdam_hyper_util.py::MINDSSC default, i.e. the original "
                              "(pre-DINO) ConvexAdam pipeline's own default - NOT the same as "
                              "the r=1 default used elsewhere in this repo for MIND as an "
                              "*auxiliary* term.")
    parser.add_argument("--mind-d", type=int, default=2,
                         help="MIND-SSC dilation for the primary feature. Default 2, matching "
                              "the original ConvexAdam repo.")
    args = parser.parse_args()
    if args.lambda_ncc > 0 and args.adam is None:
        parser.error("--lambda-ncc > 0 requires --adam (it's only wired into the Adam refinement stage)")
    main(args.gpu, args.configfile, args.adam, args.lambda_ncc, args.ncc_win, args.mind_r, args.mind_d)
