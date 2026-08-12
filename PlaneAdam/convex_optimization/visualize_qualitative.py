# Standalone qualitative-results viewer: runs the ConvexAdam(+DINO[+NCC][+MIND]) pipeline for a
# single registration pair at a single hyperparameter setting (no sweep) and saves a PNG grid of
# axial (D-axis) slices comparing fixed vs moving vs warped-moving image/segmentation, plus a
# fixed-vs-warped segmentation contour overlap panel. This is the qualitative counterpart to the
# Dice numbers the sweep scripts (convex_run_paired_dino_*.py) already save to
# convex_search_results.pth / summary.txt, but that never save anything visual.
#
# Reuses adam_refine_displacement / reduce_channels_pca / GRID_SP_ADAM / LAMBDA_WEIGHT directly
# from convex_run_paired_dino_planecycle_chunked.py (imported as a module - this script must live
# alongside it) rather than re-implementing them, so the displacement field visualized here comes
# from the exact same code path a sweep run would use. The discrete stage
# (correlate/coupled_convex/inverse_consistency) is copied inline from that file's main() since
# it isn't factored into a standalone function there.
#
# Usage:
#   python visualize_qualitative.py <gpu> <configfile> --pair-idx 0 \
#       --grid-sp 3 --disp-hw 2 --adam 80 --grid-sp-adam 2 --lambda-weight 5.0 \
#       --lambda-ncc 1.0 --ncc-win 9 --lambda-mind 2.5 --mind-r 1 --mind-d 2
#
# Pick --grid-sp/--disp-hw (or --grid-sp-adam/--lambda-weight/--lambda-ncc/--lambda-mind if
# --adam is set) to match whichever run's summary.txt / folder name you want to visualize - e.g.
# a folder named dice0.781_..._gridspadam2_lambda5.00_..._ncc1w9_mind2.5r1d2_planecycle512_...
# (an older run, from before the resolution was pushed to 896 - see the resize call below; for
# new planecycle896_... folders, the visualization resolution here already matches)
# implies --adam ... --grid-sp-adam 2 --lambda-weight 5.0 --lambda-ncc 1.0 --ncc-win 9
# --lambda-mind 2.5 --mind-r 1 --mind-d 2 (grid_sp/disp_hw fixed at FIXED_GRID_SP/FIXED_DISP_HW
# from that file when --adam is active, currently 6/4).
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Script's own directory is on sys.path[0] automatically when run directly, so this sibling-module
# import works the same way convex_run_paired_dino_planecycle_chunked.py's own doc comment
# describes for its own invocation.
import convex_run_paired_dino_planecycle_chunked as pipeline
import models.hub.backbones as backbones_module
from convexAdam_hyper_util import correlate, coupled_convex, dice_coeff, inverse_consistency
from PlaneAdam.Dataset.load_dataset_OASIS import REPO_ROOT, get_data_train, to_volume_tensor
from PlaneAdam.feature_extract.dino_extract import DinoBackboneExtractor


def _bare_axes(ax):
    """Strip ticks/spines but keep set_ylabel usable (unlike ax.axis('off'))."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def overlay_seg(ax, img_slice, seg_slice, num_labels, title=None, alpha=0.45):
    ax.imshow(img_slice.T, cmap='gray', origin='lower')
    masked = np.ma.masked_where(seg_slice.T == 0, seg_slice.T)
    ax.imshow(masked, cmap='nipy_spectral', alpha=alpha, origin='lower', vmin=0, vmax=max(num_labels, 1))
    if title:
        ax.set_title(title, fontsize=10)
    _bare_axes(ax)


def overlap_panel(ax, img_slice, seg_a_slice, seg_b_slice, title=None):
    ax.imshow(img_slice.T, cmap='gray', origin='lower')
    ax.contour(seg_a_slice.T > 0, colors='red', linewidths=0.8)
    ax.contour(seg_b_slice.T > 0, colors='cyan', linewidths=0.8)
    if title:
        ax.set_title(title, fontsize=10)
    _bare_axes(ax)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gpu", type=int)
    parser.add_argument("configfile", type=str)
    parser.add_argument("--pair-idx", type=int, default=0,
                         help="Which pair in the config's pair list to visualize.")
    parser.add_argument("--pca", type=int, default=None)
    parser.add_argument("--adam", type=int, default=None,
                         help="Adam refinement iterations. Omit to visualize the discrete-stage-only result.")
    parser.add_argument("--plane-chunk-size", type=int, default=None)
    parser.add_argument("--grid-sp", type=int, default=6,
                         help="Discrete-stage grid_sp. Matches FIXED_GRID_SP in the sweep script when --adam is set.")
    parser.add_argument("--disp-hw", type=int, default=4,
                         help="Discrete-stage disp_hw. Matches FIXED_DISP_HW in the sweep script when --adam is set.")
    parser.add_argument("--grid-sp-adam", type=int, default=pipeline.GRID_SP_ADAM)
    parser.add_argument("--lambda-weight", type=float, default=pipeline.LAMBDA_WEIGHT)
    parser.add_argument("--lambda-ncc", type=float, default=0.0)
    parser.add_argument("--ncc-win", type=int, default=9)
    parser.add_argument("--lambda-mind", type=float, default=0.0)
    parser.add_argument("--mind-r", type=int, default=1)
    parser.add_argument("--mind-d", type=int, default=2)
    parser.add_argument("--num-slices", type=int, default=4,
                         help="Number of evenly-spaced axial (D-axis) slices to plot.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output PNG path. Defaults to qualitative_pair<idx>.png next to the config's output dir.")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.configfile, 'r') as f:
        config = json.load(f)
    pairs = config['pairs']
    num_labels = config['num_labels'] - 1
    if args.pair_idx >= len(pairs):
        raise ValueError(f"--pair-idx {args.pair_idx} out of range (config has {len(pairs)} pairs)")

    imgs_fixed, segs_fixed, masks_fixed, imgs_moving, segs_moving, masks_moving = get_data_train(
        [pairs[args.pair_idx]], config['HWD'])

    backbone_fn = getattr(backbones_module, config['backbone'])
    checkpoint_path = os.path.join(REPO_ROOT, config['checkpoint'])
    backbone = backbone_fn(pretrained=True, weights=checkpoint_path)
    extractor = DinoBackboneExtractor(backbone, plane_chunk_size=args.plane_chunk_size)
    print(f'using backbone {extractor.arch_name} on {extractor.device}')

    img_fixed = imgs_fixed[0].cuda()
    seg_fixed = segs_fixed[0].cuda()
    img_moving = imgs_moving[0].cuda()
    seg_moving = segs_moving[0].cuda()
    H, W, D = img_moving.shape
    grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, H, W, D), align_corners=False)

    with torch.no_grad():
        # --- feature extraction (same as main()'s per-pair block) ---
        fixed_volume = to_volume_tensor(imgs_fixed[0]).unsqueeze(0).to(extractor.device)
        moving_volume = to_volume_tensor(imgs_moving[0]).unsqueeze(0).to(extractor.device)
        # Kept in sync with convex_run_paired_dino_planecycle_chunked.py's resize (currently 896x896,
        # MedDINOv3's second baseline step) — must match whichever resolution the run being
        # visualized actually used, or PCA/features will be extracted at the wrong scale.
        fixed_volume = F.interpolate(fixed_volume, size=(D, 896, 896), mode='trilinear', align_corners=False)
        moving_volume = F.interpolate(moving_volume, size=(D, 896, 896), mode='trilinear', align_corners=False)

        features_fix_native = extractor.extract_feature_planecycle(fixed_volume).permute(0, 4, 2, 3, 1).contiguous()
        features_mov_native = extractor.extract_feature_planecycle(moving_volume).permute(0, 4, 2, 3, 1).contiguous()
        if args.pca is not None:
            features_fix_native, features_mov_native = pipeline.reduce_channels_pca(
                features_fix_native, features_mov_native, args.pca)

        pooled_size = (H // args.grid_sp, W // args.grid_sp, D // args.grid_sp)
        features_fix_smooth = F.interpolate(features_fix_native, size=pooled_size, mode='trilinear', align_corners=False)
        features_mov_smooth = F.interpolate(features_mov_native, size=pooled_size, mode='trilinear', align_corners=False)
        n_ch = features_fix_smooth.shape[1]

        # --- discrete stage (same as main()'s per-pair block) ---
        ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, args.disp_hw, args.grid_sp, (H, W, D), n_ch)
        disp_mesh_t = F.affine_grid(
            args.disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
            (1, 1, args.disp_hw * 2 + 1, args.disp_hw * 2 + 1, args.disp_hw * 2 + 1),
            align_corners=True).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)
        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, args.grid_sp, (H, W, D))

        scale = torch.tensor([H // args.grid_sp - 1, W // args.grid_sp - 1, D // args.grid_sp - 1]
                              ).view(1, 3, 1, 1, 1).cuda().half() / 2
        ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, args.disp_hw, args.grid_sp, (H, W, D), n_ch)
        disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, args.grid_sp, (H, W, D))
        disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)
        disp_hr = F.interpolate(disp_ice.flip(1) * scale * args.grid_sp, size=(H, W, D),
                                 mode='trilinear', align_corners=False)

        # --- optional Adam refinement, incl. NCC/MIND terms if requested ---
        if args.adam is not None:
            disp_hr = pipeline.adam_refine_displacement(
                disp_hr, features_fix_native, features_mov_native, H, W, D,
                grid_sp_adam=args.grid_sp_adam, n_iters=args.adam, lambda_weight=args.lambda_weight,
                device=extractor.device, img_fixed=img_fixed, img_moving=img_moving,
                lambda_ncc=args.lambda_ncc, ncc_win=args.ncc_win,
                lambda_mind=args.lambda_mind, mind_r=args.mind_r, mind_d=args.mind_d,
            )

        scale1 = torch.tensor([D - 1, W - 1, H - 1]).cuda() / 2
        warp_grid = grid0 + disp_hr.permute(0, 2, 3, 4, 1).flip(-1).div(scale1)
        seg_warped = F.grid_sample(seg_moving.view(1, 1, H, W, D), warp_grid, mode='nearest').squeeze()
        img_warped = F.grid_sample(img_moving.view(1, 1, H, W, D), warp_grid, mode='bilinear').squeeze()

    dice = dice_coeff(seg_fixed, seg_warped, num_labels + 1)
    print(f'Pair {args.pair_idx}: mean Dice = {dice.mean().item():.3f}')

    img_fixed_np = img_fixed.detach().cpu().numpy()
    img_moving_np = img_moving.detach().cpu().numpy()
    img_warped_np = img_warped.detach().cpu().numpy()
    seg_fixed_np = seg_fixed.detach().cpu().numpy()
    seg_moving_np = seg_moving.detach().cpu().numpy()
    seg_warped_np = seg_warped.detach().cpu().numpy()

    slice_indices = np.linspace(int(0.15 * D), int(0.85 * D), args.num_slices).astype(int)

    fig, axes = plt.subplots(args.num_slices, 4, figsize=(16, 4 * args.num_slices))
    if args.num_slices == 1:
        axes = axes[None, :]
    for row, z in enumerate(slice_indices):
        header = row == 0
        overlay_seg(axes[row, 0], img_fixed_np[:, :, z], seg_fixed_np[:, :, z], num_labels, 'Fixed' if header else None)
        overlay_seg(axes[row, 1], img_moving_np[:, :, z], seg_moving_np[:, :, z], num_labels, 'Moving' if header else None)
        overlay_seg(axes[row, 2], img_warped_np[:, :, z], seg_warped_np[:, :, z], num_labels, 'Warped moving' if header else None)
        overlap_panel(axes[row, 3], img_fixed_np[:, :, z], seg_fixed_np[:, :, z], seg_warped_np[:, :, z],
                      'Overlap: red=fixed, cyan=warped' if header else None)
        axes[row, 0].set_ylabel(f'z={z}', fontsize=9)

    title = (f'Pair {args.pair_idx} — Dice={dice.mean().item():.3f} '
             f'(grid_sp={args.grid_sp}, disp_hw={args.disp_hw}')
    if args.adam:
        title += f', adam={args.adam}, grid_sp_adam={args.grid_sp_adam}, lambda_weight={args.lambda_weight}'
    if args.lambda_ncc > 0:
        title += f', lambda_ncc={args.lambda_ncc}'
    if args.lambda_mind > 0:
        title += f', lambda_mind={args.lambda_mind}'
    title += ')'
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = args.out or os.path.join(os.path.dirname(config['output']) or '.', f'qualitative_pair{args.pair_idx}.png')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved qualitative comparison to {out_path}')


if __name__ == "__main__":
    main()
