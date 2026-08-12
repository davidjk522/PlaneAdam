# Smoke test for decoder_method's FMIR model (README "Method 2": a VoxelMorph-style CNN
# decoder predicting the deformation field from PlaneCycle/DINO features, as opposed to
# Method 1's ConvexAdam test-time optimization).
#
# This is a plumbing check, not a training run: load one OASIS pair, extract DINO features,
# PCA-reduce them (the README's suggested fix for feeding large DINO feature maps into a
# conv decoder), build FMIR + its VoxelMorph warp/integration layers, run one forward pass,
# warp the moving image with the predicted flow, and confirm a loss/backward/optimizer step
# all complete with sane shapes and finite values.
#
# Run from the repo root: python -m PlaneAdam.scripts.smoke_test_fmir_decoder
import json
import os

import torch
import torch.nn.functional as F

from models.hub.backbones import dinov3_vits16
from PlaneAdam.Dataset.load_dataset_OASIS import REPO_ROOT, OASIS_CONFIG_PATH, get_data_train
from PlaneAdam.feature_extract.dino_extract import DinoBackboneExtractor
from PlaneAdam.decoder_method.FMIR import FMIR
from PlaneAdam.decoder_method.backbones import SpatialTransformer, VecInt

LOCAL_CHECKPOINT = os.path.join(REPO_ROOT, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

CUBE = 64          # isotropic smoke-test resolution; divisible by UPSCALE**(N_LEVELS-1)
PCA_DIM = 64       # PCA-reduced feature channels, matching --pca 64 in convex_optimization
UPSCALE = (2, 2, 2)
N_LEVELS = 5
N_STEPS = 3        # a couple of optimizer steps, just to prove gradients flow and shrink the loss


def to_volume_tensor(img):
    # nib volume is (H, W, D); the backbone wants (B, C, D, H, W) with in_chans=3.
    x = img.permute(2, 0, 1)  # (D, H, W)
    x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    return x.repeat(1, 3, 1, 1, 1)  # (1, 3, D, H, W)


def reduce_channels_pca(features_fix, features_mov, k):
    """
    Mirrors convex_optimization/convex_run_paired_dino.py's reduce_channels_pca: fit PCA
    jointly on both volumes' native-grid voxels, project both onto it.
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


def smoothness_loss(flow):
    dz = (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).pow(2).mean()
    dy = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).pow(2).mean()
    dx = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).pow(2).mean()
    return dz + dy + dx


def main():
    with open(OASIS_CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Just the first pair — enough to prove the pipeline works end-to-end.
    imgs_fixed, _segs_fixed, _masks_fixed, imgs_moving, _segs_moving, _masks_moving = get_data_train(
        config["pairs"][:1], config["HWD"]
    )

    backbone = dinov3_vits16(pretrained=True, weights=LOCAL_CHECKPOINT)
    extractor = DinoBackboneExtractor(backbone)
    print(f"backbone: {extractor.arch_name}, device: {extractor.device}")

    fixed_volume = to_volume_tensor(imgs_fixed[0])
    moving_volume = to_volume_tensor(imgs_moving[0])

    # (B, D, Hp, Wp, C) -> (B, C, Hp, Wp, D), matching convex_optimization's native-feature layout.
    xf_fixed = extractor.extract_feature_planecycle(fixed_volume).permute(0, 4, 2, 3, 1).contiguous()
    xf_moving = extractor.extract_feature_planecycle(moving_volume).permute(0, 4, 2, 3, 1).contiguous()
    print(f"native feature map shapes: fixed={tuple(xf_fixed.shape)}, moving={tuple(xf_moving.shape)}")

    feat_fixed, feat_moving = reduce_channels_pca(xf_fixed, xf_moving, PCA_DIM)

    cube_size = (CUBE, CUBE, CUBE)
    feat_fixed = F.interpolate(feat_fixed.float(), size=cube_size, mode='trilinear', align_corners=False)
    feat_moving = F.interpolate(feat_moving.float(), size=cube_size, mode='trilinear', align_corners=False)

    # imgs_fixed[0]/imgs_moving[0] are (H, W, D); resize to the same isotropic cube.
    device = extractor.device
    img_fixed = imgs_fixed[0][None, None].float().to(device)
    img_moving = imgs_moving[0][None, None].float().to(device)
    img_fixed = F.interpolate(img_fixed, size=cube_size, mode='trilinear', align_corners=False)
    img_moving = F.interpolate(img_moving, size=cube_size, mode='trilinear', align_corners=False)

    x_in = torch.cat([img_fixed, feat_fixed], dim=1)   # (1, PCA_DIM+1, C, C, C)
    y_in = torch.cat([img_moving, feat_moving], dim=1)
    print(f"decoder input shapes: x={tuple(x_in.shape)}, y={tuple(y_in.shape)}")

    model = FMIR(
        img_size=str(cube_size),
        start_channel='8',
        in_channels=str(PCA_DIM + 1),
    ).to(device)

    level_sizes = [CUBE // (2 ** i) for i in range(N_LEVELS)]
    transformers = [SpatialTransformer((s, s, s)).to(device) for s in level_sizes]
    integrates = [VecInt((s, s, s), nsteps=7).to(device) for s in level_sizes]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(N_STEPS):
        optimizer.zero_grad()
        int_flows, pos_flows = model(x_in, y_in, transformers, integrates, UPSCALE, registration=False)
        flow = pos_flows[0]  # finest, full-resolution displacement field
        warped_moving = transformers[0](img_moving, flow)

        recon_loss = F.mse_loss(warped_moving, img_fixed)
        smooth_loss = smoothness_loss(flow)
        loss = recon_loss + 0.01 * smooth_loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
        optimizer.step()

        print(
            f"step {step}: flow shape={tuple(flow.shape)}, recon={recon_loss.item():.6f}, "
            f"smooth={smooth_loss.item():.6f}, loss={loss.item():.6f}, grad_norm={grad_norm.item():.4f}"
        )

    assert torch.isfinite(loss), "loss is not finite"
    print("FMIR decoder smoke test passed: forward, warp, backward, and optimizer step all succeeded.")


if __name__ == "__main__":
    main()
