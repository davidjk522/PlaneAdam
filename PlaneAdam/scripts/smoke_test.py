# Smoke test: load one OASIS pair, extract two feature maps via PlaneCycle,
# and sanity-check them with cosine similarity.
#
# Run from the repo root: python -m PlaneAdam.scripts.smoke_test
import json
import os

from models.hub.backbones import dinov3_vits16
from PlaneAdam.develop import simple_cosine_similarity
from PlaneAdam.feature_extract.dino_extract import DinoBackboneExtractor
from PlaneAdam.Dataset.load_dataset_OASIS import REPO_ROOT, OASIS_CONFIG_PATH, get_data_train

LOCAL_CHECKPOINT = os.path.join(REPO_ROOT, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")


def to_volume_tensor(img):
    # nib volume is (H, W, D); the backbone wants (B, C, D, H, W) with in_chans=3.
    x = img.permute(2, 0, 1)  # (D, H, W)
    x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    return x.repeat(1, 3, 1, 1, 1)  # (1, 3, D, H, W)


def main():
    with open(OASIS_CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Just the first pair — enough to prove the pipeline works end-to-end.
    imgs_fixed, _segs_fixed, _masks_fixed, imgs_moving, _segs_moving, _masks_moving = get_data_train(
        config["pairs"][:1], config["HWD"]
    )
    fixed_volume = to_volume_tensor(imgs_fixed[0])
    moving_volume = to_volume_tensor(imgs_moving[0])

    backbone = dinov3_vits16(pretrained=True, weights=LOCAL_CHECKPOINT)
    extractor = DinoBackboneExtractor(backbone)
    print(f"backbone: {extractor.arch_name}, device: {extractor.device}")

    xf_fixed = extractor.extract_feature_planecycle(fixed_volume)
    xf_moving = extractor.extract_feature_planecycle(moving_volume)
    print(f"fixed feature map shape: {tuple(xf_fixed.shape)}")
    print(f"moving feature map shape: {tuple(xf_moving.shape)}")

    self_similarity = simple_cosine_similarity(xf_fixed, xf_fixed)
    pair_similarity = simple_cosine_similarity(xf_fixed, xf_moving)
    print(f"cosine similarity (fixed vs fixed, expect ~1.0): {self_similarity:.4f}")
    print(f"cosine similarity (fixed vs moving): {pair_similarity:.4f}")


if __name__ == "__main__":
    main()
