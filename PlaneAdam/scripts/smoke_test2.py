# Smoke test: load one OASIS pair, extract two feature maps via PlaneCycle,
# and sanity-check them with cosine similarity.
#
# Run from the repo root: python -m PlaneAdam.scripts.smoke_test
import json
import os
import torch
import nibabel as nib

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

    # list the cosine similarity between all the pairs in the config file
    for pair in config["pairs"]:
        fixed_path = os.path.join(REPO_ROOT, pair['fixed'])
        moving_path = os.path.join(REPO_ROOT, pair['moving'])

        img_fixed = torch.from_numpy(nib.load(fixed_path).get_fdata()).float().contiguous()
        img_moving = torch.from_numpy(nib.load(moving_path).get_fdata()).float().contiguous()

        fixed_volume = to_volume_tensor(img_fixed)
        moving_volume = to_volume_tensor(img_moving)

        backbone = dinov3_vits16(pretrained=True, weights=LOCAL_CHECKPOINT)
        extractor = DinoBackboneExtractor(backbone)
        print(f"Processing pair: fixed={pair['fixed']}, moving={pair['moving']}")
        print(f"backbone: {extractor.arch_name}, device: {extractor.device}")

        xf_fixed = extractor.extract_feature_planecycle(fixed_volume)
        xf_moving = extractor.extract_feature_planecycle(moving_volume)
        print(f"fixed feature map shape: {tuple(xf_fixed.shape)}")
        print(f"moving feature map shape: {tuple(xf_moving.shape)}")

        self_similarity = simple_cosine_similarity(xf_fixed, xf_fixed)
        pair_similarity = simple_cosine_similarity(xf_fixed, xf_moving)
        print(f"cosine similarity (fixed vs fixed, expect ~1.0): {self_similarity:.4f}")
        print(f"cosine similarity (fixed vs moving): {pair_similarity:.4f}")
        print("-" * 50)
        print()  # Add an empty line for better readability between pairs


if __name__ == "__main__":
    main()
