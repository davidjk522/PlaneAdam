# load dataset function
import json
import os

import nibabel as nib
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OASIS_CONFIG_PATH = os.path.join(REPO_ROOT, "OASIS_config.json")

class LoadDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load(self):
        if self.dataset_name == "OASIS":
            return self.get_oasis_config(OASIS_CONFIG_PATH)
        # elif to be addeed
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")

    def get_oasis_config(self, configfile):
        """
        Read and return the OASIS dataset configuration from OASIS_config.json.
        """
        with open(configfile, "r") as f:
            config = json.load(f)
        pairs = config['pairs']
        num_labels = config['num_labels'] -1
        imgs_fixed, segs_fixed, masks_fixed, imgs_moving, segs_moving, masks_moving = get_data_train(pairs, config['HWD'])

        return imgs_fixed, segs_fixed, masks_fixed, imgs_moving, segs_moving, masks_moving, num_labels
        
# must extract the neccessary data from the dataset using functions like get_data_train, get_data_test, etc. and return them in the format required by the model.        
def get_data_train(pairs, HWD):
    H, W, D = HWD[0], HWD[1], HWD[2]

    # images, segmentations, and masks for fixed and moving images
    imgs_fixed = [] 
    segs_fixed = []
    masks_fixed = []
    imgs_moving = []
    segs_moving = []
    masks_moving = []

    for pair in tqdm(pairs):
        fixed_path = os.path.join(REPO_ROOT, pair['fixed'])
        moving_path = os.path.join(REPO_ROOT, pair['moving'])

        img_fixed = torch.from_numpy(nib.load(fixed_path).get_fdata()).float().contiguous()
        seg_fixed = torch.from_numpy(nib.load(fixed_path.replace('imagesTr', 'labelsTr')).get_fdata()).float().contiguous()
        mask_fixed = torch.from_numpy(nib.load(fixed_path.replace('imagesTr', 'masksTr')).get_fdata()).float().contiguous()
        imgs_fixed.append(img_fixed)
        segs_fixed.append(seg_fixed)
        masks_fixed.append(mask_fixed)

        img_moving = torch.from_numpy(nib.load(moving_path).get_fdata()).float().contiguous()
        seg_moving = torch.from_numpy(nib.load(moving_path.replace('imagesTr', 'labelsTr')).get_fdata()).float().contiguous()
        mask_moving = torch.from_numpy(nib.load(moving_path.replace('imagesTr', 'masksTr')).get_fdata()).float().contiguous()
        imgs_moving.append(img_moving)
        segs_moving.append(seg_moving)
        masks_moving.append(mask_moving)

    return imgs_fixed, segs_fixed, masks_fixed, imgs_moving, segs_moving, masks_moving


def to_volume_tensor(img: torch.Tensor) -> torch.Tensor:
    """
    (H, W, D) -> (C, D, H, W), channel-expanded to 3 to match the DINOv3
    backbone's in_chans.
    """
    x = img.permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
    return x.repeat(3, 1, 1, 1)  # (3, D, H, W)


class OASISPairDataset(torch.utils.data.Dataset):
    """
    Wraps loaded OASIS fixed/moving volumes (and their segmentations/masks) for
    use with a DataLoader, yielding dict batches with 'fixed'/'moving' volume
    tensors shaped (B, C, D, H, W) for DinoBackboneExtractor.
    """

    def __init__(self, imgs_fixed, segs_fixed, masks_fixed, imgs_moving, segs_moving, masks_moving):
        self.imgs_fixed = imgs_fixed
        self.segs_fixed = segs_fixed
        self.masks_fixed = masks_fixed
        self.imgs_moving = imgs_moving
        self.segs_moving = segs_moving
        self.masks_moving = masks_moving

    def __len__(self):
        return len(self.imgs_fixed)

    def __getitem__(self, idx):
        return {
            "fixed": to_volume_tensor(self.imgs_fixed[idx]),
            "moving": to_volume_tensor(self.imgs_moving[idx]),
            "seg_fixed": self.segs_fixed[idx],
            "seg_moving": self.segs_moving[idx],
            "mask_fixed": self.masks_fixed[idx],
            "mask_moving": self.masks_moving[idx],
        }


def get_oasis_dataloader(batch_size: int = 1, shuffle: bool = False, num_workers: int = 0) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader over OASIS fixed/moving pairs, ready for
    develop.extract_features(model, data_loader, device).
    """
    imgs_fixed, segs_fixed, masks_fixed, imgs_moving, segs_moving, masks_moving, _num_labels = (
        LoadDataset("OASIS").load()
    )
    dataset = OASISPairDataset(imgs_fixed, segs_fixed, masks_fixed, imgs_moving, segs_moving, masks_moving)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
