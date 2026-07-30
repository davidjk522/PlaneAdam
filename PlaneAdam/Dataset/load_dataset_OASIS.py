# load dataset function
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OASIS_CONFIG_PATH = os.path.join(REPO_ROOT, "OASIS_config.json")

class LoadDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load(self):
        if self.dataset_name == "OASIS":
            return self.load_oasis()
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
        img_fixed = torch.from_numpy(nib.load(pair['fixed']).get_fdata()).float().contiguous()
        seg_fixed = torch.from_numpy(nib.load(pair['fixed'].replace('imagesTr', 'labelsTr')).get_fdata()).float().contiguous()
        mask_fixed = torch.from_numpy(nib.load(pair['fixed'].replace('imagesTr', 'masksTr')).get_fdata()).float().contiguous()
        imgs_fixed.append(img_fixed)
        segs_fixed.append(seg_fixed)
        masks_fixed.append(mask_fixed)

        img_moving = torch.from_numpy(nib.load(pair['moving']).get_fdata()).float().contiguous()
        seg_moving = torch.from_numpy(nib.load(pair['moving'].replace('imagesTr', 'labelsTr')).get_fdata()).float().contiguous()
        mask_moving = torch.from_numpy(nib.load(pair['moving'].replace('imagesTr', 'masksTr')).get_fdata()).float().contiguous()
        imgs_moving.append(img_moving)
        segs_moving.append(seg_moving)
        masks_moving.append(mask_moving)

    return imgs_fixed, segs_fixed, masks_fixed, imgs_moving, segs_moving, masks_moving
