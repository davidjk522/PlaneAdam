import os
import torch
import itertools
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage import zoom

class abdomenorireg_loader(Dataset):

    def __init__(self,
            root_dir = './../../../data/abdomenreg/',
            split = 'train', # train, val or test
            clips = [-500, 800],
        ):

        self.root_dir = root_dir
        self.split = split
        self.clips = clips

        if self.split == 'train':
            idxs = np.arange(1,21)
        elif self.split == 'val':
            idxs = np.arange(21,24)
        elif self.split == 'test':
            idxs = np.arange(24,31)

        img_fps = [os.path.join(root_dir,'img',"img%s.nii.gz" % (str(idx).zfill(4))) for idx in idxs]
        lbl_fps = [os.path.join(root_dir,'label',"label%s.nii.gz" % (str(idx).zfill(4))) for idx in idxs]

        save_fp = os.path.join(root_dir,'save')
        os.makedirs(save_fp, exist_ok=True)
        save_fps = [os.path.join(save_fp,"subject%s.npz" % (str(idx).zfill(4))) for idx in idxs]

        self.save_fps = {idx:save_fp for idx, save_fp in zip(idxs, save_fps)}
        self.img_fps = list(itertools.permutations(img_fps, 2))
        self.lbl_fps = list(itertools.permutations(lbl_fps, 2))
        self.sub_idx = list(itertools.permutations(idxs, 2))

        print('----->>>> %s set has %d subjects' % (self.split, len(img_fps)))
        print('----->>>> %s set has %d pairs' % (self.split, len(self.sub_idx)))

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx):

        sub_idx1, sub_idx2 = self.sub_idx[idx]

        img_fp1, img_fp2 = self.img_fps[idx]
        lbl_fp1, lbl_fp2 = self.lbl_fps[idx]

        img1 = np.array(nib.load(img_fp1).get_fdata(), dtype='float32')
        lbl1 = np.array(nib.load(lbl_fp1).get_fdata(), dtype='float32')
        img1 = np.clip(img1, self.clips[0], self.clips[1])
        img1 = (img1 - self.clips[0]) / (self.clips[1] - self.clips[0])
        img1 = img1[None, ...]
        lbl1 = lbl1[None, ...]

        img2 = np.array(nib.load(img_fp2).get_fdata(), dtype='float32')
        lbl2 = np.array(nib.load(lbl_fp2).get_fdata(), dtype='float32')
        img2 = np.clip(img2, self.clips[0], self.clips[1])
        img2 = (img2 - self.clips[0]) / (self.clips[1] - self.clips[0])
        img2 = img2[None, ...]
        lbl2 = lbl2[None, ...]

        src_img, src_lbl = torch.from_numpy(img1), torch.from_numpy(lbl1)
        tgt_img, tgt_lbl = torch.from_numpy(img2), torch.from_numpy(lbl2)

        return src_img, src_lbl, tgt_img, tgt_lbl, sub_idx1, sub_idx2

if __name__ == '__main__':

    pass
