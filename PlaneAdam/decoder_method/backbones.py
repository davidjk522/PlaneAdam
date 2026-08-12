"""
Self-contained replacement for the `models.backbones.*` package that FMIR.py/regdino.py
were originally written against (a full separate registration-model repo that isn't part
of PlaneAdam). Only the two pieces decoder_method actually needs are reimplemented here:

  - `encoder`:            the plain Conv3d+norm+activation block used to build FMIR's
                           feature encoder (`models.backbones.layers.encoder`).
  - `SpatialTransformer`: grid_sample-based warping by a dense displacement field
                           (`models.backbones.voxelmorph.torch.layers.SpatialTransformer`).
  - `VecInt`:             scaling-and-squaring integration of a stationary velocity field
                           into a diffeomorphic displacement field (same module's `VecInt`).

SpatialTransformer/VecInt follow the standard VoxelMorph formulation (Dalca et al.,
"Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder(nn.Module):
    """Conv3d -> InstanceNorm3d -> LeakyReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class SpatialTransformer(nn.Module):
    """Warps a 3D volume by a dense displacement field via trilinear grid_sample."""

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        vectors = [torch.arange(0, s, dtype=torch.float32) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # (3, D, H, W)
        grid = grid.unsqueeze(0)  # (1, 3, D, H, W)
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Normalize voxel coordinates to [-1, 1] for grid_sample.
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # grid_sample expects the channel dim last, ordered (W, H, D) i.e. reversed.
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """Integrates a stationary velocity field into a displacement field by scaling & squaring."""

    def __init__(self, inshape, nsteps=7):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec
