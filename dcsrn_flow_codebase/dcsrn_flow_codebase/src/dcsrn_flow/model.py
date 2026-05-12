from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossStitch(nn.Module):
    """Learned feature mixing across U, V, W, and magnitude/mask branches.

    For four branch feature tensors with the same shape, this computes:

        out_i = sum_j alpha[i, j] * in_j

    The matrix is initialized close to identity so each branch starts mostly independent,
    then learns how much to share with other branches.
    """

    def __init__(self, num_branches: int = 4, init_off_diagonal: float = 0.05):
        super().__init__()
        alpha = torch.eye(num_branches) * (1.0 - init_off_diagonal)
        alpha += torch.ones(num_branches, num_branches) * (init_off_diagonal / max(1, num_branches - 1))
        alpha.fill_diagonal_(1.0 - init_off_diagonal)
        self.alpha = nn.Parameter(alpha)

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(xs) != self.alpha.shape[0]:
            raise ValueError(f"Expected {self.alpha.shape[0]} branches, got {len(xs)}")
        mixed = []
        for i in range(len(xs)):
            y = 0.0
            for j, x in enumerate(xs):
                y = y + self.alpha[i, j] * x
            mixed.append(y)
        return mixed


class MultiBranchDenseBlock(nn.Module):
    """Paper-inspired dense block for U/V/W/M branches.

    Each branch has its own Conv3D layers, but new features are mixed through
    cross-stitch layers before being concatenated back to the branch stream.
    """

    def __init__(
        self,
        in_ch: int,
        growth_rate: int = 16,
        layers_per_block: int = 5,
        num_branches: int = 4,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.growth_rate = int(growth_rate)
        self.layers_per_block = int(layers_per_block)
        self.num_branches = int(num_branches)

        self.layers = nn.ModuleList()
        self.stitches = nn.ModuleList()
        current_ch = in_ch
        for _ in range(layers_per_block):
            branch_convs = nn.ModuleList(
                [ConvBNAct3D(current_ch, growth_rate, kernel_size=3) for _ in range(num_branches)]
            )
            self.layers.append(branch_convs)
            self.stitches.append(CrossStitch(num_branches=num_branches))
            current_ch += growth_rate

        self.out_ch = current_ch

    def forward(self, branches: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(branches) != self.num_branches:
            raise ValueError(f"Expected {self.num_branches} branches, got {len(branches)}")

        xs = branches
        for branch_convs, stitch in zip(self.layers, self.stitches):
            new_features = [conv(x) for conv, x in zip(branch_convs, xs)]
            new_features = stitch(new_features)
            xs = [torch.cat([old, new], dim=1) for old, new in zip(xs, new_features)]
        return xs


class BranchTransition(nn.Module):
    """Compress branch features after a dense block."""

    def __init__(self, in_ch: int, out_ch: int, num_branches: int = 4):
        super().__init__()
        self.transitions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                for _ in range(num_branches)
            ]
        )

    def forward(self, branches: List[torch.Tensor]) -> List[torch.Tensor]:
        return [layer(x) for layer, x in zip(self.transitions, branches)]


class DCSRNFlow(nn.Module):
    """div-mDCSRN-Flow inspired model.

    Input:
        lr_input: [B, 4, X_lr, Y_lr, Z_lr]
            channels = [u, v, w, magnitude]

    Output:
        velocity: [B, 3, X_hr, Y_hr, Z_hr]
        mask:     [B, 1, X_hr, Y_hr, Z_hr]

    Notes:
        The model uses four parallel branches for U, V, W, and magnitude/mask.
        Cross-stitch layers allow branches to share information while preserving
        direction-specific representations.
    """

    def __init__(
        self,
        growth_rate: int = 16,
        base_features: int = 32,
        layers_per_block: int = 5,
        num_hr_blocks: int = 3,
        scale_factor: int = 2,
        use_residual: bool = True,
    ):
        super().__init__()
        self.growth_rate = int(growth_rate)
        self.base_features = int(base_features)
        self.layers_per_block = int(layers_per_block)
        self.num_hr_blocks = int(num_hr_blocks)
        self.scale_factor = int(scale_factor)
        self.use_residual = bool(use_residual)
        self.num_branches = 4

        # One stem per branch: U, V, W, magnitude.
        self.stems = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct3D(1, base_features, kernel_size=3),
                    ConvBNAct3D(base_features, base_features, kernel_size=3),
                )
                for _ in range(self.num_branches)
            ]
        )

        # Denoising dense block at low resolution.
        self.lr_block = MultiBranchDenseBlock(
            in_ch=base_features,
            growth_rate=growth_rate,
            layers_per_block=layers_per_block,
            num_branches=self.num_branches,
        )
        self.lr_transition = BranchTransition(self.lr_block.out_ch, base_features, self.num_branches)

        # Dense blocks after trilinear upsampling.
        self.hr_blocks = nn.ModuleList()
        self.hr_transitions = nn.ModuleList()
        for _ in range(num_hr_blocks):
            block = MultiBranchDenseBlock(
                in_ch=base_features,
                growth_rate=growth_rate,
                layers_per_block=layers_per_block,
                num_branches=self.num_branches,
            )
            self.hr_blocks.append(block)
            self.hr_transitions.append(BranchTransition(block.out_ch, base_features, self.num_branches))

        # Output heads for U, V, W.
        self.velocity_heads = nn.ModuleList(
            [nn.Conv3d(base_features, 1, kernel_size=3, padding=1) for _ in range(3)]
        )

        # Mask head uses magnitude branch + estimated speed.
        self.mask_head = nn.Sequential(
            nn.Conv3d(base_features + 1, base_features, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_features, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, lr_input: torch.Tensor) -> dict:
        if lr_input.ndim != 5 or lr_input.shape[1] != 4:
            raise ValueError(f"Expected lr_input [B,4,X,Y,Z], got {tuple(lr_input.shape)}")

        # Split into U/V/W/M branches.
        branch_inputs = torch.chunk(lr_input, chunks=4, dim=1)
        branches = [stem(x) for stem, x in zip(self.stems, branch_inputs)]

        # Low-resolution denoising dense block.
        branches = self.lr_block(branches)
        branches = self.lr_transition(branches)

        # Internal 2x upsampling using trilinear interpolation.
        branches = [
            F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="trilinear",
                align_corners=False,
            )
            for x in branches
        ]

        # Remaining high-resolution dense blocks.
        for block, transition in zip(self.hr_blocks, self.hr_transitions):
            residual = branches
            branches = block(branches)
            branches = transition(branches)
            branches = [x + r for x, r in zip(branches, residual)]

        velocity_channels = [head(branches[i]) for i, head in enumerate(self.velocity_heads)]
        velocity = torch.cat(velocity_channels, dim=1)

        if self.use_residual:
            lr_vel = lr_input[:, :3]
            lr_vel_up = F.interpolate(
                lr_vel,
                scale_factor=self.scale_factor,
                mode="trilinear",
                align_corners=False,
            )
            velocity = velocity + lr_vel_up

        speed = torch.sqrt(torch.sum(velocity ** 2, dim=1, keepdim=True) + 1e-8)
        mask = self.mask_head(torch.cat([branches[3], speed], dim=1))

        return {"velocity": velocity, "mask": mask}
