from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from src.dcsrn_flow.io import load_4dflow_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_file", required=True)
    parser.add_argument("--mat_var_path", nargs="*", default=None, help="Example: mrStruct dataAy")
    parser.add_argument("--layout", default="XYZCT")
    parser.add_argument("--sr_key", default=None, help="Use this for output .mat files, e.g. velocity_sr")
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--z", type=int, default=None)
    parser.add_argument("--out_png", default="slice.png")
    args = parser.parse_args()

    if args.sr_key:
        mat = loadmat(args.mat_file)
        data = mat[args.sr_key]
    else:
        if args.mat_var_path is None:
            raise ValueError("Provide --mat_var_path for raw input files, or --sr_key for output files.")
        data = load_4dflow_mat(args.mat_file, args.mat_var_path, args.layout)

    if data.ndim != 5:
        raise ValueError(f"Expected data shape [X,Y,Z,3,T], got {data.shape}")
    z = args.z if args.z is not None else data.shape[2] // 2
    vol = data[:, :, z, :, args.time]
    speed = np.sqrt(np.sum(vol.astype(np.float32) ** 2, axis=2))

    plt.figure(figsize=(6, 5))
    plt.imshow(speed.T, origin="lower")
    plt.colorbar(label="speed")
    plt.title(f"Speed magnitude, z={z}, time={args.time}")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print("Saved:", args.out_png)


if __name__ == "__main__":
    main()
