from __future__ import annotations

import argparse
from typing import Any

import h5py
from scipy.io import loadmat


def print_h5(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: dataset shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{name}: group")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_file", required=True, type=str)
    args = parser.parse_args()

    print("Inspecting:", args.mat_file)
    print("\nTrying scipy.io.loadmat...")
    try:
        mat = loadmat(args.mat_file, squeeze_me=True, struct_as_record=False)
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            print(f"Top-level key: {key}, type={type(value)}, shape={getattr(value, 'shape', None)}")
            if hasattr(value, "__dict__"):
                fields = [k for k in value.__dict__.keys() if not k.startswith("_")]
                print(f"  struct fields: {fields}")
                for f in fields:
                    v = getattr(value, f)
                    print(f"    {key}.{f}: type={type(v)}, shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")
    except Exception as e:
        print("scipy could not read this file:", repr(e))

    print("\nTrying h5py for MATLAB v7.3...")
    try:
        with h5py.File(args.mat_file, "r") as f:
            f.visititems(print_h5)
    except Exception as e:
        print("h5py could not read this file:", repr(e))


if __name__ == "__main__":
    main()
