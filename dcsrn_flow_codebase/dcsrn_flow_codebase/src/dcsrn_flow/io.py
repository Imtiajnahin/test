from __future__ import annotations

import os
from typing import Any, Iterable, List, Sequence

import h5py
import numpy as np
from scipy.io import loadmat, savemat


def parse_var_path(path_like: Any) -> List[str]:
    """Accepts ['mrStruct','dataAy'] or 'mrStruct.dataAy'."""
    if isinstance(path_like, (list, tuple)):
        return [str(x) for x in path_like]
    if isinstance(path_like, str):
        return [x for x in path_like.split(".") if x]
    raise TypeError(f"Unsupported MAT variable path: {path_like!r}")


def _navigate(obj: Any, path: Sequence[str]) -> Any:
    current = obj
    for key in path:
        if isinstance(current, dict):
            current = current[key]
        elif isinstance(current, (h5py.File, h5py.Group)):
            current = current[key]
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            # scipy sometimes represents MATLAB structs as numpy void/object arrays
            try:
                current = current[key]
            except Exception as exc:
                raise KeyError(
                    f"Could not access '{key}' in object of type {type(current)}. "
                    f"Full requested path: {'.'.join(path)}"
                ) from exc
    return current


def load_mat_array(mat_path: str, mat_var_path: Any) -> np.ndarray:
    """Load a numeric array from either classic MATLAB files or v7.3 HDF5 MATLAB files."""
    var_path = parse_var_path(mat_var_path)

    scipy_error = None
    try:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        arr = _navigate(mat, var_path)
        arr = np.asarray(arr)
        if arr.dtype == object:
            arr = np.asarray(arr.tolist())
        return arr
    except NotImplementedError as exc:
        scipy_error = exc
    except Exception as exc:
        scipy_error = exc

    try:
        with h5py.File(mat_path, "r") as f:
            arr = _navigate(f, var_path)
            arr = np.asarray(arr)
            # h5py often returns MATLAB arrays reversed. We do not transpose here because
            # users can control layout explicitly after inspecting the shape.
            return arr
    except Exception as h5_error:
        raise RuntimeError(
            f"Could not load variable path {var_path} from {mat_path}\n"
            f"scipy error: {scipy_error}\n"
            f"h5py error: {h5_error}"
        )


def to_xyzct(data: np.ndarray, layout: str) -> np.ndarray:
    """Convert supported 5D layouts to [X, Y, Z, 3, T]."""
    data = np.asarray(data)
    layout = layout.upper()

    if data.ndim != 5:
        raise ValueError(f"Expected 5D 4D-flow data, got shape {data.shape}")

    if layout == "XYZCT":
        out = data
    elif layout == "XYZTC":
        out = np.transpose(data, (0, 1, 2, 4, 3))
    else:
        raise ValueError("layout must be 'XYZCT' or 'XYZTC'")

    if out.shape[3] != 3:
        raise ValueError(
            f"Expected component dimension to be 3 after layout conversion, got shape {out.shape}. "
            "Check data.layout in the config."
        )

    return np.asarray(out, dtype=np.float32)


def load_4dflow_mat(mat_path: str, mat_var_path: Any, layout: str) -> np.ndarray:
    arr = load_mat_array(mat_path, mat_var_path)
    return to_xyzct(arr, layout)


def save_4dflow_mat(
    output_path: str,
    velocity_sr: np.ndarray,
    mask_sr: np.ndarray | None = None,
    velocity_lr_original: np.ndarray | None = None,
    extra: dict | None = None,
) -> None:
    """Save super-resolved 4D-flow arrays in MATLAB-compatible format."""
    payload = {"velocity_sr": np.asarray(velocity_sr, dtype=np.float32)}
    if mask_sr is not None:
        payload["mask_sr"] = np.asarray(mask_sr, dtype=np.float32)
    if velocity_lr_original is not None:
        payload["velocity_lr_original"] = np.asarray(velocity_lr_original, dtype=np.float32)
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    savemat(output_path, payload, do_compression=True)
