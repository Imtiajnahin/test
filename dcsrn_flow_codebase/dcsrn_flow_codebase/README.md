# div-mDCSRN-Flow for 4D Flow `.mat` Data

This is a functional PyTorch codebase for super-resolving 4D flow MRI velocity fields with a paper-inspired div-mDCSRN-Flow architecture.

It is designed for your data structure:

```text
X × Y × Z × 3 × T
```

where:

- `X, Y, Z` are the 3D spatial dimensions
- `3` is the velocity-component dimension, for example AP/LR/FH or vx/vy/vz
- `T` is the number of time frames / cardiac phases

Example:

```text
400 × 300 × 25 × 3 × 10
```

The code treats each time frame as a 3D vector field:

```text
X × Y × Z × 3
```

The model predicts all three velocity directions together and uses a divergence loss so that the predicted 3D velocity field is more physically consistent.

---

## What is implemented

This codebase includes:

1. `.mat` loading for nested MATLAB structs such as `mrStruct.dataAy`
2. Support for data layouts:
   - `XYZCT`: `[X, Y, Z, Component, Time]`
   - `XYZTC`: `[X, Y, Z, Time, Component]`
3. Random 3D velocity patch sampling from each time frame
4. Synthetic low-resolution generation from high-resolution patches
5. A paper-inspired multi-branch 3D dense network:
   - U branch
   - V branch
   - W branch
   - magnitude/mask branch
   - cross-stitch mixing between branches
6. Loss function:
   - velocity MSE loss
   - mask BCE loss
   - divergence regularization loss
7. Training script
8. Full-case inference script that reconstructs:

```text
SR_X × SR_Y × SR_Z × 3 × T
```

---

## Important note

This is a faithful, practical implementation based on the architecture and loss described in Patel et al. 2025, but it is not the authors' official code. Some exact implementation choices are not fully specified in the paper, so the model is built to match the described behavior while remaining runnable and understandable.

---

## Installation

Create a fresh environment:

```bash
conda create -n dcsrn_flow python=3.10 -y
conda activate dcsrn_flow
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version if needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only testing:

```bash
pip install torch torchvision
```

---

## Step 1: Inspect one `.mat` file

Use this first to confirm your actual variable path.

```bash
python scripts/inspect_mat.py --mat_file /path/to/case01.mat
```

If your data is stored as:

```matlab
mrStruct.dataAy
```

then use this in the config:

```yaml
mat_var_path: ["mrStruct", "dataAy"]
```

If your data is stored directly as:

```matlab
data
```

then use:

```yaml
mat_var_path: ["data"]
```

---

## Step 2: Edit the config

Open:

```text
configs/default.yaml
```

Change these fields:

```yaml
data:
  train_mat_dir: "./data/train_mat"
  val_mat_dir: "./data/val_mat"
  mat_var_path: ["mrStruct", "dataAy"]
  layout: "XYZCT"
```

For your common data shape:

```text
400 × 300 × 25 × 3 × 10
```

use:

```yaml
layout: "XYZCT"
```

---

## Step 3: Train directly from `.mat` files

```bash
python train.py --config configs/default.yaml
```

This will:

1. Load your `.mat` files
2. Randomly sample high-resolution 3D velocity patches
3. Downsample them internally to create low-resolution patches
4. Train the model to reconstruct the high-resolution velocity patch

---

## Optional: Precompute patches

If loading `.mat` files repeatedly is slow, precompute `.npz` patches:

```bash
python scripts/prepare_dataset.py --config configs/default.yaml --split train --num_patches 5000
python scripts/prepare_dataset.py --config configs/default.yaml --split val --num_patches 1000
```

Then set in the config:

```yaml
data:
  use_precomputed: true
```

and train:

```bash
python train.py --config configs/default.yaml
```

---

## Step 4: Run inference on one full 4D flow `.mat` file

```bash
python infer_full_case.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pt \
  --input_mat /path/to/case01.mat \
  --output_mat /path/to/case01_DCSRN_Flow_SR.mat
```

The output `.mat` file contains:

```matlab
velocity_sr
mask_sr
velocity_lr_original
```

where `velocity_sr` has shape:

```text
SR_X × SR_Y × SR_Z × 3 × T
```

For scale factor 2, an input case of:

```text
400 × 300 × 25 × 3 × 10
```

will produce approximately:

```text
800 × 600 × 50 × 3 × 10
```

---

## Recommended starting settings for your data

Because your Z dimension can be around 25, use:

```yaml
patch_size_hr: [64, 64, 16]
scale_factor: 2
```

This means the model input patch is:

```text
32 × 32 × 8 × 4
```

where 4 input channels are:

```text
u, v, w, magnitude
```

and the model output patch is:

```text
64 × 64 × 16 × 3
```

---

## Notes about time

This first version is time-preserving but not recurrent/time-aware.

Each time frame is processed as a separate 3D vector field:

```text
time 0 → super-resolved velocity field
time 1 → super-resolved velocity field
...
time T-1 → super-resolved velocity field
```

The final 4D structure is restored using the time index:

```text
SR_data[:, :, :, :, t]
```

A future extension could add temporal windows:

```text
[t-1, t, t+1]
```

as extra channels or use ConvLSTM/3D+time convolutions. This code keeps the first implementation simpler and closer to the paper's per-time-frame patch description.
