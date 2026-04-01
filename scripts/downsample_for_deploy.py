"""Downsample last_frost_dates.npz for lightweight Streamlit deployment.

Reduces spatial resolution from 100m to 400m (4x block-mean) and
thresholds from 1% to 5% steps (100 → 20 levels).

Outputs:
    data/output/frost_climatology/last_frost_dates_400m.npz
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_DIR

FROST_CLIM_DIR = OUTPUT_DIR / "frost_climatology"
SRC_PATH = FROST_CLIM_DIR / "last_frost_dates.npz"
DST_PATH = FROST_CLIM_DIR / "last_frost_dates_400m.npz"

BLOCK = 4  # 100m × 4 = 400m
THRESH_STEP = 0.05  # 5% increments


def block_mean(arr, block):
    """Downsample 2D array by block-averaging, ignoring NaNs."""
    h, w = arr.shape
    h_trim = (h // block) * block
    w_trim = (w // block) * block
    trimmed = arr[:h_trim, :w_trim]
    blocks = trimmed.reshape(h_trim // block, block, w_trim // block, block)
    return np.nanmean(blocks, axis=(1, 3)).astype(np.float32)


def block_any(mask, block):
    """Downsample boolean mask: True if ANY pixel in the block is True."""
    h, w = mask.shape
    h_trim = (h // block) * block
    w_trim = (w // block) * block
    trimmed = mask[:h_trim, :w_trim]
    blocks = trimmed.reshape(h_trim // block, block, w_trim // block, block)
    return np.any(blocks, axis=(1, 3))


def block_majority(mask, block):
    """Downsample boolean mask: True if majority of pixels are True."""
    h, w = mask.shape
    h_trim = (h // block) * block
    w_trim = (w // block) * block
    trimmed = mask[:h_trim, :w_trim]
    blocks = trimmed.reshape(h_trim // block, block, w_trim // block, block)
    return np.mean(blocks, axis=(1, 3)) >= 0.5


def main():
    print(f"Loading {SRC_PATH} ...")
    d = np.load(SRC_PATH)

    grids = d["last_frost_grids"]       # (100, H, W)
    thresholds = d["thresholds"]         # (100,)
    valid_mask = d["valid_mask"]         # (H, W)
    elev_mask = d["elev_valid_mask"] if "elev_valid_mask" in d else None

    n_thresh, h, w = grids.shape
    print(f"  Source: {n_thresh} thresholds, {h}×{w} grid (100m)")

    # ── Subsample thresholds to 5% steps ────────────────────────────────────
    new_thresholds = np.arange(THRESH_STEP, 1.0 + THRESH_STEP / 2, THRESH_STEP,
                               dtype=np.float32)
    # Find nearest existing threshold for each target
    keep_idx = [np.argmin(np.abs(thresholds - t)) for t in new_thresholds]
    grids = grids[keep_idx]
    new_thresholds = thresholds[keep_idx]
    print(f"  Thresholds: {n_thresh} → {len(new_thresholds)}  "
          f"({new_thresholds[0]:.2f} to {new_thresholds[-1]:.2f})")

    # ── Block-mean spatial downsampling ─────────────────────────────────────
    n_new = len(new_thresholds)
    sample_ds = block_mean(grids[0], BLOCK)
    new_h, new_w = sample_ds.shape

    grids_ds = np.empty((n_new, new_h, new_w), dtype=np.float32)
    for i in range(n_new):
        grids_ds[i] = block_mean(grids[i], BLOCK)

    valid_ds = block_any(valid_mask, BLOCK)
    elev_ds = block_majority(elev_mask, BLOCK) if elev_mask is not None else None

    print(f"  Spatial: {h}×{w} → {new_h}×{new_w}  ({BLOCK*100}m)")

    # ── Save ────────────────────────────────────────────────────────────────
    save_dict = {
        "last_frost_grids": grids_ds,
        "thresholds": new_thresholds,
        "valid_mask": valid_ds,
    }
    if elev_ds is not None:
        save_dict["elev_valid_mask"] = elev_ds

    np.savez_compressed(DST_PATH, **save_dict)

    import os
    size_mb = os.path.getsize(DST_PATH) / 1e6
    print(f"\nSaved {DST_PATH}")
    print(f"  File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
