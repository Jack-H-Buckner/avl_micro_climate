"""Test the climatology + Tukey left-tail filter on the first 10 ECOSTRESS scenes.

Produces a multi-panel figure with before/after heatmaps for each scene,
plus columns showing the climatology lower bound and residuals.

Output: eda/tukey_filter_heatmap_comparison.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PROCESSED_DIR

from src.data.filter_ecostress import (
    _load_gridmet_tmin,
    _load_climatology,
    _gridmet_tmin_for_date,
    _climatology_for_month,
    _reproject_tmin_to_scene,
    filter_scene,
    ECOSTRESS_SCENES_DIR,
    SCENE_INVENTORY_PATH,
    GRIDMET_CLIM_ZARR,
)

K = 1.25
Z = 3.0
N_SCENES = 10
EDA_DIR = Path(__file__).resolve().parent.parent / "eda"


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_parquet(SCENE_INVENTORY_PATH, engine="fastparquet")
    inventory["datetime_utc"] = pd.to_datetime(inventory["datetime_utc"])

    ds = _load_gridmet_tmin()

    # Load climatology (optional — graceful fallback)
    clim_ds = None
    try:
        clim_ds = _load_climatology()
        print(f"Loaded climatology from {GRIDMET_CLIM_ZARR}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    # Cache reprojected climatology grids per month
    clim_cache: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}

    MIN_VALID_FRAC = 0.25  # only plot scenes with >25% valid pixels in ROI

    # Collect data for plotting
    panels = []

    for _, row in inventory.iterrows():
        if len(panels) >= N_SCENES:
            break

        scene_path = ECOSTRESS_SCENES_DIR / row["filename"]
        if not scene_path.exists():
            print(f"Missing: {scene_path}")
            continue

        scene_date = np.datetime64(row["datetime_utc"].date(), "ns")
        scene_month = int(row["datetime_utc"].month)

        tmin_data = _gridmet_tmin_for_date(ds, scene_date)
        if tmin_data is None:
            print(f"No gridMET Tmin for {scene_date}")
            continue

        tmin_arr, tmin_meta = tmin_data

        with rasterio.open(scene_path) as src:
            lst_before = src.read(1).astype(np.float32)
            profile = dict(src.profile)

        nodata = profile.get("nodata")
        if nodata is not None:
            lst_before[lst_before == nodata] = np.nan

        # Skip scenes with insufficient valid-pixel coverage
        n_total = lst_before.size
        n_valid = int(np.count_nonzero(~np.isnan(lst_before)))
        if n_valid / n_total <= MIN_VALID_FRAC:
            print(f"Skipped (valid {n_valid / n_total:.0%} <= {MIN_VALID_FRAC:.0%}): {row['filename']}")
            continue

        tmin_reprojected = _reproject_tmin_to_scene(tmin_arr, tmin_meta, profile)

        # Compute residuals for the "before" image
        valid = ~np.isnan(lst_before) & ~np.isnan(tmin_reprojected)
        residuals = np.full_like(lst_before, np.nan)
        if valid.sum() > 0:
            residuals[valid] = lst_before[valid] - tmin_reprojected[valid]

        # Reproject climatology for this month
        clim_mean_reproj = None
        clim_std_reproj = None
        clim_lower_bound = None
        if clim_ds is not None:
            if scene_month not in clim_cache:
                (mean_arr, mean_meta), (std_arr, std_meta) = _climatology_for_month(
                    clim_ds, scene_month
                )
                clim_cache[scene_month] = (
                    _reproject_tmin_to_scene(mean_arr, mean_meta, profile),
                    _reproject_tmin_to_scene(std_arr, std_meta, profile),
                )
            clim_mean_reproj, clim_std_reproj = clim_cache[scene_month]
            clim_lower_bound = clim_mean_reproj - Z * clim_std_reproj

        result = filter_scene(
            scene_path,
            tmin_reprojected,
            k=K,
            clim_mean_reprojected=clim_mean_reproj,
            clim_std_reprojected=clim_std_reproj,
            z=Z,
        )
        if result is None:
            print(f"Skipped (too few pixels after filter): {row['filename']}")
            continue

        lst_after, stats = result

        label = row["filename"].replace(".tif", "")
        panels.append({
            "label": label,
            "before": lst_before,
            "after": lst_after,
            "residuals": residuals,
            "clim_lower_bound": clim_lower_bound,
            "stats": stats,
        })

    ds.close()
    if clim_ds is not None:
        clim_ds.close()

    if not panels:
        print("No scenes to plot.")
        return

    # ── Plot ────────────────────────────────────────────────────────────
    has_clim = panels[0]["clim_lower_bound"] is not None
    ncols = 4 if has_clim else 3
    n = len(panels)
    fig, axes = plt.subplots(n, ncols, figsize=(4 * ncols, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    # Shared LST colour range across all panels
    all_lst = np.concatenate([p["before"].ravel() for p in panels])
    vmin_lst = float(np.nanpercentile(all_lst, 2))
    vmax_lst = float(np.nanpercentile(all_lst, 98))

    # Shared residual colour range
    all_res = np.concatenate([p["residuals"].ravel() for p in panels])
    vmax_res = float(np.nanpercentile(np.abs(all_res[~np.isnan(all_res)]), 98))

    cmap_lst = plt.cm.inferno.copy()
    cmap_lst.set_bad("0.85")

    cmap_res = plt.cm.RdBu_r.copy()
    cmap_res.set_bad("0.85")

    cmap_clim = plt.cm.coolwarm.copy()
    cmap_clim.set_bad("0.85")

    for i, panel in enumerate(panels):
        col = 0

        # Before
        ax_before = axes[i, col]
        im0 = ax_before.imshow(
            panel["before"], cmap=cmap_lst, vmin=vmin_lst, vmax=vmax_lst,
        )
        ax_before.set_ylabel(panel["label"], fontsize=7, rotation=0, ha="right", va="center")
        ax_before.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if i == 0:
            ax_before.set_title("Before filter (LST °C)")
        col += 1

        # Climatology lower bound (own color scale per panel)
        if has_clim:
            ax_clim = axes[i, col]
            clim_lb = panel["clim_lower_bound"]
            clim_vmin = float(np.nanmin(clim_lb)) if not np.all(np.isnan(clim_lb)) else 0
            clim_vmax = float(np.nanmax(clim_lb)) if not np.all(np.isnan(clim_lb)) else 1
            im_clim = ax_clim.imshow(
                clim_lb, cmap=cmap_clim, vmin=clim_vmin, vmax=clim_vmax,
            )
            ax_clim.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            if i == 0:
                ax_clim.set_title(f"Clim. lower bound\n(mean − {Z}σ)")
            s = panel["stats"]
            info = f"clim_removed={s['pixels_removed_climatology']}"
            ax_clim.text(
                0.02, 0.02, info, transform=ax_clim.transAxes,
                fontsize=6, color="white", va="bottom",
                bbox=dict(facecolor="black", alpha=0.6, pad=1),
            )
            col += 1

        # Residuals
        ax_resid = axes[i, col]
        s = panel["stats"]
        fence = s["lower_fence"]

        im1 = ax_resid.imshow(
            panel["residuals"], cmap=cmap_res, vmin=-vmax_res, vmax=vmax_res,
        )
        ax_resid.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if i == 0:
            ax_resid.set_title("Residuals (LST − Tmin)")
        info = f"fence={fence:.1f}  tukey={s['pixels_removed_tukey']}"
        ax_resid.text(
            0.02, 0.02, info, transform=ax_resid.transAxes,
            fontsize=6, color="white", va="bottom",
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )
        col += 1

        # After
        ax_after = axes[i, col]
        im2 = ax_after.imshow(
            panel["after"], cmap=cmap_lst, vmin=vmin_lst, vmax=vmax_lst,
        )
        ax_after.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if i == 0:
            ax_after.set_title("After filter (LST °C)")

    # Colour bars
    fig.colorbar(im0, ax=axes[:, 0].tolist(), label="LST (°C)", shrink=0.6, pad=0.01)
    if has_clim:
        fig.colorbar(im_clim, ax=axes[:, 1].tolist(), label="Lower bound (°C)", shrink=0.6, pad=0.01)
    fig.colorbar(im1, ax=axes[:, -2].tolist(), label="Residual (°C)", shrink=0.6, pad=0.01)
    fig.colorbar(im2, ax=axes[:, -1].tolist(), label="LST (°C)", shrink=0.6, pad=0.01)

    title = f"Climatology (z={Z}) + Tukey (k={K}) filter: first {n} scenes"
    fig.suptitle(title, fontsize=12, y=1.01)

    out_path = EDA_DIR / "tukey_filter_heatmap_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
