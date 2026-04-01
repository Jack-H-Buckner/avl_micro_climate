"""Frost probability time series chart — cumulative last-frost view."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def plot_frost_timeseries(timeseries, threshold, location_label=None):
    """Plot cumulative last-frost probability vs. week-of-year.

    Parameters
    ----------
    timeseries : list of (week_num, date_label, cumulative_prob) tuples
    threshold : float, confidence level (e.g. 0.50)
    location_label : optional string for the title

    Returns
    -------
    matplotlib Figure
    """
    week_nums = [t[0] for t in timeseries]
    date_labels = [t[1] for t in timeseries]
    cum_probs = [t[2] if t[2] is not None else np.nan for t in timeseries]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(week_nums, cum_probs, color="#2196F3", linewidth=2.5, marker="o",
            markersize=3, zorder=3, label="Cumulative confidence")
    ax.fill_between(week_nums, cum_probs, alpha=0.12, color="#2196F3")

    # Threshold line
    thresh_pct = int(round(threshold * 100))
    ax.axhline(y=threshold, color="#E53935", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Threshold: {thresh_pct}%", zorder=2)

    # Find and annotate last frost date
    cum_arr = np.array(cum_probs)
    if np.any(np.isfinite(cum_arr)):
        for i, (wn, dlabel, cp) in enumerate(timeseries):
            if cp is not None and cp >= threshold:
                ax.axvline(x=wn, color="#4CAF50", linestyle=":", linewidth=1.5, zorder=2)
                ax.annotate(
                    f"Last frost ~{dlabel}\n({thresh_pct}% confidence)",
                    xy=(wn, threshold),
                    xytext=(wn + 2, threshold - 0.15),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1.5),
                    color="#4CAF50",
                    fontweight="bold",
                )
                break

    # X-axis
    tick_positions = week_nums[::3]
    tick_labels = date_labels[::3]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    ax.set_ylabel("Confidence last frost has passed", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)

    title = "Cumulative last frost probability"
    if location_label:
        title = f"{title} — {location_label}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    fig.tight_layout()
    return fig
