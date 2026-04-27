"""EDA figure generation. Saves PNGs into the project ``figures/`` folder so they
can be embedded directly into the LaTeX report.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config_v1

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")


def fig_monthly_trend(df: pd.DataFrame) -> None:
    monthly = df.groupby("month_date").size().rename("count").reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly["month_date"], monthly["count"], marker="o", linewidth=2)
    ax.set_title("Sheffield monthly crime volume")
    ax.set_xlabel("Month")
    ax.set_ylabel("Reported crimes")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(config_v1.FIGURE_DIR / "eda_monthly_trend.png", dpi=140)
    plt.close(fig)


def fig_category_breakdown(df: pd.DataFrame) -> None:
    counts = df["category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    counts.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Crimes by category (whole period)")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(config_v1.FIGURE_DIR / "eda_category_breakdown.png", dpi=140)
    plt.close(fig)


def fig_seasonal_heatmap(df: pd.DataFrame) -> None:
    pivot = (
        df.groupby(["year", "month_num"]).size().unstack(fill_value=0).sort_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "crimes"})
    ax.set_title("Crime count — year × month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    fig.tight_layout()
    fig.savefig(config_v1.FIGURE_DIR / "eda_year_month_heatmap.png", dpi=140)
    plt.close(fig)


def fig_spatial_density(df: pd.DataFrame) -> None:
    """Hexbin plot of report density across Sheffield."""
    fig, ax = plt.subplots(figsize=(9, 9))
    hb = ax.hexbin(df["longitude"], df["latitude"], gridsize=60, cmap="magma", mincnt=1)
    ax.set_title("Spatial density of crime reports")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(hb, ax=ax, label="crimes per cell")
    fig.tight_layout()
    fig.savefig(config_v1.FIGURE_DIR / "eda_spatial_density.png", dpi=140)
    plt.close(fig)


def fig_category_trends_small_multiples(df: pd.DataFrame) -> None:
    """One mini time-series per category — great diagnostic figure."""
    agg = df.groupby(["month_date", "category"]).size().rename("count").reset_index()
    cats = sorted(agg["category"].unique())
    ncols = 3
    nrows = -(-len(cats) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.4 * nrows), sharex=True)
    axes = axes.flatten()
    for ax, cat in zip(axes, cats):
        sub = agg[agg["category"] == cat]
        ax.plot(sub["month_date"], sub["count"], color="#2b6cb0")
        ax.set_title(cat, fontsize=11)
        ax.tick_params(labelsize=8)
    for ax in axes[len(cats):]:
        ax.axis("off")
    fig.suptitle("Monthly trend by category", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(config_v1.FIGURE_DIR / "eda_category_trends.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_all(df: pd.DataFrame) -> None:
    logger.info("Generating EDA figures …")
    fig_monthly_trend(df)
    fig_category_breakdown(df)
    fig_seasonal_heatmap(df)
    fig_spatial_density(df)
    fig_category_trends_small_multiples(df)
    logger.info("Saved EDA figures to %s", config_v1.FIGURE_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from src.data.features import engineer
    run_all(engineer())
