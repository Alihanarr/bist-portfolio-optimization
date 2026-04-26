"""
visualize.py
------------
Tüm grafikleri üretir ve outputs/ klasörüne kaydeder.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Tema ─────────────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#0d1117",
    "surface":   "#161b22",
    "border":    "#30363d",
    "text":      "#e6edf3",
    "subtext":   "#8b949e",
    "accent":    "#58a6ff",
    "green":     "#3fb950",
    "orange":    "#d29922",
    "red":       "#f85149",
    "purple":    "#bc8cff",
    "gradient":  ["#58a6ff", "#bc8cff", "#3fb950"],
}

def _apply_theme(fig, ax_list):
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in ax_list if isinstance(ax_list, (list, np.ndarray)) else [ax_list]:
        ax.set_facecolor(COLORS["surface"])
        ax.tick_params(colors=COLORS["subtext"], labelsize=9)
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])


# ── 1. Korelasyon Isı Haritası ────────────────────────────────────────────────

def plot_correlation(returns: pd.DataFrame):
    corr = returns.corr()
    tickers = [t.replace(".IS", "") for t in corr.columns]

    fig, ax = plt.subplots(figsize=(14, 11))
    _apply_theme(fig, ax)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))   # üçgeni gizle

    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f",
        linewidths=0.5, linecolor=COLORS["border"],
        xticklabels=tickers, yticklabels=tickers,
        ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8, "color": COLORS["text"]},
    )
    ax.set_title("Hisse Korelasyon Matrisi (5 Yıllık Günlük Getiriler)",
                 fontsize=14, pad=16, color=COLORS["text"], fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✓ Kaydedildi: {path}")


# ── 2. Etkin Sınır ────────────────────────────────────────────────────────────

def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    random_df:   pd.DataFrame,
    max_sharpe:  dict,
    min_vol:     dict,
):
    fig, ax = plt.subplots(figsize=(12, 8))
    _apply_theme(fig, ax)

    # Monte Carlo noktaları
    sc = ax.scatter(
        random_df["vol"] * 100,
        random_df["return"] * 100,
        c=random_df["sharpe"],
        cmap="plasma", alpha=0.35, s=8, zorder=1,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Sharpe Oranı", color=COLORS["text"], fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=COLORS["subtext"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["subtext"])

    # Etkin sınır çizgisi
    ax.plot(
        frontier_df["vol"] * 100,
        frontier_df["return"] * 100,
        color=COLORS["accent"], lw=2.5, zorder=2, label="Etkin Sınır",
    )

    # Max Sharpe noktası
    ax.scatter(
        max_sharpe["vol"] * 100, max_sharpe["return"] * 100,
        marker="*", s=400, color=COLORS["green"], zorder=5,
        label=f"Max Sharpe ({max_sharpe['sharpe']:.2f})",
    )
    ax.annotate(
        f"  Max Sharpe\n  Getiri: {max_sharpe['return']*100:.1f}%\n  Vol: {max_sharpe['vol']*100:.1f}%",
        xy=(max_sharpe["vol"] * 100, max_sharpe["return"] * 100),
        color=COLORS["green"], fontsize=8.5,
    )

    # Min Volatilite noktası
    ax.scatter(
        min_vol["vol"] * 100, min_vol["return"] * 100,
        marker="D", s=150, color=COLORS["orange"], zorder=5,
        label=f"Min Volatilite ({min_vol['vol']*100:.1f}%)",
    )
    ax.annotate(
        f"  Min Vol\n  Getiri: {min_vol['return']*100:.1f}%\n  Vol: {min_vol['vol']*100:.1f}%",
        xy=(min_vol["vol"] * 100, min_vol["return"] * 100),
        color=COLORS["orange"], fontsize=8.5,
    )

    ax.set_xlabel("Yıllık Volatilite (%)", fontsize=11)
    ax.set_ylabel("Yıllık Beklenen Getiri (%)", fontsize=11)
    ax.set_title("Markowitz Etkin Sınır — BIST Portföyü (5 Yıl)",
                 fontsize=14, pad=14, fontweight="bold")
    ax.legend(facecolor=COLORS["surface"], edgecolor=COLORS["border"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, color=COLORS["border"], alpha=0.4, linestyle="--")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "efficient_frontier.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✓ Kaydedildi: {path}")


# ── 3. Ağırlık Dağılımı (Bar) ────────────────────────────────────────────────

def plot_weights(max_sharpe: dict, min_vol: dict, sector_map: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    _apply_theme(fig, axes)

    sector_colors = {
        "Enerji":             COLORS["green"],
        "Bankacılık":         COLORS["accent"],
        "Yatırım Kuruluşu":   COLORS["purple"],
        "Sanayi":             COLORS["orange"],
        "Savunma/Teknoloji":  "#ff7b72",
        "Perakende/Tüketim":  "#ffa657",
    }

    for ax, portfolio, title in zip(
        axes,
        [max_sharpe, min_vol],
        ["Max Sharpe Portföyü", "Min Volatilite Portföyü"],
    ):
        weights = portfolio["weights"]
        weights = weights[weights > 0.001].sort_values(ascending=True)
        tickers = [t.replace(".IS", "") for t in weights.index]
        bar_colors = [sector_colors.get(sector_map.get(t + ".IS", ""), COLORS["accent"])
                      for t in tickers]

        bars = ax.barh(tickers, weights.values * 100,
                       color=bar_colors, edgecolor=COLORS["border"], height=0.65)

        for bar, val in zip(bars, weights.values * 100):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", color=COLORS["text"], fontsize=8.5)

        ax.set_xlabel("Ağırlık (%)", fontsize=10)
        ax.set_title(
            f"{title}\nGetiri: {portfolio['return']*100:.1f}%  |  "
            f"Vol: {portfolio['vol']*100:.1f}%  |  Sharpe: {portfolio['sharpe']:.2f}",
            fontsize=11, pad=10, fontweight="bold",
        )
        ax.set_xlim(0, weights.values.max() * 100 * 1.2)
        ax.grid(True, axis="x", color=COLORS["border"], alpha=0.4, linestyle="--")

    # Sektör renk açıklaması
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s)
                       for s, c in sector_colors.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               facecolor=COLORS["surface"], edgecolor=COLORS["border"],
               labelcolor=COLORS["text"], fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Portföy Ağırlık Dağılımı — Sektör Renklendirmesi",
                 fontsize=13, color=COLORS["text"], fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "portfolio_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✓ Kaydedildi: {path}")


# ── 4. Kümülatif Getiri ──────────────────────────────────────────────────────

def plot_cumulative_returns(
    returns: pd.DataFrame,
    max_sharpe_w: pd.Series,
    min_vol_w:    pd.Series,
    bench_returns: pd.Series,
):
    fig, ax = plt.subplots(figsize=(14, 7))
    _apply_theme(fig, ax)

    common_idx = returns.index.intersection(bench_returns.index)
    ret_aligned   = returns.loc[common_idx]
    bench_aligned = bench_returns.loc[common_idx].pct_change().dropna()

    port_ms  = (ret_aligned[max_sharpe_w.index] @ max_sharpe_w).rename("Max Sharpe")
    port_mv  = (ret_aligned[min_vol_w.index]    @ min_vol_w).rename("Min Volatilite")
    bench_r  = bench_aligned.rename("BIST100")

    for series, color, lw, ls in [
        (port_ms, COLORS["green"],  2.5, "-"),
        (port_mv, COLORS["orange"], 2.5, "--"),
        (bench_r, COLORS["accent"], 1.8, ":"),
    ]:
        cum = (1 + series).cumprod() - 1
        ax.plot(cum.index, cum * 100, color=color, lw=lw, ls=ls,
                label=f"{series.name}  (toplam: {cum.iloc[-1]*100:.1f}%)")

    ax.axhline(0, color=COLORS["border"], lw=1)
    ax.set_xlabel("Tarih", fontsize=10)
    ax.set_ylabel("Kümülatif Getiri (%)", fontsize=10)
    ax.set_title("Kümülatif Getiri Karşılaştırması — Portföyler vs BIST100",
                 fontsize=13, pad=12, fontweight="bold")
    ax.legend(facecolor=COLORS["surface"], edgecolor=COLORS["border"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, color=COLORS["border"], alpha=0.4, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cumulative_returns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✓ Kaydedildi: {path}")


# ── 5. Risk Metrikleri Özet ───────────────────────────────────────────────────

def plot_risk_summary(stats_df: pd.DataFrame):
    """Hisse bazlı yıllık getiri vs volatilite scatter."""
    fig, ax = plt.subplots(figsize=(12, 8))
    _apply_theme(fig, ax)

    x = stats_df["Yıllık Volatilite (%)"]
    y = stats_df["Yıllık Getiri (%)"]
    s = stats_df["Sharpe Oranı"]

    sc = ax.scatter(x, y, c=s, cmap="RdYlGn", s=120, zorder=3,
                    edgecolors=COLORS["border"], linewidth=0.8)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Sharpe Oranı", color=COLORS["text"])
    cbar.ax.yaxis.set_tick_params(color=COLORS["subtext"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["subtext"])

    for ticker, row in stats_df.iterrows():
        ax.annotate(
            ticker.replace(".IS", ""),
            (row["Yıllık Volatilite (%)"], row["Yıllık Getiri (%)"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, color=COLORS["text"],
        )

    ax.axhline(0, color=COLORS["red"], lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Yıllık Volatilite (%)", fontsize=11)
    ax.set_ylabel("Yıllık Getiri (%)", fontsize=11)
    ax.set_title("Risk-Getiri Dağılımı — Hisse Bazlı",
                 fontsize=13, pad=12, fontweight="bold")
    ax.grid(True, color=COLORS["border"], alpha=0.4, linestyle="--")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "risk_return_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✓ Kaydedildi: {path}")
