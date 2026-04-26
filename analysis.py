"""
analysis.py
-----------
Ana analiz scripti — tüm adımları sırasıyla çalıştırır.
Jupyter Notebook olarak da kullanılabilir (her bölüm bir hücreye karşılık gelir).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np

from src.data_loader import (
    fetch_prices, compute_returns, annualized_stats, SECTOR_MAP
)
from src.portfolio import (
    max_sharpe_portfolio, min_volatility_portfolio,
    efficient_frontier, random_portfolios,
    compute_var_cvar, compute_beta,
)
from src.visualize import (
    plot_correlation, plot_efficient_frontier,
    plot_weights, plot_cumulative_returns, plot_risk_summary,
)


# ════════════════════════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("BIST Portföy Optimizasyonu & Risk Analizi")
print("=" * 60)

prices, bench = fetch_prices()
returns       = compute_returns(prices)
bench_returns = bench  # fiyat serisi; cumulative'de pct_change alınacak


# ════════════════════════════════════════════════════════════════════════════════
# 2. TEK HİSSE İSTATİSTİKLERİ
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Hisse Bazlı Yıllık İstatistikler ──")
stats = annualized_stats(returns)
print(stats.to_string())

plot_risk_summary(stats)


# ════════════════════════════════════════════════════════════════════════════════
# 3. KORELASYON ANALİZİ
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Korelasyon Matrisi hesaplanıyor ──")
plot_correlation(returns)


# ════════════════════════════════════════════════════════════════════════════════
# 4. PORTFÖY OPTİMİZASYONU
# ════════════════════════════════════════════════════════════════════════════════
mean_returns = returns.mean()
cov_matrix   = returns.cov()

print("\n── Optimizasyon çalışıyor ──")
ms  = max_sharpe_portfolio(mean_returns, cov_matrix)
mv  = min_volatility_portfolio(mean_returns, cov_matrix)
ef  = efficient_frontier(mean_returns, cov_matrix, n_points=200)
rnd = random_portfolios(mean_returns, cov_matrix, n_portfolios=6000)

print(f"\n✓ Max Sharpe  → Getiri: {ms['return']*100:.2f}%  "
      f"Vol: {ms['vol']*100:.2f}%  Sharpe: {ms['sharpe']:.3f}")
print(f"✓ Min Volatilite → Getiri: {mv['return']*100:.2f}%  "
      f"Vol: {mv['vol']*100:.2f}%  Sharpe: {mv['sharpe']:.3f}")

plot_efficient_frontier(ef, rnd, ms, mv)
plot_weights(ms, mv, SECTOR_MAP)


# ════════════════════════════════════════════════════════════════════════════════
# 5. RİSK METRİKLERİ (VaR, CVaR, Beta)
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Risk Metrikleri ──")

for label, port in [("Max Sharpe", ms), ("Min Volatilite", mv)]:
    w_aligned = port["weights"].reindex(returns.columns).fillna(0).values
    var_cvar  = compute_var_cvar(returns, w_aligned)

    port_ret  = pd.Series(returns.values @ w_aligned, index=returns.index)
    beta_val  = compute_beta(port_ret, bench.pct_change().dropna())

    print(f"\n  [{label}]")
    for k, v in var_cvar.items():
        print(f"    {k}: {v}%")
    print(f"    Beta (BIST100'e karşı): {beta_val}")


# ════════════════════════════════════════════════════════════════════════════════
# 6. KÜMÜLATİF GETİRİ
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Kümülatif getiri grafiği ──")
plot_cumulative_returns(returns, ms["weights"], mv["weights"], bench)


# ════════════════════════════════════════════════════════════════════════════════
# ÖZET RAPOR
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALİZ TAMAMLANDI")
print("=" * 60)
print(f"  Kullanılan hisse sayısı : {len(returns.columns)}")
print(f"  Veri aralığı            : {returns.index[0].date()} → {returns.index[-1].date()}")
print(f"  Üretilen grafikler      : outputs/ klasörüne kaydedildi")
print("\n  Portföy Özeti:")
for label, p in [("Max Sharpe", ms), ("Min Volatilite", mv)]:
    top3 = p["weights"].nlargest(3)
    top3_str = ", ".join([f"{t.replace('.IS','')} ({v*100:.1f}%)"
                          for t, v in top3.items()])
    print(f"    {label}: {top3_str}")
