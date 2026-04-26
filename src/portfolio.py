"""
portfolio.py
------------
Markowitz portföy optimizasyonu: Etkin Sınır, Max-Sharpe, Min-Volatilite.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252
RISK_FREE     = 0.40   # Türkiye yaklaşık risk-free rate


# ── Portföy performans metrikleri ────────────────────────────────────────────

def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> tuple[float, float, float]:
    """
    Returns
    -------
    ret : yıllık beklenen getiri
    vol : yıllık volatilite
    sharpe : Sharpe oranı
    """
    ret    = np.dot(weights, mean_returns) * TRADING_DAYS
    vol    = np.sqrt(weights @ cov_matrix.values @ weights) * np.sqrt(TRADING_DAYS)
    sharpe = (ret - RISK_FREE) / vol
    return ret, vol, sharpe


# ── Optimizasyon yardımcıları ─────────────────────────────────────────────────

def _neg_sharpe(weights, mean_returns, cov_matrix):
    _, _, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    return -sharpe


def _portfolio_vol(weights, mean_returns, cov_matrix):
    _, vol, _ = portfolio_performance(weights, mean_returns, cov_matrix)
    return vol


def _optimize(
    objective,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    bounds=None,
    target_return: float = None,
) -> np.ndarray:
    n = len(mean_returns)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: np.dot(w, mean_returns) * TRADING_DAYS - target_return,
        })
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))

    result = minimize(
        objective,
        x0=np.ones(n) / n,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    return result.x


# ── Ana optimizasyon fonksiyonları ────────────────────────────────────────────

def max_sharpe_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> dict:
    """Maksimum Sharpe oranına sahip portföyü döner."""
    weights = _optimize(_neg_sharpe, mean_returns, cov_matrix)
    ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    return {
        "weights": pd.Series(weights, index=mean_returns.index),
        "return":  ret,
        "vol":     vol,
        "sharpe":  sharpe,
        "label":   "Max Sharpe",
    }


def min_volatility_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> dict:
    """Minimum volatiliteye sahip portföyü döner."""
    weights = _optimize(_portfolio_vol, mean_returns, cov_matrix)
    ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    return {
        "weights": pd.Series(weights, index=mean_returns.index),
        "return":  ret,
        "vol":     vol,
        "sharpe":  sharpe,
        "label":   "Min Volatilite",
    }


def efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Etkin sınırı oluşturan (volatilite, getiri) noktalarını döner.
    """
    ret_min = mean_returns.min() * TRADING_DAYS
    ret_max = mean_returns.max() * TRADING_DAYS
    target_returns = np.linspace(ret_min, ret_max, n_points)

    results = []
    for target in target_returns:
        try:
            w = _optimize(_portfolio_vol, mean_returns, cov_matrix,
                          target_return=target)
            r, v, s = portfolio_performance(w, mean_returns, cov_matrix)
            results.append({"return": r, "vol": v, "sharpe": s})
        except Exception:
            continue

    return pd.DataFrame(results)


def random_portfolios(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo simülasyonu: rastgele portföy ağırlıkları üretir.
    """
    rng = np.random.default_rng(seed)
    n   = len(mean_returns)
    records = []
    for _ in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        r, v, s = portfolio_performance(w, mean_returns, cov_matrix)
        records.append({"return": r, "vol": v, "sharpe": s})
    return pd.DataFrame(records)


# ── Risk metrikleri ───────────────────────────────────────────────────────────

def compute_var_cvar(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """
    Günlük portföy getirisi üzerinden VaR ve CVaR hesaplar.

    Parameters
    ----------
    returns    : günlük getiri DataFrame
    weights    : portföy ağırlıkları
    confidence : güven düzeyi (varsayılan 0.95)
    """
    port_ret = returns.values @ weights
    var      = np.percentile(port_ret, (1 - confidence) * 100)
    cvar     = port_ret[port_ret <= var].mean()
    return {
        "VaR_95 (günlük %)":  round(var  * 100, 4),
        "CVaR_95 (günlük %)": round(cvar * 100, 4),
        "VaR_95 (yıllık %)":  round(var  * 100 * np.sqrt(TRADING_DAYS), 4),
    }


def compute_beta(
    port_returns: pd.Series,
    bench_returns: pd.Series,
) -> float:
    """Portföyün BIST100'e karşı betasını hesaplar."""
    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
    cov_mat = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return round(cov_mat[0, 1] / cov_mat[1, 1], 4)
