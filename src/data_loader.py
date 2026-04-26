"""
data_loader.py
--------------
BIST hisse senedi verilerini yfinance üzerinden çeker ve temizler.
"""

import yfinance as yf
import pandas as pd
import os

# ── Hisse listesi ──────────────────────────────────────────────────────────────
TICKERS = {
    "Enerji":              ["AKSEN.IS", "ZOREN.IS", "AYEN.IS", "EUPWR.IS"],
    "Bankacılık":          ["GARAN.IS", "ISCTR.IS", "AKBNK.IS",
                            "YKBNK.IS", "HALKB.IS", "VAKBN.IS", "TSKB.IS"],
    "Yatırım Kuruluşu":    ["ISMEN.IS", "GEDIK.IS"],
    "Sanayi":              ["EREGL.IS", "TOASO.IS"],
    "Savunma/Teknoloji":   ["ASELS.IS", "LOGO.IS"],
    "Perakende/Tüketim":   ["BIMAS.IS", "MGROS.IS"],
}

BENCHMARK = "XU100.IS"   # BIST100 endeksi

ALL_TICKERS = [t for tickers in TICKERS.values() for t in tickers]
SECTOR_MAP  = {t: s for s, tickers in TICKERS.items() for t in tickers}


def fetch_prices(
    tickers: list[str] = ALL_TICKERS,
    benchmark: str = BENCHMARK,
    period: str = "5y",
    interval: str = "1d",
    data_dir: str = "data",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Kapanış fiyatlarını indirir; eksik veri > %20 olan hisseleri düşürür.

    Returns
    -------
    prices    : pd.DataFrame  – günlük kapanış fiyatları (hisseler sütun)
    benchmark : pd.Series     – BIST100 kapanış fiyatları
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "prices_raw.parquet")

    # ── Önbellekten yükle ──────────────────────────────────────────────────
    if os.path.exists(cache_path):
        print("✓ Önbellekten yükleniyor...")
        df = pd.read_parquet(cache_path)
    else:
        print(f"↓ {len(tickers)} hisse + benchmark indiriliyor...")
        raw = yf.download(
            tickers + [benchmark],
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )["Close"]
        raw.to_parquet(cache_path)
        df = raw

    # ── Benchmark ayır ────────────────────────────────────────────────────
    bench_series = df[benchmark].dropna()
    df = df.drop(columns=[benchmark], errors="ignore")

    # ── Eksik veri filtresi (%20 üzeri → çıkar) ───────────────────────────
    missing_pct = df.isna().mean()
    dropped = missing_pct[missing_pct > 0.20].index.tolist()
    if dropped:
        print(f"⚠ Eksik veri nedeniyle çıkarıldı: {dropped}")
    df = df.drop(columns=dropped)

    # ── Tamamen boş sütunları çıkar ───────────────────────────────────────
    empty_cols = [c for c in df.columns if df[c].dropna().empty]
    if empty_cols:
        print(f"⚠ Hiç veri gelmeyen hisseler çıkarıldı: {empty_cols}")
    df = df.drop(columns=empty_cols, errors="ignore")

    # ── Forward-fill → kalan NaN'ları temizle ─────────────────────────────
    df = df.ffill().dropna()

    if df.empty:
        raise ValueError("Veri çekme başarısız: tüm hisseler boş döndü.")

    print(f"✓ Veri hazır: {df.shape[0]} gün × {df.shape[1]} hisse")
    print(f"  Tarih aralığı: {df.index[0].date()} → {df.index[-1].date()}")
    return df, bench_series


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Günlük log getirilerini hesaplar."""
    return prices.pct_change().dropna()


def annualized_stats(returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Her hisse için yıllık getiri ve volatilite hesaplar.

    Returns
    -------
    pd.DataFrame: index=ticker, columns=[annual_return, annual_vol, sharpe]
    """
    mu  = returns.mean() * trading_days
    sig = returns.std()  * (trading_days ** 0.5)
    rf  = 0.40           # Türkiye kısa vadesi yaklaşık risk-free (güncel politika faizi)
    sharpe = (mu - rf) / sig

    stats = pd.DataFrame({
        "Yıllık Getiri (%)":    (mu  * 100).round(2),
        "Yıllık Volatilite (%)": (sig * 100).round(2),
        "Sharpe Oranı":          sharpe.round(3),
    })
    stats.index.name = "Hisse"
    return stats.sort_values("Sharpe Oranı", ascending=False)


if __name__ == "__main__":
    prices, bench = fetch_prices()
    returns = compute_returns(prices)
    stats   = annualized_stats(returns)
    print("\n── Hisse İstatistikleri ──")
    print(stats.to_string())
