"""Trend + momentum screen and portfolio-fit ranking. Pure functions, no network.

Scores rank stocks against each other on past price behaviour only. They are a shortlist for
further study, not a forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import analytics as A

MIN_HISTORY = 260            # about one year of trading days
RULES = {"rsi_max": 75.0, "max_below_high": 0.20}


METRIC_COLS = ["ticker", "price", "r1m", "r3m", "r6m", "r12_1", "sma50", "sma200", "sma200_slope", "rsi", "vol",
               "below_high", "last_date"]


def metrics(closes: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker trend and momentum measures from daily (adjusted) closes."""
    rows = []
    for t in closes.columns:
        s = closes[t].dropna()
        if len(s) < MIN_HISTORY:
            continue
        sma50, sma200 = A.sma(s, 50), A.sma(s, 200)
        r = s.pct_change().dropna()
        p = float(s.iloc[-1])
        high = float(s.iloc[-252:].max())
        rows.append({
            "ticker": t, "price": p,
            "r1m": s.iloc[-1] / s.iloc[-22] - 1,
            "r3m": s.iloc[-1] / s.iloc[-64] - 1,
            "r6m": s.iloc[-1] / s.iloc[-127] - 1,
            "r12_1": s.iloc[-22] / s.iloc[-253] - 1,      # 12-month return skipping the last month
            "sma50": float(sma50.iloc[-1]), "sma200": float(sma200.iloc[-1]),
            "sma200_slope": float(sma200.iloc[-1] / sma200.iloc[-21] - 1),
            "rsi": float(A.rsi(s).iloc[-1]),
            "vol": float(r.iloc[-63:].std() * np.sqrt(A.TRADING_DAYS)),
            "below_high": p / high - 1,
            "last_date": s.index[-1],
        })
    return pd.DataFrame(rows, columns=METRIC_COLS)


def trend_ok(m: pd.DataFrame) -> pd.Series:
    return ((m["price"] > m["sma200"]) & (m["sma50"] > m["sma200"]) & (m["sma200_slope"] > 0)
            & (m["rsi"] <= RULES["rsi_max"]) & (m["below_high"] >= -RULES["max_below_high"]) & (m["r12_1"] > 0))


def _z(x: pd.Series) -> pd.Series:
    sd = x.std()
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=x.index)
    return ((x - x.mean()) / sd).clip(-3, 3)


def score(m: pd.DataFrame) -> pd.DataFrame:
    """Add eligible flag and a 0-100 momentum score (percentile among eligible stocks)."""
    m = m.copy()
    m["eligible"] = trend_ok(m)
    e = m[m["eligible"]]
    raw = pd.Series(np.nan, index=m.index)
    if not e.empty:
        raw.loc[e.index] = (0.40 * _z(e["r12_1"]) + 0.30 * _z(e["r6m"]) + 0.15 * _z(e["r3m"])
                            + 0.15 * _z(e["r6m"] / e["vol"].replace(0, np.nan)).fillna(0)
                            - 0.30 * (e["rsi"] > 70) - 0.20 * (e["vol"] > e["vol"].quantile(0.9)))
    m["raw_score"] = raw
    m["score"] = raw.rank(pct=True) * 100
    return m


def reasons(r: pd.Series) -> list[str]:
    out = [f"{r['r12_1']:+.0%} over 12 months (excluding the last month), {r['r6m']:+.0%} over 6 months",
           "Above its 50- and 200-day averages, 200-day average rising"]
    out.append(f"RSI {r['rsi']:.0f}" + (" (getting stretched)" if r["rsi"] > 70 else " (not overbought)"))
    out.append("At its 1-year high" if r["below_high"] > -0.01 else f"{-r['below_high']:.0%} below its 1-year high")
    out.append(f"Volatility {r['vol']:.0%} a year")
    return out


def _cap_per_sector(df: pd.DataFrame, n: int, per_sector: int) -> pd.DataFrame:
    taken, counts = [], {}
    for i, r in df.iterrows():
        k = r.get("sector") or "Other"
        if counts.get(k, 0) >= per_sector:
            continue
        counts[k] = counts.get(k, 0) + 1
        taken.append(i)
        if len(taken) == n:
            break
    return df.loc[taken]


def standalone(scored: pd.DataFrame, universe: pd.DataFrame, held: set[str], n: int = 10,
               per_sector: int = 3) -> pd.DataFrame:
    """Best trend + momentum scores across the universe, at most per_sector per sector."""
    df = scored[scored["eligible"]].merge(universe, on="ticker", how="left").sort_values("score", ascending=False)
    df["held"] = df["ticker"].isin(held)
    out = _cap_per_sector(df, n, per_sector).copy()
    out["reasons"] = [reasons(r) for _, r in out.iterrows()]
    return out.reset_index(drop=True)


def portfolio_returns(closes: pd.DataFrame, weights: pd.Series, days: int = 126) -> pd.Series:
    """Daily returns of the current portfolio mix (current weights) over the last `days` days."""
    w = weights[weights.index.isin(closes.columns)].dropna()
    if w.empty:
        return pd.Series(dtype=float)
    r = closes[w.index].ffill().pct_change(fill_method=None).iloc[-days:].fillna(0.0)
    return r @ (w / w.sum())


def portfolio_fit(scored: pd.DataFrame, universe: pd.DataFrame, closes: pd.DataFrame, pos: pd.DataFrame,
                  targets_pct: dict | None = None, n: int = 10, per_sector: int = 2,
                  add_weight: float = 0.05) -> pd.DataFrame:
    """Eligible stocks you don't hold, ranked by momentum plus how much they diversify your portfolio.

    fit = 0.45 momentum + 0.30 low correlation with your portfolio + 0.15 sector gap + 0.10 new country.
    """
    valued = pos.dropna(subset=["value"]) if not pos.empty else pos
    if valued.empty:
        return pd.DataFrame()
    held = set(valued["ticker"])
    weights = valued.set_index("ticker")["weight"]
    port_r = portfolio_returns(closes, weights)
    sector_w = A.group_weights(valued, "sector").set_index("sector")["weight"]
    country_w = A.group_weights(valued, "country").set_index("country")["weight"]
    targets = {k: v / 100 for k, v in (targets_pct or {}).items() if v}

    df = scored[scored["eligible"] & ~scored["ticker"].isin(held)].merge(universe, on="ticker", how="left")
    if df.empty:
        return df
    rets = closes[df["ticker"]].ffill().pct_change(fill_method=None).iloc[-len(port_r):] if len(port_r) else None
    df["corr"] = [rets[t].corr(port_r) if rets is not None and len(port_r) > 40 else np.nan for t in df["ticker"]]
    df["sector_weight"] = df["sector"].map(sector_w).fillna(0.0)
    df["country_weight"] = df["country"].map(country_w).fillna(0.0)
    if targets:
        gap = (df["sector"].map(targets).fillna(0.0) - df["sector_weight"]).clip(lower=0)
    else:
        gap = (0.15 - df["sector_weight"]).clip(lower=0)
    m = df["score"] / 100
    d = ((1 - df["corr"].fillna(0.5)) / 2).clip(0, 1)
    s = (gap / 0.15).clip(0, 1)
    c = (df["country_weight"] < 0.20).astype(float)
    df["fit"] = (0.45 * m + 0.30 * d + 0.15 * s + 0.10 * c) * 100
    hhi_now = A.hhi(weights)
    df["eff_n_after"] = 1 / (hhi_now * (1 - add_weight) ** 2 + add_weight ** 2)
    df = df.sort_values("fit", ascending=False)
    out = _cap_per_sector(df, n, per_sector).copy()

    def why(r):
        rs = []
        if pd.notna(r["corr"]):
            label = "moves fairly independently" if r["corr"] < 0.3 else "partly follows your holdings" if r["corr"] < 0.6 else "moves with your holdings"
            rs.append(f"Correlation with your portfolio {r['corr']:.2f} ({label})")
        if r["sector_weight"] < 0.10:
            rs.append(f"Adds {r['sector']}: {r['sector_weight']:.0%} of your portfolio today")
        if r["country_weight"] < 0.20:
            rs.append(f"Adds exposure to {r['country']} ({r['country_weight']:.0%} today)")
        rs.append(f"Effective positions {1 / hhi_now:.1f} → {r['eff_n_after']:.1f} if bought at {add_weight:.0%}")
        return rs + reasons(r)[:2]

    out["reasons"] = [why(r) for _, r in out.iterrows()]
    return out.reset_index(drop=True)
