"""Long shots: stocks trading far below the analysts' average target price. Pure functions, no network.

High risk by design. A big gap to the target often means the market doubts the analysts, and targets
are frequently too optimistic. The score ranks how big and how well supported the gap is; risk is shown
next to it, not rewarded.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

MIN_ANALYSTS = 3
MAX_AGE_DAYS = 30          # older targets are not used for ranking
KNIFE_DROP = -0.15         # 3-month return below this, in a downtrend = "falling knife"
FEE_WARN = 0.03            # buy + sell fees above 3% of the money invested

RATINGS = {"strong_buy": "Strong buy", "buy": "Buy", "hold": "Hold", "underperform": "Underperform",
           "sell": "Sell", "strong_sell": "Strong sell", "none": ""}

COLS = ["ticker", "name", "sector", "country", "currency", "index", "price", "target_mean", "target_low",
        "target_high", "upside", "upside_low", "spread", "n_analysts", "rec_mean", "rating", "vol", "below_high",
        "r3m", "rsi", "falling_knife", "held", "score", "targets_date"]


def falling_knife(m: pd.DataFrame) -> pd.Series:
    """Price below a falling trend (price < 50-day < 200-day average) and down more than 15% in 3 months."""
    return (m["price"] < m["sma50"]) & (m["sma50"] < m["sma200"]) & (m["r3m"] < KNIFE_DROP)


def _pct(x: pd.Series) -> pd.Series:
    return x.rank(pct=True)


def rank(metrics: pd.DataFrame, targets: pd.DataFrame, universe: pd.DataFrame, held: set[str],
         min_analysts: int = MIN_ANALYSTS, max_age_days: int = MAX_AGE_DAYS, now: datetime | None = None) -> pd.DataFrame:
    """Every stock with usable targets and a positive gap, best score first.

    score (0-100, percentile) = 0.50 upside to the mean target + 0.20 analysts agreeing (narrow low-high
    range) + 0.15 average rating + 0.15 number of analysts.
    """
    if metrics.empty or targets.empty:
        return pd.DataFrame(columns=COLS)
    now = now or datetime.now()
    t = targets.rename(columns={"mean": "target_mean", "low": "target_low", "high": "target_high",
                                "price": "yahoo_price", "currency": "target_currency"})
    d = metrics.merge(t, on="ticker", how="inner").merge(universe, on="ticker", how="left")
    d["targets_date"] = pd.to_datetime(d["fetched_at"])
    fresh = d["targets_date"] >= pd.Timestamp(now - timedelta(days=max_age_days))
    # Yahoo's own price must be in the same unit as ours (pence vs pounds, wrong listing): within 3x
    ratio = d["price"] / d["yahoo_price"].where(d["yahoo_price"] > 0)
    same_unit = ratio.isna() | ratio.between(1 / 3, 3)
    d = d[fresh & same_unit & (d["n_analysts"].fillna(0) >= min_analysts) & (d["target_mean"] > 0) & (d["price"] > 0)]
    d = d.copy()
    d["upside"] = d["target_mean"] / d["price"] - 1
    d = d[d["upside"] > 0]
    if d.empty:
        return pd.DataFrame(columns=COLS)
    d["upside_low"] = d["target_low"] / d["price"] - 1
    d["spread"] = (d["target_high"] - d["target_low"]) / d["target_mean"]
    raw = (0.50 * _pct(d["upside"]) + 0.20 * _pct(-d["spread"]).fillna(0.5)
           + 0.15 * _pct(-d["rec_mean"]).fillna(0.5) + 0.15 * _pct(d["n_analysts"]))
    d["score"] = raw.rank(pct=True) * 100
    d["rating"] = d["rec_key"].fillna("").map(lambda k: RATINGS.get(k, k.replace("_", " ").capitalize()))
    d["falling_knife"] = falling_knife(d)
    d["held"] = d["ticker"].isin(held)
    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    d["reasons"] = [reasons(r) for _, r in d.iterrows()]
    return d[[c for c in COLS if c in d.columns] + ["reasons"]]


def reasons(r: pd.Series) -> list[str]:
    out = [f"Average analyst target {r['target_mean']:,.2f} vs price {r['price']:,.2f}: {r['upside']:+.0%}",
           f"{int(r['n_analysts'])} analysts" + (f", average rating {r['rating']}" if r.get("rating") else "")
           + f"; targets range {r['target_low']:,.2f} to {r['target_high']:,.2f}"]
    if pd.notna(r.get("upside_low")):
        out.append("Even the lowest target is above the price" if r["upside_low"] > 0 else
                   f"The lowest target is {r['upside_low']:+.0%} from the price")
    if r.get("falling_knife"):
        out.append(f"Falling knife: {r['r3m']:+.0%} in 3 months, below its 50- and 200-day averages")
    else:
        out.append(f"{r['r3m']:+.0%} in 3 months, {-r['below_high']:.0%} below its 1-year high")
    out.append(f"Volatility {r['vol']:.0%} a year")
    return out


def stake_view(df: pd.DataFrame, stake: float, fee_base: float, rate_to_base: dict[str, float]) -> pd.DataFrame:
    """What a stake (in base currency) buys at today's price, whole shares only, with buy + sell fees.

    Adds: price_base, shares, invested, fees_pct (both fees / invested), break_even (price needed to get
    the money back after both fees, in the stock's currency), needed (break_even vs price), too_dear
    (cannot buy one share), fees_high (fees above FEE_WARN).
    """
    if df.empty:
        return df.assign(price_base=[], shares=[], invested=[], fees_pct=[], break_even=[], needed=[],
                         too_dear=[], fees_high=[])
    d = df.copy()
    rate = d["currency"].map(lambda c: rate_to_base.get(c, np.nan)).astype(float)
    d["price_base"] = d["price"] * rate
    d["shares"] = np.floor(stake / d["price_base"]).fillna(0)
    d["invested"] = d["shares"] * d["price_base"]
    has = d["shares"] > 0
    d["fees_pct"] = np.where(has, 2 * fee_base / d["invested"].where(has), np.nan)
    d["break_even"] = np.where(has, (d["invested"] + 2 * fee_base) / d["shares"].where(has) / rate, np.nan)
    d["needed"] = d["break_even"] / d["price"] - 1
    d["too_dear"] = ~has & rate.notna()
    d["fees_high"] = d["fees_pct"] > FEE_WARN
    return d


def narrow(df: pd.DataFrame, max_price: float | None = None, min_upside: float = 0.0, min_analysts: int = 0,
           hide_knives: bool = False, affordable_only: bool = False) -> pd.DataFrame:
    """Long-shot specific filters (sector/region/index/held go through ideas.apply_filters)."""
    if df.empty:
        return df
    d = df
    if max_price is not None:
        d = d[d["price"] <= max_price]
    d = d[(d["upside"] >= min_upside) & (d["n_analysts"] >= min_analysts)]
    if hide_knives:
        d = d[~d["falling_knife"].fillna(False).astype(bool)]
    if affordable_only and "too_dear" in d:
        d = d[~d["too_dear"].fillna(False).astype(bool)]
    return d
