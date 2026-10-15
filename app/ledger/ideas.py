"""Filtering and grouping of scan results for the Daily ideas page. Pure functions."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EUROPE = {"Austria", "Belgium", "Denmark", "Finland", "France", "Germany", "Greece", "Ireland", "Luxembourg",
          "Netherlands", "Norway", "Portugal", "Spain", "Sweden", "Switzerland", "United Kingdom"}
REGIONS = ["United States", "Italy", "Europe (ex Italy)", "Other"]


def region_of(country: str) -> str:
    c = str(country or "").strip()
    if c in ("United States", "USA", "US"):
        return "United States"
    if c == "Italy":
        return "Italy"
    if c in EUROPE:
        return "Europe (ex Italy)"
    return "Other"


@dataclass
class Filters:
    sectors: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    indexes: list[str] = field(default_factory=list)
    min_score: float = 0.0
    max_rsi: float = 100.0
    max_vol: float | None = None          # fraction, 0.40 = 40% a year
    max_below_high: float | None = None   # fraction, 0.10 = at most 10% under the 1-year high
    hide_held: bool = False


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    if "region" not in d:
        d["region"] = d["country"].map(region_of)
    if f.sectors:
        d = d[d["sector"].isin(f.sectors)]
    if f.regions:
        d = d[d["region"].isin(f.regions)]
    if f.countries:
        d = d[d["country"].isin(f.countries)]
    if f.indexes:
        d = d[d["index"].fillna("").map(lambda v: any(k in v.split(",") for k in f.indexes))]
    if "score" in d:
        d = d[d["score"].fillna(0) >= f.min_score]
    if "rsi" in d:
        d = d[d["rsi"].fillna(0) <= f.max_rsi]
    if f.max_vol is not None and "vol" in d:
        d = d[d["vol"] <= f.max_vol]
    if f.max_below_high is not None and "below_high" in d:
        d = d[d["below_high"] >= -f.max_below_high]
    if f.hide_held and "held" in d:
        d = d[~d["held"].fillna(False).astype(bool)]
    return d


def ranked(df: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    return df.sort_values(by, ascending=False).head(n).reset_index(drop=True)


def best_per_sector(df: pd.DataFrame, by: str, k: int) -> pd.DataFrame:
    """Top k per sector; sectors ordered by their best stock, stocks by rank inside the sector."""
    if df.empty:
        return df
    d = df.sort_values(by, ascending=False).copy()
    d["sector_rank"] = d.groupby("sector").cumcount() + 1
    d = d[d["sector_rank"] <= k]
    order = d.groupby("sector")[by].max().sort_values(ascending=False).index
    d["sector"] = pd.Categorical(d["sector"], categories=list(order), ordered=True)
    d = d.sort_values(["sector", "sector_rank"])
    d["sector"] = d["sector"].astype(str)
    return d.reset_index(drop=True)


def breadth(scanned: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Per sector: stocks scanned, stocks in an uptrend and their share, after region/country/index filters."""
    if scanned.empty:
        return pd.DataFrame(columns=["sector", "scanned", "uptrend", "share"])
    geo = Filters(regions=f.regions, countries=f.countries, indexes=f.indexes, sectors=f.sectors)
    d = apply_filters(scanned, geo)
    g = d.groupby("sector").agg(scanned=("ticker", "size"), uptrend=("eligible", "sum")).reset_index()
    g["share"] = g["uptrend"] / g["scanned"]
    return g.sort_values("share", ascending=False).reset_index(drop=True)
