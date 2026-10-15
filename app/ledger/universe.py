"""Index member lists (S&P 500/400/600, Nasdaq 100, Euro Stoxx 50, FTSE MIB) read from Wikipedia.

Lists are cached in the database and refreshed at most once a week; if Wikipedia cannot be reached
the last cached copy is used.
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timedelta

import pandas as pd

from .db import Store

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "StockLedger/0.1 (personal portfolio tool)"}
INDEXES = {
    "sp500": ("S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    "ndx": ("Nasdaq 100", "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"),
    "sx5e": ("Euro Stoxx 50", "https://en.wikipedia.org/wiki/EURO_STOXX_50"),
    "ftsemib": ("FTSE MIB", "https://en.wikipedia.org/wiki/FTSE_MIB"),
    "sp400": ("S&P 400 MidCap", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"),
    "sp600": ("S&P 600 SmallCap", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"),
}
US_LISTS = {"sp500": 400, "sp400": 350, "sp600": 550}       # same table layout; minimum rows expected
REFRESH_DAYS = 7

# Keyword -> GICS sector, for ICB names and sub-sector labels used on some pages.
_SECTOR_KEYWORDS = [
    ("real estate", "Real Estate"), ("reit", "Real Estate"),
    ("telecom", "Communication Services"), ("media", "Communication Services"),
    ("communication", "Communication Services"),
    ("semiconductor", "Information Technology"), ("software", "Information Technology"),
    ("technology", "Information Technology"), ("information", "Information Technology"),
    ("bank", "Financials"), ("insurance", "Financials"), ("financ", "Financials"),
    ("asset management", "Financials"), ("investment", "Financials"),
    ("oil", "Energy"), ("gas", "Energy"), ("energy", "Energy"),
    ("utilit", "Utilities"), ("electric", "Utilities"),
    ("pharma", "Health Care"), ("health", "Health Care"), ("medical", "Health Care"), ("biotech", "Health Care"),
    ("food", "Consumer Staples"), ("beverage", "Consumer Staples"), ("tobacco", "Consumer Staples"),
    ("staples", "Consumer Staples"), ("personal care", "Consumer Staples"), ("drug retail", "Consumer Staples"),
    ("automobile", "Consumer Discretionary"), ("auto", "Consumer Discretionary"), ("luxury", "Consumer Discretionary"),
    ("apparel", "Consumer Discretionary"), ("retail", "Consumer Discretionary"), ("travel", "Consumer Discretionary"),
    ("leisure", "Consumer Discretionary"), ("discretionary", "Consumer Discretionary"),
    ("chemical", "Materials"), ("material", "Materials"), ("steel", "Materials"), ("mining", "Materials"),
    ("metal", "Materials"), ("construction material", "Materials"),
    ("aerospace", "Industrials"), ("defense", "Industrials"), ("industrial", "Industrials"),
    ("construction", "Industrials"), ("transport", "Industrials"), ("engineering", "Industrials"),
]


def to_gics(label: str) -> str:
    s = str(label or "").strip().lower()
    if not s or s == "nan":
        return "Other"
    for key, sector in _SECTOR_KEYWORDS:
        if key in s:
            return sector
    return "Other"


def _tables(url: str) -> list[pd.DataFrame]:
    import requests

    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def _pick(tables: list[pd.DataFrame], need: str, min_rows: int) -> pd.DataFrame:
    for t in tables:
        t.columns = [re.sub(r"\[.*?\]", "", str(c)).strip() for c in t.columns]
        if need in t.columns and len(t) >= min_rows:
            return t
    raise ValueError(f"no table with column {need!r} and at least {min_rows} rows")


def fetch_index(key: str) -> pd.DataFrame:
    """One index as rows of ticker (Yahoo symbol), name, sector (GICS), country, currency."""
    _, url = INDEXES[key]
    tabs = _tables(url)
    if key in US_LISTS:
        t = _pick(tabs, "Symbol", US_LISTS[key])
        out = pd.DataFrame({"ticker": t["Symbol"].str.replace(".", "-", regex=False), "name": t["Security"],
                            "sector": t["GICS Sector"].map(to_gics), "country": "United States", "currency": "USD"})
    elif key == "ndx":
        t = _pick(tabs, "Ticker", 90)
        industry = next(c for c in t.columns if c.startswith("ICB Industry") or c.startswith("GICS Sector"))
        out = pd.DataFrame({"ticker": t["Ticker"].str.replace(".", "-", regex=False), "name": t["Company"],
                            "sector": t[industry].map(to_gics), "country": "United States", "currency": "USD"})
    elif key == "sx5e":
        t = _pick(tabs, "Ticker", 40)
        out = pd.DataFrame({"ticker": t["Ticker"], "name": t["Name"], "sector": t["Sector"].map(to_gics),
                            "country": t["Registered office"], "currency": "EUR"})
    elif key == "ftsemib":
        t = _pick(tabs, "Ticker", 30)
        sector_col = next(c for c in t.columns if "Sector" in c)
        out = pd.DataFrame({"ticker": t["Ticker"], "name": t["Company"], "sector": t[sector_col].map(to_gics),
                            "country": "Italy", "currency": "EUR"})
    else:
        raise KeyError(key)
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out = out[out["ticker"].str.match(r"^[A-Z0-9][A-Z0-9.\-]*$")]
    out["index"] = key
    return out.drop_duplicates("ticker").reset_index(drop=True)


def load_universe(store: Store, keys: list[str], force: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Members of the chosen indexes, one row per ticker (index column lists every index it is in).

    Returns (universe, notes) where notes explain any index served from cache or missing.
    """
    notes, frames = [], []
    cached = store.universe()
    for key in keys:
        mine = cached[cached["index"] == key]
        fresh = (not mine.empty and pd.Timestamp(mine["updated_at"].iloc[0])
                 > pd.Timestamp(datetime.now() - timedelta(days=REFRESH_DAYS)))
        if fresh and not force:
            frames.append(mine.drop(columns=["updated_at"]))
            continue
        try:
            df = fetch_index(key)
            store.save_universe(key, df)
            frames.append(df)
        except Exception as exc:  # network, page layout change
            log.warning("could not refresh %s: %s", key, exc)
            if mine.empty:
                notes.append(f"{INDEXES[key][0]}: could not download the member list and no saved copy exists, skipped.")
            else:
                notes.append(f"{INDEXES[key][0]}: using the saved member list from {mine['updated_at'].iloc[0][:10]}.")
                frames.append(mine.drop(columns=["updated_at"]))
    if not frames:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country", "currency", "index"]), notes
    allrows = pd.concat(frames, ignore_index=True)
    idx = allrows.groupby("ticker")["index"].apply(lambda s: ",".join(sorted(set(s))))
    uni = allrows.drop_duplicates("ticker").set_index("ticker")
    uni["index"] = idx
    return uni.reset_index(), notes
