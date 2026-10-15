"""Market data from Yahoo Finance via yfinance: price history, FX and ticker lookup."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Yahoo sector names -> GICS sector names used across the app (and the web artifact).
YAHOO_TO_GICS = {
    "Technology": "Information Technology",
    "Communication Services": "Communication Services",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Energy": "Energy",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Industrials": "Industrials",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}

# Quotes Yahoo gives in minor units: (major currency, divisor).
MINOR_UNITS = {"GBp": ("GBP", 100), "GBX": ("GBP", 100), "ZAc": ("ZAR", 100), "ILA": ("ILS", 100)}


def major_currency(cur: str | None) -> tuple[str, float]:
    """Return (ISO currency, divisor) so price / divisor is in major units."""
    cur = (cur or "").strip()
    if cur in MINOR_UNITS:
        return MINOR_UNITS[cur]
    return cur.upper(), 1.0


def fx_ticker(cur: str, base: str) -> str:
    return f"{cur}{base}=X"


def empty_closes() -> pd.DataFrame:
    """No data, but with a date index so date filters and comparisons still work."""
    return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))


def download_closes(tickers: list[str], period: str = "2y", adjusted: bool = True,
                    start: str | None = None) -> pd.DataFrame:
    """Daily closes, one column per ticker. Missing tickers are dropped.

    start (YYYY-MM-DD) fetches from that date to today instead of a fixed period; fetching only what
    is needed is much faster than period="max".

    adjusted=True lowers past prices by later dividends and splits (right for returns and risk);
    adjusted=False gives the prices actually traded that day (right for comparing with your orders).
    """
    import yfinance as yf

    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return empty_closes()
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # unknown symbols are reported in the UI instead
    span = {"start": start} if start else {"period": period}
    try:
        raw = yf.download(tickers, auto_adjust=adjusted, progress=False, threads=True, group_by="column", **span)
    except Exception as exc:  # network errors, Yahoo throttling
        log.warning("yfinance download failed: %s", exc)
        return empty_closes()
    if raw is None or raw.empty:
        return empty_closes()
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    closes = closes.dropna(axis=1, how="all").sort_index()
    return closes


def fx_history(currencies: set[str], base: str, period: str = "2y", start: str | None = None) -> pd.DataFrame:
    """Rate series (1 unit of currency in base currency) for each foreign currency."""
    foreign = sorted({c for c in currencies if c and c != base})
    if not foreign:
        return empty_closes()
    closes = download_closes([fx_ticker(c, base) for c in foreign], period, start=start)
    return closes.rename(columns={fx_ticker(c, base): c for c in foreign})


def cap_bucket(market_cap_usd: float | None) -> str:
    if not market_cap_usd:
        return ""
    if market_cap_usd >= 200e9:
        return "Mega"
    if market_cap_usd >= 10e9:
        return "Large"
    if market_cap_usd >= 2e9:
        return "Mid"
    return "Small"


def lookup_basic(ticker: str) -> dict:
    """Name, currency and type from Yahoo's chart data: the same service that serves prices, and less
    often refused than the detailed company profile. No sector, country or size."""
    import yfinance as yf

    try:
        t = yf.Ticker(ticker)
        if t.history(period="5d").empty:
            return {}
        meta = t.history_metadata or {}
    except Exception as exc:
        log.warning("basic lookup %s failed: %s", ticker, exc)
        return {}
    if not meta.get("currency"):
        return {}
    return {"ticker": ticker.upper(), "name": meta.get("longName") or meta.get("shortName") or "",
            "sector": "ETF / Fund" if meta.get("instrumentType") in ("ETF", "MUTUALFUND") else "",
            "industry": "", "country": "", "currency": meta["currency"], "cap_bucket": "",
            "source": "basic"}


def lookup(ticker: str) -> dict:
    """Name, sector, country, currency and size for a ticker; falls back to lookup_basic() when the
    detailed profile is unavailable. Empty dict when Yahoo has nothing at all."""
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        log.warning("lookup %s failed: %s", ticker, exc)
        info = {}
    if not info or not (info.get("shortName") or info.get("longName")):
        return lookup_basic(ticker)
    cur = info.get("currency") or ""
    mcap = info.get("marketCap")
    iso, _ = major_currency(cur)
    if mcap and iso and iso != "USD":
        rate = download_closes([fx_ticker(iso, "USD")], "5d")
        mcap = mcap * float(rate.iloc[-1, 0]) if not rate.empty else None
    quote_type = info.get("quoteType", "")
    sector = "ETF / Fund" if quote_type in ("ETF", "MUTUALFUND") else YAHOO_TO_GICS.get(info.get("sector", ""), "")
    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName") or "",
        "sector": sector,
        "industry": info.get("industry", "") or "",
        "country": info.get("country", "") or "",
        "currency": cur,
        "cap_bucket": cap_bucket(mcap),
    }
