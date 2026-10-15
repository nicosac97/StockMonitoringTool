"""Analyst price targets from Yahoo Finance, kept in the database and refreshed about once a week.

Yahoo answers one stock per request and refuses bursts, so each run fetches only the targets that are
missing or older than a week, a limited number per run, and stops early when Yahoo keeps refusing.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta

import pandas as pd

from .db import Store

log = logging.getLogger(__name__)

STALE_DAYS = 7
MAX_CALLS = 400          # per run; a first fill of ~1,600 stocks takes about four mornings
MAX_FAILURES = 15        # consecutive refusals before giving up for this run (rate limit)
SAVE_EVERY = 25


def fetch(ticker: str) -> dict | None:
    """Targets for one ticker. None when Yahoo refused or knows nothing about it."""
    import yfinance as yf

    info = yf.Ticker(ticker).info or {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not info or not (info.get("shortName") or info.get("longName") or price):
        return None
    return {"ticker": ticker, "mean": info.get("targetMeanPrice"), "median": info.get("targetMedianPrice"),
            "high": info.get("targetHighPrice"), "low": info.get("targetLowPrice"),
            "n_analysts": int(info.get("numberOfAnalystOpinions") or 0), "rec_mean": info.get("recommendationMean"),
            "rec_key": info.get("recommendationKey") or "", "currency": info.get("currency") or "", "price": price}


def due(cached: pd.DataFrame, tickers: list[str], stale_days: int = STALE_DAYS, now: datetime | None = None) -> list[str]:
    """Tickers needing a fetch: never fetched first, then the oldest; fresh ones left out."""
    now = now or datetime.now()
    when = dict(zip(cached["ticker"], pd.to_datetime(cached["fetched_at"]))) if not cached.empty else {}
    limit = pd.Timestamp(now - timedelta(days=stale_days))
    # never fetched: in a fixed mixed order, so a first fill covers all lists and letters, not A to C
    missing = sorted((t for t in tickers if t not in when), key=lambda t: hashlib.md5(t.encode()).hexdigest())
    old = sorted((t for t in tickers if t in when and when[t] < limit), key=lambda t: when[t])
    return missing + old


def refresh(store: Store, tickers: list[str], max_calls: int = MAX_CALLS, stale_days: int = STALE_DAYS,
            fetcher=fetch, pause_s: float = 0.15, max_failures: int = MAX_FAILURES, say=None) -> dict:
    """Fetch targets that are due. Returns counts: fetched, failed, pending (still due), stopped (rate limited)."""
    todo = due(store.targets(), sorted(set(tickers)), stale_days)
    batch, rows, fetched, failed, streak, stopped = todo[:max_calls], [], 0, 0, 0, False
    for i, t in enumerate(batch, 1):
        try:
            row = fetcher(t)
        except Exception as exc:  # Yahoo refusal ("Invalid Crumb", 401, 429) or network
            log.debug("targets %s failed: %s", t, exc)
            row = None
        if row is None:
            failed += 1
            streak += 1
            if streak >= max_failures:
                stopped = True
                log.warning("targets: %d refusals in a row, stopping after %d of %d", streak, i, len(batch))
                break
        else:
            streak = 0
            fetched += 1
            rows.append({**row, "fetched_at": datetime.now().isoformat(timespec="seconds")})
        if len(rows) >= SAVE_EVERY:
            store.save_targets(rows)
            rows = []
        if say and i % 20 == 0:
            say(i / len(batch), f"Analyst targets: {i} of {len(batch)} stocks")
        if pause_s:
            time.sleep(pause_s)
    if rows:
        store.save_targets(rows)
    return {"fetched": fetched, "failed": failed, "pending": len(todo) - fetched, "stopped": stopped}
