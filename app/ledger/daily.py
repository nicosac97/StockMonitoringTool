"""The daily scan: download prices for the chosen indexes, screen, rank, store, comment, notify."""
from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from . import analytics as A
from . import commentary, longshots, prices, screener, targets
from .db import Store
from .universe import INDEXES, load_universe

log = logging.getLogger(__name__)


DEFAULT_SCAN = {"indexes": ["sp500", "ndx", "sx5e", "ftsemib"], "top_n": 10, "commentary": True,
                "longshot_indexes": ["sp500", "ndx", "sx5e", "ftsemib", "sp400", "sp600"]}


def scan_settings(settings: dict) -> dict:
    """Scan settings with defaults filled in (also safe when an older settings module is loaded)."""
    return {**DEFAULT_SCAN, **(settings.get("scan") or {})}


def current_positions(store: Store, closes: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Positions exactly as the app computes them (orders first, hand-entered holdings otherwise)."""
    s = store.settings()
    holdings, orders = store.holdings(), store.orders()
    quote_cur = dict(zip(holdings["ticker"], holdings["currency"]))
    book = A.order_book(orders, quote_cur, fx, s["base_currency"], s["fx_overrides"])
    held_now = A.holdings_with_orders(holdings, book)
    return A.positions(held_now, closes, fx, s["base_currency"], s["fx_overrides"])


def _plain(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        elif isinstance(v, pd.Timestamp):
            out[k] = v.date().isoformat()
        else:
            out[k] = v
    return out


MIN_SHARE = 0.5                   # below this share of stocks downloaded, the scan counts as failed
RETRY_WAIT_S = (60, 120, 240)     # after waking from sleep the network can take minutes to come back


def download_with_retry(tickers: list[str], say, wait_s=RETRY_WAIT_S) -> pd.DataFrame:
    closes = prices.download_closes(tickers, "2y")
    for i, w in enumerate(wait_s, 1):
        if len(closes.columns) >= MIN_SHARE * len(tickers):
            break
        say(0.10, f"Only {len(closes.columns)} of {len(tickers)} stocks downloaded; retrying in {w} s ({i}/{len(wait_s)})")
        log.warning("download got %d of %d, retry %d in %ds", len(closes.columns), len(tickers), i, w)
        time.sleep(w)
        retry = prices.download_closes(tickers, "2y")
        if len(retry.columns) > len(closes.columns):
            closes = retry
    return closes


def longshot_step(store: Store, run_id: int, held: set[str], closes: pd.DataFrame | None = None,
                  metrics: pd.DataFrame | None = None, say=None, top_n: int = 10,
                  max_calls: int = targets.MAX_CALLS) -> tuple[list[dict], str]:
    """Rank long shots and attach them to scan run_id. Reuses prices and metrics already computed.

    Returns (top picks, note for the run). Never raises: a failure only costs the long-shot lists.
    """
    say = say or (lambda f, m: log.info("%3.0f%% %s", f * 100, m))
    try:
        s = store.settings()
        cfg = scan_settings(s)
        uni, notes = load_universe(store, cfg["longshot_indexes"])
        if uni.empty:
            return [], "Long shots: no index member lists available."
        have = set(metrics["ticker"]) if metrics is not None and not metrics.empty else set()
        extra = sorted(set(uni["ticker"]) - have - (set(closes.columns) if closes is not None else set()))
        frames = [metrics[screener.METRIC_COLS]] if have else []
        if closes is not None:
            known = [t for t in uni["ticker"] if t in closes.columns and t not in have]
            if known:
                frames.append(screener.metrics(closes[known]))
        if extra:
            say(0.80, f"Long shots: downloading prices for {len(extra)} more stocks (mid and small caps)")
            more = prices.download_closes(extra, "2y")
            if not more.empty:
                frames.append(screener.metrics(more))
        m = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        if m.empty:
            return [], "Long shots: no price data."
        m = m.drop_duplicates("ticker")
        say(0.82, "Long shots: collecting analyst price targets")
        stats = targets.refresh(store, m["ticker"].tolist(), max_calls=max_calls,
                                say=lambda f, msg: say(0.82 + 0.03 * f, msg))
        min_analysts = int((s.get("longshots") or {}).get("min_analysts", longshots.MIN_ANALYSTS))
        ranked = longshots.rank(m, store.targets(), uni[["ticker", "name", "sector", "country", "currency", "index"]],
                                held, min_analysts=min_analysts)
        rows = [_plain(r) for r in ranked.to_dict("records")]
        store.add_picks(run_id, {"longshot": rows[:top_n], "longshot_all": rows})
        have_targets = len(m) - stats["pending"]
        note = f"Long shots: {len(rows)} stocks below their analyst target"
        if stats["pending"]:
            note += (f"; analyst targets collected for {have_targets} of {len(m)} stocks so far, the rest follow "
                     "in the next runs")
        if stats["stopped"]:
            note += " (Yahoo refused more requests today)"
        return rows[:top_n], " ".join(notes + [note + "."])
    except Exception as exc:  # never let this step break the morning scan
        log.exception("long shots failed")
        return [], f"Long shots failed: {exc}"


def run_scan(store: Store | None = None, with_commentary: bool | None = None, notify: bool = True,
             progress=None, retry_wait_s=RETRY_WAIT_S) -> int:
    """Run one scan and return its run id. progress(fraction, message) is called along the way."""
    store = store or Store()
    say = progress or (lambda f, m: log.info("%3.0f%% %s", f * 100, m))
    t0 = time.time()
    s = store.settings()
    cfg = scan_settings(s)
    base = s["base_currency"]
    if with_commentary is None:
        with_commentary = bool(cfg.get("commentary", True))

    say(0.02, "Loading index member lists")
    uni, notes = load_universe(store, cfg["indexes"])
    held = set(store.holdings()["ticker"]) | set(store.orders()["ticker"])
    tickers = sorted(set(uni["ticker"]) | held)
    if not tickers:
        run_id = store.save_scan({"run_at": datetime.now().isoformat(timespec="seconds"), "status": "failed",
                                  "notes": " ".join(notes) or "No stocks to scan."}, {})
        return run_id

    say(0.10, f"Downloading 2 years of prices for {len(tickers)} stocks (about a minute)")
    closes = download_with_retry(tickers, say, wait_s=retry_wait_s)
    got = len(closes.columns)
    if got < MIN_SHARE * len(tickers):
        msg = (f"Price download failed: got {got} of {len(tickers)} stocks after {len(retry_wait_s) + 1} attempts "
               "(no internet connection, or Yahoo Finance not answering).")
        log.error(msg)
        run_id = store.save_scan({"run_at": datetime.now().isoformat(timespec="seconds"), "status": "failed",
                                  "n_universe": len(uni), "n_scanned": 0, "n_eligible": 0,
                                  "duration_s": round(time.time() - t0, 1), "notes": msg}, {})
        if notify:
            toast("Stock Ledger: morning scan failed", "No price data could be downloaded. Open Daily ideas and press "
                  "Run scan now when you are online.")
        return run_id
    if got < len(tickers):
        notes.append(f"Prices missing for {len(tickers) - got} of {len(tickers)} stocks.")
    currencies = {prices.major_currency(c)[0] for c in store.holdings()["currency"]} | {s["fee"]["currency"]}
    fx = prices.fx_history({c for c in currencies if c}, base, "2y")

    say(0.75, "Scoring trend and momentum")
    universe_cols = uni[["ticker", "name", "sector", "country", "currency", "index"]]
    scored = screener.score(screener.metrics(closes[[t for t in uni["ticker"] if t in closes.columns]]))
    pos = current_positions(store, closes, fx)
    top_n = int(cfg.get("top_n", 10))
    alone = screener.standalone(scored, universe_cols, set(pos["ticker"]) if not pos.empty else held, n=top_n)
    fit = screener.portfolio_fit(scored, universe_cols, closes, pos, s["sector_targets"], n=top_n)
    missing = len(uni) - int(len(scored))
    if missing > 0:
        notes.append(f"{missing} stocks skipped (no data or less than a year of history).")

    every = screener.standalone(scored, universe_cols, set(pos["ticker"]) if not pos.empty else held,
                                n=10_000, per_sector=10_000)          # all stocks in an uptrend, for filtering
    fit_all = screener.portfolio_fit(scored, universe_cols, closes, pos, s["sector_targets"],
                                     n=10_000, per_sector=10_000)    # every candidate, for filtering
    scanned = scored[["ticker", "eligible"]].merge(universe_cols[["ticker", "sector", "country", "index"]],
                                                   on="ticker", how="left") if not scored.empty else scored
    picks = {"standalone": [_plain(r) for r in alone.to_dict("records")],
             "portfolio": [_plain(r) for r in fit.to_dict("records")],
             "all": [_plain(r) for r in every.to_dict("records")],
             "portfolio_all": [_plain(r) for r in fit_all.to_dict("records")],
             "scanned": [_plain(r) for r in scanned.to_dict("records")]}
    run = {"run_at": datetime.now().isoformat(timespec="seconds"), "status": "ok", "n_universe": len(uni),
           "n_scanned": int(len(scored)), "n_eligible": int(scored["eligible"].sum()) if not scored.empty else 0,
           "duration_s": round(time.time() - t0, 1), "notes": " ".join(notes), "commentary": "", "commentary_note": ""}
    run_id = store.save_scan(run, picks)

    shots, ls_note = longshot_step(store, run_id, set(pos["ticker"]) if not pos.empty else held, closes,
                                   scored[screener.METRIC_COLS] if not scored.empty else None, say, top_n)
    run["notes"] = " ".join(x for x in (run["notes"], ls_note) if x)
    run["duration_s"] = round(time.time() - t0, 1)
    store.update_scan(run_id, notes=run["notes"], duration_s=run["duration_s"])

    if with_commentary:
        say(0.85, "Asking Claude for commentary")
        valued = pos.dropna(subset=["value"]) if not pos.empty else pos
        summary = {"base_currency": base, "positions": len(valued),
                   "effective_positions": round(A.effective_n(valued["weight"]), 1) if len(valued) else 0,
                   "by_sector_pct": {r.sector: round(r.weight * 100, 1) for r in A.group_weights(valued, "sector").itertuples()} if len(valued) else {},
                   "by_country_pct": {r.country: round(r.weight * 100, 1) for r in A.group_weights(valued, "country").itertuples()} if len(valued) else {},
                   "broker_fee_per_order": s["fee"]}
        text, note = commentary.write_commentary(picks["standalone"], picks["portfolio"], summary, shots[:5])
        store.update_scan(run_id, commentary=text, commentary_note=note)

    say(1.0, "Done")
    if notify:
        top = ", ".join(p["ticker"] for p in picks["standalone"][:3]) or "no stock passed the trend filter"
        shot = f" Top long shot: {shots[0]['ticker']} ({shots[0]['upside']:+.0%} to target)." if shots else ""
        toast("Stock Ledger: daily ideas ready",
              f"{run['n_eligible']} of {run['n_scanned']} stocks in an uptrend. Top momentum: {top}.{shot}")
    return run_id


def toast(title: str, body: str, url: str = "http://localhost:8501/ideas") -> bool:
    """Windows notification through PowerShell; clicking it opens the Daily ideas page. Never raises."""
    xml = (f"<toast activationType='protocol' launch='{escape(url)}'><visual><binding template='ToastGeneric'>"
           f"<text>{escape(title)}</text><text>{escape(body)}</text></binding></visual></toast>")
    ps = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null;"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null;"
        "$x = New-Object Windows.Data.Xml.Dom.XmlDocument; $x.LoadXml($env:SL_TOAST_XML);"
        "$t = [Windows.UI.Notifications.ToastNotification]::new($x);"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        "'{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe').Show($t)"
    )
    try:
        import os
        env = {**os.environ, "SL_TOAST_XML": xml}
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], env=env,
                           capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode == 0
    except Exception as exc:
        log.warning("notification failed: %s", exc)
        return False


def index_names(keys: list[str]) -> str:
    return ", ".join(INDEXES[k][0] for k in keys if k in INDEXES)
