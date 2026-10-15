"""SQLite storage for holdings, watchlist, manual price log and settings."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .prices import MINOR_UNITS

DEFAULT_DB = Path(os.environ.get("STOCK_LEDGER_DB")
                  or Path(__file__).resolve().parent.parent / "data" / "portfolio.db")

HOLDING_COLS = ["ticker", "name", "shares", "avg_cost", "currency", "sector", "industry",
                "country", "cap_bucket", "added_at", "note"]
WATCH_COLS = ["ticker", "name", "currency", "target_price", "sector", "country", "note", "added_at"]

DEFAULT_SETTINGS = {
    "base_currency": "EUR",
    "benchmark": "URTH",          # iShares MSCI World ETF, a broad global yardstick
    "risk_free_pct": 2.0,
    "sector_targets": {},         # sector -> percent
    "fx_overrides": {},           # currency -> rate to base, used when Yahoo has no quote
    "plan": {"start": 0.0, "monthly": 300.0, "years": 20, "returns": [3.0, 6.0, 9.0]},
    "fee": {"amount": 1.12, "currency": "USD"},          # broker commission per buy and per sell order
    "alerts": {"stock_pct": 25.0, "portfolio_pct": 15.0},  # loss vs money paid that triggers a sell alert
    "scan": {"indexes": ["sp500", "ndx", "sx5e", "ftsemib"], "top_n": 10, "commentary": True,
             "longshot_indexes": ["sp500", "ndx", "sx5e", "ftsemib", "sp400", "sp600"]},
    "longshots": {"stake": 150.0, "cheap_price": 20.0, "min_analysts": 3},
}

TARGET_COLS = ["ticker", "mean", "median", "high", "low", "n_analysts", "rec_mean", "rec_key", "currency", "price",
               "fetched_at"]
ORDER_COLS = ["ticker", "side", "day", "shares", "price", "fee", "fee_currency", "note"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings (
    ticker TEXT PRIMARY KEY,
    name TEXT DEFAULT '',
    shares REAL NOT NULL,
    avg_cost REAL NOT NULL,
    currency TEXT DEFAULT '',
    sector TEXT DEFAULT '',
    industry TEXT DEFAULT '',
    country TEXT DEFAULT '',
    cap_bucket TEXT DEFAULT '',
    added_at TEXT DEFAULT '',
    note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY,
    name TEXT DEFAULT '',
    currency TEXT DEFAULT '',
    target_price REAL,
    sector TEXT DEFAULT '',
    country TEXT DEFAULT '',
    note TEXT DEFAULT '',
    added_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS price_log (
    ticker TEXT NOT NULL,
    day TEXT NOT NULL,
    price REAL NOT NULL,
    PRIMARY KEY (ticker, day)
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    day TEXT NOT NULL,
    shares REAL NOT NULL CHECK (shares > 0),
    price REAL NOT NULL CHECK (price >= 0),
    fee REAL NOT NULL DEFAULT 0,
    fee_currency TEXT NOT NULL DEFAULT 'USD',
    note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS universe (
    "index" TEXT NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT DEFAULT '',
    sector TEXT DEFAULT '',
    country TEXT DEFAULT '',
    currency TEXT DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY ("index", ticker)
);
CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    status TEXT NOT NULL,
    n_universe INTEGER, n_scanned INTEGER, n_eligible INTEGER,
    duration_s REAL,
    notes TEXT DEFAULT '',
    commentary TEXT DEFAULT '',
    commentary_note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS scan_picks (
    run_id INTEGER NOT NULL,
    list TEXT NOT NULL,
    rank INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (run_id, list, rank)
);
CREATE TABLE IF NOT EXISTS targets (
    ticker TEXT PRIMARY KEY,
    mean REAL, median REAL, high REAL, low REAL,
    n_analysts INTEGER DEFAULT 0,
    rec_mean REAL,
    rec_key TEXT DEFAULT '',
    currency TEXT DEFAULT '',
    price REAL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def normalize_currency(cur: str | None) -> str:
    """Upper-case ISO codes but keep Yahoo's minor-unit codes (GBp = pence) as they are."""
    cur = (cur or "").strip()
    return cur if cur in MINOR_UNITS else cur.upper()


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- holdings ----
    def holdings(self) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM holdings ORDER BY ticker", c)
        return df

    def upsert_holding(self, h: dict) -> None:
        row = {k: h.get(k) for k in HOLDING_COLS}
        row["ticker"] = str(row["ticker"]).strip().upper()
        row["currency"] = normalize_currency(row.get("currency"))
        row["added_at"] = row.get("added_at") or date.today().isoformat()
        for k in HOLDING_COLS:
            if row[k] is None and k not in ("shares", "avg_cost"):
                row[k] = ""
        with self._conn() as c:
            c.execute(
                f"INSERT OR REPLACE INTO holdings ({','.join(HOLDING_COLS)}) "
                f"VALUES ({','.join('?' * len(HOLDING_COLS))})",
                [row[k] for k in HOLDING_COLS],
            )

    def delete_holding(self, ticker: str) -> None:
        """Remove a holding together with its orders."""
        with self._conn() as c:
            c.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
            c.execute("DELETE FROM orders WHERE ticker = ?", (ticker,))

    # ---- orders ----
    def orders(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM orders ORDER BY day, id", c)

    def add_order(self, o: dict) -> int:
        side = str(o["side"]).strip().upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {o['side']!r}")
        row = {"ticker": str(o["ticker"]).strip().upper(), "side": side, "day": str(o["day"])[:10],
               "shares": float(o["shares"]), "price": float(o["price"]), "fee": float(o.get("fee") or 0),
               "fee_currency": (o.get("fee_currency") or "USD").upper(), "note": o.get("note") or ""}
        with self._conn() as c:
            cur = c.execute(f"INSERT INTO orders ({','.join(ORDER_COLS)}) VALUES ({','.join('?' * len(ORDER_COLS))})",
                            [row[k] for k in ORDER_COLS])
            return int(cur.lastrowid)

    def delete_orders(self, ids: list[int]) -> None:
        with self._conn() as c:
            c.executemany("DELETE FROM orders WHERE id = ?", [(int(i),) for i in ids])

    # ---- watchlist ----
    def watchlist(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM watchlist ORDER BY ticker", c)

    def upsert_watch(self, w: dict) -> None:
        row = {k: w.get(k) for k in WATCH_COLS}
        row["ticker"] = str(row["ticker"]).strip().upper()
        row["currency"] = normalize_currency(row.get("currency"))
        row["added_at"] = row.get("added_at") or date.today().isoformat()
        for k in WATCH_COLS:
            if row[k] is None and k != "target_price":
                row[k] = ""
        with self._conn() as c:
            c.execute(
                f"INSERT OR REPLACE INTO watchlist ({','.join(WATCH_COLS)}) "
                f"VALUES ({','.join('?' * len(WATCH_COLS))})",
                [row[k] for k in WATCH_COLS],
            )

    def delete_watch(self, ticker: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
            c.execute("DELETE FROM price_log WHERE ticker = ?", (ticker,))

    # ---- manual price log (from the artifact, or for tickers Yahoo does not cover) ----
    def log_price(self, ticker: str, day: str, price: float) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO price_log (ticker, day, price) VALUES (?,?,?)",
                      (ticker.upper(), day, float(price)))

    def price_log(self, ticker: str | None = None) -> pd.DataFrame:
        with self._conn() as c:
            if ticker:
                return pd.read_sql_query("SELECT * FROM price_log WHERE ticker = ? ORDER BY day",
                                         c, params=(ticker.upper(),))
            return pd.read_sql_query("SELECT * FROM price_log ORDER BY ticker, day", c)

    # ---- index members ----
    def universe(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query('SELECT * FROM universe', c)

    def save_universe(self, key: str, df: pd.DataFrame) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute('DELETE FROM universe WHERE "index" = ?', (key,))
            c.executemany('INSERT OR REPLACE INTO universe ("index", ticker, name, sector, country, currency, updated_at) '
                          'VALUES (?,?,?,?,?,?,?)',
                          [(key, r.ticker, r.name, r.sector, r.country, r.currency, now) for r in df.itertuples()])

    # ---- daily scans ----
    def save_scan(self, run: dict, picks: dict[str, list[dict]]) -> int:
        cols = ["run_at", "status", "n_universe", "n_scanned", "n_eligible", "duration_s", "notes",
                "commentary", "commentary_note"]
        with self._conn() as c:
            cur = c.execute(f"INSERT INTO scan_runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                            [run.get(k) for k in cols])
            run_id = int(cur.lastrowid)
            c.executemany("INSERT INTO scan_picks (run_id, list, rank, ticker, data) VALUES (?,?,?,?,?)",
                          [(run_id, lst, i + 1, p["ticker"], json.dumps(p, default=str))
                           for lst, rows in picks.items() for i, p in enumerate(rows)])
        return run_id

    def add_picks(self, run_id: int, picks: dict[str, list[dict]]) -> None:
        """Add (or replace) lists of picks on an existing run."""
        with self._conn() as c:
            c.executemany("DELETE FROM scan_picks WHERE run_id = ? AND list = ?", [(run_id, lst) for lst in picks])
            c.executemany("INSERT INTO scan_picks (run_id, list, rank, ticker, data) VALUES (?,?,?,?,?)",
                          [(run_id, lst, i + 1, p["ticker"], json.dumps(p, default=str))
                           for lst, rows in picks.items() for i, p in enumerate(rows)])

    def update_scan(self, run_id: int, **fields) -> None:
        with self._conn() as c:
            c.execute(f"UPDATE scan_runs SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                      [*fields.values(), run_id])

    def scan_runs(self, limit: int = 60) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM scan_runs ORDER BY id DESC LIMIT ?", c, params=(limit,))

    def scan_picks(self, run_id: int) -> dict[str, list[dict]]:
        with self._conn() as c:
            rows = c.execute("SELECT list, data FROM scan_picks WHERE run_id = ? ORDER BY list, rank", (run_id,)).fetchall()
        out: dict[str, list[dict]] = {}
        for lst, data in rows:
            out.setdefault(lst, []).append(json.loads(data))
        return out

    # ---- analyst price targets ----
    def targets(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM targets", c)

    def save_targets(self, rows: list[dict]) -> None:
        cols = TARGET_COLS
        with self._conn() as c:
            c.executemany(f"INSERT OR REPLACE INTO targets ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                          [[r.get(k) for k in cols] for r in rows])

    # ---- settings ----
    def settings(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT key, value FROM settings").fetchall()
        out = json.loads(json.dumps(DEFAULT_SETTINGS))
        for k, v in rows:
            v = json.loads(v)
            # dict settings: stored values over defaults, so keys added in later versions are always present
            out[k] = {**out[k], **v} if isinstance(out.get(k), dict) and isinstance(v, dict) and k not in ("sector_targets", "fx_overrides") else v
        return out

    def set_setting(self, key: str, value) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                      (key, json.dumps(value)))

    def wipe(self) -> None:
        with self._conn() as c:
            for t in ("holdings", "watchlist", "price_log", "orders", "settings"):
                c.execute(f"DELETE FROM {t}")
