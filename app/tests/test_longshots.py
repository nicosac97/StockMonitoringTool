import math
from datetime import datetime, timedelta

import pandas as pd
import pytest

from ledger import longshots as L
from ledger import targets as T
from ledger.db import Store

NOW = datetime(2026, 10, 15, 8, 0)
FRESH = (NOW - timedelta(days=2)).isoformat()


def metrics(rows):
    # ticker, price, sma50, sma200, r3m
    return pd.DataFrame([{"ticker": t, "price": p, "sma50": s50, "sma200": s200, "r3m": r3, "vol": 0.4,
                          "below_high": -0.3, "rsi": 45.0} for t, p, s50, s200, r3 in rows])


def tgt(rows):
    # ticker, mean, low, high, n, rec_mean, fetched_at, yahoo price
    return pd.DataFrame([{"ticker": t, "mean": m, "median": m, "low": lo, "high": hi, "n_analysts": n, "rec_mean": rm,
                          "rec_key": "buy", "currency": "USD", "price": yp, "fetched_at": f}
                         for t, m, lo, hi, n, rm, f, yp in rows])


UNI = pd.DataFrame({"ticker": list("ABCDEFG"), "name": list("ABCDEFG"), "sector": ["Energy"] * 7,
                    "country": ["United States"] * 7, "currency": ["USD"] * 7, "index": ["sp600"] * 7})


def test_rank_orders_by_gap_and_drops_unusable_rows():
    m = metrics([("A", 10, 11, 12, 0.0), ("B", 10, 11, 12, 0.0), ("C", 10, 11, 12, 0.0), ("D", 10, 11, 12, 0.0),
                 ("E", 10, 11, 12, 0.0), ("F", 10, 11, 12, 0.0), ("G", 10, 11, 12, 0.0)])
    t = tgt([("A", 20, 15, 25, 10, 1.8, FRESH, 10),     # +100%
             ("B", 13, 11, 15, 10, 1.8, FRESH, 10),     # +30%
             ("C", 30, 15, 45, 2, 1.8, FRESH, 10),      # too few analysts
             ("D", 9, 7, 11, 10, 2.5, FRESH, 10),       # below price
             ("E", 30, 20, 40, 10, 1.8, (NOW - timedelta(days=40)).isoformat(), 10),   # stale targets
             ("F", 30, 20, 40, 10, 1.8, FRESH, 1000),   # Yahoo price in pence: unit mismatch
             ("G", 15, 12, 18, 10, 1.5, FRESH, 10)])    # +50%
    out = L.rank(m, t, UNI, held={"B"}, now=NOW)
    assert out["ticker"].tolist() == ["A", "G", "B"]
    a = out.set_index("ticker")
    assert a.loc["A", "upside"] == pytest.approx(1.0) and a.loc["A", "upside_low"] == pytest.approx(0.5)
    assert a.loc["B", "held"] and not a.loc["A", "held"]
    assert a.loc["A", "rating"] == "Buy"
    assert any("Even the lowest target" in r for r in a.loc["A", "reasons"])


def test_falling_knife_flag():
    m = metrics([("A", 8, 10, 12, -0.30), ("B", 8, 10, 12, -0.05), ("C", 14, 12, 10, -0.30)])
    k = L.falling_knife(m).tolist()
    assert k == [True, False, False]          # only a deep drop below falling averages


def test_stake_view_fees_and_break_even():
    df = pd.DataFrame({"ticker": ["A", "B", "C"], "price": [10.0, 400.0, 5.0], "currency": ["USD", "USD", "CHF"]})
    v = L.stake_view(df, stake=150, fee_base=1.0, rate_to_base={"USD": 0.5}).set_index("ticker")
    # A: 10 USD = 5 EUR -> 30 shares, 150 EUR invested, fees 2 EUR
    assert v.loc["A", "shares"] == 30 and v.loc["A", "fees_pct"] == pytest.approx(2 / 150)
    assert v.loc["A", "break_even"] == pytest.approx((150 + 2) / 30 / 0.5)
    assert v.loc["A", "needed"] == pytest.approx(152 / 150 - 1)
    assert v.loc["B", "too_dear"] and v.loc["B", "shares"] == 0          # 200 EUR a share
    assert not v.loc["C", "too_dear"] and math.isnan(v.loc["C", "break_even"])   # no rate: unknown, not "too dear"
    small = L.stake_view(df, stake=50, fee_base=1.0, rate_to_base={"USD": 0.5}).set_index("ticker")
    assert small.loc["A", "fees_high"]                                   # 2 EUR on 50 EUR = 4%


def test_narrow_filters():
    df = pd.DataFrame({"ticker": list("ABC"), "price": [5.0, 50.0, 8.0], "upside": [0.5, 0.8, 0.1],
                       "n_analysts": [5, 12, 8], "falling_knife": [True, False, False], "too_dear": [False, True, False]})
    assert L.narrow(df, max_price=20)["ticker"].tolist() == ["A", "C"]
    assert L.narrow(df, min_upside=0.3, min_analysts=6)["ticker"].tolist() == ["B"]
    assert L.narrow(df, hide_knives=True, affordable_only=True)["ticker"].tolist() == ["C"]


def test_due_puts_missing_first_then_oldest():
    cached = pd.DataFrame({"ticker": ["A", "B", "C"], "fetched_at": [(NOW - timedelta(days=20)).isoformat(),
                                                                     (NOW - timedelta(days=1)).isoformat(),
                                                                     (NOW - timedelta(days=9)).isoformat()]})
    assert T.due(cached, ["A", "B", "C", "D"], 7, NOW) == ["D", "A", "C"]


def test_refresh_caps_calls_saves_and_stops_on_refusals(tmp_path):
    s = Store(tmp_path / "t.db")
    calls = []

    def ok(t):
        calls.append(t)
        return {"ticker": t, "mean": 10.0, "n_analysts": 4}

    r = T.refresh(s, ["A", "B", "C"], max_calls=2, fetcher=ok, pause_s=0)
    assert len(calls) == 2 and r == {"fetched": 2, "failed": 0, "pending": 1, "stopped": False}
    assert set(s.targets()["ticker"]) == set(calls)
    T.refresh(s, ["A", "B", "C"], fetcher=ok, pause_s=0)
    assert sorted(calls) == ["A", "B", "C"]                    # the first two are fresh: not fetched again

    def refused(t):
        raise RuntimeError("Invalid Crumb")

    r = T.refresh(s, list("DEFGH"), fetcher=refused, pause_s=0, max_failures=3)
    assert r["stopped"] and r["failed"] == 3 and r["pending"] == 5


def test_add_picks_replaces_list(tmp_path):
    s = Store(tmp_path / "p.db")
    rid = s.save_scan({"run_at": "2026-10-15T08:00:00", "status": "ok"}, {"standalone": [{"ticker": "X"}]})
    s.add_picks(rid, {"longshot": [{"ticker": "A"}, {"ticker": "B"}]})
    s.add_picks(rid, {"longshot": [{"ticker": "C"}]})
    p = s.scan_picks(rid)
    assert [x["ticker"] for x in p["longshot"]] == ["C"] and p["standalone"][0]["ticker"] == "X"
    assert s.settings()["longshots"]["stake"] == 150.0
