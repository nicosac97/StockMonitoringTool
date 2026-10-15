import math

import numpy as np
import pandas as pd
import pytest

from ledger import analytics as A


def make_holdings(rows):
    cols = ["ticker", "name", "shares", "avg_cost", "currency", "sector", "industry", "country", "cap_bucket"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


@pytest.fixture
def three_equal():
    h = make_holdings([
        ("AAA", "A", 10, 10, "EUR", "Energy", "", "Italy", "Large"),
        ("BBB", "B", 10, 10, "EUR", "Financials", "", "Italy", "Large"),
        ("CCC", "C", 10, 10, "USD", "Utilities", "", "United States", "Mid"),
    ])
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    closes = pd.DataFrame({"AAA": [10, 10, 10], "BBB": [9, 9.5, 10], "CCC": [11, 11, 12.5]}, index=idx)
    fx = pd.DataFrame({"USD": [0.8, 0.8, 0.8]}, index=idx)
    return h, closes, fx


def test_positions_value_weight_and_fx(three_equal):
    h, closes, fx = three_equal
    pos = A.positions(h, closes, fx, "EUR")
    by = pos.set_index("ticker")
    assert by.loc["AAA", "value"] == pytest.approx(100)
    assert by.loc["CCC", "value"] == pytest.approx(10 * 12.5 * 0.8)   # 100 EUR
    assert pos["weight"].sum() == pytest.approx(1)
    assert by.loc["BBB", "day_change"] == pytest.approx(10 / 9.5 - 1)
    assert by.loc["CCC", "pl_pct"] == pytest.approx(0.25)


def test_equal_weights_hhi_and_effective_n(three_equal):
    h, closes, fx = three_equal
    pos = A.positions(h, closes, fx, "EUR")
    assert A.hhi(pos["weight"]) == pytest.approx(1 / 3)
    assert A.effective_n(pos["weight"]) == pytest.approx(3)


def test_missing_fx_excluded_and_reported(three_equal):
    h, closes, _ = three_equal
    pos = A.positions(h, closes, None, "EUR")
    assert math.isnan(pos.set_index("ticker").loc["CCC", "value"])
    assert pos["weight"].sum() == pytest.approx(1)
    assert any(lvl == "error" and "USD" in msg for lvl, msg in A.checks(pos))


def test_fx_override_used_when_no_quote(three_equal):
    h, closes, _ = three_equal
    pos = A.positions(h, closes, None, "EUR", fx_overrides={"USD": 0.5})
    assert pos.set_index("ticker").loc["CCC", "value"] == pytest.approx(62.5)


def test_pence_quotes_converted_to_pounds():
    h = make_holdings([("VOD.L", "Vodafone", 100, 80, "GBp", "Communication Services", "", "UK", "Mid")])
    closes = pd.DataFrame({"VOD.L": [90.0]}, index=pd.date_range("2025-01-01", periods=1))
    pos = A.positions(h, closes, None, "GBP")
    r = pos.iloc[0]
    assert r["currency"] == "GBP" and r["price"] == pytest.approx(0.9) and r["value"] == pytest.approx(90)


def test_manual_price_fallback():
    h = make_holdings([("XYZ", "", 2, 5, "EUR", "", "", "", "")])
    pos = A.positions(h, pd.DataFrame(), None, "EUR", manual_prices={"XYZ": (7.0, "2025-02-01")})
    assert pos.iloc[0]["price"] == 7.0 and pos.iloc[0]["price_source"] == "Manual"


def test_checks_flag_concentration():
    h = make_holdings([("BIG", "", 90, 1, "EUR", "Energy", "", "Italy", ""),
                       ("SML", "", 10, 1, "EUR", "Utilities", "", "Italy", "")])
    pos = A.positions(h, pd.DataFrame(), None, "EUR")
    text = " ".join(m for _, m in A.checks(pos))
    assert "BIG is 90.0%" in text and "Energy is 90.0%" in text and "Italy is 100.0%" in text


def test_rebalance_and_new_money(three_equal):
    h, closes, fx = three_equal
    pos = A.positions(h, closes, fx, "EUR")   # 100 each, total 300
    targets = {"Energy": 50, "Financials": 25, "Utilities": 25}
    rb = A.rebalance(pos, targets).set_index("sector")
    assert rb.loc["Energy", "change"] == pytest.approx(50)
    assert rb.loc["Financials", "change"] == pytest.approx(-25)
    alloc = A.allocate_new_money(pos, targets, 100)   # new total 400: Energy needs 100
    assert alloc["Energy"] == pytest.approx(100) and alloc["Financials"] == 0
    assert alloc.sum() == pytest.approx(100)


def test_risk_measures():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    s = pd.Series([100, 120, 90, 95, 130], index=idx, dtype=float)
    assert A.max_drawdown(s) == pytest.approx(-0.25)
    r = A.daily_returns(s)
    assert A.beta(pd.concat([r] * 10, ignore_index=True), pd.concat([r] * 10, ignore_index=True)) == pytest.approx(1)


def test_correlation_and_diversification_ratio():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=300)
    a = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    b = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    closes = pd.DataFrame({"A": a, "B": b, "A2": a * 2}, index=idx)
    corr = A.correlation(closes)
    assert corr.loc["A", "A2"] == pytest.approx(1)
    assert abs(corr.loc["A", "B"]) < 0.2
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert A.diversification_ratio(w, closes) > 1.2          # independent stocks help
    assert A.diversification_ratio(pd.Series({"A": 0.5, "A2": 0.5}), closes) == pytest.approx(1)


def test_rsi_bounds_and_trend():
    idx = pd.bdate_range("2024-01-01", periods=260)
    up = pd.Series(np.linspace(50, 150, 260), index=idx)
    assert A.rsi(up).iloc[-1] == pytest.approx(100)
    assert A.trend_signal(up) == "Uptrend"
    assert A.trend_signal(up[::-1].set_axis(idx)) == "Downtrend"
    assert A.trend_signal(up.iloc[:50]) == "Not enough history"


def test_period_returns():
    idx = pd.bdate_range("2024-12-02", periods=60)
    s = pd.Series(np.arange(100, 160, dtype=float), index=idx)
    assert A.period_return(s, 21) == pytest.approx(159 / 138 - 1)
    last_2024 = s[s.index.year == 2024].iloc[-1]
    assert A.ytd_return(s) == pytest.approx(159 / last_2024 - 1)


def test_projection_and_monthly_needed():
    p = A.projection(0, 100, 1, 0)
    assert p["value"].iloc[-1] == pytest.approx(1200)
    p = A.projection(1000, 0, 10, 7)
    assert p["value"].iloc[-1] == pytest.approx(1000 * 1.07 ** 10)
    need = A.monthly_needed(50000, 5000, 10, 6)
    assert A.projection(5000, need, 10, 6)["value"].iloc[-1] == pytest.approx(50000)
    assert A.monthly_needed(100, 1000, 5, 5) == 0


def test_portfolio_history_uses_fx_series(three_equal):
    h, closes, fx = three_equal
    hist = A.portfolio_history(h, closes, fx, "EUR")
    assert hist.iloc[0] == pytest.approx(100 + 90 + 10 * 11 * 0.8)
    assert hist.iloc[-1] == pytest.approx(300)
