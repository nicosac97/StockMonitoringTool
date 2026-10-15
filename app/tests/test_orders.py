import math

import pandas as pd
import pytest

from ledger import analytics as A
from ledger.db import Store

IDX = pd.to_datetime(["2025-01-02", "2025-06-02", "2026-01-02"])
FX = pd.DataFrame({"USD": [0.90, 0.80, 0.85]}, index=IDX)   # 1 USD in EUR on each day


def orders(rows):
    return pd.DataFrame([dict(id=i, ticker=t, side=s, day=d, shares=q, price=p, fee=f, fee_currency=fc)
                         for i, (t, s, d, q, p, f, fc) in enumerate(rows, 1)])


def test_fx_on_uses_rate_of_order_day():
    assert A.fx_on(FX, "USD", "EUR", "2025-03-15") == 0.90      # last quote on or before
    assert A.fx_on(FX, "USD", "EUR", "2024-12-01") == 0.90      # before history: first quote
    assert A.fx_on(FX, "EUR", "EUR", "2025-03-15") == 1.0
    assert A.fx_on(None, "CHF", "EUR", "2025-03-15", {"CHF": 1.05}) == 1.05
    assert A.fx_on(None, "CHF", "EUR", "2025-03-15") is None


def test_average_cost_with_fees_and_partial_sell():
    o = orders([
        ("AAPL", "BUY", "2025-01-02", 10, 100, 1.12, "USD"),
        ("AAPL", "BUY", "2025-06-02", 10, 120, 1.12, "USD"),
        ("AAPL", "SELL", "2026-01-02", 5, 150, 1.12, "USD"),
    ])
    b = A.order_book(o, {"AAPL": "USD"}, FX, "EUR").set_index("ticker").loc["AAPL"]
    # cost in EUR: buy1 1000*0.9 + 1.12*0.9 = 901.008 ; buy2 1200*0.8 + 1.12*0.8 = 960.896
    invested_before = 901.008 + 960.896
    sold_cost = invested_before * 5 / 20
    proceeds = 5 * 150 * 0.85 - 1.12 * 0.85
    assert b["shares"] == 15
    assert b["invested_base"] == pytest.approx(invested_before - sold_cost)
    assert b["realized_base"] == pytest.approx(proceeds - sold_cost)
    assert b["fees_base"] == pytest.approx(1.12 * (0.9 + 0.8 + 0.85))
    # avg cost in USD incl. buy fees: (1000 + 1.12 + 1200 + 1.12*0.8/0.8) / 20
    assert b["avg_cost"] == pytest.approx((1001.12 + 1201.12) / 20)
    assert b["orders"] == 3 and not b["oversold"]


def test_full_sell_closes_position_and_oversell_is_flagged():
    o = orders([("ENI.MI", "BUY", "2025-01-02", 100, 10, 1.12, "USD"),
                ("ENI.MI", "SELL", "2025-06-02", 150, 12, 1.12, "USD")])
    b = A.order_book(o, {"ENI.MI": "EUR"}, FX, "EUR").set_index("ticker").loc["ENI.MI"]
    assert b["shares"] == 0 and b["invested_base"] == 0 and b["oversold"]
    assert b["realized_base"] == pytest.approx(100 * 12 - 0.896 - (1000 + 1.008))
    h = pd.DataFrame([{"ticker": "ENI.MI", "name": "Eni", "shares": 1, "avg_cost": 1, "currency": "EUR",
                       "sector": "Energy", "industry": "", "country": "Italy", "cap_bucket": ""}])
    assert A.holdings_with_orders(h, A.order_book(o, {"ENI.MI": "EUR"}, FX, "EUR")).empty


def test_pence_orders():
    o = orders([("VOD.L", "BUY", "2025-01-02", 100, 70, 0, "GBP")])
    b = A.order_book(o, {"VOD.L": "GBp"}, None, "GBP").set_index("ticker").loc["VOD.L"]
    assert b["invested_base"] == pytest.approx(70) and b["avg_cost"] == pytest.approx(70)


def test_positions_cost_is_money_paid_not_todays_fx():
    o = orders([("AAPL", "BUY", "2025-01-02", 10, 100, 1.12, "USD")])
    book = A.order_book(o, {"AAPL": "USD"}, FX, "EUR")
    h = pd.DataFrame([{"ticker": "AAPL", "name": "Apple", "shares": 0, "avg_cost": 0, "currency": "USD",
                       "sector": "", "industry": "", "country": "", "cap_bucket": ""}])
    hh = A.holdings_with_orders(h, book)
    closes = pd.DataFrame({"AAPL": [100.0, 100.0, 100.0]}, index=IDX)
    pos = A.positions(hh, closes, FX, "EUR").iloc[0]
    assert pos["cost"] == pytest.approx(901.008)               # paid at 0.90
    assert pos["value"] == pytest.approx(850)                  # worth at 0.85 today
    assert pos["pl"] == pytest.approx(850 - 901.008)           # the FX loss shows up


def test_history_follows_shares_actually_held_and_net_invested():
    o = orders([("X", "BUY", "2025-06-02", 10, 10, 0, "EUR"), ("X", "SELL", "2026-01-02", 4, 10, 0, "EUR")])
    held = A.shares_by_day(o)
    h = pd.DataFrame([{"ticker": "X", "shares": 6, "avg_cost": 10, "currency": "EUR"}])
    closes = pd.DataFrame({"X": [8.0, 10.0, 11.0]}, index=IDX)
    hist = A.portfolio_history(h, closes, None, "EUR", held=held)
    assert list(hist.index) == list(IDX[1:])                   # starts at first order
    assert hist.tolist() == pytest.approx([100.0, 66.0])
    flows = A.cash_flows(o, {"X": "EUR"}, None, "EUR")
    assert A.net_invested(flows, hist.index).tolist() == pytest.approx([100.0, 60.0])


def test_xirr():
    one_year = [(pd.Timestamp("2025-01-01"), -1000), (pd.Timestamp("2026-01-01"), 1100)]
    assert A.xirr(one_year) == pytest.approx(0.10, abs=1e-3)
    assert math.isnan(A.xirr([(pd.Timestamp("2025-01-01"), -1000)]))


def test_sellout_break_even_and_alerts():
    pos = pd.DataFrame([
        {"ticker": "UP", "shares": 10, "price": 12.0, "currency": "EUR", "fx": 1.0, "value": 120.0, "cost": 100.0},
        {"ticker": "DOWN", "shares": 10, "price": 7.0, "currency": "USD", "fx": 0.5, "value": 35.0, "cost": 50.0},
    ])
    sell = A.sellout(pos, fee_base=1.0, stock_alert_pct=25).set_index("ticker")
    assert sell.loc["UP", "net_result"] == pytest.approx(19)
    assert sell.loc["UP", "status"] == "Above break-even"
    assert sell.loc["UP", "break_even"] == pytest.approx(10.1)          # (100 + 1) / 10
    assert sell.loc["DOWN", "net_pct"] == pytest.approx(34 / 50 - 1)   # -32%
    assert sell.loc["DOWN", "status"] == "ALERT"
    assert sell.loc["DOWN", "break_even"] == pytest.approx(51 / (10 * 0.5))
    alerts = A.sell_alerts(sell.reset_index(), 25, 15)
    assert any("DOWN" in m for _, m in alerts)
    assert not any("Whole portfolio" in m for _, m in alerts)          # 153 vs 150 overall
    assert any("Whole portfolio" in m for _, m in A.sell_alerts(sell.reset_index(), 25, -5))   # +2% <= +5%


def test_store_orders_round_trip(tmp_path):
    s = Store(tmp_path / "o.db")
    i = s.add_order({"ticker": "aapl", "side": "buy", "day": "2025-01-02", "shares": 3, "price": 150, "fee": 1.12})
    assert s.orders().iloc[0][["ticker", "side", "fee_currency"]].tolist() == ["AAPL", "BUY", "USD"]
    with pytest.raises(ValueError):
        s.add_order({"ticker": "X", "side": "hold", "day": "2025-01-02", "shares": 1, "price": 1})
    s.upsert_holding({"ticker": "AAPL", "shares": 3, "avg_cost": 150})
    s.delete_holding("AAPL")
    assert s.orders().empty
    s.add_order({"ticker": "X", "side": "SELL", "day": "2025-01-02", "shares": 1, "price": 1})
    s.delete_orders([i + 1])
    assert s.orders().empty


def test_fx_fallback_uses_manual_then_last_saved_rate():
    last = {"USD": {"rate": 0.89, "day": "2026-10-08"}, "CHF": {"rate": 1.07, "day": "2026-10-08"}}
    over, notes = A.fx_fallback(pd.DataFrame(), "EUR", {"USD", "CHF", "GBP"}, last, {"CHF": 1.05})
    assert over == {"CHF": 1.05, "USD": 0.89}           # manual wins, saved fills the gap, GBP stays missing
    assert any("last saved rate 0.8900 from 2026-10-08" in n for n in notes)
    assert A.latest_rates(FX) == {"USD": {"rate": 0.85, "day": "2026-01-02"}}
    over, notes = A.fx_fallback(FX, "EUR", {"USD"}, last, {})
    assert over == {} and notes == []                    # Yahoo answered: nothing to fall back on


def test_zero_average_cost_is_flagged_not_counted_as_profit():
    h = pd.DataFrame([{"ticker": "AMZN", "name": "", "shares": 1, "avg_cost": 0, "currency": "EUR", "sector": "x",
                       "industry": "", "country": "", "cap_bucket": ""},
                      {"ticker": "ENI", "name": "", "shares": 10, "avg_cost": 10, "currency": "EUR", "sector": "x",
                       "industry": "", "country": "", "cap_bucket": ""}])
    closes = pd.DataFrame({"AMZN": [200.0], "ENI": [9.0]}, index=IDX[:1])
    pos = A.positions(h, closes, None, "EUR")
    assert math.isnan(pos.set_index("ticker").loc["AMZN", "pl"])
    sell = A.sellout(pos, 1.0, 25).set_index("ticker")
    assert sell.loc["AMZN", "status"] == "No cost entered" and math.isnan(sell.loc["AMZN", "net_pct"])
    assert A.sell_summary(sell.reset_index())["invested"] == pytest.approx(100)    # AMZN left out
    assert any("No average cost for AMZN" in m for _, m in A.checks(pos))
