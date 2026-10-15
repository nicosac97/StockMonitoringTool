import math

import pandas as pd
import pytest

from ledger import analytics as A
from ledger import charts


def orders():
    rows = [("X", "BUY", "2025-01-02", 10, 10.0, 0), ("X", "BUY", "2025-02-03", 10, 20.0, 0),
            ("X", "SELL", "2025-03-03", 20, 25.0, 0), ("X", "BUY", "2025-04-01", 5, 8.0, 0),
            ("Y", "BUY", "2025-01-02", 1, 99.0, 0)]
    return pd.DataFrame([dict(id=i, ticker=t, side=s, day=d, shares=q, price=p, fee=f, fee_currency="EUR")
                         for i, (t, s, d, q, p, f) in enumerate(rows, 1)])


def test_avg_cost_steps_follow_orders_and_break_when_flat():
    s = A.avg_cost_steps(orders(), "X", {"X": "EUR"}, None, "EUR")
    assert list(s.index) == list(pd.to_datetime(["2025-01-02", "2025-02-03", "2025-03-03", "2025-04-01"]))
    assert s.iloc[0] == pytest.approx(10) and s.iloc[1] == pytest.approx(15)
    assert math.isnan(s.iloc[2])                 # sold out: line breaks
    assert s.iloc[3] == pytest.approx(8)         # fresh position starts again


def test_chart_has_buy_and_sell_markers_with_hover_vs_today():
    o = orders()
    x = o[o["ticker"] == "X"]
    price = pd.Series([10.0, 20.0, 25.0, 8.0, 12.0], index=pd.to_datetime(
        ["2025-01-02", "2025-02-03", "2025-03-03", "2025-04-01", "2025-05-02"]))
    fig = charts.price_with_orders(price, x, A.avg_cost_steps(o, "X", {"X": "EUR"}, None, "EUR"), "X", "EUR")
    by_name = {t.name: t for t in fig.data}
    assert len(by_name["Buy"].x) == 3 and len(by_name["Sell"].x) == 1
    assert by_name["Buy"].marker.symbol == "triangle-up" and by_name["Sell"].marker.symbol == "triangle-down"
    assert "Now +20.0% vs this price" in by_name["Buy"].text[0]      # 12 vs bought at 10
    assert by_name["Your average cost"].x[-1] == pd.Timestamp("2025-05-02")   # carried to today


def test_empty_history_keeps_a_date_index():
    from ledger.prices import empty_closes
    e = empty_closes()
    assert isinstance(e.index, pd.DatetimeIndex)
    h = A.portfolio_history(pd.DataFrame({"ticker": ["X"], "shares": [1], "currency": ["EUR"]}), e, None, "EUR")
    assert h.empty and isinstance(h.index, pd.DatetimeIndex)
    assert h[h.index >= pd.Timestamp("2025-01-01")].empty      # the comparison that used to crash
