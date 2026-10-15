"""Pure portfolio math: valuation, concentration, risk, trend signals and projections.

Nothing here touches the network or the database, so it is easy to test.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .prices import major_currency

TRADING_DAYS = 252
LIMITS = {"position": 0.10, "sector": 0.30, "country": 0.60}
SECTORS = ["Information Technology", "Communication Services", "Consumer Discretionary",
           "Consumer Staples", "Energy", "Financials", "Health Care", "Industrials",
           "Materials", "Real Estate", "Utilities", "ETF / Fund", "Other"]
CAPS = ["Mega", "Large", "Mid", "Small"]


# ---------------------------------------------------------------- valuation

def last_valid(series: pd.Series | None) -> tuple[float | None, pd.Timestamp | None]:
    if series is None:
        return None, None
    s = series.dropna()
    if s.empty:
        return None, None
    return float(s.iloc[-1]), s.index[-1]


def fx_rate(iso: str, base: str, fx: pd.DataFrame | None, overrides: dict | None) -> float | None:
    if not iso or iso == base:
        return 1.0
    if fx is not None and iso in fx.columns:
        rate, _ = last_valid(fx[iso])
        if rate:
            return rate
    rate = (overrides or {}).get(iso)
    return float(rate) if rate else None


def positions(holdings: pd.DataFrame, closes: pd.DataFrame | None, fx: pd.DataFrame | None,
              base: str, fx_overrides: dict | None = None,
              manual_prices: dict | None = None) -> pd.DataFrame:
    """One row per holding with price, value in base currency, P/L and weight."""
    cols = ["ticker", "name", "sector", "industry", "country", "currency", "cap_bucket", "shares",
            "avg_cost", "price", "price_date", "price_source", "day_change", "fx", "value_local",
            "value", "cost", "pl", "pl_pct", "weight"]
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for h in holdings.to_dict("records"):
        iso, div = major_currency(h.get("currency") or base)
        iso = iso or base
        series = closes[h["ticker"]] if closes is not None and h["ticker"] in closes.columns else None
        price, when = last_valid(series)
        source = "Yahoo"
        day_change = None
        if price is not None:
            s = series.dropna()
            if len(s) >= 2 and s.iloc[-2]:
                day_change = s.iloc[-1] / s.iloc[-2] - 1
        elif manual_prices and h["ticker"] in manual_prices:
            price, when = manual_prices[h["ticker"]]
            when = pd.Timestamp(when)
            source = "Manual"
        else:
            price, when, source = float(h["avg_cost"]), None, "None"
        price /= div
        avg_cost = float(h["avg_cost"]) / div
        shares = float(h["shares"])
        rate = fx_rate(iso, base, fx, fx_overrides)
        value_local = shares * price
        cost_local = shares * avg_cost
        invested = h.get("invested_base")
        has_invested = invested is not None and not pd.isna(invested)
        rows.append({
            **{k: h.get(k, "") for k in ("ticker", "name", "sector", "industry", "country", "cap_bucket")},
            "currency": iso, "shares": shares, "avg_cost": avg_cost, "price": price,
            "price_date": when, "price_source": source, "day_change": day_change, "fx": rate,
            "value_local": value_local,
            "value": value_local * rate if rate else np.nan,
            # money really paid (order-day FX, fees) when orders exist; otherwise cost at today's FX
            "cost": float(invested) if (has_invested and rate) else (cost_local * rate if rate else np.nan),
            "pl_pct": (value_local / cost_local - 1) if cost_local else np.nan,
        })
    df = pd.DataFrame(rows)
    df["pl"] = (df["value"] - df["cost"]).where(df["cost"] > 0)   # no average cost entered: P/L unknown
    total = df["value"].sum(skipna=True)
    df["weight"] = df["value"] / total if total else np.nan
    return df.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)[cols]


def hhi(weights) -> float:
    w = pd.Series(weights, dtype=float).dropna()
    return float((w ** 2).sum())


def effective_n(weights) -> float:
    h = hhi(weights)
    return 1 / h if h else 0.0


def group_weights(pos: pd.DataFrame, column: str) -> pd.DataFrame:
    valued = pos.dropna(subset=["value"])
    if valued.empty:
        return pd.DataFrame(columns=[column, "value", "weight"])
    keys = valued[column].fillna("").replace("", "Unclassified")
    g = valued.groupby(keys)["value"].sum().sort_values(ascending=False)
    out = g.reset_index()
    out.columns = [column, "value"]
    out["weight"] = out["value"] / out["value"].sum()
    return out


def checks(pos: pd.DataFrame) -> list[tuple[str, str]]:
    """Concentration and data-quality findings as (level, message); level is error, warning or info."""
    out: list[tuple[str, str]] = []
    if pos.empty:
        return out
    missing_fx = sorted(pos.loc[pos["fx"].isna(), "currency"].unique())
    if missing_fx:
        out.append(("error", f"No FX rate for {', '.join(missing_fx)}. Those holdings are left out of "
                             "totals. Add a manual rate in Data & settings."))
    valued = pos.dropna(subset=["value"])
    for r in valued[valued["weight"] > LIMITS["position"]].itertuples():
        out.append(("warning", f"{r.ticker} is {r.weight:.1%} of the portfolio (guide: at most {LIMITS['position']:.0%} per stock)."))
    for r in group_weights(pos, "sector").itertuples():
        if r.weight > LIMITS["sector"] and r.sector != "ETF / Fund":
            out.append(("warning", f"{r.sector} is {r.weight:.1%} of the portfolio (guide: at most {LIMITS['sector']:.0%} per sector)."))
    for r in group_weights(pos, "country").itertuples():
        if r.weight > LIMITS["country"]:
            out.append(("warning", f"{r.country} is {r.weight:.1%} of the portfolio (guide: at most {LIMITS['country']:.0%} in one country)."))
    n = effective_n(valued["weight"])
    if len(valued) and n < 8:
        out.append(("warning", f"The portfolio behaves like {n:.1f} equal-sized positions. Below about 8, "
                               "one company's bad news moves the whole portfolio."))
    unclassified = pos.loc[pos["sector"].fillna("") == "", "ticker"].tolist()
    if unclassified:
        out.append(("info", f"No sector for {', '.join(unclassified)}. Edit the holding or use Look up."))
    no_cost = pos.loc[pos["cost"].fillna(0) <= 0, "ticker"].tolist()
    if no_cost:
        out.append(("warning", f"No average cost for {', '.join(no_cost)}: their profit/loss and sell-out results "
                               "can't be computed. Edit the holding, or record the buy on the Orders page."))
    no_price = pos.loc[pos["price_source"] == "None", "ticker"].tolist()
    if no_price:
        out.append(("info", f"No market price for {', '.join(no_price)}; using average cost. Check the ticker symbol."))
    return out


# ---------------------------------------------------------------- targets

def rebalance(pos: pd.DataFrame, targets_pct: dict) -> pd.DataFrame:
    """Per sector: actual vs target weight and the amount to buy (+) or sell (-) to reach target."""
    g = group_weights(pos, "sector").set_index("sector")
    total = g["value"].sum() if not g.empty else 0.0
    sectors = list(dict.fromkeys(list(g.index) + [s for s, v in targets_pct.items() if v]))
    rows = []
    for s in sectors:
        actual_value = float(g["value"].get(s, 0.0)) if not g.empty else 0.0
        t = targets_pct.get(s)
        target_value = (t / 100 * total) if t else None
        rows.append({"sector": s, "actual_pct": actual_value / total * 100 if total else 0.0,
                     "target_pct": t, "actual_value": actual_value,
                     "change": (target_value - actual_value) if target_value is not None else None})
    return pd.DataFrame(rows)


def allocate_new_money(pos: pd.DataFrame, targets_pct: dict, cash: float) -> pd.Series:
    """Split new cash across sectors furthest below target, without selling anything."""
    if cash <= 0 or not any(targets_pct.values()):
        return pd.Series(dtype=float)
    g = group_weights(pos, "sector").set_index("sector")["value"] if not pos.empty else pd.Series(dtype=float)
    new_total = float(g.sum()) + cash
    deficit = pd.Series({s: max(0.0, t / 100 * new_total - float(g.get(s, 0.0)))
                         for s, t in targets_pct.items() if t})
    if deficit.sum() <= 0:
        return deficit
    return deficit * min(1.0, cash / deficit.sum())


# ---------------------------------------------------------------- history & risk

def portfolio_history(holdings: pd.DataFrame, closes: pd.DataFrame, fx: pd.DataFrame | None,
                      base: str, fx_overrides: dict | None = None,
                      held: pd.DataFrame | None = None) -> pd.Series:
    """Daily value in base currency.

    Tickers with a column in `held` (shares by day, from orders) use the shares actually held each
    day; the others use today's shares over the whole window.
    """
    if holdings.empty or closes is None or closes.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    parts = []
    for h in holdings.to_dict("records"):
        t = h["ticker"]
        if t not in closes.columns:
            continue
        iso, div = major_currency(h.get("currency") or base)
        iso = iso or base
        px = closes[t] / div
        if iso == base:
            rate = 1.0
        elif fx is not None and iso in fx.columns:
            rate = fx[iso].reindex(closes.index).ffill().bfill()
        elif (fx_overrides or {}).get(iso):
            rate = float(fx_overrides[iso])
        else:
            continue
        if held is not None and t in held.columns:
            qty = held[t].reindex(held.index.union(closes.index)).ffill().reindex(closes.index).fillna(0.0)
            parts.append((px.ffill().fillna(0.0) * rate * qty).rename(t))
        else:
            parts.append((px * rate * float(h["shares"])).rename(t))
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    df = pd.concat(parts, axis=1).ffill().dropna()
    total = df.sum(axis=1)
    if held is not None and not held.empty:
        total = total[total.index >= held.index.min()]
    return total


def daily_returns(series: pd.Series | pd.DataFrame):
    return series.ffill().pct_change(fill_method=None).dropna(how="all")


def annual_volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.std() * math.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan")


def max_drawdown(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float((s / s.cummax() - 1).min())


def cagr(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return float("nan")
    years = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")


def sharpe(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    excess = r - risk_free_annual / TRADING_DAYS
    return float(excess.mean() / r.std() * math.sqrt(TRADING_DAYS))


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    df = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(df) < 20:
        return float("nan")
    var = df.iloc[:, 1].var()
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / var) if var else float("nan")


def correlation(closes: pd.DataFrame) -> pd.DataFrame:
    return daily_returns(closes).corr(min_periods=40)


def average_correlation(corr: pd.DataFrame) -> float:
    if corr.shape[0] < 2:
        return float("nan")
    vals = corr.values[np.triu_indices_from(corr.values, k=1)]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def diversification_ratio(weights: pd.Series, closes: pd.DataFrame) -> float:
    """Weighted average of stock volatilities divided by portfolio volatility. 1 = no benefit."""
    w = weights.dropna()
    w = w[w.index.isin(closes.columns)]
    if len(w) < 2:
        return float("nan")
    w = w / w.sum()
    r = daily_returns(closes[w.index]).dropna()
    if len(r) < 40:
        return float("nan")
    cov = r.cov().values * TRADING_DAYS
    vols = np.sqrt(np.diag(cov))
    port_vol = math.sqrt(float(w.values @ cov @ w.values))
    return float((w.values @ vols) / port_vol) if port_vol else float("nan")


def asset_risk(closes: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    bench_r = daily_returns(benchmark) if benchmark is not None else None
    for t in closes.columns:
        s = closes[t].dropna()
        if len(s) < 2:
            continue
        r = daily_returns(s)
        rows.append({"ticker": t, "volatility": annual_volatility(r),
                     "max_drawdown": max_drawdown(s),
                     "return_1y": period_return(s, TRADING_DAYS),
                     "beta": beta(r, bench_r) if bench_r is not None else float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- trend

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's relative strength index."""
    d = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def period_return(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) <= days:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-1 - days] - 1)


def ytd_return(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    prior = s[s.index < pd.Timestamp(year=s.index[-1].year, month=1, day=1)]
    return float(s.iloc[-1] / prior.iloc[-1] - 1) if not prior.empty else float("nan")


def trend_signal(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) < 200:
        return "Not enough history"
    p, m50, m200 = s.iloc[-1], sma(s, 50).iloc[-1], sma(s, 200).iloc[-1]
    if p > m200 and m50 > m200:
        return "Uptrend"
    if p < m200 and m50 < m200:
        return "Downtrend"
    return "Mixed"


# ---------------------------------------------------------------- planning

def projection(start: float, monthly: float, years: int, annual_return_pct: float) -> pd.DataFrame:
    """Month-by-month value of a monthly investment plan (contribution at month end)."""
    rm = (1 + annual_return_pct / 100) ** (1 / 12) - 1
    value, contributed, rows = float(start), float(start), []
    for m in range(1, int(years * 12) + 1):
        value = value * (1 + rm) + monthly
        contributed += monthly
        rows.append({"month": m, "year": m / 12, "contributed": contributed, "value": value})
    return pd.DataFrame(rows)


def monthly_needed(goal: float, start: float, years: int, annual_return_pct: float) -> float:
    """Monthly contribution needed to reach goal; 0 if the start amount alone gets there."""
    n = int(years * 12)
    if n <= 0:
        return float("nan")
    rm = (1 + annual_return_pct / 100) ** (1 / 12) - 1
    grown_start = start * (1 + rm) ** n
    if grown_start >= goal:
        return 0.0
    factor = n if rm == 0 else ((1 + rm) ** n - 1) / rm
    return (goal - grown_start) / factor


# ---------------------------------------------------------------- orders

def fx_on(fx: pd.DataFrame | None, cur: str, base: str, day, overrides: dict | None = None) -> float | None:
    """Rate (1 unit of cur in base) on a given day: last quote on or before it, else the first one after."""
    if not cur or cur == base:
        return 1.0
    if fx is not None and cur in fx.columns:
        q = fx[cur].dropna()
        if not q.empty:
            before = q[q.index <= pd.Timestamp(day)]
            return float(before.iloc[-1] if not before.empty else q.iloc[0])
    r = (overrides or {}).get(cur)
    return float(r) if r else None


def order_book(orders: pd.DataFrame, quote_currency: dict, fx: pd.DataFrame | None, base: str,
               overrides: dict | None = None) -> pd.DataFrame:
    """Replay orders with the average-cost method (costo medio ponderato).

    quote_currency maps ticker -> currency as Yahoo quotes it (e.g. "GBp"). Order prices are in those
    quote units. Fees are in each order's fee_currency; they add to the cost of a buy and reduce the
    proceeds of a sell. Base-currency amounts use the FX rate of the order day.
    Returns one row per ticker: shares, avg_cost (quote units, buy fees included), invested_base
    (money paid for the shares still held), realized_base, fees_base, first_buy, orders, oversold.
    """
    cols = ["ticker", "shares", "avg_cost", "invested_base", "realized_base", "fees_base", "first_buy",
            "orders", "oversold"]
    if orders is None or orders.empty:
        return pd.DataFrame(columns=cols)
    book: dict[str, dict] = {}
    order_cols = ["day", "id"] if "id" in orders.columns else ["day"]
    for o in orders.sort_values(order_cols).to_dict("records"):
        t = o["ticker"]
        iso, div = major_currency(quote_currency.get(t) or base)
        iso = iso or base
        b = book.setdefault(t, {"ticker": t, "shares": 0.0, "cost_local": 0.0, "invested_base": 0.0,
                                "realized_base": 0.0, "fees_base": 0.0, "first_buy": None, "orders": 0,
                                "oversold": False})
        b["orders"] += 1
        day = pd.Timestamp(o["day"])
        rate = fx_on(fx, iso, base, day, overrides)
        fee_rate = fx_on(fx, (o.get("fee_currency") or base).upper(), base, day, overrides)
        rate = np.nan if rate is None else rate
        fee_rate = np.nan if fee_rate is None else fee_rate
        px = float(o["price"]) / div
        q = float(o["shares"])
        fee_base = float(o.get("fee") or 0) * fee_rate
        fee_local = fee_base / rate
        b["fees_base"] += fee_base
        if o["side"] == "BUY":
            b["shares"] += q
            b["cost_local"] += q * px + fee_local
            b["invested_base"] += q * px * rate + fee_base
            b["first_buy"] = b["first_buy"] or day
        else:
            if q > b["shares"] + 1e-9:
                b["oversold"] = True
                q = b["shares"]
            if b["shares"] <= 0:
                continue
            frac = q / b["shares"]
            cost_b = b["invested_base"] * frac
            b["realized_base"] += q * px * rate - fee_base - cost_b
            b["cost_local"] -= b["cost_local"] * frac
            b["invested_base"] -= cost_b
            b["shares"] -= q
    rows = []
    for t, b in book.items():
        div = major_currency(quote_currency.get(t) or base)[1]
        shares = b["shares"] if b["shares"] > 1e-9 else 0.0
        rows.append({**{k: b[k] for k in ("ticker", "realized_base", "fees_base", "first_buy", "orders", "oversold")},
                     "shares": shares,
                     "avg_cost": b["cost_local"] / shares * div if shares else 0.0,
                     "invested_base": b["invested_base"] if shares else 0.0})
    return pd.DataFrame(rows)[cols]


def holdings_with_orders(holdings: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    """Holdings where shares and cost come from orders; fully sold positions are dropped."""
    h = holdings.copy()
    if "invested_base" not in h.columns:
        h["invested_base"] = np.nan
    if book is None or book.empty:
        return h
    b = book.set_index("ticker")
    missing = [t for t in b.index if t not in set(h["ticker"])]
    if missing:
        h = pd.concat([h, pd.DataFrame({"ticker": missing, "name": "", "currency": "", "sector": "",
                                        "industry": "", "country": "", "cap_bucket": ""})], ignore_index=True)
    m = h["ticker"].isin(b.index)
    for col in ("shares", "avg_cost", "invested_base"):
        h[col] = h[col].astype(float)
        h.loc[m, col] = h.loc[m, "ticker"].map(b[col]).astype(float)
    return h[~m | (h["shares"] > 0)].reset_index(drop=True)


def shares_by_day(orders: pd.DataFrame) -> pd.DataFrame:
    """Shares held at the end of each order day, one column per ticker."""
    if orders is None or orders.empty:
        return pd.DataFrame()
    o = orders.copy()
    o["day"] = pd.to_datetime(o["day"])
    o["signed"] = np.where(o["side"] == "BUY", o["shares"], -o["shares"])
    daily = o.groupby(["day", "ticker"])["signed"].sum().unstack(fill_value=0.0).sort_index()
    return daily.cumsum().clip(lower=0)


def cash_flows(orders: pd.DataFrame, quote_currency: dict, fx: pd.DataFrame | None, base: str,
               overrides: dict | None = None) -> pd.DataFrame:
    """Money paid in (buys, negative) and taken out (sells, positive) per order, in base currency, fees included."""
    rows = []
    for o in orders.to_dict("records"):
        iso, div = major_currency(quote_currency.get(o["ticker"]) or base)
        day = pd.Timestamp(o["day"])
        rate = fx_on(fx, iso or base, base, day, overrides) or np.nan
        fee_rate = fx_on(fx, (o.get("fee_currency") or base).upper(), base, day, overrides) or np.nan
        fee = float(o.get("fee") or 0) * fee_rate
        gross = float(o["shares"]) * float(o["price"]) / div * rate
        rows.append({"day": day, "amount": -(gross + fee) if o["side"] == "BUY" else gross - fee})
    return pd.DataFrame(rows, columns=["day", "amount"])


def net_invested(flows: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Running total of money put in minus money taken out, on the given dates."""
    if flows.empty:
        return pd.Series(dtype=float)
    run = (-flows.groupby("day")["amount"].sum()).sort_index().cumsum()
    return run.reindex(run.index.union(index)).ffill().reindex(index).fillna(0.0)


def xirr(flows: list[tuple]) -> float:
    """Money-weighted yearly return of dated cash flows (negative = paid in). NaN if undefined."""
    flows = [(pd.Timestamp(d), float(a)) for d, a in flows if a and not pd.isna(a)]
    if not flows or min(a for _, a in flows) >= 0 or max(a for _, a in flows) <= 0:
        return float("nan")
    t0 = min(d for d, _ in flows)
    yrs = [((d - t0).days / 365.25, a) for d, a in flows]
    if max(y for y, _ in yrs) < 7 / 365.25:
        return float("nan")

    def npv(r: float) -> float:
        return sum(a / (1 + r) ** y for y, a in yrs)

    lo, hi = -0.9999, 100.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(300):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo < 1e-10:
            break
    return float((lo + hi) / 2)


# ---------------------------------------------------------------- sell-out

SELL_COLS = ["ticker", "shares", "price", "currency", "value", "sell_fee", "net_proceeds", "invested",
             "net_result", "net_pct", "break_even", "to_break_even", "fee_share", "status"]


def sellout(pos: pd.DataFrame, fee_base: float, stock_alert_pct: float) -> pd.DataFrame:
    """What selling each whole position today brings after the broker fee, vs the money paid.

    break_even is the price (major units, like pos.price) at which a sale returns at least the money
    paid including both fees; to_break_even is the price move needed to get there.
    """
    v = pos.dropna(subset=["value", "cost"]).copy()
    if v.empty:
        return pd.DataFrame(columns=SELL_COLS)
    v["sell_fee"] = fee_base
    v["net_proceeds"] = v["value"] - fee_base
    v["invested"] = v["cost"]
    v["net_result"] = v["net_proceeds"] - v["invested"]
    known = v["invested"] > 0
    v["net_pct"] = (v["net_result"] / v["invested"]).where(known)
    v["break_even"] = ((v["invested"] + fee_base) / (v["shares"] * v["fx"])).where(known)
    v["to_break_even"] = v["break_even"] / v["price"] - 1
    v["fee_share"] = fee_base / v["value"]
    limit = -stock_alert_pct / 100
    v["status"] = np.where(~known, "No cost entered", np.where(v["net_pct"] <= limit, "ALERT",
                           np.where(v["net_result"] >= 0, "Above break-even", "Below break-even")))
    return v[SELL_COLS].reset_index(drop=True)


def sell_summary(sell: pd.DataFrame) -> dict:
    """Totals over positions with a known cost (positions without one would count as pure profit)."""
    sell = sell[sell["invested"] > 0]
    invested, net = float(sell["invested"].sum()), float(sell["net_proceeds"].sum())
    return {"invested": invested, "net_proceeds": net, "net_result": net - invested,
            "net_pct": net / invested - 1 if invested else float("nan")}


def sell_alerts(sell: pd.DataFrame, stock_alert_pct: float, portfolio_alert_pct: float) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if sell.empty:
        return out
    total = sell_summary(sell)["net_pct"]
    if total <= -portfolio_alert_pct / 100:
        out.append(("error", f"Whole portfolio: selling everything now returns {total:+.1%} vs what you paid "
                             f"(alert at -{portfolio_alert_pct:g}%)."))
    for r in sell[sell["status"] == "ALERT"].itertuples():
        out.append(("error", f"{r.ticker}: selling now returns {r.net_pct:+.1%} vs what you paid "
                             f"(alert at -{stock_alert_pct:g}%). Break-even price {r.break_even:,.2f} {r.currency}."))
    return out


def avg_cost_steps(orders: pd.DataFrame, ticker: str, quote_currency: dict, fx: pd.DataFrame | None,
                   base: str, overrides: dict | None = None) -> pd.Series:
    """Average cost per share (quote units, buy fees included) after each order day of one ticker.

    NaN where the position was fully sold, so a chart line breaks instead of joining across it.
    """
    o = orders[orders["ticker"] == ticker]
    if o.empty:
        return pd.Series(dtype=float)
    o = o.sort_values(["day", "id"] if "id" in o.columns else ["day"])
    out = {}
    for day in sorted(o["day"].unique()):
        b = order_book(o[o["day"] <= day], quote_currency, fx, base, overrides)
        r = b.iloc[0]
        out[pd.Timestamp(day)] = r["avg_cost"] if r["shares"] > 0 else np.nan
    return pd.Series(out, dtype=float).sort_index()



def latest_rates(fx: pd.DataFrame | None) -> dict:
    """Last quoted rate and its date per currency: {"USD": {"rate": 0.8927, "day": "2026-10-09"}}."""
    out = {}
    if fx is None or fx.empty:
        return out
    for c in fx.columns:
        q = fx[c].dropna()
        if not q.empty:
            out[c] = {"rate": round(float(q.iloc[-1]), 6), "day": q.index[-1].date().isoformat()}
    return out


def fx_fallback(fx: pd.DataFrame | None, base: str, needed: set, last: dict, manual: dict) -> tuple[dict, list[str]]:
    """Rates to use when Yahoo has no quote: your manual rates first, else the last rate saved from Yahoo.

    Returns (overrides, notes). Notes tell the user which saved rates stand in for missing quotes.
    """
    have = set(fx.columns) if fx is not None else set()
    overrides, notes = dict(manual or {}), []
    for c in sorted(needed - have):
        if c in overrides:
            notes.append(f"Yahoo Finance gave no {c}→{base} rate just now; using your manual rate {overrides[c]:g}.")
        elif c in (last or {}):
            overrides[c] = last[c]["rate"]
            notes.append(f"Yahoo Finance gave no {c}→{base} rate just now; using the last saved rate "
                         f"{last[c]['rate']:.4f} from {last[c]['day']}.")
    return overrides, notes
