"""Stock Ledger — local portfolio monitor.

Run:  .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st


def _reload_changed_ledger() -> None:
    """Streamlit re-runs app.py on every change but keeps ledger/* modules from the first load, so an update
    to them only showed after restarting run.bat. Reload the ones whose file changed, and everything that
    depends on them (listed after them), before app.py imports from them."""
    import importlib
    import os
    import sys
    order = ["prices", "db", "analytics", "universe", "screener", "ideas", "targets", "longshots", "commentary",
             "charts", "transfer", "daily"]
    changed = False
    for name in order:
        mod = sys.modules.get(f"ledger.{name}")
        if mod is None or not getattr(mod, "__file__", None):
            continue
        mtime = os.path.getmtime(mod.__file__)
        if changed or mtime != getattr(mod, "_loaded_mtime", mtime):
            mod = importlib.reload(mod)
            changed = True
        mod._loaded_mtime = mtime


_reload_changed_ledger()

from ledger import analytics as A  # noqa: E402  (after the reload above)
from ledger import charts, prices
from ledger.db import Store
from ledger.transfer import export_holdings_csv, export_json, import_csv, import_json, import_orders_csv

st.set_page_config(page_title="Stock Ledger", page_icon="📒", layout="wide")


# ---------------------------------------------------------------- data access

def get_store() -> Store:
    # Not cached on purpose: a cached instance survives code reloads and keeps the old class
    # (e.g. without methods added later). Creating one only opens the file and ensures the schema.
    return Store()


@st.cache_data(ttl=900, show_spinner="Fetching prices from Yahoo Finance…")
def load_closes(tickers: tuple[str, ...], period: str, start: str | None = None) -> pd.DataFrame:
    return prices.download_closes(list(tickers), period, start=start)


@st.cache_data(ttl=900, show_spinner="Fetching traded prices…")
def load_traded_closes(ticker: str, period: str = "max", start: str | None = None) -> pd.Series:
    """Unadjusted closes (prices as traded), to line up with order prices."""
    c = prices.download_closes([ticker], period, adjusted=False, start=start)
    return c[ticker].dropna() if ticker in c.columns else pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def since(series: pd.Series, start) -> pd.Series:
    """Rows on or after start; safe on empty or non-date series."""
    if start is None or series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return series
    return series[series.index >= pd.Timestamp(start)]


@st.cache_data(ttl=900, show_spinner="Fetching FX rates…")
def load_fx(currencies: tuple[str, ...], base: str, period: str, start: str | None = None) -> pd.DataFrame:
    return prices.fx_history(set(currencies), base, period, start=start)


@st.cache_data(ttl=86400, show_spinner="Looking up ticker…")
def _cached_lookup(ticker: str) -> dict:
    return prices.lookup(ticker)


def load_lookup(ticker: str) -> dict:
    """Ticker details, cached for a day, except failures: those are retried on the next attempt."""
    info = _cached_lookup(ticker)
    if not info:
        _cached_lookup.clear()
    return info


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def money(v, cur: str, digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "–"
    sign = "−" if v < 0 else ""
    return f"{sign}{abs(v):,.{digits}f} {cur}"


def pct(v, digits: int = 1, signed: bool = False) -> str:
    if v is None or pd.isna(v):
        return "–"
    return f"{v:+.{digits}%}" if signed else f"{v:.{digits}%}"


WINDOW_DAYS = {"1w": 7, "1m": 31, "1y": 365, "2y": 730, "5y": 1826}
YAHOO_PERIOD = {"1w": "5d", "1m": "1mo", "1y": "1y", "2y": "2y", "5y": "5y"}


def context() -> dict:
    store = get_store()
    s = store.settings()
    base = s["base_currency"]
    period = st.session_state.get("period", "2y")
    holdings, watch, orders = store.holdings(), store.watchlist(), store.orders()
    # fetch far enough back to price every order on its own day, but no further
    fetch, start = YAHOO_PERIOD[period], None
    if not orders.empty and pd.Timestamp(orders["day"].min()) < pd.Timestamp.today() - pd.Timedelta(days=WINDOW_DAYS[period]):
        start = (pd.Timestamp(orders["day"].min()) - pd.Timedelta(days=10)).date().isoformat()
    tickers = set(holdings["ticker"]) | set(watch["ticker"]) | set(orders["ticker"])
    if s["benchmark"]:
        tickers.add(s["benchmark"])
    closes = load_closes(tuple(sorted(tickers)), fetch, start) if tickers else prices.empty_closes()
    currencies = {prices.major_currency(c)[0] for c in pd.concat([holdings["currency"], watch["currency"]])}
    currencies |= set(orders["fee_currency"]) | {s["fee"]["currency"]}
    fx = load_fx(tuple(sorted(c for c in currencies if c)), base, fetch, start)
    notes = []
    needed = {c for c in currencies if c and c != base}
    missing = needed - set(fx.columns)
    if missing:
        load_fx.clear()          # don't keep a failed download for 15 minutes: retry on the next rerun
    if tickers and closes.empty:
        load_closes.clear()
        notes.append("Yahoo Finance returned no prices just now, so values use average cost. Press **Refresh prices** "
                     "in the sidebar in a minute.")
    s["fx_overrides"], fx_notes = A.fx_fallback(fx, base, needed, s.get("fx_last", {}).get(base, {}), s["fx_overrides"])
    notes += fx_notes
    latest = A.latest_rates(fx)
    if latest and latest != {k: v for k, v in s.get("fx_last", {}).get(base, {}).items() if k in latest}:
        store.set_setting("fx_last", {**s.get("fx_last", {}), base: {**s.get("fx_last", {}).get(base, {}), **latest}})
    log = store.price_log()
    manual = {t: (g["price"].iloc[-1], g["day"].iloc[-1]) for t, g in log.groupby("ticker")} if not log.empty else {}
    quote_cur = dict(zip(holdings["ticker"], holdings["currency"]))
    book = A.order_book(orders, quote_cur, fx, base, s["fx_overrides"])
    held_now = A.holdings_with_orders(holdings, book)
    pos = A.positions(held_now, closes, fx, base, s["fx_overrides"], manual)
    fee_rate = A.fx_on(fx, s["fee"]["currency"], base, pd.Timestamp.today(), s["fx_overrides"])
    fee_base = float(s["fee"]["amount"]) * (fee_rate or 1.0)
    sell = A.sellout(pos, fee_base, s["alerts"]["stock_pct"])
    alerts = A.sell_alerts(sell, s["alerts"]["stock_pct"], s["alerts"]["portfolio_pct"])
    return dict(store=store, s=s, base=base, period=period, holdings=holdings, watch=watch, orders=orders,
                quote_cur=quote_cur, book=book, held_now=held_now, closes=closes, fx=fx, log=log,
                manual=manual, pos=pos, fee_base=fee_base, sell=sell, alerts=alerts, notes=notes)


def show_notes(ctx: dict) -> None:
    for n in ctx["notes"]:
        st.warning(n, icon="⚠️")


NOTHING_VALUED = ("None of your holdings could be valued in {base}: no exchange rate for {curs}. Yahoo Finance did "
                  "not answer and there is no saved rate yet. Press **Refresh prices** in the sidebar, or add a manual "
                  "rate on **Data & settings**.")


def series_for(ctx: dict, ticker: str) -> pd.Series:
    """Yahoo closes for a ticker, or the manual price log when Yahoo has none."""
    if ticker in ctx["closes"].columns:
        return ctx["closes"][ticker].dropna()
    log = ctx["log"]
    g = log[log["ticker"] == ticker]
    return pd.Series(g["price"].values, index=pd.to_datetime(g["day"]), name=ticker)


# ---------------------------------------------------------------- holding / watch forms

FORM_KEYS = ["name", "currency", "sector", "industry", "country", "cap_bucket"]


def apply_lookup(prefix: str) -> None:
    t = st.session_state.get(f"{prefix}_ticker", "").strip().upper()
    if not t:
        st.session_state[f"{prefix}_msg"] = ("warning", "Type a ticker first.")
        return
    info = load_lookup(t)
    if not info:
        st.session_state[f"{prefix}_msg"] = ("warning", f"Yahoo Finance returned nothing for {t}. Check the symbol "
                                                      "(Milan listings end in .MI, Amsterdam .AS, Xetra .DE); if it is right, "
                                                      "Yahoo may be busy: try again in a minute.")
        return
    for k in FORM_KEYS:
        if info.get(k):
            st.session_state[f"{prefix}_{k}"] = info[k]
    if info.get("source") == "basic":
        st.session_state[f"{prefix}_msg"] = ("info", f"Found {info.get('name') or t} ({info['currency']}), but Yahoo's company "
                                                     "profile is unavailable right now: set sector and country yourself, or "
                                                     "look up again later.")
    else:
        st.session_state[f"{prefix}_msg"] = ("success", f"Filled from Yahoo Finance: {info.get('name', t)}. Check before saving.")


def load_holding_into_form() -> None:
    choice = st.session_state.get("h_pick")
    store = get_store()
    h = store.holdings().set_index("ticker").to_dict("index").get(choice) if choice and choice != "New holding" else None
    st.session_state["h_ticker"] = choice if h else ""
    st.session_state["h_shares"] = float(h["shares"]) if h else 0.0
    st.session_state["h_avg_cost"] = float(h["avg_cost"]) if h else 0.0
    for k in FORM_KEYS:
        st.session_state[f"h_{k}"] = (h.get(k) or "") if h else ""
    st.session_state["h_msg"] = None


def remove_selected(kind: str) -> None:
    """Callback: delete the tickers picked in the remove box (h = holdings, w = watchlist)."""
    store = get_store()
    picked = st.session_state.get(f"{kind}_remove", [])
    for t in picked:
        (store.delete_holding if kind == "h" else store.delete_watch)(t)
    st.session_state[f"{kind}_remove"] = []
    st.session_state[f"{kind}_remove_ok"] = False
    if kind == "h" and st.session_state.get("h_pick") in picked:
        st.session_state["h_pick"] = "New holding"
        load_holding_into_form()
    if picked:
        st.toast(f"Removed {', '.join(picked)}")


def remove_box(kind: str, tickers: list[str], what: str) -> None:
    with st.expander(f"Remove {what}", icon="🗑️"):
        picked = st.multiselect(f"Choose {what} to remove", tickers, key=f"{kind}_remove")
        ok = st.checkbox("Yes, remove them permanently", key=f"{kind}_remove_ok", disabled=not picked)
        st.button(f"Remove {len(picked) or ''} {what}".replace("  ", " "), type="primary",
                  disabled=not (picked and ok), on_click=remove_selected, args=(kind,), key=f"{kind}_remove_btn")


def show_msg(prefix: str) -> None:
    m = st.session_state.get(f"{prefix}_msg")
    if m:
        getattr(st, m[0])(m[1])


def sector_select(key: str) -> None:
    opts = [""] + A.SECTORS
    if st.session_state.get(key) not in opts:
        st.session_state[key] = ""
    st.selectbox("Sector", opts, key=key, format_func=lambda x: x or "Not set")


# ---------------------------------------------------------------- pages

def page_portfolio() -> None:
    ctx = context()
    pos, base = ctx["pos"], ctx["base"]
    st.title("Portfolio")
    show_notes(ctx)
    for lvl, msg in ctx["alerts"]:
        getattr(st, lvl)(msg, icon="🚨")
    if ctx["alerts"]:
        st.page_link(PAGE_SELLOUT, label="Open Sell-out for details", icon="➡️")
    if pos.empty:
        st.info("No holdings yet. Add your first one below, or import the JSON you exported from the "
                "Stock Ledger web page on the **Data & settings** page.")
    elif pos["value"].isna().all():
        st.error(NOTHING_VALUED.format(base=base, curs=", ".join(sorted(set(pos["currency"])))))
    else:
        valued = pos.dropna(subset=["value"])
        total, cost = valued["value"].sum(), valued["cost"].sum()
        day_pl = (valued["value"] * valued["day_change"].fillna(0) / (1 + valued["day_change"].fillna(0))).sum()
        c = st.columns(4)
        c[0].metric("Portfolio value", money(total, base, 0), f"{money(day_pl, base, 0)} today")
        priced = valued[valued["cost"] > 0]       # positions without an average cost would count as pure profit
        p_val, p_cost = priced["value"].sum(), priced["cost"].sum()
        c[1].metric("Unrealized P/L", money(p_val - p_cost, base, 0), pct((p_val - p_cost) / p_cost if p_cost else None, signed=True),
                    help=None if len(priced) == len(valued) else f"Leaves out {len(valued) - len(priced)} holdings without an average cost.")
        c[2].metric("Positions", len(pos), f"effective {A.effective_n(valued['weight']):.1f}", delta_color="off")
        top = valued.iloc[0]
        c[3].metric("Largest position", top["ticker"], pct(top["weight"]), delta_color="off")
        book = ctx["book"]
        if not book.empty:
            flows = A.cash_flows(ctx["orders"], ctx["quote_cur"], ctx["fx"], base, ctx["s"]["fx_overrides"])
            irr = A.xirr(list(zip(flows["day"], flows["amount"])) + [(pd.Timestamp.today().normalize(), total)])
            c = st.columns(4)
            c[0].metric("Realized P/L (sold)", money(book["realized_base"].sum(), base, 0),
                        help="Gains or losses already locked in by sell orders, after fees.")
            c[1].metric("Fees paid", money(book["fees_base"].sum(), base, 2),
                        help=f"All broker fees on your orders ({ctx['s']['fee']['amount']:g} {ctx['s']['fee']['currency']} per order by default).")
            c[2].metric("Money-weighted return", pct(irr, signed=True) + " / yr" if pd.notna(irr) else "–",
                        help="XIRR: yearly return that accounts for when you put money in and took it out.")
            c[3].metric("Orders", int(book["orders"].sum()), f"since {book['first_buy'].min():%d %b %Y}", delta_color="off")
            if book["oversold"].any():
                st.warning("Some sell orders sell more shares than you held at that date: "
                           + ", ".join(book.loc[book["oversold"], "ticker"]) + ". Check them on the Orders page.")

        table = pos.assign(
            day_change=pos["day_change"] * 100, pl_pct=pos["pl_pct"] * 100, weight=pos["weight"] * 100,
            price_date=pos["price_date"].apply(lambda d: d.date().isoformat() if pd.notna(d) else ""),
        )
        st.dataframe(
            table[["ticker", "name", "shares", "avg_cost", "price", "currency", "day_change", "value",
                   "pl", "pl_pct", "weight", "sector", "price_date", "price_source"]],
            hide_index=True, width="stretch",
            column_config={
                "ticker": "Ticker", "name": "Name", "shares": st.column_config.NumberColumn("Shares", format="%.4g"),
                "avg_cost": st.column_config.NumberColumn("Avg cost", format="%.2f"),
                "price": st.column_config.NumberColumn("Price", format="%.2f"), "currency": "Ccy",
                "day_change": st.column_config.NumberColumn("Day", format="%+.2f%%"),
                "value": st.column_config.NumberColumn(f"Value ({base})", format="%.0f"),
                "pl": st.column_config.NumberColumn(f"P/L ({base})", format="%+.0f"),
                "pl_pct": st.column_config.NumberColumn("P/L %", format="%+.1f%%", help="In the stock's own currency"),
                "weight": st.column_config.ProgressColumn("Weight", format="%.1f%%", min_value=0, max_value=100),
                "sector": "Sector", "price_date": "Price date", "price_source": "Source",
            })
        st.caption("Prices are Yahoo Finance closes (delayed, dividend-adjusted history). P/L % is in the "
                   "stock's own currency; values convert at the latest FX rate.")
        remove_box("h", ctx["holdings"]["ticker"].tolist(), "holdings")
        st.caption("Removing a holding also removes all of its orders.")

        if not ctx["orders"].empty:
            st.subheader("Your real history")
            held = A.shares_by_day(ctx["orders"])
            hist = A.portfolio_history(ctx["held_now"], ctx["closes"], ctx["fx"], base, ctx["s"]["fx_overrides"], held=held)
            if hist.empty:
                st.caption("No price history for your order dates yet.")
            else:
                flows = A.cash_flows(ctx["orders"], ctx["quote_cur"], ctx["fx"], base, ctx["s"]["fx_overrides"])
                inv = A.net_invested(flows, hist.index)
                st.plotly_chart(charts.value_vs_invested(hist, inv, base), width="stretch")
                st.caption("Value of the shares you actually held each day (from your orders) against the money you "
                           "had put in minus money taken out. Holdings without orders count from their current shares.")

        st.subheader("Today's holdings vs benchmark")
        hist = A.portfolio_history(ctx["held_now"], ctx["closes"], ctx["fx"], base, ctx["s"]["fx_overrides"])
        hist = since(hist, pd.Timestamp.today() - pd.Timedelta(days=WINDOW_DAYS[ctx["period"]]))
        bench = ctx["s"]["benchmark"]
        bench_s = ctx["closes"][bench] if bench in ctx["closes"].columns else None
        if hist.empty:
            st.caption("No price history available yet.")
        else:
            st.plotly_chart(charts.indexed_performance(hist, bench_s, bench), width="stretch")
            r = A.daily_returns(hist)
            c = st.columns(4)
            c[0].metric("Return over window", pct(hist.iloc[-1] / hist.iloc[0] - 1, signed=True))
            c[1].metric("Volatility (yearly)", pct(A.annual_volatility(r)))
            c[2].metric("Worst drop (max drawdown)", pct(A.max_drawdown(hist)))
            c[3].metric(f"Beta vs {bench}", f"{A.beta(r, A.daily_returns(bench_s)):.2f}" if bench_s is not None else "–")
            st.caption("What-if view: today's shares held over the whole window, in "
                       f"{base}. Useful to compare your current mix with the benchmark.")

    st.subheader("Add, edit or remove")
    tickers = ["New holding"] + ctx["holdings"]["ticker"].tolist()
    st.selectbox("Holding", tickers, key="h_pick", on_change=load_holding_into_form)
    if "h_ticker" not in st.session_state:
        load_holding_into_form()
    editing = st.session_state.get("h_pick", "New holding") != "New holding"
    managed = editing and st.session_state["h_pick"] in set(ctx["book"]["ticker"])
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.text_input("Ticker (Yahoo symbol)", key="h_ticker", disabled=editing,
                  placeholder="AAPL, ENI.MI, ASML.AS, VWCE.DE")
    c2.button("Look up on Yahoo", on_click=apply_lookup, args=("h",), width="stretch")
    show_msg("h")
    if managed:
        st.info("Shares and average cost of this stock come from its orders. Change them on the Orders page; "
                "here you can edit name, sector and the other details.")
    with st.form("holding_form", border=True):
        c = st.columns(3)
        c[0].text_input("Name", key="h_name")
        c[1].number_input("Shares", key="h_shares", min_value=0.0, step=1.0, format="%.4f", disabled=managed)
        c[2].number_input("Average cost per share", key="h_avg_cost", min_value=0.0, format="%.4f", disabled=managed,
                          help="In the quote's units. London (.L) stocks quote in pence (GBp).")
        c = st.columns(3)
        c[0].text_input("Currency", key="h_currency", placeholder=base, help="As Yahoo reports it, e.g. EUR, USD, GBp")
        with c[1]:
            sector_select("h_sector")
        c[2].selectbox("Company size", [""] + A.CAPS, key="h_cap_bucket", format_func=lambda x: x or "Not set")
        c = st.columns(2)
        c[0].text_input("Industry", key="h_industry")
        c[1].text_input("Country", key="h_country")
        if st.form_submit_button("Save holding", type="primary"):
            t = st.session_state["h_ticker"].strip().upper()
            if not t or (st.session_state["h_shares"] <= 0 and not managed):
                st.error("Enter a ticker and a number of shares above zero.")
            else:
                ctx["store"].upsert_holding({
                    "ticker": t, "shares": st.session_state["h_shares"], "avg_cost": st.session_state["h_avg_cost"],
                    **{k: st.session_state[f"h_{k}"] for k in FORM_KEYS},
                    "currency": st.session_state["h_currency"] or base,
                })
                st.toast(f"Saved {t}")
                st.rerun()


def page_diversification() -> None:
    ctx = context()
    pos, base, s = ctx["pos"], ctx["base"], ctx["s"]
    st.title("Diversification")
    show_notes(ctx)
    if pos.empty:
        st.info("Add holdings on the Portfolio page first.")
        return
    if pos["value"].isna().all():
        st.error(NOTHING_VALUED.format(base=base, curs=", ".join(sorted(set(pos["currency"])))))
        return
    valued = pos.dropna(subset=["value"])
    w = valued.set_index("ticker")["weight"]
    held = [t for t in w.index if t in ctx["closes"].columns]
    closes = ctx["closes"][held] if held else pd.DataFrame()
    corr = A.correlation(closes) if len(held) >= 2 else pd.DataFrame()

    c = st.columns(4)
    n = A.effective_n(w)
    c[0].metric("Effective positions", f"{n:.1f}", "well spread" if n >= 15 else "moderate" if n >= 8 else "concentrated",
                delta_color="normal" if n >= 15 else "off" if n >= 8 else "inverse",
                help="1 ÷ HHI. How many equal-sized positions your portfolio behaves like.")
    c[1].metric("Concentration (HHI)", f"{A.hhi(w):.3f}", help="Sum of squared weights. 1 means a single stock.")
    c[2].metric("Average correlation", f"{A.average_correlation(corr):.2f}" if not corr.empty else "–",
                help="Average of daily-return correlations between your stocks. Lower means they move "
                     "more independently; above 0.6 they tend to fall together.")
    dr = A.diversification_ratio(w, closes) if not closes.empty else float("nan")
    c[3].metric("Diversification ratio", f"{dr:.2f}" if pd.notna(dr) else "–",
                help="Average stock volatility ÷ portfolio volatility. 1.0 = no benefit; higher is better.")

    st.subheader("Checks")
    findings = A.checks(pos)
    if not any(lvl in ("error", "warning") for lvl, _ in findings):
        st.success("No concentration limits breached.")
    for lvl, msg in findings:
        getattr(st, lvl)(msg)

    st.subheader("Allocation")
    targets = {k: v for k, v in s["sector_targets"].items() if v}
    c = st.columns(2)
    with c[0]:
        st.markdown("**By sector**")
        st.plotly_chart(charts.allocation_bars(A.group_weights(pos, "sector"), "sector", A.LIMITS["sector"], targets),
                        width="stretch")
        st.markdown("**By currency**")
        st.plotly_chart(charts.allocation_bars(A.group_weights(pos, "currency"), "currency"), width="stretch")
    with c[1]:
        st.markdown("**By country**")
        st.plotly_chart(charts.allocation_bars(A.group_weights(pos, "country"), "country", A.LIMITS["country"]),
                        width="stretch")
        st.markdown("**By company size**")
        st.plotly_chart(charts.allocation_bars(A.group_weights(pos, "cap_bucket"), "cap_bucket"), width="stretch")
    st.caption("Orange bars are above the guide limit (30% per sector, 60% per country). Grey ticks are your targets.")

    if not corr.empty:
        st.subheader("How your stocks move together")
        st.plotly_chart(charts.correlation_heatmap(corr, dark=is_dark()), width="stretch")
        st.caption("Correlation of daily returns over the history window. Red pairs move together and "
                   "add little diversification; blue pairs move independently or in opposite directions.")

    st.subheader("Risk per stock")
    bench = s["benchmark"]
    risk = A.asset_risk(closes, ctx["closes"][bench] if bench in ctx["closes"].columns else None)
    if not risk.empty:
        risk = risk.merge(valued[["ticker", "weight"]], on="ticker")
        for col in ("volatility", "max_drawdown", "return_1y", "weight"):
            risk[col] *= 100
        st.dataframe(risk.sort_values("weight", ascending=False), hide_index=True, width="stretch",
                     column_config={
                         "ticker": "Ticker",
                         "weight": st.column_config.NumberColumn("Weight", format="%.1f%%"),
                         "volatility": st.column_config.NumberColumn("Volatility (yearly)", format="%.1f%%"),
                         "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="%.1f%%"),
                         "return_1y": st.column_config.NumberColumn("1-year return", format="%+.1f%%"),
                         "beta": st.column_config.NumberColumn(f"Beta vs {bench}", format="%.2f"),
                     })

    st.subheader("Sector targets and rebalancing")
    st.caption("Set the share you want in each sector. Totals should add up to 100%.")
    current = A.group_weights(pos, "sector").set_index("sector")["weight"]
    editor = pd.DataFrame({"sector": A.SECTORS})
    editor["actual_pct"] = editor["sector"].map(current).fillna(0) * 100
    editor["target_pct"] = editor["sector"].map(lambda x: s["sector_targets"].get(x))
    edited = st.data_editor(
        editor, hide_index=True, width="stretch", key="targets_editor", disabled=["sector", "actual_pct"],
        column_config={"sector": "Sector",
                       "actual_pct": st.column_config.NumberColumn("Actual", format="%.1f%%"),
                       "target_pct": st.column_config.NumberColumn("Target %", min_value=0, max_value=100, step=1.0)})
    total_t = edited["target_pct"].fillna(0).sum()
    c = st.columns([1, 3], vertical_alignment="center")
    if c[0].button("Save targets", type="primary"):
        ctx["store"].set_setting("sector_targets", {r.sector: float(r.target_pct) for r in edited.itertuples()
                                                    if pd.notna(r.target_pct) and r.target_pct > 0})
        st.toast("Targets saved")
        st.rerun()
    (c[1].success if abs(total_t - 100) < 0.01 else c[1].warning)(f"Targets add up to {total_t:.1f}%")

    if targets:
        rb = A.rebalance(pos, targets)
        rb["action"] = rb["change"].apply(lambda v: "–" if pd.isna(v) else "On target" if abs(v) < valued["value"].sum() * 0.005
                                          else f"Buy {money(v, base, 0)}" if v > 0 else f"Sell {money(-v, base, 0)}")
        st.dataframe(rb[["sector", "actual_pct", "target_pct", "action"]], hide_index=True, width="stretch",
                     column_config={"sector": "Sector",
                                    "actual_pct": st.column_config.NumberColumn("Actual", format="%.1f%%"),
                                    "target_pct": st.column_config.NumberColumn("Target", format="%.1f%%"),
                                    "action": "To reach target"})
        cash = st.number_input(f"New money to invest without selling ({base})", min_value=0.0, step=100.0)
        if cash > 0:
            alloc = A.allocate_new_money(pos, targets, cash)
            alloc = alloc[alloc > 0.5]
            if alloc.empty:
                st.info("Every sector is already at or above target.")
            else:
                st.dataframe(alloc.rename("amount").reset_index().rename(columns={"index": "sector"}),
                             hide_index=True, column_config={
                                 "sector": "Sector", "amount": st.column_config.NumberColumn(f"Invest ({base})", format="%.0f")})


def page_watchlist() -> None:
    ctx = context()
    store, watch = ctx["store"], ctx["watch"]
    st.title("Watchlist")
    if watch.empty:
        st.info("Nothing on your watchlist yet. Add tickers you are considering below.")
    else:
        rows = []
        for w in watch.to_dict("records"):
            sr = series_for(ctx, w["ticker"])
            last = sr.iloc[-1] if not sr.empty else None
            tgt = w["target_price"] if pd.notna(w["target_price"]) else None
            r = A.rsi(sr).iloc[-1] if len(sr) > 15 else None
            rows.append({"ticker": w["ticker"], "name": w["name"], "price": last,
                         "m1": A.period_return(sr, 21) * 100, "m3": A.period_return(sr, 63) * 100,
                         "y1": A.period_return(sr, 252) * 100, "ytd": A.ytd_return(sr) * 100,
                         "rsi": r, "trend": A.trend_signal(sr), "target": tgt,
                         "to_target": (last / tgt - 1) * 100 if (tgt and last) else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "ticker": "Ticker", "name": "Name", "price": st.column_config.NumberColumn("Price", format="%.2f"),
            "m1": st.column_config.NumberColumn("1M", format="%+.1f%%"),
            "m3": st.column_config.NumberColumn("3M", format="%+.1f%%"),
            "y1": st.column_config.NumberColumn("1Y", format="%+.1f%%"),
            "ytd": st.column_config.NumberColumn("YTD", format="%+.1f%%"),
            "rsi": st.column_config.NumberColumn("RSI 14", format="%.0f", help="Above 70 often overbought, below 30 oversold"),
            "trend": st.column_config.TextColumn("Trend", help="Price and 50-day average vs 200-day average"),
            "target": st.column_config.NumberColumn("Buy target", format="%.2f"),
            "to_target": st.column_config.NumberColumn("Above target", format="%+.1f%%"),
        })
        remove_box("w", watch["ticker"].tolist(), "watchlist tickers")

    st.subheader("Chart")
    options = list(dict.fromkeys(watch["ticker"].tolist() + ctx["holdings"]["ticker"].tolist()))
    if options:
        t = st.selectbox("Ticker", options)
        sr = series_for(ctx, t)
        wrow = watch.set_index("ticker").to_dict("index").get(t, {})
        tgt = wrow.get("target_price") if wrow and pd.notna(wrow.get("target_price")) else None
        if len(sr) < 2:
            st.caption("No price history for this ticker. Check the symbol, or log prices by hand below.")
        else:
            st.plotly_chart(charts.price_with_trend(sr, A.sma(sr, 50), A.sma(sr, 200), tgt, t), width="stretch")
            st.plotly_chart(charts.rsi_chart(A.rsi(sr)), width="stretch")
            st.caption(f"Trend: {A.trend_signal(sr)}. Signals describe the past and do not predict prices.")
            if t in set(ctx["orders"]["ticker"]):
                st.info(f"You have orders for {t}. See them marked on the price chart on the **Orders** page.", icon="🧾")
        if wrow.get("note"):
            st.markdown(f"**Note:** {wrow['note']}")
        with st.expander("Log a price by hand (for tickers Yahoo does not cover)"):
            c = st.columns(3, vertical_alignment="bottom")
            d = c[0].date_input("Date", value=date.today(), key="log_d")
            p = c[1].number_input("Price", min_value=0.0, format="%.4f", key="log_p")
            if c[2].button("Log price") and p > 0:
                store.log_price(t, d.isoformat(), p)
                st.toast(f"Logged {t} at {p}")
                st.rerun()

    st.subheader("Add to watchlist")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.text_input("Ticker (Yahoo symbol)", key="w_ticker", placeholder="NVDA, RACE.MI")
    c2.button("Look up on Yahoo", key="w_lookup_btn", on_click=apply_lookup, args=("w",), width="stretch")
    show_msg("w")
    with st.form("watch_form"):
        c = st.columns(3)
        c[0].text_input("Name", key="w_name")
        c[1].text_input("Currency", key="w_currency")
        c[2].number_input("Buy target price (0 = none)", min_value=0.0, format="%.2f", key="w_target")
        c = st.columns(3)
        with c[0]:
            sector_select("w_sector")
        c[1].text_input("Country", key="w_country")
        c[2].text_input("Note", key="w_note", placeholder="Why you are watching it")
        if st.form_submit_button("Add to watchlist", type="primary"):
            t = st.session_state.get("w_ticker", "").strip().upper()
            if not t:
                st.error("Enter a ticker.")
            else:
                store.upsert_watch({"ticker": t, "name": st.session_state["w_name"],
                                    "currency": st.session_state["w_currency"],
                                    "target_price": st.session_state["w_target"] or None,
                                    "sector": st.session_state["w_sector"], "country": st.session_state["w_country"],
                                    "note": st.session_state["w_note"]})
                st.toast(f"Watching {t}")
                st.rerun()


def page_planner() -> None:
    ctx = context()
    store, s, base = ctx["store"], ctx["s"], ctx["base"]
    plan = s["plan"]
    st.title("Investment planner")
    st.caption("Projects a monthly investment plan (PAC) at three assumed yearly returns. Results are "
               "before tax and fees and in today's prices only if you subtract inflation from the returns.")
    current = ctx["pos"]["value"].sum() if not ctx["pos"].empty else 0.0
    c = st.columns(3)
    start = c[0].number_input(f"Starting amount ({base})", min_value=0.0, step=1000.0,
                              value=float(plan.get("start") or round(current, -2)))
    monthly = c[1].number_input(f"Monthly contribution ({base})", min_value=0.0, step=50.0, value=float(plan["monthly"]))
    years = c[2].slider("Years", 1, 40, int(plan["years"]))
    c = st.columns(3)
    rates = [c[i].number_input(lbl, value=float(plan["returns"][i]), step=0.5, format="%.1f")
             for i, lbl in enumerate(["Cautious return %", "Middle return %", "Optimistic return %"])]
    if st.button("Remember these inputs"):
        store.set_setting("plan", {"start": start, "monthly": monthly, "years": years, "returns": rates})
        st.toast("Plan saved")

    scen = {f"{r:.1f}% a year": A.projection(start, monthly, years, r) for r in rates}
    st.plotly_chart(charts.projection_chart(scen), width="stretch")
    first = next(iter(scen.values()))
    marks = sorted({y for y in (5, 10, 15, 20, 25, 30, 40) if y <= years} | {years})
    table = pd.DataFrame({"Year": marks, "Paid in": [first.loc[first["month"] == y * 12, "contributed"].iloc[0] for y in marks]})
    for label, df in scen.items():
        table[label] = [df.loc[df["month"] == y * 12, "value"].iloc[0] for y in marks]
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(k, format="%.0f") for k in table.columns if k != "Year"})

    st.subheader("Reach a goal")
    c = st.columns(3)
    goal = c[0].number_input(f"Goal ({base})", min_value=0.0, step=10000.0, value=100000.0)
    gy = c[1].slider("In years", 1, 40, 15, key="goal_years")
    gr = c[2].number_input("Assumed return %", value=float(rates[1]), step=0.5, format="%.1f", key="goal_rate")
    need = A.monthly_needed(goal, start, gy, gr)
    st.metric("Monthly contribution needed", money(need, base, 0))


def fill_order_price() -> None:
    """Callback: put the Yahoo close of the order day into the price field."""
    t = st.session_state.get("o_ticker", "").strip().upper()
    d = pd.Timestamp(st.session_state.get("o_day", date.today()))
    if not t:
        st.session_state["o_msg"] = ("warning", "Type a ticker first.")
        return
    closes = load_closes((t,), "1mo", (d - pd.Timedelta(days=10)).date().isoformat())
    if closes.empty or t not in closes.columns:
        st.session_state["o_msg"] = ("warning", f"No Yahoo prices for {t}. Type the price yourself.")
        return
    s = closes[t].dropna()
    s = s[s.index <= d]
    if s.empty:
        st.session_state["o_msg"] = ("warning", f"No Yahoo price for {t} on or before {d:%d %b %Y}.")
        return
    st.session_state["o_price"] = round(float(s.iloc[-1]), 4)
    st.session_state["o_msg"] = ("info", f"Close of {s.index[-1]:%d %b %Y}: {s.iloc[-1]:.4f}. Replace it with your "
                                         "real execution price if you have it. Note: past closes are adjusted for dividends.")


def remove_orders() -> None:
    picked = st.session_state.get("o_remove", [])
    get_store().delete_orders([int(x.split()[0].lstrip("#")) for x in picked])
    st.session_state["o_remove"] = []
    st.session_state["o_remove_ok"] = False
    if picked:
        st.toast(f"Removed {len(picked)} orders")


def page_orders() -> None:
    ctx = context()
    store, s, base = ctx["store"], ctx["s"], ctx["base"]
    fee = s["fee"]
    st.title("Orders")
    st.caption("Record every buy and sell. Shares, average cost and the money you paid are then computed from "
               f"your orders (average-cost method, FX of the order day, {fee['amount']:g} {fee['currency']} broker "
               "fee per order by default).")

    st.subheader("New order")
    held = ctx["book"].set_index("ticker")["shares"] if not ctx["book"].empty else pd.Series(dtype=float)
    c = st.columns([1, 2, 1.3, 1.3], vertical_alignment="bottom")
    side = c[0].segmented_control("Side", ["BUY", "SELL"], default="BUY", key="o_side")
    c[1].text_input("Ticker (Yahoo symbol)", key="o_ticker", placeholder="AAPL, ENI.MI")
    c[2].date_input("Date", key="o_day", value=date.today(), max_value=date.today(), format="DD/MM/YYYY")
    c[3].button("Use close of that day", on_click=fill_order_price, width="stretch")
    show_msg("o")
    with st.form("order_form"):
        c = st.columns(4)
        c[0].number_input("Shares", key="o_shares", min_value=0.0, step=1.0, format="%.4f")
        c[1].number_input("Price per share", key="o_price", min_value=0.0, format="%.4f",
                          help="In the quote's units (pence for London .L stocks).")
        c[2].number_input("Fee", key="o_fee", min_value=0.0, value=float(fee["amount"]), format="%.2f")
        c[3].text_input("Fee currency", key="o_fee_cur", value=fee["currency"])
        c = st.columns([1, 3])
        c[0].text_input("Currency (new tickers only)", key="o_cur", placeholder="from Yahoo",
                        help="Used only if Yahoo cannot tell the currency of a ticker you do not hold yet.")
        c[1].text_input("Note", key="o_note")
        if st.form_submit_button("Save order", type="primary"):
            t = st.session_state["o_ticker"].strip().upper()
            q, p = st.session_state["o_shares"], st.session_state["o_price"]
            if not t or q <= 0 or p <= 0 or not side:
                st.error("Enter side, ticker, shares and price.")
            elif side == "SELL" and q > float(held.get(t, 0.0)) + 1e-9:
                st.error(f"You hold {float(held.get(t, 0.0)):g} shares of {t} from orders; you cannot sell {q:g}.")
            else:
                if t not in set(ctx["holdings"]["ticker"]):
                    info = load_lookup(t) or {}
                    cur = info.get("currency") or st.session_state["o_cur"].strip()
                    if not cur:
                        st.error(f"Yahoo Finance did not answer for {t} just now, so its currency is unknown. Type the "
                                 "currency (e.g. USD) in 'Currency (new tickers only)' and save again, or retry in a minute.")
                        st.stop()
                    store.upsert_holding({**{k: info.get(k, "") for k in FORM_KEYS}, "ticker": t, "shares": 0, "avg_cost": 0,
                                          "currency": cur})
                elif t not in set(ctx["book"]["ticker"]) and float(ctx["holdings"].set_index("ticker").loc[t, "shares"]) > 0:
                    st.session_state["o_warn"] = (f"{t} had shares entered by hand. From now on its orders decide the "
                                                  "position: add an opening BUY for shares you already had (see below).")
                store.add_order({"ticker": t, "side": side, "day": st.session_state["o_day"].isoformat(), "shares": q,
                                 "price": p, "fee": st.session_state["o_fee"],
                                 "fee_currency": st.session_state["o_fee_cur"] or fee["currency"],
                                 "note": st.session_state["o_note"]})
                st.toast(f"{side} {q:g} {t} @ {p:g} saved")
                st.rerun()
    if st.session_state.get("o_warn"):
        st.warning(st.session_state.pop("o_warn"))

    orders = ctx["orders"]
    if orders.empty:
        st.info("No orders yet.")
    else:
        st.subheader("Positions from orders")
        b = ctx["book"].merge(ctx["holdings"][["ticker", "name", "currency"]], on="ticker", how="left")
        st.dataframe(b[["ticker", "name", "shares", "avg_cost", "currency", "invested_base", "realized_base",
                        "fees_base", "orders", "first_buy"]], hide_index=True, width="stretch", column_config={
            "ticker": "Ticker", "name": "Name", "shares": st.column_config.NumberColumn("Shares", format="%.4g"),
            "avg_cost": st.column_config.NumberColumn("Avg cost (incl. fees)", format="%.4f"), "currency": "Ccy",
            "invested_base": st.column_config.NumberColumn(f"Money in ({base})", format="%.2f",
                                                           help="Paid for the shares still held, at the FX of each order day"),
            "realized_base": st.column_config.NumberColumn(f"Realized ({base})", format="%+.2f"),
            "fees_base": st.column_config.NumberColumn(f"Fees ({base})", format="%.2f"),
            "orders": "Orders", "first_buy": st.column_config.DateColumn("First buy", format="DD/MM/YYYY")})

        st.subheader("Chart with your orders")
        c = st.columns([2, 1, 1], vertical_alignment="bottom")
        tk = c[0].selectbox("Stock", sorted(orders["ticker"].unique()), key="oc_ticker")
        show_avg = c[1].checkbox("50/200-day averages", key="oc_sma")
        spans = ["Last week", "Last month", "From 3 months before first order", "From 1 year before first order",
                 "All history"]
        margin = c[2].selectbox("Show", spans, index=2, key="oc_start",
                                help="Shorter spans download less data and load faster.")
        o_t = orders[orders["ticker"] == tk]
        cur = ctx["quote_cur"].get(tk) or base
        first, today = pd.Timestamp(o_t["day"].min()), pd.Timestamp.today().normalize()
        start = {"Last week": today - pd.Timedelta(days=7), "Last month": today - pd.DateOffset(months=1),
                 "From 3 months before first order": first - pd.DateOffset(months=3),
                 "From 1 year before first order": first - pd.DateOffset(years=1)}.get(margin)
        # moving averages need ~200 trading days before the first visible day
        fetch_from = None if start is None else start - pd.Timedelta(days=300 if show_avg else 5)
        price = load_traded_closes(tk, "max", fetch_from.date().isoformat() if fetch_from is not None else None)
        if price.empty:
            st.warning(f"Yahoo returned no prices for {tk} in this span (unknown or delisted symbol, or Yahoo is "
                       "throttling requests; try again in a minute). Only your orders are shown.")
        sma50, sma200 = since(A.sma(price, 50), start), since(A.sma(price, 200), start)
        price = since(price, start)
        steps = A.avg_cost_steps(orders, tk, ctx["quote_cur"], ctx["fx"], base, ctx["s"]["fx_overrides"])
        if start is not None and not steps.empty:
            before = steps[steps.index < start]
            steps = since(steps, start)
            if not before.empty:   # average cost already in force when the span begins
                steps = pd.concat([pd.Series({start: before.iloc[-1]}), steps]).sort_index()
            hidden = int((pd.to_datetime(o_t["day"]) < start).sum())
            o_t = o_t[pd.to_datetime(o_t["day"]) >= start]
            if hidden:
                st.caption(f"{hidden} earlier orders are outside this span; pick a longer span to see them.")
        st.plotly_chart(charts.price_with_orders(price, o_t, steps, tk, cur,
                                                 sma50 if show_avg else None, sma200 if show_avg else None),
                        width="stretch", key="orders_chart")
        if not price.empty and not steps.empty and pd.notna(steps.iloc[-1]):
            now, avg = float(price.iloc[-1]), float(steps.iloc[-1])
            c = st.columns(3)
            c[0].metric("Price now", f"{now:,.2f} {cur}")
            c[1].metric("Your average cost", f"{avg:,.2f} {cur}", help="Includes buy fees")
            c[2].metric("Price vs average cost", pct(now / avg - 1, signed=True),
                        help="Price only, in the stock's currency. See Sell-out for the result after the sell fee and FX.")
        st.caption("▲ buys and ▼ sells sit at the price you entered; hover one to see how today's price compares. "
                   "The line uses prices as traded (not lowered for dividends), so markers line up with it. "
                   "Yahoo still adjusts these prices for stock splits: orders placed before a split will look off.")

        st.subheader("All orders")
        t_filter = st.multiselect("Filter by ticker", sorted(orders["ticker"].unique()))
        view = orders[orders["ticker"].isin(t_filter)] if t_filter else orders
        view = view.assign(gross=view["shares"] * view["price"]).sort_values(["day", "id"], ascending=False)
        st.dataframe(view[["id", "day", "side", "ticker", "shares", "price", "gross", "fee", "fee_currency", "note"]],
                     hide_index=True, width="stretch", column_config={
                         "id": "#", "day": "Date", "side": "Side", "ticker": "Ticker",
                         "shares": st.column_config.NumberColumn("Shares", format="%.4g"),
                         "price": st.column_config.NumberColumn("Price", format="%.4f"),
                         "gross": st.column_config.NumberColumn("Amount (quote ccy)", format="%.2f"),
                         "fee": st.column_config.NumberColumn("Fee", format="%.2f"), "fee_currency": "Fee ccy",
                         "note": "Note"})
        with st.expander("Remove orders", icon="🗑️"):
            labels = [f"#{r.id} {r.day} {r.side} {r.shares:g} {r.ticker} @ {r.price:g}" for r in orders.itertuples()]
            picked = st.multiselect("Choose orders to remove", labels, key="o_remove")
            ok = st.checkbox("Yes, remove them permanently", key="o_remove_ok", disabled=not picked)
            st.button(f"Remove {len(picked)} orders", type="primary", disabled=not (picked and ok), on_click=remove_orders)

    snap = ctx["holdings"][(ctx["holdings"]["shares"] > 0) & ~ctx["holdings"]["ticker"].isin(set(orders["ticker"]))]
    if not snap.empty:
        st.subheader("Turn hand-entered holdings into orders")
        st.caption("These holdings have shares typed in by hand. Converting creates one opening BUY at the average "
                   "cost on the date you choose, so they join the order history. Fee 0 unless you set one.")
        c = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
        pick = c[0].multiselect("Holdings", snap["ticker"].tolist(), default=snap["ticker"].tolist())
        d = c[1].date_input("Bought on", value=date.today(), max_value=date.today(), format="DD/MM/YYYY", key="conv_day")
        f = c[2].number_input("Fee per order", min_value=0.0, value=0.0, format="%.2f", key="conv_fee")
        if c[3].button("Convert", type="primary", disabled=not pick):
            for h in snap[snap["ticker"].isin(pick)].to_dict("records"):
                store.add_order({"ticker": h["ticker"], "side": "BUY", "day": d.isoformat(), "shares": h["shares"],
                                 "price": h["avg_cost"], "fee": f, "fee_currency": fee["currency"], "note": "Opening position"})
            st.toast(f"Converted {len(pick)} holdings")
            st.rerun()

    st.subheader("Import orders from your broker")
    st.caption("CSV with columns date, ticker, side (buy/sell or acquisto/vendita), shares, price and optionally fee. "
               "Dates like 2025-01-31 or 31/01/2025. Orders already stored are skipped.")
    up = st.file_uploader("Orders CSV", type=["csv"], key="orders_csv")
    if up is not None and st.button("Import orders"):
        st.success(import_orders_csv(store, up.getvalue().decode("utf-8-sig"), float(fee["amount"]), fee["currency"]))
        st.cache_data.clear()


def page_sellout() -> None:
    ctx = context()
    s, base, sell, fee_base = ctx["s"], ctx["base"], ctx["sell"], ctx["fee_base"]
    al = s["alerts"]
    st.title("Sell-out")
    show_notes(ctx)
    st.caption(f"What you would get by selling today, after the broker fee of {s['fee']['amount']:g} {s['fee']['currency']} "
               f"(≈ {money(fee_base, base)}), compared with what you paid including the buy fees. "
               f"Alerts: one stock at −{al['stock_pct']:g}% or worse, whole portfolio at −{al['portfolio_pct']:g}% or worse. "
               "Change these on Data & settings. Before tax.")
    if sell.empty:
        st.info("No priced positions yet.")
        return
    tot = A.sell_summary(sell)
    c = st.columns(4)
    c[0].metric("Money paid", money(tot["invested"], base, 0))
    c[1].metric("You would get", money(tot["net_proceeds"], base, 0), f"after {len(sell)} sell fees", delta_color="off")
    c[2].metric("Result", money(tot["net_result"], base, 0), pct(tot["net_pct"], signed=True))
    breached = int((sell["status"] == "ALERT").sum())
    c[3].metric("Alerts", breached + (tot["net_pct"] <= -al["portfolio_pct"] / 100),
                "none" if not ctx["alerts"] else "check below", delta_color="off" if not ctx["alerts"] else "inverse")

    if ctx["alerts"]:
        for lvl, msg in ctx["alerts"]:
            getattr(st, lvl)(msg, icon="🚨")
    else:
        st.success("No sell alerts. No stock is down 25% or more and the portfolio is above its threshold.".replace(
            "25", f"{al['stock_pct']:g}"))

    st.plotly_chart(charts.sell_result_bars(sell.dropna(subset=["net_pct"]), al["stock_pct"]), width="stretch")
    label = {"ALERT": "🔴 Alert", "Above break-even": "🟢 Above break-even", "Below break-even": "🟠 Below break-even",
             "No cost entered": "⚪ No cost entered"}
    t = sell.assign(status=sell["status"].map(label), net_pct=sell["net_pct"] * 100,
                    to_break_even=sell["to_break_even"] * 100, fee_share=sell["fee_share"] * 100)
    st.dataframe(t[["status", "ticker", "shares", "price", "break_even", "to_break_even", "currency", "value",
                    "sell_fee", "net_proceeds", "invested", "net_result", "net_pct", "fee_share"]],
                 hide_index=True, width="stretch", column_config={
                     "status": "Status", "ticker": "Ticker", "shares": st.column_config.NumberColumn("Shares", format="%.4g"),
                     "price": st.column_config.NumberColumn("Price now", format="%.2f"),
                     "break_even": st.column_config.NumberColumn("Break-even price", format="%.2f",
                                                                 help="Lowest price at which selling everything gives back at least what you paid, fees included"),
                     "to_break_even": st.column_config.NumberColumn("Move to break-even", format="%+.1f%%"),
                     "currency": "Ccy",
                     "value": st.column_config.NumberColumn(f"Value ({base})", format="%.2f"),
                     "sell_fee": st.column_config.NumberColumn(f"Sell fee ({base})", format="%.2f"),
                     "net_proceeds": st.column_config.NumberColumn(f"You get ({base})", format="%.2f"),
                     "invested": st.column_config.NumberColumn(f"Paid ({base})", format="%.2f"),
                     "net_result": st.column_config.NumberColumn(f"Result ({base})", format="%+.2f"),
                     "net_pct": st.column_config.NumberColumn("Result %", format="%+.1f%%"),
                     "fee_share": st.column_config.NumberColumn("Fee % of sale", format="%.2f%%",
                                                                help="How much of the sale the fee eats. Small positions pay relatively more."),
                 })

    st.subheader("Simulate a sale")
    c = st.columns(3)
    tk = c[0].selectbox("Stock", sell["ticker"].tolist())
    row = sell.set_index("ticker").loc[tk]
    q = c[1].number_input("Shares to sell", min_value=0.0, max_value=float(row["shares"]), value=float(row["shares"]),
                          format="%.4f", key=f"sim_q_{tk}")
    p = c[2].number_input(f"Sale price ({row['currency']})", min_value=0.0, value=float(row["price"]), format="%.4f",
                          key=f"sim_p_{tk}")
    fx_now = ctx["pos"].set_index("ticker").loc[tk, "fx"]
    if q > 0:
        frac = q / row["shares"]
        gross = q * p * fx_now
        cost = row["invested"] * frac
        net = gross - fee_base
        c = st.columns(4)
        c[0].metric("Sale amount", money(gross, base))
        c[1].metric("You get after fee", money(net, base))
        c[2].metric("Paid for these shares", money(cost, base))
        c[3].metric("Result", money(net - cost, base), pct((net - cost) / cost if cost else None, signed=True))
        be = (cost + fee_base) / (q * fx_now)
        st.caption(f"Break-even price for {q:g} shares: {be:,.4f} {row['currency']}. Selling in several small orders "
                   f"costs {money(fee_base, base)} each time.")


TASK_NAME = "StockLedgerDailyScan"


def task_status() -> str | None:
    """Next run time of the scheduled scan, or None when no task is registered."""
    import subprocess
    ps = (f"$i = Get-ScheduledTaskInfo -TaskName {TASK_NAME} -ErrorAction Stop; "
          "$i.NextRunTime.ToString('yyyy-MM-dd HH:mm')")      # same output whatever the Windows language
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                           text=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return None
    if r.returncode != 0:
        return None
    out = r.stdout.strip()
    return f"{pd.Timestamp(out):%a %d %b %H:%M}" if out else "registered"


def run_task_script(*args: str) -> str:
    import subprocess
    from pathlib import Path
    script = Path(__file__).resolve().parent / "install_daily_task.ps1"
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *args],
                       capture_output=True, text=True, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return (r.stdout + r.stderr).strip()


PICK_COLUMNS = {
    "ticker": "Ticker", "name": "Name", "sector": "Sector", "country": "Country",
    "price": st.column_config.NumberColumn("Price", format="%.2f"), "currency": "Ccy",
    "r1m": st.column_config.NumberColumn("1M", format="%+.1f%%"),
    "r3m": st.column_config.NumberColumn("3M", format="%+.1f%%"),
    "r6m": st.column_config.NumberColumn("6M", format="%+.1f%%"),
    "r12_1": st.column_config.NumberColumn("12M ex last", format="%+.0f%%", help="12-month return excluding the last month"),
    "rsi": st.column_config.NumberColumn("RSI", format="%.0f"),
    "vol": st.column_config.NumberColumn("Volatility", format="%.0f%%"),
    "below_high": st.column_config.NumberColumn("From 1Y high", format="%+.0f%%"),
    "score": st.column_config.ProgressColumn("Momentum", format="%.0f", min_value=0, max_value=100),
    "fit": st.column_config.ProgressColumn("Fit", format="%.0f", min_value=0, max_value=100),
    "corr": st.column_config.NumberColumn("Corr. w/ portfolio", format="%.2f"),
    "sector_weight": st.column_config.NumberColumn("Your sector %", format="%.0f%%"),
    "eff_n_after": st.column_config.NumberColumn("Eff. positions after", format="%.1f"),
    "held": st.column_config.CheckboxColumn("Held"),
    "sector_rank": st.column_config.NumberColumn("#", format="%d", help="Rank inside its sector"),
    "index": "Index",
}


def picks_table(rows: list[dict], cols: list[str], key: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No stock in this list.")
        return df
    for c in ("r1m", "r3m", "r6m", "r12_1", "vol", "below_high", "sector_weight"):
        if c in df:
            df[c] = df[c] * 100
    show = [c for c in cols if c in df.columns]
    st.dataframe(df[show], hide_index=True, width="stretch", key=key,
                 column_config={c: PICK_COLUMNS.get(c, c) for c in show})
    return df


def pick_detail(df: pd.DataFrame, key: str, ctx: dict) -> None:
    if df.empty:
        return
    t = st.selectbox("Look closer at", df["ticker"].tolist(), key=f"{key}_pick",
                     format_func=lambda x: f"{x}  {df.set_index('ticker').loc[x, 'name']}")
    row = df.set_index("ticker").loc[t]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Why it is on the list**")
        st.markdown("\n".join(f"- {r}" for r in row["reasons"]))
        fee = ctx["fee_base"]
        st.caption(f"Broker fee {money(fee, ctx['base'])} per order: an order of at least "
                   f"{money(fee / 0.005, ctx['base'], 0)} keeps it under 0.5%.")
        in_watch = t in set(ctx["watch"]["ticker"])
        if st.button("On your watchlist" if in_watch else "Add to watchlist", key=f"{key}_watch", disabled=in_watch):
            ctx["store"].upsert_watch({"ticker": t, "name": row.get("name", ""), "currency": row.get("currency", ""),
                                       "sector": row.get("sector", ""), "country": row.get("country", ""),
                                       "note": f"From daily ideas ({date.today():%d %b %Y})"})
            st.toast(f"Added {t} to the watchlist")
            st.rerun()
    with c2:
        sr = load_closes((t,), "1y")
        sr = sr[t].dropna() if t in sr.columns else pd.Series(dtype=float)
        if len(sr) > 2:
            st.plotly_chart(charts.price_with_trend(sr, A.sma(sr, 50), A.sma(sr, 200), None, t),
                            width="stretch", key=f"{key}_chart")


def text_or_empty(v) -> str:
    """Database text that may be NULL (read back as None or NaN)."""
    return v if isinstance(v, str) else ""


def ideas_explorer(picks: dict, ctx: dict) -> None:
    """Ranked or best-per-sector view of every stock in an uptrend, with filters."""
    from ledger import ideas as I
    from ledger.universe import INDEXES

    mom = pd.DataFrame(picks.get("all") or picks.get("standalone", []))
    fit = pd.DataFrame(picks.get("portfolio_all") or picks.get("portfolio", []))
    if mom.empty:
        st.info("No stock passed the trend filter in this scan.")
        return
    if "all" not in picks or "portfolio_all" not in picks:
        st.caption("This older scan stored only the top picks; run a new scan to filter the full list.")
    for d in (mom, fit):
        if not d.empty:
            d["region"] = d["country"].map(I.region_of)

    st.subheader("Ideas")
    c = st.columns([1.3, 1.3, 1.4], vertical_alignment="bottom")
    by = c[0].segmented_control("Rank by", ["Momentum", "Fit for my portfolio"], default="Momentum", key="ix_by")
    view = c[1].segmented_control("Show", ["Best per sector", "Ranked list"], default="Best per sector", key="ix_view")
    if view == "Ranked list":
        n = c[2].slider("How many", 10, 100, 25, step=5, key="ix_n")
    else:
        k = c[2].slider("Per sector", 1, 5, 2, key="ix_k")
    use_fit = by == "Fit for my portfolio"
    base = fit if use_fit else mom
    if use_fit and base.empty:
        st.info("Add holdings to rank stocks by how well they fit your portfolio.")
        return

    f = st.columns(4)
    flt = I.Filters(
        sectors=f[0].multiselect("Sector", sorted(base["sector"].dropna().unique()), key="ix_sec"),
        regions=f[1].multiselect("Region", [r for r in I.REGIONS if r in set(base["region"])], key="ix_reg"),
        countries=f[2].multiselect("Country", sorted(base["country"].dropna().unique()), key="ix_cty"),
        indexes=f[3].multiselect("Index", list(INDEXES), format_func=lambda x: INDEXES[x][0], key="ix_idx"),
    )
    with st.expander("More filters"):
        g = st.columns(5)
        flt.min_score = g[0].slider("Min momentum score", 0, 100, 0, step=5, key="ix_min")
        flt.max_rsi = g[1].slider("Max RSI", 30, 75, 75, key="ix_rsi", help="Lower = avoid stocks that just ran up")
        vol = g[2].slider("Max volatility %/yr", 10, 100, 100, step=5, key="ix_vol")
        flt.max_vol = None if vol >= 100 else vol / 100
        hi = g[3].slider("Max % below 1Y high", 1, 20, 20, key="ix_hi")
        flt.max_below_high = None if hi >= 20 else hi / 100
        flt.hide_held = g[4].checkbox("Hide stocks I hold", key="ix_held", value=use_fit)

    scanned = pd.DataFrame(picks.get("scanned", []))
    if not scanned.empty:
        b = I.breadth(scanned, flt)
        with st.expander("Which sectors are strong", expanded=False, icon="📊"):
            st.plotly_chart(charts.breadth_bars(b), width="stretch", key="ix_breadth")
            st.caption("Share of each sector's scanned stocks that are in an uptrend, for the region/country/index "
                       "filters above. A sector where most stocks trend up is a broad move, not one company.")

    sel = I.apply_filters(base, flt)
    rank_col = "fit" if use_fit else "score"
    out = I.ranked(sel, rank_col, n) if view == "Ranked list" else I.best_per_sector(sel, rank_col, k)
    st.caption(f"{len(sel)} stocks match the filters; showing {len(out)}"
               + ("" if view == "Ranked list" else f", the best {k} of each sector") + ".")
    cols = (["sector_rank"] if view == "Best per sector" else []) + ["sector", "ticker", "name", "country", "price",
            "currency"] + (["fit", "score", "corr", "sector_weight", "eff_n_after"] if use_fit else
                           ["score", "r3m", "r6m", "r12_1", "rsi", "vol", "below_high", "held"])
    df = picks_table(out.to_dict("records"), cols, "ix_table")
    pick_detail(df, "ix", ctx)


def page_ideas() -> None:
    from ledger import commentary as C
    from ledger.daily import index_names, run_scan, scan_settings
    from ledger.universe import INDEXES
    ctx = context()
    store, s = ctx["store"], ctx["s"]
    cfg = scan_settings(s)
    st.title("Daily ideas")
    st.caption("Every weekday morning the scan checks the members of " + index_names(cfg["indexes"]) + " for a healthy "
               "uptrend (price and 50-day average above a rising 200-day average, RSI at most 75, within 20% of the "
               "1-year high, positive 12-month return) and ranks them by momentum. A second list ranks them by how much "
               "they would diversify your portfolio. Past prices only: a shortlist to study, not advice.")

    runs = store.scan_runs()
    c = st.columns([2, 1.2, 1.2], vertical_alignment="bottom")
    with_c = c[1].checkbox("With Claude commentary", value=C.available() and bool(cfg.get("commentary", True)),
                           disabled=not C.available(),
                           help=None if C.available() else "Set the OPENROUTER_API_KEY or ANTHROPIC_API_KEY environment variable to enable it.")
    if c[2].button("Run scan now", type="primary", width="stretch"):
        with st.status("Scanning…", expanded=True) as box:
            bar = st.progress(0.0)
            def progress(f, msg):
                bar.progress(min(1.0, f), text=msg)
            run_scan(store, with_commentary=with_c, notify=False, progress=progress, retry_wait_s=(5,))
            box.update(label="Scan finished", state="complete")
        st.rerun()
    if runs.empty:
        c[0].info("No scan yet. Run one now, or install the daily task below.")
    else:
        labels = {int(r.id): f"{pd.Timestamp(r.run_at):%a %d %b %Y %H:%M}" for r in runs.itertuples()}
        rid = c[0].selectbox("Scan", list(labels), format_func=labels.get, key="ideas_run")
        run = runs.set_index("id").loc[rid]
        if run["status"] != "ok":
            st.error(f"This scan failed: {run['notes']}")
        else:
            m = st.columns(4)
            m[0].metric("Stocks scanned", int(run["n_scanned"]), f"of {int(run['n_universe'])} in the lists", delta_color="off")
            m[1].metric("In an uptrend", int(run["n_eligible"]),
                        f"{run['n_eligible'] / max(run['n_scanned'], 1):.0%} of scanned", delta_color="off")
            m[2].metric("Scan time", f"{run['duration_s']:.0f} s")
            age = (pd.Timestamp.now() - pd.Timestamp(run["run_at"])).days
            m[3].metric("Prices as of", f"{pd.Timestamp(run['run_at']):%d %b}", "today" if age == 0 else f"{age} days ago",
                        delta_color="off" if age <= 1 else "inverse")
            if text_or_empty(run["notes"]):
                st.caption(run["notes"])
            picks = store.scan_picks(rid)
            note_text, note_meta = text_or_empty(run["commentary"]), text_or_empty(run["commentary_note"])
            if note_text:
                with st.expander("Claude's morning note", expanded=True, icon="🗒️"):
                    st.markdown(note_text)
                    st.caption(f"{note_meta} Based on the numbers and Claude's general knowledge; no live news.".strip())
            elif note_meta:
                st.caption(f"No commentary: {note_meta}")

            ideas_explorer(picks, ctx)

    st.subheader("Morning schedule")
    nxt = task_status()
    c = st.columns([2, 1, 1], vertical_alignment="bottom")
    if nxt:
        c[0].success(f"Daily scan is scheduled (Monday–Friday). Next run: {nxt}.")
    else:
        c[0].info("No daily task yet. Install it to scan every weekday morning, even with the app closed.")
    at = c[1].time_input("Time", value=pd.Timestamp("08:00").time(), step=900, key="task_time")
    if c[2].button("Install / update task" if not nxt else "Update time", width="stretch"):
        st.code(run_task_script("-Time", at.strftime("%H:%M")))
    if nxt and st.button("Remove daily task"):
        st.code(run_task_script("-Remove"))
    with st.expander("Scan settings"):
        idx = st.multiselect("Lists to scan", list(INDEXES), default=cfg["indexes"], format_func=lambda k: INDEXES[k][0])
        ls_idx = st.multiselect("Lists for Long shots (analyst targets)", list(INDEXES), default=cfg["longshot_indexes"],
                                format_func=lambda k: INDEXES[k][0],
                                help="S&P 400 and 600 add about 1,000 mid and small caps: more extreme gaps to target, "
                                     "slower first collection of analyst targets.")
        top = st.slider("Stocks per list", 5, 25, int(cfg.get("top_n", 10)))
        com = st.checkbox("Claude commentary in the morning run", value=bool(cfg.get("commentary", True)))
        st.caption("Claude commentary uses Claude Opus 5.5 (roughly 5–15 US cents per morning) through OpenRouter or the "
                   "Anthropic API. It needs the OPENROUTER_API_KEY or ANTHROPIC_API_KEY environment variable: "
                   + {"openrouter": "OpenRouter key found.", "anthropic": "Anthropic key found."}.get(C.provider(), "not set yet."))
        if st.button("Save scan settings"):
            store.set_setting("scan", {"indexes": idx or cfg["indexes"], "top_n": top, "commentary": com,
                                       "longshot_indexes": ls_idx or cfg["longshot_indexes"]})
            st.toast("Scan settings saved")
            st.rerun()


LONGSHOT_COLUMNS = {
    "sector_rank": st.column_config.NumberColumn("#", format="%d", help="Rank inside its sector"),
    "sector": "Sector", "ticker": "Ticker", "name": "Name", "country": "Country", "currency": "Ccy",
    "price": st.column_config.NumberColumn("Price", format="%.2f"),
    "target_mean": st.column_config.NumberColumn("Avg target", format="%.2f"),
    "upside": st.column_config.NumberColumn("To target", format="%+.0f%%", help="Average analyst target vs price"),
    "upside_low": st.column_config.NumberColumn("To lowest", format="%+.0f%%", help="Lowest analyst target vs price"),
    "n_analysts": st.column_config.NumberColumn("Analysts", format="%d"),
    "rating": "Rating",
    "score": st.column_config.ProgressColumn("Score", format="%.0f", min_value=0, max_value=100,
                                             help="Gap to target, analyst agreement, rating and coverage"),
    "vol": st.column_config.NumberColumn("Volatility", format="%.0f%%"),
    "r3m": st.column_config.NumberColumn("3M", format="%+.0f%%"),
    "falling_knife": st.column_config.CheckboxColumn("Falling knife", help="Down >15% in 3 months, below falling averages"),
    "shares": st.column_config.NumberColumn("Shares for stake", format="%d"),
    "fees_pct": st.column_config.NumberColumn("Fees in+out", format="%.1f%%", help="Buy + sell fee vs money invested"),
    "needed": st.column_config.NumberColumn("Rise to break even", format="%+.1f%%"),
    "held": st.column_config.CheckboxColumn("Held"),
}


def longshot_run(store: Store):
    """Latest successful scan that has long-shot lists: (run row, picks) or (None, {})."""
    runs = store.scan_runs()
    for r in runs[runs["status"] == "ok"].itertuples():
        picks = store.scan_picks(int(r.id))
        if "longshot_all" in picks:
            return r, picks
    return None, {}


def page_longshots() -> None:
    from ledger import ideas as I
    from ledger import longshots as L
    from ledger.daily import longshot_step
    from ledger.universe import INDEXES
    ctx = context()
    store, s, base = ctx["store"], ctx["s"], ctx["base"]
    cfg = s.get("longshots") or {}
    st.title("Long shots")
    st.warning("**High risk, small money only.** These stocks trade far below the average price target of the "
               "analysts who follow them. Targets are often too optimistic, and a big gap usually means the market "
               "doubts them: the company may have a real problem. Small caps can lose most of their value. Keep the "
               "total you put here to an amount you can afford to lose.", icon="🎯")

    run, picks = longshot_run(store)
    held = set(ctx["pos"]["ticker"]) if not ctx["pos"].empty else set(ctx["holdings"]["ticker"])
    c = st.columns([3, 1.3], vertical_alignment="bottom")
    if run is None:
        c[0].info("No long-shot list yet. It is built by the morning scan, or press the button.")
    else:
        c[0].caption(f"From the scan of {pd.Timestamp(run.run_at):%a %d %b %Y %H:%M}. "
                     + next((n for n in text_or_empty(run.notes).split(". ") if n.startswith("Long shots")), ""))
    target_run = run.id if run is not None else (int(store.scan_runs().iloc[0]["id"]) if not store.scan_runs().empty else None)
    if c[1].button("Collect targets and re-rank", width="stretch", disabled=target_run is None,
                   help="Downloads prices for every stock in the lists and up to 400 analyst targets (a few minutes)."):
        with st.status("Collecting…", expanded=True) as box:
            bar = st.progress(0.0)
            _, note = longshot_step(store, int(target_run), held, say=lambda f, m: bar.progress(min(1.0, max(0.0, (f - 0.8) / 0.05)), text=m))
            box.update(label=note, state="complete")
        st.rerun()
    if target_run is None:
        st.caption("Run a scan on **Daily ideas** first.")
        return

    df = pd.DataFrame(picks.get("longshot_all", []))
    if df.empty:
        st.info("No stock with enough analyst coverage is below its target yet. Targets are collected a few hundred "
                "per run; press the button above to collect more.")
        return

    st.subheader("Your stake")
    c = st.columns(4)
    stake = c[0].number_input(f"Money per idea ({base})", min_value=10.0, value=float(cfg.get("stake", 150.0)), step=10.0,
                              key="ls_stake")
    cheap = c[1].number_input("Cheap share: price up to", min_value=1.0, value=float(cfg.get("cheap_price", 20.0)),
                              step=1.0, key="ls_cheap", help="In the stock's own currency")
    cheap_only = c[2].toggle("Cheap shares only", key="ls_cheap_only")
    affordable = c[3].toggle("Only what the stake can buy", value=True, key="ls_afford")
    curs = sorted(c_ for c_ in df["currency"].dropna().unique() if c_)
    fx = load_fx(tuple(curs), base, "1mo")
    rates = {c_: A.fx_on(fx, c_, base, pd.Timestamp.today(), s["fx_overrides"]) for c_ in curs}
    df = L.stake_view(df, stake, ctx["fee_base"], {k: v for k, v in rates.items() if v})
    st.caption(f"Broker fee {money(ctx['fee_base'], base)} to buy and again to sell: on {money(stake, base, 0)} that is "
               f"{2 * ctx['fee_base'] / stake:.1%} of the stake before the stock has to move. Whole shares only.")

    df["region"] = df["country"].map(I.region_of)
    c = st.columns([1.3, 1.4], vertical_alignment="bottom")
    view = c[0].segmented_control("Show", ["Ranked list", "Best per sector"], default="Ranked list", key="ls_view")
    if view == "Best per sector":
        k = c[1].slider("Per sector", 1, 5, 2, key="ls_k")
    else:
        n = c[1].slider("How many", 10, 100, 25, step=5, key="ls_n")
    f = st.columns(4)
    flt = I.Filters(
        sectors=f[0].multiselect("Sector", sorted(df["sector"].dropna().unique()), key="ls_sec"),
        regions=f[1].multiselect("Region", [r for r in I.REGIONS if r in set(df["region"])], key="ls_reg"),
        countries=f[2].multiselect("Country", sorted(df["country"].dropna().unique()), key="ls_cty"),
        indexes=f[3].multiselect("Index", list(INDEXES), format_func=lambda x: INDEXES[x][0], key="ls_idx"),
    )
    with st.expander("More filters"):
        g = st.columns(4)
        min_up = g[0].slider("Min gap to target %", 0, 200, 20, step=5, key="ls_up") / 100
        min_n = g[1].slider("Min analysts", 3, 30, int(cfg.get("min_analysts", 3)), key="ls_minn")
        knives = g[2].checkbox("Hide falling knives", key="ls_knife")
        flt.hide_held = g[3].checkbox("Hide stocks I hold", key="ls_held")
    sel = L.narrow(I.apply_filters(df, flt), max_price=cheap if cheap_only else None, min_upside=min_up,
                   min_analysts=min_n, hide_knives=knives, affordable_only=affordable)
    out = I.ranked(sel, "score", n) if view == "Ranked list" else I.best_per_sector(sel, "score", k)
    st.caption(f"{len(sel)} of {len(df)} stocks match; showing {len(out)}. "
               f"{int(df['falling_knife'].sum())} of all are falling knives.")
    if out.empty:
        st.info("Nothing matches. Loosen the filters or raise the stake.")
        return
    show = out.copy()
    for col in ("upside", "upside_low", "vol", "r3m", "fees_pct", "needed"):
        show[col] = show[col] * 100
    cols = (["sector_rank"] if view == "Best per sector" else []) + [
        "sector", "ticker", "name", "price", "currency", "target_mean", "upside", "upside_low", "n_analysts", "rating",
        "score", "vol", "r3m", "falling_knife", "shares", "fees_pct", "held"]
    cols = [c_ for c_ in cols if c_ in show.columns]
    st.dataframe(show[cols], hide_index=True, width="stretch", key="ls_table",
                 column_config={c_: LONGSHOT_COLUMNS.get(c_, c_) for c_ in cols})

    t = st.selectbox("Look closer at", out["ticker"].tolist(), key="ls_pick",
                     format_func=lambda x: f"{x}  {out.set_index('ticker').loc[x, 'name']}")
    row = out.set_index("ticker").loc[t]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Why it is on the list**")
        st.markdown("\n".join(f"- {r}" for r in row["reasons"]))
        if row["shares"] > 0:
            st.markdown(f"**With {money(stake, base, 0)}:** {int(row['shares'])} shares for "
                        f"{money(row['invested'], base)}; fees in and out {money(2 * ctx['fee_base'], base)} "
                        f"({row['fees_pct']:.1%}). Break-even price {row['break_even']:,.2f} {row['currency']} "
                        f"({row['needed']:+.1%}).")
        else:
            st.markdown(f"**One share costs more than {money(stake, base, 0)}.**")
        if row["fees_high"]:
            st.caption("Fees eat more than 3% of this stake: a bigger stake, or a cheaper stock, wastes less.")
        in_watch = t in set(ctx["watch"]["ticker"])
        if st.button("On your watchlist" if in_watch else "Add to watchlist", key="ls_watch", disabled=in_watch):
            store.upsert_watch({"ticker": t, "name": row.get("name", ""), "currency": row.get("currency", ""),
                                "sector": row.get("sector", ""), "country": row.get("country", ""),
                                "note": f"Long shot ({date.today():%d %b %Y}): avg analyst target {row['target_mean']:,.2f}"})
            st.toast(f"Added {t} to the watchlist")
            st.rerun()
    with c2:
        sr = load_closes((t,), "1y")
        sr = sr[t].dropna() if t in sr.columns else pd.Series(dtype=float)
        if len(sr) > 2:
            st.plotly_chart(charts.price_with_trend(sr, A.sma(sr, 50), A.sma(sr, 200), None, t,
                                                    analyst={"Low": row["target_low"], "Average": row["target_mean"],
                                                             "High": row["target_high"]}),
                            width="stretch", key="ls_chart")
        st.caption(f"Analyst targets as of {pd.Timestamp(row['targets_date']):%d %b %Y} (Yahoo Finance).")

    if st.button("Save stake, cheap price and min analysts as defaults"):
        store.set_setting("longshots", {"stake": stake, "cheap_price": cheap, "min_analysts": min_n})
        st.toast("Saved")


def page_data() -> None:
    store = get_store()
    s = store.settings()
    st.title("Data & settings")

    st.subheader("Settings")
    c = st.columns(3)
    base_opts = ["EUR", "USD", "GBP", "CHF"]
    base = c[0].selectbox("Base currency", base_opts, index=base_opts.index(s["base_currency"]) if s["base_currency"] in base_opts else 0)
    bench = c[1].text_input("Benchmark ticker", s["benchmark"], help="URTH = MSCI World ETF; ^GSPC = S&P 500; FTSEMIB.MI = FTSE MIB")
    rf = c[2].number_input("Risk-free rate % (for Sharpe)", value=float(s["risk_free_pct"]), step=0.25)
    st.markdown("**Broker fee and sell alerts**")
    c = st.columns(4)
    fee_amt = c[0].number_input("Fee per order", min_value=0.0, value=float(s["fee"]["amount"]), step=0.01, format="%.2f",
                                help="Charged on every buy and every sell. You can still change it per order.")
    fee_cur = c[1].text_input("Fee currency", s["fee"]["currency"])
    a_stock = c[2].number_input("Alert: one stock down %", min_value=1.0, max_value=100.0,
                                value=float(s["alerts"]["stock_pct"]), step=1.0,
                                help="Alert when selling a stock now would return this much less than you paid, fees included.")
    a_port = c[3].number_input("Alert: whole portfolio down %", min_value=1.0, max_value=100.0,
                               value=float(s["alerts"]["portfolio_pct"]), step=1.0)
    st.markdown("**Manual FX rates** (used only when Yahoo has no quote)")
    fx_df = pd.DataFrame({"currency": list(s["fx_overrides"].keys()), "rate": list(s["fx_overrides"].values())})
    fx_edit = st.data_editor(fx_df, num_rows="dynamic", hide_index=True, key="fx_editor",
                             column_config={"currency": "Currency (ISO)", "rate": st.column_config.NumberColumn(f"1 unit in {base}", format="%.4f")})
    if st.button("Save settings", type="primary"):
        store.set_setting("base_currency", base)
        store.set_setting("benchmark", bench.strip().upper())
        store.set_setting("risk_free_pct", rf)
        store.set_setting("fee", {"amount": fee_amt, "currency": (fee_cur or "USD").strip().upper()})
        store.set_setting("alerts", {"stock_pct": a_stock, "portfolio_pct": a_port})
        store.set_setting("fx_overrides", {str(r.currency).upper(): float(r.rate) for r in fx_edit.itertuples()
                                           if r.currency and pd.notna(r.rate) and r.rate > 0})
        st.cache_data.clear()
        st.toast("Settings saved")
        st.rerun()

    st.subheader("Import")
    st.caption("A JSON export (from this app or the Stock Ledger web page), or a broker CSV of current holdings "
               "with columns such as ticker, shares, avg_cost, currency. Separators , and ; and decimal commas work. "
               "A ticker you already hold is replaced. Import order history on the Orders page.")
    up = st.file_uploader("Choose a file", type=["json", "csv"])
    skip = st.checkbox("Skip example entries", value=True)
    if up is not None and st.button("Import file"):
        text = up.getvalue().decode("utf-8-sig")
        try:
            msg = import_json(store, text, skip) if up.name.lower().endswith(".json") else import_csv(store, text)
            st.success(msg)
            st.cache_data.clear()
        except Exception as exc:
            st.error(f"Could not import {up.name}: {exc}")

    st.subheader("Export")
    c = st.columns(2)
    c[0].download_button("Download JSON (everything)", export_json(store, context()["held_now"]), f"stock-ledger-{date.today()}.json",
                         "application/json", width="stretch")
    c[1].download_button("Download holdings CSV", export_holdings_csv(store), f"stock-ledger-holdings-{date.today()}.csv",
                         "text/csv", width="stretch")
    st.caption(f"The database file is {store.path}. Copy it to back up everything.")

    st.subheader("Delete everything")
    confirm = st.checkbox("Yes, delete all holdings, orders, watchlist entries, price logs and settings")
    if st.button("Delete all data", disabled=not confirm):
        store.wipe()
        st.cache_data.clear()
        st.toast("All data deleted")
        st.rerun()


# ---------------------------------------------------------------- shell

with st.sidebar:
    st.markdown("### 📒 Stock Ledger")
    st.radio("History window", ["1w", "1m", "1y", "2y", "5y"], index=3, key="period", horizontal=True,
             help="Shorter windows download much faster. Risk figures, correlations and 200-day averages need "
                  "at least a few months of history and show '–' on 1w/1m.")
    if st.button("Refresh prices", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Prices from Yahoo Finance, refreshed every 15 minutes. For personal tracking and study, "
               "not financial advice.")

PAGE_SELLOUT = st.Page(page_sellout, title="Sell-out", icon="🚦", url_path="sellout")
nav = st.navigation([
    st.Page(page_portfolio, title="Portfolio", icon="💼", url_path="portfolio", default=True),
    st.Page(page_orders, title="Orders", icon="🧾", url_path="orders"),
    PAGE_SELLOUT,
    st.Page(page_ideas, title="Daily ideas", icon="💡", url_path="ideas"),
    st.Page(page_longshots, title="Long shots", icon="🎯", url_path="longshots"),
    st.Page(page_diversification, title="Diversification", icon="🧩", url_path="diversification"),
    st.Page(page_watchlist, title="Watchlist", icon="👀", url_path="watchlist"),
    st.Page(page_planner, title="Planner", icon="📈", url_path="planner"),
    st.Page(page_data, title="Data & settings", icon="⚙️", url_path="data"),
])
nav.run()
