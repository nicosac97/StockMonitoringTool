"""Plotly figures. Colors follow one fixed categorical order so a series keeps its color everywhere."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # blue, orange, aqua, yellow
MUTED = "#898781"
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#d9d8d3"], [1.0, "#e34948"]]  # blue <-> red, gray midpoint


def _layout(fig: go.Figure, height: int = 340, **kw) -> go.Figure:
    opts = dict(height=height, margin=dict(l=8, r=8, t=30, b=8), hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_layout(**{**opts, **kw})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1)
    return fig


def indexed_performance(portfolio: pd.Series, benchmark: pd.Series | None, bench_name: str) -> go.Figure:
    """Both series rebased to 100 on one axis (never two y-axes)."""
    fig = go.Figure()
    p = portfolio.dropna()
    if p.empty:
        return _layout(fig)
    fig.add_scatter(x=p.index, y=p / p.iloc[0] * 100, name="Your holdings", line=dict(color=S1, width=2))
    if benchmark is not None:
        b = benchmark.dropna()
        b = b[b.index >= p.index[0]]
        if not b.empty:
            fig.add_scatter(x=b.index, y=b / b.iloc[0] * 100, name=bench_name, line=dict(color=S2, width=2))
    fig.update_yaxes(title_text="Indexed (start = 100)")
    return _layout(fig)


def value_vs_invested(value: pd.Series, invested: pd.Series, base: str) -> go.Figure:
    """Market value against net money paid in, same currency, one axis."""
    fig = go.Figure()
    fig.add_scatter(x=invested.index, y=invested, name="Money paid in (net)", line=dict(color=MUTED, width=2, shape="hv"),
                    fill="tozeroy", fillcolor="rgba(137,135,129,0.12)")
    fig.add_scatter(x=value.index, y=value, name="Market value", line=dict(color=S1, width=2))
    fig.update_yaxes(title_text=base)
    return _layout(fig)


def sell_result_bars(sell: pd.DataFrame, alert_pct: float) -> go.Figure:
    """Result of selling each position now vs money paid, with the alert threshold marked."""
    d = sell.sort_values("net_pct")
    colors = ["#d03b3b" if s == "ALERT" else (S1 if r >= 0 else S2) for s, r in zip(d["status"], d["net_pct"])]
    fig = go.Figure(go.Bar(x=d["net_pct"] * 100, y=d["ticker"], orientation="h", marker_color=colors,
                           text=[f"{v:+.1%}" for v in d["net_pct"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:+.1f}% after fees<extra></extra>", name="Result if sold now"))
    fig.add_vline(x=-alert_pct, line=dict(color="#d03b3b", width=1, dash="dash"),
                  annotation_text=f"Alert −{alert_pct:g}%", annotation_position="top")
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    lo = min(-alert_pct * 1.2, float(d["net_pct"].min()) * 120)
    hi = max(10.0, float(d["net_pct"].max()) * 120)
    fig.update_xaxes(range=[lo, hi], ticksuffix="%")
    return _layout(fig, height=max(200, 34 * len(d) + 70), hovermode="closest", showlegend=False, bargap=0.35)


def allocation_bars(df: pd.DataFrame, label_col: str, limit: float | None = None,
                    targets: dict | None = None) -> go.Figure:
    d = df.head(8).copy()
    if len(df) > 8:
        d.loc[len(d)] = {label_col: f"Other ({len(df) - 8})", "value": df["value"].iloc[8:].sum(),
                         "weight": df["weight"].iloc[8:].sum()}
    d = d.iloc[::-1]
    colors = [S2 if (limit and w > limit and k != "ETF / Fund") else S1 for k, w in zip(d[label_col], d["weight"])]
    fig = go.Figure(go.Bar(x=d["weight"] * 100, y=d[label_col], orientation="h", marker_color=colors,
                           text=[f"{w:.1%}" for w in d["weight"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>", name="Weight"))
    if targets:
        t = [(k, v) for k, v in targets.items() if k in set(d[label_col]) and v]
        if t:
            fig.add_scatter(x=[v for _, v in t], y=[k for k, _ in t], mode="markers", name="Target",
                            marker=dict(symbol="line-ns", size=18, line=dict(width=3, color=MUTED)),
                            hovertemplate="%{y} target: %{x:.1f}%<extra></extra>")
    top = max([*(d["weight"] * 100), *([v for v in (targets or {}).values() if v] or [0])], default=10)
    fig.update_xaxes(range=[0, top * 1.25], ticksuffix="%")
    return _layout(fig, height=max(180, 34 * len(d) + 60), hovermode="closest", showlegend=bool(targets),
                   bargap=0.35)


def correlation_heatmap(corr: pd.DataFrame, dark: bool = False) -> go.Figure:
    scale = [DIVERGING[0], [0.5, "#383835" if dark else "#f0efec"], DIVERGING[2]]   # neutral middle per theme
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale=scale, text=corr.round(2).values, texttemplate="%{text}",
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                               colorbar=dict(title="Corr.", thickness=10)))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=max(300, 34 * len(corr) + 80), hovermode="closest")


def price_with_trend(series: pd.Series, sma50: pd.Series, sma200: pd.Series, target: float | None,
                     name: str, analyst: dict[str, float] | None = None) -> go.Figure:
    """Price with moving averages; optional buy target and analyst targets {"Low": x, "Average": y, "High": z}."""
    fig = go.Figure()
    fig.add_scatter(x=series.index, y=series, name=name, line=dict(color=S1, width=2))
    if sma50.notna().any():
        fig.add_scatter(x=sma50.index, y=sma50, name="50-day average", line=dict(color=S2, width=2))
    if sma200.notna().any():
        fig.add_scatter(x=sma200.index, y=sma200, name="200-day average", line=dict(color=S3, width=2))
    if target:
        fig.add_hline(y=target, line=dict(color=MUTED, width=1, dash="dash"),
                      annotation_text=f"Buy target {target:g}", annotation_position="top left")
    for label, y in (analyst or {}).items():
        if y and pd.notna(y):
            fig.add_hline(y=y, line=dict(color=S4 if label == "Average" else MUTED, width=1.5 if label == "Average" else 1,
                                         dash="dash" if label == "Average" else "dot"),
                          annotation_text=f"{label} analyst target {y:,.2f}", annotation_position="top left")
    return _layout(fig, height=360)


def rsi_chart(r: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=r.index, y=r, name="RSI 14", line=dict(color=S1, width=2)))
    for level, text in ((70, "Overbought 70"), (30, "Oversold 30")):
        fig.add_hline(y=level, line=dict(color=MUTED, width=1, dash="dot"),
                      annotation_text=text, annotation_position="top left")
    fig.update_yaxes(range=[0, 100])
    return _layout(fig, height=200, showlegend=False)


def projection_chart(scenarios: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    colors = [S1, S2, S3, S4]
    first = next(iter(scenarios.values()), None)
    if first is not None:
        fig.add_scatter(x=first["year"], y=first["contributed"], name="Money paid in",
                        line=dict(color=MUTED, width=2), fill="tozeroy", fillcolor="rgba(137,135,129,0.12)")
    for (label, df), c in zip(scenarios.items(), colors):
        fig.add_scatter(x=df["year"], y=df["value"], name=label, line=dict(color=c, width=2))
    fig.update_xaxes(title_text="Years")
    return _layout(fig, height=380)


BUY_COLOR, SELL_COLOR = S3, S2   # aqua up-triangles, orange down-triangles: shape and label carry the meaning too


def price_with_orders(price: pd.Series, orders: pd.DataFrame, avg_cost: pd.Series, name: str, currency: str,
                      sma50: pd.Series | None = None, sma200: pd.Series | None = None) -> go.Figure:
    """Traded price with every BUY/SELL of one stock marked at its date and price, plus the average cost."""
    fig = go.Figure()
    fig.add_scatter(x=price.index, y=price, name=f"{name} price", line=dict(color=S1, width=2),
                    hovertemplate="%{x|%d %b %Y}: %{y:,.2f} " + currency + "<extra></extra>")
    if sma50 is not None and sma50.notna().any():
        fig.add_scatter(x=sma50.index, y=sma50, name="50-day average", line=dict(color=S4, width=1.5),
                        hoverinfo="skip")
    if sma200 is not None and sma200.notna().any():
        fig.add_scatter(x=sma200.index, y=sma200, name="200-day average", line=dict(color=MUTED, width=1.5),
                        hoverinfo="skip")
    if not avg_cost.empty:
        steps = avg_cost.copy()
        if not price.empty:
            steps.loc[price.index[-1]] = steps.iloc[-1]   # carry today's average cost to the right edge
        fig.add_scatter(x=steps.index, y=steps, name="Your average cost", mode="lines",
                        line=dict(color=MUTED, width=2, dash="dash", shape="hv"),
                        hovertemplate="Average cost %{y:,.2f} " + currency + "<extra></extra>")
    last = float(price.iloc[-1]) if not price.empty else None
    for side, symbol, color in (("BUY", "triangle-up", BUY_COLOR), ("SELL", "triangle-down", SELL_COLOR)):
        o = orders[orders["side"] == side]
        if o.empty:
            continue
        vs_now = [(last / p - 1) if (last and p) else None for p in o["price"]]
        text = [f"{side} {q:g} @ {p:,.2f} {currency}<br>{pd.Timestamp(d):%d %b %Y}"
                + (f"<br>Now {v:+.1%} vs this price" if v is not None else "")
                for q, p, d, v in zip(o["shares"], o["price"], o["day"], vs_now)]
        fig.add_scatter(x=pd.to_datetime(o["day"]), y=o["price"], mode="markers", name=side.title(),
                        marker=dict(symbol=symbol, size=14, color=color, line=dict(width=2, color="rgba(255,255,255,0.9)")),
                        text=text, hovertemplate="%{text}<extra></extra>")
    fig.update_xaxes(rangeselector=dict(buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(step="all", label="All")]))
    fig.update_yaxes(title_text=currency)
    return _layout(fig, height=440, hovermode="closest")


def breadth_bars(b: pd.DataFrame) -> go.Figure:
    """Share of each sector's stocks in an uptrend (one series, one colour)."""
    d = b.iloc[::-1]
    fig = go.Figure(go.Bar(x=d["share"] * 100, y=d["sector"], orientation="h", marker_color=S1,
                           text=[f"{u:.0f} of {n:.0f}" for u, n in zip(d["uptrend"], d["scanned"])],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.0f}% in an uptrend<extra></extra>", name="In uptrend"))
    fig.update_xaxes(range=[0, 115], ticksuffix="%")
    return _layout(fig, height=max(220, 30 * len(d) + 60), hovermode="closest", showlegend=False, bargap=0.35)
