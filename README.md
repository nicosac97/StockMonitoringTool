# Stock Monitoring Tool (Stock Ledger)

A personal tool to monitor a stock portfolio, follow market trends and plan investments. It runs
locally on Windows as a [Streamlit](https://streamlit.io) app. Prices and FX rates come from Yahoo
Finance, and your data stays in a SQLite file on your PC.

> For personal tracking and study only. Not financial advice. Prices are delayed daily closes.

## What it does

| Area | Capabilities |
|------|--------------|
| **Portfolio** | Current value, day change, unrealized and realized P/L, fees paid, money-weighted return (XIRR), value vs money invested over time, comparison with a benchmark (default MSCI World ETF `URTH`). |
| **Orders** | Record buys and sells (date, shares, price, fee). The price can be filled from Yahoo's close on that day. Shares, average cost and money paid are computed from your orders using the average-cost method, the FX rate of each order day and fees. Import broker order CSVs. A chart per stock marks each BUY/SELL on the price line so you can see gains and losses. |
| **Sell-out** | What selling now would return after the broker fee, compared with what you paid. Break-even price per stock and a sale simulator. Alerts when one stock is down 25% or the whole portfolio 15%; both levels are configurable. |
| **Daily ideas** | Weekday-morning scan of S&P 500, Nasdaq 100, Euro Stoxx 50 and FTSE MIB (about 600 stocks). Keeps stocks in a healthy uptrend and ranks them by momentum. A second list ranks them by how well they would diversify your portfolio. Filter by sector, region, country, index, RSI and volatility, or show the best per sector. Optional written morning note from Claude. |
| **Long shots** | High-risk, small-stake ideas: stocks trading far below their analysts' average target price. Includes the S&P 400 MidCap and S&P 600 SmallCap. Each stock shows a "falling knife" flag, analyst agreement and coverage, and for your stake: shares bought, buy + sell fees as a share of it, and break-even price. Price chart with low / average / high analyst targets. |
| **Diversification** | Effective number of positions (1/HHI), average correlation, diversification ratio, concentration warnings. Allocation by sector, country, currency and size, plus a correlation heatmap. Volatility, drawdown and beta per stock. Sector targets with suggested buy/sell amounts and a split for new money. |
| **Watchlist** | 1M/3M/1Y/YTD returns, RSI 14, trend (50/200-day averages), distance to your buy target, price chart. Manual price log for tickers Yahoo does not cover. |
| **Planner** | Monthly savings plan (PAC) projection at three assumed returns, and the monthly amount needed to reach a goal. |
| **Data & settings** | Base currency, benchmark, broker fee, alert levels, manual FX rates. Import/export as JSON or CSV. Delete everything. |

Multi-currency throughout: USD, EUR, GBP (including London quotes in pence, `GBp`), CHF and others are
converted to your base currency. If Yahoo has no FX quote, the last saved or manual rate is used.

## Requirements

- Windows 10/11 (the app itself runs anywhere Python runs; the scheduled scan and notifications are Windows-only)
- Python 3.11 or newer on `PATH`
- Internet access for Yahoo Finance and Wikipedia (index member lists)

## Install and start

```
git clone https://github.com/nicosac97/StockMonitoringTool.git
cd StockMonitoringTool\app
run.bat
```

The first `run.bat` creates a virtual environment in `app\.venv`, installs `requirements.txt`, and
opens http://localhost:8501. After that, double-click `app\run.bat` to start the app.

Manual start:

```
cd app
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
```

## First steps

1. **Data & settings:** set base currency (default EUR), broker fee per order (default 1.12 USD,
   applied to every buy and every sell), and alert levels.
2. **Orders:** add your buys and sells. Type a ticker and press *Look up* to fill name, currency,
   sector and country from Yahoo. Press *Use close of that day* to fill the price. Or import a broker
   CSV.
3. **Portfolio**, **Sell-out** and **Diversification** then update from those orders.
4. **Daily ideas:** press *Run scan now* once, then *Install / update task* to scan every weekday morning.

### Tickers

Use Yahoo Finance symbols:

| Market | Example |
|--------|---------|
| US | `AAPL`, `NVDA` |
| Milan | `ENI.MI`, `ISP.MI` |
| Amsterdam | `ASML.AS` |
| Xetra | `SAP.DE` |
| London (quoted in pence) | `VOD.L`: enter prices in pence too |
| ETFs | `VWCE.DE`, `URTH` |

### Import formats

- **Holdings CSV:** columns such as `ticker, shares, avg_cost, currency`. `,` or `;` separators;
  decimal commas are understood with `;` files.
- **Orders CSV** (Orders page): `date, ticker, side (BUY/SELL, also Italian Acquisto/Vendita), shares, price`, optional `fee`; common Italian broker headers (`data`, `tipo`, ...) are recognised.
- **JSON:** the app's own export, or the export of the web artifact version (`artifact/`).

## Morning scan

`app\daily_scan.py` runs the scan without the app open:

1. Downloads 2 years of prices for every index member.
2. Keeps stocks in a healthy uptrend: price and 50-day average above a rising 200-day average, RSI at
   most 75, within 20% of the 1-year high, positive 12-month return.
3. Ranks them by momentum: 12-month return excluding the last month, 6- and 3-month returns, return
   per unit of volatility.
4. Ranks every candidate by fit with your portfolio: 45% momentum, 30% low correlation with your
   holdings, 15% missing sector, 10% new country.
5. Collects analyst targets for Long shots (see below).
6. Asks Claude for a short written note, if an API key is set.
7. Saves everything to the database and shows a Windows notification that opens the Daily ideas page.

Schedule (Monday–Friday 08:00; runs as soon as the PC is back on if it was asleep, and waits for the
network):

```
cd app
powershell -ExecutionPolicy Bypass -File install_daily_task.ps1            # install, 08:00
powershell -ExecutionPolicy Bypass -File install_daily_task.ps1 -Time 07:30
powershell -ExecutionPolicy Bypass -File install_daily_task.ps1 -Remove
```

The Daily ideas page has the same buttons. Run by hand:
`.venv\Scripts\python daily_scan.py [--no-commentary] [--no-notify]`. Log: `app\data\scan.log`.

### Claude's morning note (optional)

The note covers the market picture, momentum picks, portfolio-fit picks, long shots and watch-outs.
Set **one** API key as a Windows user environment variable, then sign out and back in so the
scheduled task sees it:

```
setx OPENROUTER_API_KEY "sk-or-v1-..."      # via OpenRouter
setx ANTHROPIC_API_KEY  "sk-ant-..."        # or directly via Anthropic
setx OPENROUTER_MODEL   "anthropic/claude-sonnet-5.5"   # optional, OpenRouter model to use
```

The default model is Claude Opus 5.5. A note costs a few US cents. If both keys are set, the
Anthropic key is used. Without a key, the scan still runs and the note is skipped. Keys are read from
the environment only; they are never stored in the database or exports.

## Long shots

Stocks trading well below the average price target of the analysts who cover them. Source: Yahoo
Finance analyst data, at least 3 analysts, targets no older than 30 days.

- **Score:** 50% gap to the average target, 20% analyst agreement (narrow low–high range), 15% average
  rating, 15% number of analysts. Risk is shown, not rewarded.
- **Falling knife:** price below a falling 50/200-day average and down more than 15% in 3 months. The
  market may know something the targets do not. There is a filter to hide these.
- **Stake helper:** money per idea (default €150), whole shares only. Shows the buy + sell fees as a
  share of the stake and the price needed to break even. A warning appears when fees are above 3%.
  There is also a "Cheap shares only" filter.
- **Target collection:** Yahoo serves targets one stock at a time, so each morning collects up to 400
  (about 5 minutes) and refreshes each stock weekly. A full first fill of about 1,600 stocks takes
  about four mornings. *Collect targets and re-rank* on the page does one more batch on demand.

Analyst targets are often too optimistic, and small caps can lose most of their value. Use only
money you can afford to lose.

## Project layout

```
app/
  app.py                 Streamlit UI (all pages)
  daily_scan.py          command-line morning scan (Task Scheduler entry point)
  install_daily_task.ps1 installs/removes the Windows scheduled task
  run.bat                first-run setup + start
  ledger/
    analytics.py         portfolio math: positions, average cost, XIRR, risk, diversification (pure functions)
    prices.py            Yahoo Finance prices, FX, ticker lookup
    db.py                SQLite storage and settings
    transfer.py          JSON/CSV import and export
    charts.py            Plotly figures
    universe.py          index member lists from Wikipedia (cached weekly)
    screener.py          trend + momentum screen and portfolio fit
    ideas.py             filters and grouping for Daily ideas
    targets.py           analyst price targets cache
    longshots.py         Long shots ranking, falling-knife flag, stake/fee maths
    daily.py             the scan pipeline and Windows notification
    commentary.py        Claude note via Anthropic API or OpenRouter
  tests/                 pytest suite (no network needed)
artifact/
  stock-monitor.html     earlier single-page web version (claude.ai artifact), JSON-compatible with the app
```

## Data and privacy

- Everything you enter lives in `app\data\portfolio.db`. Back it up by copying the file. Set
  `STOCK_LEDGER_DB` to use another location.
- `app\data\` is git-ignored: your holdings, orders and scan log are never committed.
- The only data sent out: ticker symbols to Yahoo Finance and Wikipedia, and, if you enable the
  note, scan results plus a summary of your portfolio weights to the Claude API.

## Tests

```
cd app
.venv\Scripts\python -m pytest
```

## Troubleshooting

- **"Yahoo Finance returned nothing" / missing prices:** Yahoo rate-limits now and then. Wait a minute
  and press *Refresh prices* in the sidebar. Check the ticker suffix (`.MI`, `.DE`, ...).
- **No morning notification:** check the task on Daily ideas (next run time). The PC must be on or
  asleep, not shut down. See `app\data\scan.log`.
- **Claude note missing:** the Daily ideas page shows the reason (no key, key rejected, no credit).
  After `setx`, sign out and back in.
- **After updating the code:** the app reloads changed modules automatically. If a page still errors,
  restart `run.bat`.
