"""Import and export. The JSON format matches the Stock Ledger web artifact, so data moves both ways."""
from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from datetime import date, datetime

from .analytics import CAPS, SECTORS
from .db import Store


def parse_num(v, decimal_comma: bool = False) -> float | None:
    """Parse 1234.5, 1.234,5, 1,234.5, 12,5 and values with currency symbols.

    decimal_comma=True (European files, usually ;-separated) reads every dot as a thousands separator.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[\s'€$£%]", "", str(v))
    if not s:
        return None
    c, d = s.rfind(","), s.rfind(".")
    if decimal_comma:
        s = s.replace(".", "").replace(",", ".")
    elif c > -1 and d > -1:
        s = s.replace(".", "").replace(",", ".") if c > d else s.replace(",", "")
    elif c > -1:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


ALIASES = {
    "ticker": ["ticker", "symbol", "simbolo", "titolo"],
    "name": ["name", "nome", "description", "descrizione", "company"],
    "shares": ["shares", "quantity", "qty", "quantita", "units", "position"],
    "avg_cost": ["avgcost", "averagecost", "avgprice", "averageprice", "prezzomedio", "pmc", "cost",
                 "purchaseprice", "buyprice"],
    "currency": ["currency", "valuta", "ccy"],
    "price": ["lastprice", "price", "currentprice", "last", "prezzo", "close"],
    "price_date": ["pricedate", "date", "data"],
    "sector": ["sector", "settore"],
    "industry": ["industry"],
    "country": ["country", "paese"],
    "cap_bucket": ["capbucket", "size"],
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return re.sub(r"[^a-z]", "", s)


def import_json(store: Store, text: str, skip_examples: bool = True) -> str:
    data = json.loads(text)
    holdings = data if isinstance(data, list) else data.get("holdings", [])
    watch = [] if isinstance(data, list) else data.get("watch", [])
    settings = {} if isinstance(data, list) else data.get("settings", {}) or {}
    n_h = n_w = n_skip = 0
    for h in holdings:
        if not h.get("ticker"):
            continue
        if skip_examples and h.get("example"):
            n_skip += 1
            continue
        store.upsert_holding({
            "ticker": h["ticker"], "name": h.get("name", ""),
            "shares": parse_num(h.get("shares")) or 0.0,
            "avg_cost": parse_num(h.get("avgCost", h.get("avg_cost"))) or 0.0,
            "currency": h.get("currency", ""), "sector": h.get("sector", ""),
            "industry": h.get("industry", ""), "country": h.get("country", ""),
            "cap_bucket": h.get("capBucket", h.get("cap_bucket", "")),
            "added_at": h.get("addedAt", h.get("added_at", "")), "note": h.get("note", ""),
        })
        price = parse_num(h.get("lastPrice"))
        if price and h.get("priceDate"):
            store.log_price(h["ticker"], h["priceDate"], price)
        n_h += 1
    for w in watch:
        if not w.get("ticker"):
            continue
        if skip_examples and w.get("example"):
            n_skip += 1
            continue
        store.upsert_watch({
            "ticker": w["ticker"], "name": w.get("name", ""), "currency": w.get("currency", ""),
            "target_price": parse_num(w.get("targetPrice", w.get("target_price"))),
            "sector": w.get("sector", ""), "country": w.get("country", ""), "note": w.get("note", ""),
            "added_at": w.get("addedAt", w.get("added_at", "")),
        })
        for e in w.get("history", []) or []:
            p = parse_num(e.get("p"))
            if p and e.get("d"):
                store.log_price(w["ticker"], e["d"], p)
        n_w += 1
    if settings:
        base = settings.get("base") or settings.get("base_currency")
        if base:
            store.set_setting("base_currency", base)
        if settings.get("targets") or settings.get("sector_targets"):
            store.set_setting("sector_targets", settings.get("targets") or settings.get("sector_targets"))
        fx = settings.get("fx", {})
        if base and isinstance(fx, dict) and isinstance(fx.get(base), dict):
            store.set_setting("fx_overrides", {k: parse_num(v) for k, v in fx[base].items() if parse_num(v)})
        for key in ("fee", "alerts", "benchmark", "plan"):
            if settings.get(key):
                store.set_setting(key, settings[key])
    n_o = add_orders(store, [] if isinstance(data, list) else data.get("orders", []) or [])
    msg = f"Imported {n_h} holdings, {n_w} watchlist entries and {n_o} orders."
    if n_skip:
        msg += f" Skipped {n_skip} example entries."
    return msg


def _order_key(o: dict) -> tuple:
    return (str(o["ticker"]).upper(), str(o["side"]).upper(), str(o["day"])[:10],
            round(float(o["shares"]), 6), round(float(o["price"]), 6))


def add_orders(store: Store, orders: list[dict]) -> int:
    """Add orders, skipping exact duplicates of ones already stored. Returns how many were added."""
    have = {_order_key(o) for o in store.orders().to_dict("records")}
    n = 0
    for o in orders:
        if not o.get("ticker") or not o.get("side") or not o.get("day"):
            continue
        k = _order_key(o)
        if k in have:
            continue
        store.add_order(o)
        have.add(k)
        n += 1
    return n


SIDE_WORDS = {"buy": "BUY", "b": "BUY", "acquisto": "BUY", "acq": "BUY", "a": "BUY", "+": "BUY",
              "sell": "SELL", "s": "SELL", "vendita": "SELL", "ven": "SELL", "v": "SELL", "-": "SELL"}
ORDER_ALIASES = {
    "day": ["date", "data", "tradedate", "dataoperazione", "day"],
    "ticker": ["ticker", "symbol", "simbolo", "titolo"],
    "side": ["side", "type", "tipo", "operation", "operazione", "segno", "action", "buysell"],
    "shares": ["shares", "quantity", "qty", "quantita", "units"],
    "price": ["price", "prezzo", "unitprice", "prezzounitario"],
    "fee": ["fee", "fees", "commission", "commissions", "commissione", "commissioni"],
    "fee_currency": ["feecurrency", "commissioncurrency", "valutacommissione"],
}


def _parse_day(v: str) -> str | None:
    v = v.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(v[:10], fmt).date().isoformat()
        except ValueError:
            continue
    return None


def import_orders_csv(store: Store, text: str, default_fee: float, fee_currency: str) -> str:
    """Broker order history: date, ticker, side (buy/sell, acquisto/vendita), shares, price, fee."""
    first = text.splitlines()[0] if text else ""
    sep = ";" if first.count(";") > first.count(",") else ("\t" if "\t" in first else ",")
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=sep) if any(c.strip() for c in r)]
    if len(rows) < 2:
        return "The CSV needs a header row and at least one data row."
    head = [_norm(h) for h in rows[0]]
    col = {k: next((i for i, h in enumerate(head) if h in al), None) for k, al in ORDER_ALIASES.items()}
    need = [k for k in ("day", "ticker", "side", "shares", "price") if col[k] is None]
    if need:
        return f"Missing columns: {', '.join(need)}. Found: " + ", ".join(rows[0])
    dc = sep == ";"
    parsed, bad = [], 0
    for r in rows[1:]:
        def g(k):
            i = col[k]
            return r[i].strip() if i is not None and i < len(r) else ""
        side = SIDE_WORDS.get(g("side").lower())
        day = _parse_day(g("day"))
        shares, price = parse_num(g("shares"), dc), parse_num(g("price"), dc)
        if not (side and day and g("ticker") and shares and price is not None):
            bad += 1
            continue
        fee = parse_num(g("fee"), dc) if col["fee"] is not None else default_fee
        parsed.append({"ticker": g("ticker").upper(), "side": side, "day": day, "shares": abs(shares),
                       "price": price, "fee": abs(fee or 0.0), "fee_currency": g("fee_currency") or fee_currency})
    n = add_orders(store, parsed)
    msg = f"Imported {n} orders."
    if len(parsed) - n:
        msg += f" Skipped {len(parsed) - n} already stored."
    if bad:
        msg += f" Could not read {bad} rows."
    return msg


def import_csv(store: Store, text: str) -> str:
    first = text.splitlines()[0] if text else ""
    sep = ";" if first.count(";") > first.count(",") else ("\t" if "\t" in first else ",")
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=sep) if any(c.strip() for c in r)]
    if len(rows) < 2:
        return "The CSV needs a header row and at least one data row."
    head = [_norm(h) for h in rows[0]]
    col = {k: next((i for i, h in enumerate(head) if h in al), None) for k, al in ALIASES.items()}
    if col["ticker"] is None or col["shares"] is None:
        return "Could not find ticker and shares columns. Found: " + ", ".join(rows[0])
    existing = store.holdings().set_index("ticker").to_dict("index")
    n = skipped = 0
    dc = sep == ";"
    num = lambda v: parse_num(v, decimal_comma=dc)  # noqa: E731
    for r in rows[1:]:
        def g(k):
            i = col[k]
            return r[i].strip() if i is not None and i < len(r) else ""
        ticker, shares = g("ticker").upper(), num(g("shares"))
        if not ticker or not shares:
            skipped += 1
            continue
        prev = existing.get(ticker, {})
        store.upsert_holding({
            "ticker": ticker, "shares": shares,
            "avg_cost": num(g("avg_cost")) or prev.get("avg_cost") or 0.0,
            "name": g("name") or prev.get("name", ""),
            "currency": g("currency") or prev.get("currency", ""),
            "sector": g("sector") if g("sector") in SECTORS else prev.get("sector", ""),
            "industry": g("industry") or prev.get("industry", ""),
            "country": g("country") or prev.get("country", ""),
            "cap_bucket": g("cap_bucket") if g("cap_bucket") in CAPS else prev.get("cap_bucket", ""),
            "added_at": prev.get("added_at", ""), "note": prev.get("note", ""),
        })
        price = num(g("price"))
        if price:
            store.log_price(ticker, g("price_date") or date.today().isoformat(), price)
        n += 1
    msg = f"Imported {n} holdings."
    if skipped:
        msg += f" Skipped {skipped} rows without ticker or shares."
    return msg


def export_json(store: Store, holdings_now=None) -> str:
    """Everything as JSON. holdings_now (shares/avg cost derived from orders) replaces the stored
    snapshot so the web artifact, which has no orders, still sees the right positions."""
    s = store.settings()
    log = store.price_log()
    holdings = []
    src = holdings_now if holdings_now is not None else store.holdings()
    src = src.astype(object).where(src.notna(), "")
    for h in src.to_dict("records"):
        last = log[log["ticker"] == h["ticker"]].tail(1)
        holdings.append({
            "ticker": h["ticker"], "name": h["name"], "shares": h["shares"], "avgCost": h["avg_cost"],
            "currency": h["currency"], "sector": h["sector"], "industry": h["industry"],
            "country": h["country"], "capBucket": h["cap_bucket"], "addedAt": h["added_at"],
            "note": h["note"],
            "lastPrice": float(last["price"].iloc[0]) if not last.empty else None,
            "priceDate": last["day"].iloc[0] if not last.empty else "",
        })
    watch = []
    for w in store.watchlist().to_dict("records"):
        hist = log[log["ticker"] == w["ticker"]]
        watch.append({
            "ticker": w["ticker"], "name": w["name"], "currency": w["currency"],
            "targetPrice": w["target_price"] if w["target_price"] == w["target_price"] else None,
            "sector": w["sector"], "country": w["country"], "note": w["note"], "addedAt": w["added_at"],
            "history": [{"d": d, "p": p} for d, p in zip(hist["day"], hist["price"])],
        })
    base = s["base_currency"]
    return json.dumps({
        "app": "stock-ledger", "version": 1, "exportedAt": datetime.now().isoformat(timespec="seconds"),
        "settings": {"base": base, "targets": s["sector_targets"], "fx": {base: s["fx_overrides"]},
                     "benchmark": s["benchmark"], "plan": s["plan"], "fee": s["fee"], "alerts": s["alerts"]},
        "holdings": holdings, "watch": watch,
        "orders": [{k: o[k] for k in ("ticker", "side", "day", "shares", "price", "fee", "fee_currency", "note")}
                   for o in store.orders().to_dict("records")],
    }, indent=2)


def export_holdings_csv(store: Store) -> str:
    buf = io.StringIO()
    cols = ["ticker", "name", "shares", "avg_cost", "currency", "sector", "industry", "country", "cap_bucket"]
    store.holdings()[cols].to_csv(buf, index=False)
    return buf.getvalue()
