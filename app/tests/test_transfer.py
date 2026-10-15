import json

import pytest

from ledger.db import Store
from ledger.transfer import export_json, import_csv, import_json, import_orders_csv, parse_num


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "t.db")


@pytest.mark.parametrize("raw,expected", [
    ("12,5", 12.5), ("1.234,56", 1234.56), ("1,234.56", 1234.56), ("€ 7", 7.0), ("", None), ("abc", None), (3, 3.0),
])
def test_parse_num(raw, expected):
    assert parse_num(raw) == expected


ARTIFACT_EXPORT = {
    "app": "stock-ledger", "version": 1,
    "settings": {"base": "EUR", "fx": {"EUR": {"USD": 0.86}}, "targets": {"Energy": 20}},
    "holdings": [
        {"ticker": "ENI.MI", "name": "Eni", "shares": 200, "avgCost": 13, "currency": "EUR", "lastPrice": 14.2,
         "priceDate": "2026-10-01", "sector": "Energy", "country": "Italy", "capBucket": "Large"},
        {"ticker": "AAPL", "shares": 15, "avgCost": 150, "currency": "USD", "example": True},
    ],
    "watch": [{"ticker": "RACE.MI", "targetPrice": 380, "history": [{"d": "2026-09-01", "p": 400}, {"d": "2026-10-01", "p": 390}]}],
}


def test_import_artifact_json_and_round_trip(store):
    msg = import_json(store, json.dumps(ARTIFACT_EXPORT))
    assert "1 holdings" in msg and "Skipped 1 example" in msg
    h = store.holdings().set_index("ticker")
    assert list(h.index) == ["ENI.MI"] and h.loc["ENI.MI", "avg_cost"] == 13
    s = store.settings()
    assert s["fx_overrides"] == {"USD": 0.86} and s["sector_targets"] == {"Energy": 20}
    assert len(store.price_log("RACE.MI")) == 2

    out = json.loads(export_json(store))
    assert out["holdings"][0]["lastPrice"] == 14.2
    assert out["watch"][0]["history"][-1] == {"d": "2026-10-01", "p": 390.0}

    other = Store(store.path.with_name("other.db"))
    import_json(other, json.dumps(out))
    assert other.holdings()["ticker"].tolist() == ["ENI.MI"]
    assert other.watchlist().iloc[0]["target_price"] == 380


def test_import_italian_broker_csv(store):
    text = "Simbolo;Quantità;Prezzo medio;Valuta\nENEL.MI;400;6,05;EUR\n;;;\nISP.MI;1.000;2,61;EUR\n"
    msg = import_csv(store, text)
    assert msg == "Imported 2 holdings."
    h = store.holdings().set_index("ticker")
    assert h.loc["ISP.MI", "shares"] == 1000 and h.loc["ENEL.MI", "avg_cost"] == pytest.approx(6.05)


def test_import_csv_without_required_columns(store):
    assert import_csv(store, "foo,bar\n1,2").startswith("Could not find ticker and shares")


def test_pence_currency_code_survives_storage(store):
    store.upsert_holding({"ticker": "vod.l", "shares": 1, "avg_cost": 70, "currency": "GBp"})
    store.upsert_holding({"ticker": "aapl", "shares": 1, "avg_cost": 1, "currency": "usd"})
    cur = store.holdings().set_index("ticker")["currency"]
    assert cur["VOD.L"] == "GBp" and cur["AAPL"] == "USD"


def test_orders_csv_italian_broker_and_dedupe(store):
    text = ("Data;Simbolo;Operazione;Quantità;Prezzo;Commissioni\n"
            "02/01/2025;AAPL;Acquisto;10;150,50;1,12\n"
            "15/03/2025;AAPL;Vendita;4;170;1,12\n"
            "bad;row;x;;;\n")
    msg = import_orders_csv(store, text, 1.12, "USD")
    assert msg == "Imported 2 orders. Could not read 1 rows."
    o = store.orders()
    assert o["side"].tolist() == ["BUY", "SELL"] and o["day"].tolist() == ["2025-01-02", "2025-03-15"]
    assert o["price"].tolist() == [150.5, 170.0] and o["fee"].tolist() == [1.12, 1.12]
    assert import_orders_csv(store, text, 1.12, "USD").startswith("Imported 0 orders. Skipped 2")


def test_orders_survive_json_round_trip(store, tmp_path):
    store.add_order({"ticker": "ENI.MI", "side": "BUY", "day": "2025-01-02", "shares": 100, "price": 13, "fee": 1.12})
    other = Store(tmp_path / "o2.db")
    assert "1 orders" in import_json(other, export_json(store))
    assert "0 orders" in import_json(other, export_json(store))   # no duplicates on re-import
