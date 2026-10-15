import pandas as pd

from ledger import daily, prices, screener
from ledger.db import Store

UNI = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "DDD"], "name": list("ABCD"), "sector": ["Energy"] * 4,
                    "country": ["Italy"] * 4, "currency": ["EUR"] * 4, "index": ["ftsemib"] * 4})


def fake_closes(tickers):
    idx = pd.bdate_range("2024-06-03", periods=300)
    return pd.DataFrame({t: [100 + i * 0.2 for i in range(300)] for t in tickers}, index=idx)


def setup(monkeypatch, tmp_path, downloads):
    store = Store(tmp_path / "d.db")
    calls, toasts = [], []
    monkeypatch.setattr(daily, "load_universe", lambda s, keys: (UNI, []))
    monkeypatch.setattr(prices, "download_closes", lambda t, p="2y", **k: (calls.append(1), downloads[min(len(calls), len(downloads)) - 1])[1])
    monkeypatch.setattr(prices, "fx_history", lambda *a, **k: prices.empty_closes())
    monkeypatch.setattr(daily, "toast", lambda title, body, **k: toasts.append(title) or True)
    monkeypatch.setattr(daily.time, "sleep", lambda s: None)
    return store, calls, toasts


def test_metrics_of_nothing_is_an_empty_table():
    m = screener.score(screener.metrics(prices.empty_closes()))
    assert m.empty and "eligible" in m.columns


def test_no_network_after_wake_fails_cleanly_and_notifies(monkeypatch, tmp_path):
    store, calls, toasts = setup(monkeypatch, tmp_path, [prices.empty_closes()])
    rid = daily.run_scan(store, with_commentary=False)
    run = store.scan_runs(1).iloc[0]
    assert run["id"] == rid and run["status"] == "failed" and "Price download failed" in run["notes"]
    assert len(calls) == 1 + len(daily.RETRY_WAIT_S)          # first try + every retry
    assert toasts == ["Stock Ledger: morning scan failed"]


def test_retry_recovers_when_network_comes_back(monkeypatch, tmp_path):
    store, calls, toasts = setup(monkeypatch, tmp_path, [prices.empty_closes(), fake_closes(list("ABCD") and UNI["ticker"])])
    daily.run_scan(store, with_commentary=False)
    run = store.scan_runs(1).iloc[0]
    assert run["status"] == "ok" and run["n_scanned"] == 4 and len(calls) == 2
    assert run["commentary"] == "" and run["commentary_note"] == ""   # never NULL
    assert toasts == ["Stock Ledger: daily ideas ready"]
