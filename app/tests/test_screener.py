import numpy as np
import pandas as pd
import pytest

from ledger import commentary, screener
from ledger.db import Store
from ledger.universe import to_gics

IDX = pd.bdate_range("2024-06-03", periods=300)


def path(daily_drift, noise=0.008, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + daily_drift + rng.normal(0, noise, len(IDX)))


def closes():
    return pd.DataFrame({
        "STRONG": path(0.0015, seed=5),        # steady uptrend
        "OK": path(0.0008, seed=6),            # milder uptrend
        "DOWN": path(-0.0015, seed=3),         # downtrend: must be filtered out
        "SHORT": np.r_[np.full(100, np.nan), path(0.002, seed=4)[100:]],   # under a year of data
        "SPIKE": np.r_[np.full(280, 100.0), np.linspace(100, 160, 20)],     # flat then vertical: RSI too high
    }, index=IDX)


UNIVERSE = pd.DataFrame({"ticker": ["STRONG", "OK", "DOWN", "SPIKE"], "name": ["S", "O", "D", "P"],
                         "sector": ["Information Technology", "Health Care", "Energy", "Industrials"],
                         "country": ["United States", "Germany", "Italy", "France"],
                         "currency": ["USD", "EUR", "EUR", "EUR"], "index": ["sp500", "sx5e", "ftsemib", "sx5e"]})


def test_metrics_and_trend_filter():
    m = screener.metrics(closes())
    assert "SHORT" not in set(m["ticker"])                       # not enough history
    s = screener.score(m).set_index("ticker")
    assert s.loc["STRONG", "eligible"] and s.loc["OK", "eligible"]
    assert not s.loc["DOWN", "eligible"]
    assert not s.loc["SPIKE", "eligible"]                         # RSI far above 75
    assert s.loc["STRONG", "score"] > s.loc["OK", "score"]
    r12 = closes()["STRONG"]
    assert s.loc["STRONG", "r12_1"] == pytest.approx(r12.iloc[-22] / r12.iloc[-253] - 1)


def test_standalone_caps_sector_and_flags_held():
    scored = screener.score(screener.metrics(closes()))
    out = screener.standalone(scored, UNIVERSE, held={"OK"}, n=5, per_sector=1)
    assert out["ticker"].tolist() == ["STRONG", "OK"]
    assert out.set_index("ticker").loc["OK", "held"]
    assert any("12 months" in r for r in out.loc[0, "reasons"])


def test_portfolio_fit_prefers_new_sector_and_excludes_held():
    c = closes()
    scored = screener.score(screener.metrics(c))
    pos = pd.DataFrame({"ticker": ["STRONG"], "value": [1000.0], "weight": [1.0], "sector": ["Information Technology"],
                        "country": ["United States"]})
    fit = screener.portfolio_fit(scored, UNIVERSE, c, pos, n=5)
    assert fit["ticker"].tolist() == ["OK"]                     # STRONG is held, others fail the trend filter
    r = fit.iloc[0]
    assert r["sector_weight"] == 0 and r["country_weight"] == 0
    assert r["eff_n_after"] == pytest.approx(1 / (0.95 ** 2 + 0.05 ** 2))
    assert any("Adds Health Care" in x for x in r["reasons"])
    assert screener.portfolio_fit(scored, UNIVERSE, c, pos.iloc[0:0]).empty


def test_icb_and_gics_labels_map_to_gics():
    assert to_gics("Technology") == "Information Technology"
    assert to_gics("Telecommunications") == "Communication Services"
    assert to_gics("Banks") == "Financials"
    assert to_gics("Basic Materials") == "Materials"
    assert to_gics("Health Care") == "Health Care"
    assert to_gics("") == "Other"


def test_scan_storage_round_trip(tmp_path):
    s = Store(tmp_path / "s.db")
    rid = s.save_scan({"run_at": "2026-10-08T08:00:00", "status": "ok", "n_universe": 5, "n_scanned": 4,
                       "n_eligible": 2, "duration_s": 3.2, "notes": ""},
                      {"standalone": [{"ticker": "A", "score": 90.0}, {"ticker": "B", "score": 70.0}],
                       "portfolio": [{"ticker": "C", "fit": 60.0}]})
    s.update_scan(rid, commentary="## Market read", commentary_note="test")
    assert s.scan_runs().iloc[0]["commentary"] == "## Market read"
    picks = s.scan_picks(rid)
    assert [p["ticker"] for p in picks["standalone"]] == ["A", "B"] and picks["portfolio"][0]["fit"] == 60.0


def test_commentary_skips_cleanly_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    text, note = commentary.write_commentary([], [], {})
    assert text == "" and "ANTHROPIC_API_KEY" in note and "OPENROUTER_API_KEY" in note


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self.reason, self._body = status, "reason", body

    def json(self):
        return self._body


def test_commentary_through_openrouter(monkeypatch):
    import requests

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    sent = {}

    def post(url, **kw):
        sent.update(url=url, **kw)
        return FakeResponse(200, {"model": "anthropic/claude-opus-5.5", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                                  "choices": [{"finish_reason": "stop", "message": {"content": "## Market read\nok"}}]})

    monkeypatch.setattr(requests, "post", post)
    assert commentary.provider() == "openrouter"
    text, note = commentary.write_commentary([{"ticker": "A", "score": 90.0}], [], {})
    assert text.startswith("## Market read") and "via OpenRouter (10 input / 5 output" in note
    assert sent["headers"]["Authorization"] == "Bearer sk-or-test" and sent["json"]["model"] == "anthropic/claude-opus-5.5"
    assert sent["json"]["messages"][0]["role"] == "system" and '"ticker": "A"' in sent["json"]["messages"][1]["content"]

    monkeypatch.setattr(requests, "post", lambda url, **kw: FakeResponse(401, {"error": {"code": 401, "message": "No auth"}}))
    assert commentary.write_commentary([], [], {}) == ("", "The OpenRouter API key was rejected. Check OPENROUTER_API_KEY.")
    monkeypatch.setattr(requests, "post", lambda url, **kw: FakeResponse(402, {"error": {"code": 402, "message": "credits"}}))
    assert "not enough credits" in commentary.write_commentary([], [], {})[1]


def test_settings_fill_new_keys_for_old_databases(tmp_path):
    import json
    import sqlite3
    s = Store(tmp_path / "old.db")
    with sqlite3.connect(s.path) as c:   # a database saved before 'scan' existed, with a partial 'fee'
        c.execute("INSERT INTO settings (key, value) VALUES ('fee', ?)", (json.dumps({"amount": 2.0}),))
    got = s.settings()
    assert got["scan"]["top_n"] == 10
    assert got["fee"] == {"amount": 2.0, "currency": "USD"}
    from ledger.daily import scan_settings
    assert scan_settings({})["indexes"] == ["sp500", "ndx", "sx5e", "ftsemib"]
