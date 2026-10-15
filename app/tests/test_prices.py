from ledger import prices


def test_lookup_falls_back_to_chart_data_when_profile_is_refused(monkeypatch):
    import yfinance as yf

    class Refused:
        def __init__(self, t):
            pass

        @property
        def info(self):
            raise RuntimeError("401 Unauthorized: Invalid Crumb")

    monkeypatch.setattr(yf, "Ticker", Refused)
    monkeypatch.setattr(prices, "lookup_basic", lambda t: {"ticker": t, "name": "nVent Electric plc",
                                                          "currency": "USD", "source": "basic"})
    got = prices.lookup("NVT")
    assert got["currency"] == "USD" and got["source"] == "basic"
