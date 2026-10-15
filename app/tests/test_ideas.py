import pandas as pd

from ledger import ideas as I

ROWS = pd.DataFrame([
    # ticker, sector, country, index, score, rsi, vol, below_high, held
    ("A1", "Energy", "United States", "sp500", 95, 60, 0.30, -0.02, False),
    ("A2", "Energy", "Italy", "ftsemib", 90, 72, 0.25, -0.05, True),
    ("A3", "Energy", "United States", "ndx,sp500", 70, 50, 0.20, -0.15, False),
    ("B1", "Financials", "Italy", "ftsemib", 99, 40, 0.35, -0.08, False),
    ("B2", "Financials", "Germany", "sx5e", 60, 55, 0.18, -0.01, False),
    ("C1", "Health Care", "United States", "sp500", 80, 65, 0.50, -0.12, False),
], columns=["ticker", "sector", "country", "index", "score", "rsi", "vol", "below_high", "held"])


def test_region_of():
    assert I.region_of("United States") == "United States"
    assert I.region_of("Italy") == "Italy"
    assert I.region_of("Germany") == "Europe (ex Italy)"
    assert I.region_of("Japan") == "Other"


def test_best_per_sector_orders_sectors_by_their_best_stock():
    out = I.best_per_sector(ROWS, "score", 2)
    assert out["ticker"].tolist() == ["B1", "B2", "A1", "A2", "C1"]
    assert out["sector_rank"].tolist() == [1, 2, 1, 2, 1]


def test_filters_combine():
    f = I.Filters(regions=["United States"])
    assert set(I.apply_filters(ROWS, f)["ticker"]) == {"A1", "A3", "C1"}
    f = I.Filters(indexes=["ndx"])                                   # multi-index membership string
    assert I.apply_filters(ROWS, f)["ticker"].tolist() == ["A3"]
    f = I.Filters(max_rsi=60, max_vol=0.30, hide_held=True)
    assert set(I.apply_filters(ROWS, f)["ticker"]) == {"A1", "A3", "B2"}
    f = I.Filters(max_below_high=0.05, min_score=80)
    assert set(I.apply_filters(ROWS, f)["ticker"]) == {"A1", "A2"}
    f = I.Filters(sectors=["Financials"], countries=["Germany"])
    assert I.apply_filters(ROWS, f)["ticker"].tolist() == ["B2"]


def test_ranked_list():
    assert I.ranked(ROWS, "score", 3)["ticker"].tolist() == ["B1", "A1", "A2"]


def test_breadth_counts_share_in_uptrend_per_sector_with_geo_filters():
    scanned = pd.DataFrame({"ticker": list("abcdef"), "sector": ["Energy", "Energy", "Energy", "Utilities", "Utilities", "Energy"],
                            "country": ["Italy", "Italy", "United States", "Italy", "Italy", "Germany"],
                            "index": ["ftsemib"] * 6, "eligible": [True, False, True, True, True, False]})
    b = I.breadth(scanned, I.Filters()).set_index("sector")
    assert b.loc["Energy", "scanned"] == 4 and b.loc["Energy", "uptrend"] == 2 and b.loc["Utilities", "share"] == 1
    b = I.breadth(scanned, I.Filters(regions=["Italy"])).set_index("sector")
    assert b.loc["Energy", "scanned"] == 2 and b.loc["Energy", "share"] == 0.5
