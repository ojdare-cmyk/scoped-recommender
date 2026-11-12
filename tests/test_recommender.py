"""Small tests that prove the rules in SCOPE.md hold. Run with: python -m pytest"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recommender import LaptopRecommender, cheapest_baseline, load_catalog, passes_spec_check  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG = load_catalog(os.path.join(HERE, "data", "laptops.csv"))
MODEL = LaptopRecommender(CATALOG)


def test_catalog_size_is_in_scope():
    assert 50 <= len(CATALOG) <= 150
    assert CATALOG["id"].is_unique


def test_budget_is_never_exceeded():
    top = MODEL.recommend("gaming laptop with a big gpu", max_budget=700)
    assert (top["price_usd"] <= 700).all()


def test_os_filter_is_respected():
    top = MODEL.recommend("light laptop long battery", max_budget=3000, os_choice="macOS")
    assert (top["os"] == "macOS").all()


def test_only_real_catalog_rows_come_back():
    top = MODEL.recommend("anything", max_budget=5000)
    assert set(top["id"]).issubset(set(CATALOG["id"]))


def test_explanation_names_shared_words():
    top = MODEL.recommend("nvidia gpu cuda deep learning", max_budget=1000, os_choice="Windows")
    assert "nvidia" in top.iloc[0]["why"] or "cuda" in top.iloc[0]["why"]


def test_baseline_is_cheapest_first():
    base = cheapest_baseline(CATALOG, max_budget=1000)
    assert list(base["price_usd"]) == sorted(base["price_usd"])


def test_spec_check_blank_means_no_requirement():
    laptop = CATALOG.iloc[0]
    profile = pd.Series({"max_weight_lb": None, "min_ram_gb": None, "min_battery_hr": None,
                         "needs_gpu": None, "needs_pen": None})
    assert passes_spec_check(laptop, profile)
