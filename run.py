"""
run.py: one command runs the whole project.

    python run.py

What it does, in order:
1. Loads the catalog and the 10 test profiles.
2. Gets a top-3 list for each profile from the TF-IDF recommender and from the
   "3 cheapest" baseline.
3. Runs the automatic spec check on every recommended laptop.
4. Writes the results tables and the chart into results/.
5. Writes a blank rating sheet into ratings/ (only if one does not exist yet,
   so it never overwrites ratings people have already entered).
6. If real ratings have been entered, computes THE metric from SCOPE.md:
   how many of the 10 users found at least one top-3 laptop relevant.
"""

import os

import matplotlib
matplotlib.use("Agg")  # draw charts to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from recommender import (LaptopRecommender, cheapest_baseline, load_catalog,
                         passes_spec_check)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
RATINGS = os.path.join(HERE, "ratings")


def build_lists(catalog, profiles):
    """Top 3 from each method for each profile, as one long table."""
    model = LaptopRecommender(catalog)
    rows = []
    for _, p in profiles.iterrows():
        lists = {
            "tfidf": model.recommend(p["request"], p["max_budget"], p["os"]),
            "cheapest": cheapest_baseline(catalog, p["max_budget"], p["os"]),
        }
        for method, top in lists.items():
            for rank, (_, laptop) in enumerate(top.iterrows(), start=1):
                rows.append({
                    "profile_id": p["profile_id"],
                    "persona": p["persona"],
                    "method": method,
                    "rank": rank,
                    "laptop_id": laptop["id"],
                    "laptop": f'{laptop["brand"]} {laptop["model"]}',
                    "price_usd": laptop["price_usd"],
                    "os": laptop["os"],
                    "similarity": round(laptop.get("score", float("nan")), 3),
                    "matched_words": laptop.get("why", ""),
                    "spec_check": passes_spec_check(laptop, p),
                })
    return pd.DataFrame(rows)


def summarise(recs):
    """Per profile and method: how many of the 3 passed the spec check."""
    per = (recs.groupby(["profile_id", "method"])["spec_check"]
               .sum().unstack("method").astype(int))
    per.columns = [f"{c}_spec_passes_of_3" for c in per.columns]
    return per


def draw_chart(per, path):
    """Side-by-side bars: spec-check passes out of 3, per profile, per method."""
    ax = per[["tfidf_spec_passes_of_3", "cheapest_spec_passes_of_3"]].plot(
        kind="bar", figsize=(9, 4.5), color=["#2b6cb0", "#a0aec0"], width=0.75)
    ax.set_ylim(0, 4)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Test profile")
    ax.set_ylabel("Top-3 laptops meeting the profile's specs")
    ax.set_title("Automatic spec check (a build-time check, not the user metric)")
    ax.legend(["TF-IDF recommender", "3 cheapest in budget (baseline)"],
              loc="upper center", ncol=2, frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_rating_sheet(recs, path):
    """A blank sheet for real people to rate the TF-IDF lists. Never overwrites."""
    if os.path.exists(path):
        return False
    sheet = recs[recs["method"] == "tfidf"][
        ["profile_id", "persona", "rank", "laptop_id", "laptop", "price_usd", "os"]
    ].copy()
    sheet["rater_name"] = ""
    sheet["would_seriously_consider_yes_no"] = ""
    sheet["comment"] = ""
    sheet.to_csv(path, index=False)
    return True


def score_ratings(path):
    """THE metric: users (profiles) with at least one 'yes' in their top 3."""
    if not os.path.exists(path):
        return None
    r = pd.read_csv(path, dtype=str).fillna("")
    r["yes"] = r["would_seriously_consider_yes_no"].str.strip().str.lower().isin(["yes", "y"])
    rated = r[r["would_seriously_consider_yes_no"].str.strip() != ""]
    if rated.empty:
        return None
    hits = rated.groupby("profile_id")["yes"].any()
    return int(hits.sum()), int(hits.size)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(RATINGS, exist_ok=True)
    catalog = load_catalog(os.path.join(HERE, "data", "laptops.csv"))
    profiles = pd.read_csv(os.path.join(HERE, "data", "profiles.csv"))

    recs = build_lists(catalog, profiles)
    recs.to_csv(os.path.join(RESULTS, "recommendations.csv"), index=False)

    per = summarise(recs)
    per.to_csv(os.path.join(RESULTS, "spec_check_summary.csv"))
    draw_chart(per, os.path.join(RESULTS, "spec_check_chart.png"))

    print(f"Catalog: {len(catalog)} laptops. Profiles: {len(profiles)}.\n")
    for pid, group in recs[recs["method"] == "tfidf"].groupby("profile_id"):
        print(f"{pid}  {group['persona'].iloc[0]}")
        for _, r in group.iterrows():
            mark = "spec ok " if r["spec_check"] else "spec MISS"
            print(f"   {r['rank']}. {r['laptop']:<34} ${r['price_usd']:<5} {mark}  matched: {r['matched_words']}")
    print()
    for method in ["tfidf", "cheapest"]:
        col = f"{method}_spec_passes_of_3"
        any_hit = int((per[col] > 0).sum())
        print(f"Spec check, {method:<8}: {per[col].sum()} of 30 laptops pass; "
              f"{any_hit} of 10 profiles have at least one passing laptop")

    sheet = os.path.join(RATINGS, "ratings.csv")
    if write_rating_sheet(recs, sheet):
        print("\nWrote a blank rating sheet: ratings/ratings.csv")
    result = score_ratings(sheet)
    if result is None:
        print("\nTHE METRIC (top-3 relevance from real users): not rated yet. "
              "Fill ratings/ratings.csv, then run again.")
    else:
        print(f"\nTHE METRIC (top-3 relevance from real users): {result[0]} of {result[1]} "
              "rated profiles had at least one laptop the person would seriously consider.")


if __name__ == "__main__":
    main()
