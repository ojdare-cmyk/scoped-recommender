"""
recommender.py: the whole recommender, in about 100 lines.

How it works, in plain words:
1. Every laptop has a short description. TF-IDF turns each description into a
   list of word weights. A word that shows up in only a few descriptions
   ("cuda", "stylus") gets a high weight; a word that is everywhere ("laptop")
   gets almost none.
2. The student's request goes through the same TF-IDF, so it becomes a list of
   word weights too.
3. Cosine similarity measures how much the two lists point the same way.
   1.0 means the same words with the same emphasis, 0.0 means nothing shared.
4. Before ranking, two hard filters remove anything over budget or on the wrong
   operating system. The student sets these; the model never overrides them.

No generative AI anywhere. See SCOPE.md for why.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_catalog(path="data/laptops.csv"):
    """Read the catalog and build the one text field TF-IDF will read."""
    catalog = pd.read_csv(path)
    # The description carries most of the meaning. The GPU name is added so a
    # request for "nvidia" or "rtx" can match. Brand is added so "mac" or
    # "thinkpad" requests can match.
    catalog["text"] = (
        catalog["brand"] + " " + catalog["model"] + " " +
        catalog["gpu"] + " " + catalog["description"]
    ).str.lower()
    return catalog


class LaptopRecommender:
    """Fit once on the catalog, then call recommend() for each student."""

    def __init__(self, catalog):
        self.catalog = catalog.reset_index(drop=True)
        # stop_words="english" drops words like "the" and "for".
        # ngram_range=(1, 2) also counts two-word phrases like "long battery".
        # sublinear_tf=True stops a word repeated 3 times from counting 3x.
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(self.catalog["text"])

    def _allowed(self, max_budget, os_choice):
        """Hard filters: True for rows the student would accept at all."""
        ok = self.catalog["price_usd"] <= max_budget
        if os_choice and os_choice.lower() != "any":
            ok = ok & (self.catalog["os"].str.lower() == os_choice.lower())
        return ok

    def recommend(self, request, max_budget, os_choice="any", k=3):
        """Return the top k laptops for one request, with the words that matched."""
        query = self.vectorizer.transform([request.lower()])
        scores = cosine_similarity(query, self.matrix).ravel()

        results = self.catalog.copy()
        results["score"] = scores
        results = results[self._allowed(max_budget, os_choice)]
        # Highest similarity first. If two laptops tie, the cheaper one wins,
        # because a student would rather save money for the same fit.
        results = results.sort_values(["score", "price_usd"], ascending=[False, True])
        top = results.head(k).copy()
        top["why"] = [self.matched_words(query, i) for i in top.index]
        return top

    def matched_words(self, query, row_index, n=4):
        """The words that contributed most to this match. This is the explanation."""
        contribution = query.multiply(self.matrix[row_index]).toarray().ravel()
        words = self.vectorizer.get_feature_names_out()
        best = contribution.argsort()[::-1][:n]
        return ", ".join(words[i] for i in best if contribution[i] > 0) or "(no shared words)"


def cheapest_baseline(catalog, max_budget, os_choice="any", k=3):
    """What most students actually do: sort by price and take the cheapest ones."""
    ok = catalog["price_usd"] <= max_budget
    if os_choice and os_choice.lower() != "any":
        ok = ok & (catalog["os"].str.lower() == os_choice.lower())
    return catalog[ok].sort_values("price_usd").head(k).copy()


def passes_spec_check(laptop, profile):
    """
    The supporting check from SCOPE.md. True if this laptop meets every hard
    need written into the profile. Blank means "no requirement".
    These columns are NOT used by the recommender, so this is an outside check
    on whether word-matching found machines with the right specs.
    """
    def given(col):
        return pd.notna(profile.get(col)) and str(profile.get(col)).strip() != ""

    if given("max_weight_lb") and laptop["weight_lb"] > float(profile["max_weight_lb"]):
        return False
    if given("min_ram_gb") and laptop["ram_gb"] < float(profile["min_ram_gb"]):
        return False
    if given("min_battery_hr") and laptop["battery_hr"] < float(profile["min_battery_hr"]):
        return False
    if given("needs_gpu") and laptop["gpu"] == "integrated":
        return False
    if given("needs_pen") and laptop["pen_support"] != "yes":
        return False
    return True
