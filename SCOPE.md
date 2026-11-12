# Scope: Student Laptop Recommender

Written 2026-10-20, before any code. This page decides what gets built. If a later
idea does not fit this page, it waits for version 2.

## The problem

A first-year graduate student has a budget and a few needs ("I run Python
notebooks", "I carry it all day", "I want Windows for Excel"). Retail sites show
hundreds of laptops sorted by price or "popularity". The student either picks the
cheapest one that looks fine or asks a friend. Neither answer looks at what they
actually said they need.

## Who it is for

One user: a university student choosing a laptop under a fixed budget, who can
describe their needs in a sentence or two but does not want to compare spec
sheets.

Not for: IT departments buying in bulk, gamers chasing benchmarks, or anyone who
already knows the exact model they want.

## The one metric that counts

**Top-3 relevance: out of 10 test users, how many say at least one of the 3
laptops shown is one they would seriously consider buying.**

- Rated by the person, not by me. A recommender that I grade myself is a
  recommender that agrees with me.
- Target for version 1: 7 of 10.
- Everything else (speed, a nicer screen, more data) is secondary. If a change
  does not move this number, it does not ship.

One supporting check, used only while building: an automatic "spec check" that
asks whether each top-3 laptop meets the hard needs written into the profile
(for example "weighs under 3.5 lb"). It catches obvious mistakes before a real
person sees the list. It is not the metric, because it is my definition of a
good match, not the user's.

## What is in scope

1. A catalog of 50 to 150 laptops, each with price, operating system, key specs
   and a short plain-English description.
2. A TF-IDF recommender: it turns each laptop's description and the student's
   sentence into word-weight vectors and ranks laptops by how similar they are.
3. Two hard filters the student sets: maximum budget and operating system.
4. A baseline to beat: "the 3 cheapest laptops in budget", which is what most
   students actually do.
5. 10 test profiles, a rating sheet, and one script that runs everything.

## What is out of scope (and why)

| Out | Why |
|---|---|
| Generative AI (chatbots, LLM-written advice) | See below. |
| Live prices or scraping retailer sites | Prices change weekly and scraping breaks. The catalog is a dated snapshot. |
| User accounts, saved searches, purchase links | Not needed to test whether the ranking is useful. |
| Collaborative filtering ("students like you bought...") | Needs purchase history from many users. We have none. |
| A polished web app | A script and a results table answer the question. A screen can come later. |
| Reviews and star ratings | Not reliable without a real source, and they would dominate the ranking. |

## Why generative AI is not needed here

- **The catalog is small and structured.** 100 laptops with prices and specs fit
  in a spreadsheet. Matching words in a request to words in a description is a
  solved problem; TF-IDF does it in milliseconds on a laptop.
- **Every answer must be explainable.** With TF-IDF I can show exactly which
  words caused a match ("light", "battery", "python"). An LLM's reasoning is
  harder to audit, and it can describe a laptop that is not in the catalog.
- **Wrong numbers are the worst failure.** A model that invents a price or a
  battery life sends a student to buy the wrong machine. TF-IDF only ever returns
  rows that exist.
- **Cost and privacy.** No API key, no per-query cost, nothing leaves the
  machine.
- **The bar to add it later is clear.** If real users rate fewer than 7 of 10
  lists as relevant, and the failures are about understanding the request rather
  than catalog gaps, a language model for reading the request becomes worth
  testing. Not before.

## Done means

- One command runs the whole thing and writes the results.
- 10 profiles each have a top-3 list and a rating from a real person.
- The README reports the real top-3 relevance number, whatever it is.
