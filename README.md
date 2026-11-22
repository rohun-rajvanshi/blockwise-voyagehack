# BlockWise

**Spend only where waiting costs more.**
A risk-aware engine that tells TBO.com which hotel inventory to pre-buy, hold as an allotment, fix at the supplier, or source on demand, within a fixed budget.

Team Gini, Goa Institute of Management · VoyageHack 4.0 · Inventory Optimization challenge

## The idea

For every hotel-week block, BlockWise picks one of four actions:

| Action | When |
|---|---|
| **Pre-buy** | Demand is high and certain; the bulk rate beats the on-demand rate |
| **Allotment** | Demand is high but uncertain; pay a small holding fee, release what doesn't sell |
| **Integrate supplier** | Sold-outs come from connection gaps, not real scarcity; needs no budget |
| **On demand** | Supplier APIs already serve enough; no cash at risk |

The optimiser adds rooms 5 at a time to whichever block gives the best return per rupee, after a penalty for bad-case (P10) losses, until the budget runs out.

## What's in this repo

| File | What it does |
|---|---|
| `engine.py` | Synthetic seed data, demand simulator (with a market-wide shock), profit model, greedy budget allocator, action and reason labels |
| `reproduce.py` | Reproduces every prototype number on the slides, including the held-out and forecast-error tests |

Built so far: simulator and budget allocator (pre-buy vs. allotment).
Next: LightGBM demand forecasts on TBO data, Integrate scoring, demand hidden by sold-outs, OR-Tools optimiser, Streamlit dashboard.

## Run it

```bash
pip install -r requirements.txt
python reproduce.py
```

Expected output (Rs 60L budget, 60 hotel-week blocks across 10 destinations):

| Plan | Total profit per season |
|---|---|
| On demand only | Rs 38.8L |
| Split budget by last year's sales | Rs 52.0L |
| BlockWise | Rs 58.4L (+12%) |

Return on the budget: 33% for BlockWise vs. 22% for the last-year split.
Planned on one set of 4,000 scenarios and scored on a fresh set: +12% with an accurate forecast, +9% with 20% forecast error, +3% with 40% error.

**Caveat:** all data is synthetic. The results test the decision logic, not real-world accuracy.

## AI tools used

- An AI coding and analysis assistant helped write the simulator and allocator, run the tests, and draft the slides and this README. The team reviewed the outputs.
