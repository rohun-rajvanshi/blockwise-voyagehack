"""BlockWise engine: synthetic seed data, scenario simulator and greedy budget allocator.

Everything here is NumPy only, so it runs in normal Python and in the browser (stlite/Pyodide).
Money is in rupees. A "block" is one hotel-destination x stay-week.
"""
import numpy as np

DESTS = ["Goa", "Dubai", "Bangkok", "Bali", "Singapore", "Jaipur", "Phuket", "Maldives", "Kerala", "London"]
WEEKS = ["Dec-W3", "Dec-W4", "Jan-W1", "Feb-W2", "Mar-W4", "Apr-W2"]
STEP = 5  # rooms added per greedy step


def make_world(seed=11, scenarios=4000):
    """Synthetic seed data. Same generator and seed as the numbers on the slides."""
    rng = np.random.default_rng(seed)
    blocks = [(d, w) for d in DESTS for w in WEEKS]
    n = len(blocks)
    cost = rng.uniform(3000, 9000, n)              # pre-buy (hard commit) rate per room-night
    allot = cost * 1.07                            # allotment rate, paid only when sold
    fee = 0.12 * cost                              # allotment holding fee (capital at risk)
    spot = cost * rng.uniform(1.05, 1.32, n)       # on-demand supplier API rate
    sell = cost * rng.uniform(1.18, 1.42, n)       # sell price = TBO's usual markup
    spot_share = rng.uniform(0.35, 1.0, n)         # share of demand the API can actually serve
    mu = rng.uniform(40, 220, n) * rng.uniform(0.6, 1.6, n)   # expected demand (room-nights)
    sd = mu * rng.uniform(0.2, 0.6, n)             # forecast uncertainty
    canc = rng.uniform(0.03, 0.15, n)              # cancellation rate
    last_year = np.minimum(mu * rng.lognormal(0, 0.35, n), mu * spot_share * 1.1)  # censored by sold-outs
    w = dict(blocks=blocks, n=n, cost=cost, allot=allot, fee=fee, spot=spot, sell=sell,
             spot_share=spot_share, mu=mu, sd=sd, canc=canc, last_year=last_year)
    w["D"] = draw_demand(w, rng, scenarios)
    return w


def draw_demand(w, rng, scenarios=4000, mu=None, sd=None):
    """Demand scenarios with one market-wide shock shared by all blocks."""
    mu = w["mu"] if mu is None else mu
    sd = w["sd"] if sd is None else sd
    z = rng.normal(0, 1, (scenarios, 1))
    return np.maximum(0, mu * (1 + 0.12 * z) + rng.normal(0, 1, (scenarios, w["n"])) * sd) * (1 - w["canc"])


def profit(w, D, qp, qa):
    """Profit per scenario per block: pre-bought rooms sell first, then allotments, then on-demand API."""
    sp = np.minimum(D, qp); r = D - sp
    sa = np.minimum(r, qa); r2 = r - sa
    ss = np.minimum(r2, w["mu"] * w["spot_share"])
    return (sp * (w["sell"] - w["cost"]) - (qp - sp) * w["cost"]
            + sa * (w["sell"] - w["allot"]) - qa * w["fee"]
            + ss * np.maximum(w["sell"] - w["spot"], 0))


def optimise(w, budget, risk=0.5, D=None):
    """Greedy marginal allocation: repeatedly add 5 rooms (pre-buy or allotment) to the block with the
    best risk-adjusted gain per rupee, until the budget runs out. risk = weight on the bad-case (P10) loss."""
    D = w["D"] if D is None else D
    n = w["n"]
    qp = np.zeros(n); qa = np.zeros(n); spent = 0.0
    cur = profit(w, D, qp, qa)
    unit = [STEP * w["cost"], STEP * w["fee"]]
    while True:
        best = None; bv = 0.0
        for kind in (0, 1):
            a = qp + STEP if kind == 0 else qp
            b = qa if kind == 0 else qa + STEP
            new = profit(w, D, a, b)
            d = new - cur
            gain = (d.mean(0) - risk * np.maximum(0, -np.percentile(d, 10, axis=0))) / unit[kind]
            gain[spent + unit[kind] > budget] = -np.inf
            i = int(np.argmax(gain))
            if gain[i] > bv:
                bv = gain[i]; best = (i, kind, new[:, i])
        if best is None:
            break
        i, kind, col = best
        if kind == 0: qp[i] += STEP
        else: qa[i] += STEP
        spent += unit[kind][i]; cur[:, i] = col
    return qp, qa


def last_year_plan(w, budget):
    """Baseline: split the budget by last year's sales, all as hard pre-buys."""
    share = w["last_year"] / w["last_year"].sum()
    return np.floor(budget * share / w["cost"]), np.zeros(w["n"])


def summary(w, D, qp, qa):
    p = profit(w, D, qp, qa).sum(1)
    sp = np.minimum(D, qp); r = D - sp; sa = np.minimum(r, qa); r2 = r - sa
    ss = np.minimum(r2, w["mu"] * w["spot_share"])
    served = (sp + sa + ss).sum(1)
    return dict(profit=p.mean(), p10=np.percentile(p, 10),
                unsold=(qp - sp).sum(1).mean(), served=served.mean() / D.sum(1).mean(),
                capital=float((qp * w["cost"] + qa * w["fee"]).sum()))


def action_for(w, i, qp, qa):
    if qp[i] > 0 and qp[i] >= qa[i]: return "Pre-buy"
    if qa[i] > 0: return "Allotment"
    if w["spot_share"][i] < 0.6: return "Integrate supplier"
    return "On demand"


def reason_for(w, i, action):
    cv = w["sd"][i] / w["mu"][i]
    certainty = "narrow" if cv < 0.35 else ("medium" if cv < 0.5 else "wide")
    api = int(round(w["spot_share"][i] * 100))
    if action == "Pre-buy":
        return f"Strong demand, {certainty} range; bulk rate beats spot"
    if action == "Allotment":
        return f"Strong demand but {certainty} range; small fee, release unsold"
    if action == "Integrate supplier":
        return f"API serves only {api}% of demand: fix the connection, no budget"
    return f"API serves {api}% of demand; no need to tie up cash"
