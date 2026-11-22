"""Reproduce every prototype number on the VoyageHack 4.0 slides.

Run:  python reproduce.py            (about 1-2 minutes on a laptop)
"""
import numpy as np
import engine as E

BUDGET = 6_000_000  # Rs 60 lakh


def lakh(x):
    return f"Rs {x / 1e5:.1f}L"


def main():
    w = E.make_world(seed=11, scenarios=4000)
    zero = np.zeros(w["n"])
    base_q = E.last_year_plan(w, BUDGET)

    # 1) In-sample result (plan and score on the same 4,000 scenarios)
    qp, qa = E.optimise(w, BUDGET, risk=0.5)
    o = E.summary(w, w["D"], qp, qa)
    b = E.summary(w, w["D"], *base_q)
    s = E.summary(w, w["D"], zero, zero)
    print("== Same scenarios ==")
    print(f"On demand only : {lakh(s['profit'])}")
    print(f"Last-year split: {lakh(b['profit'])}")
    print(f"BlockWise      : {lakh(o['profit'])}  ({(o['profit'] / b['profit'] - 1) * 100:+.1f}%)")

    # 2) Held-out test: plan on one set of scenarios, score on a fresh set it never saw,
    #    with the planner's forecast wrong by 0%, 20% or 40% per block.
    print("\n== Fresh scenarios, forecast error ==")
    for err in (0.0, 0.2, 0.4):
        r = np.random.default_rng(7)
        f = np.exp(r.normal(0, err, w["n"])) if err > 0 else np.ones(w["n"])
        D_plan = E.draw_demand(w, np.random.default_rng(8), 4000, mu=w["mu"] * f,
                               sd=w["sd"] * (1.3 if err > 0 else 1.0))
        D_test = E.draw_demand(w, np.random.default_rng(1007), 4000)
        qp, qa = E.optimise(w, BUDGET, risk=0.5, D=D_plan)
        o = E.summary(w, D_test, qp, qa)
        b = E.summary(w, D_test, *base_q)
        s = E.summary(w, D_test, zero, zero)
        extra_o, extra_b = o["profit"] - s["profit"], b["profit"] - s["profit"]
        print(f"error {int(err * 100):>2}%: total {lakh(o['profit'])} vs {lakh(b['profit'])} "
              f"({(o['profit'] / b['profit'] - 1) * 100:+.1f}%) | return on budget "
              f"{extra_o / BUDGET * 100:.0f}% vs {extra_b / BUDGET * 100:.0f}% | demand served "
              f"{o['served'] * 100:.0f}% vs {b['served'] * 100:.0f}% | unsold room-nights "
              f"{o['unsold']:.0f} vs {b['unsold']:.0f}")


if __name__ == "__main__":
    main()
