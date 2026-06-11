#!/usr/bin/env python3
"""
Generate a realistic (synthetic) subscription dataset for the case study.

This is clearly labelled synthetic data so the analysis runs end to end with
no network access. The METHOD is what matters and transfers directly to a real
dataset: swap in your own CSV with the same columns and re-run analyze.py.
"""
import csv
import os
import numpy as np

rng = np.random.default_rng(42)
N = 2000
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
OUT = os.path.join(HERE, "data", "subscriptions.csv")

contract = rng.choice(["month-to-month", "one-year", "two-year"], N, p=[0.55, 0.30, 0.15])
plan = rng.choice(["basic", "standard", "premium"], N, p=[0.40, 0.40, 0.20])
payment = rng.choice(["auto", "manual"], N, p=[0.60, 0.40])
tenure = rng.integers(1, 73, N)
tickets = rng.poisson(1.5, N)

base = {"basic": 30.0, "standard": 65.0, "premium": 100.0}
charges = np.array([base[p] for p in plan]) + rng.normal(0, 6, N)
charges = np.round(np.clip(charges, 18, 140), 2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# True data-generating process (unknown to the analyst at analysis time).
logit = (
    -1.0
    + 1.4 * (contract == "month-to-month")
    - 0.35 * (contract == "two-year")
    - 0.045 * tenure
    + 0.28 * tickets
    + 0.55 * (payment == "manual")
    + 0.004 * (charges - 60)
)
churn = (rng.random(N) < sigmoid(logit)).astype(int)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["customer_id", "contract", "plan", "payment_method",
                "tenure_months", "support_tickets", "monthly_charges", "churned"])
    for i in range(N):
        w.writerow([1000 + i, contract[i], plan[i], payment[i],
                    int(tenure[i]), int(tickets[i]), float(charges[i]), int(churn[i])])

print("wrote %d rows -> %s" % (N, OUT))
print("overall churn rate: %.1f%%" % (100 * churn.mean()))
