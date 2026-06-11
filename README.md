# churn-case-study

An analyst case study that turns a subscription table into a single costed recommendation: segment analysis, a from-scratch logistic regression (NumPy, no sklearn), revenue-at-risk sizing, and four charts. Runs fully offline.

**Read the write-up: [CASE_STUDY.md](CASE_STUDY.md)**

## Run

```bash
python3 generate_data.py   # creates a reproducible synthetic dataset
python3 analyze.py         # segments + model + charts/ + findings.json
```

Only `pandas`, `numpy`, and `matplotlib` are required.

## Use real data

Replace `data/subscriptions.csv` with any CSV that has these columns and re-run `analyze.py`:

```
customer_id, contract, plan, payment_method, tenure_months, support_tickets, monthly_charges, churned
```

`churned` must be 0 or 1. Everything else, including the charts and the revenue-at-risk segment, recomputes automatically.

## What it demonstrates

- Framing a vague business problem into an answerable question
- Segmentation and the confound check (why segments alone mislead)
- A transparent, dependency-free model with honest validation (AUC reported)
- Translating drivers into a costed, testable recommendation with a holdout
- Clear data-provenance and caveats
