#!/usr/bin/env python3
"""
Churn case study: from a flat subscription table to a costed recommendation.

Pipeline:
  1. Load + validate the data
  2. Segment churn rates (contract, tenure band, support tickets, payment)
  3. Fit a from-scratch logistic regression (numpy) to rank drivers
  4. Quantify revenue-at-risk and a targeted intervention
  5. Save charts to charts/ and a machine-readable findings.json

No sklearn dependency: the model is plain gradient descent so it runs anywhere.
Swap data/subscriptions.csv for a real CSV with the same columns to re-run.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CHARTS = os.path.join(HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)
INK, ACCENT = "#222222", "#b5532a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc",
                     "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(os.path.join(HERE, "data", "subscriptions.csv"))
assert df["churned"].isin([0, 1]).all(), "churned must be 0/1"
findings = {"n": int(len(df)), "overall_churn": round(100 * df["churned"].mean(), 1)}
print("Rows: %d | overall churn: %.1f%%" % (len(df), findings["overall_churn"]))


def rate_by(col):
    g = df.groupby(col)["churned"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    g["mean"] = (100 * g["mean"]).round(1)
    return g


df["tenure_band"] = pd.cut(df["tenure_months"], [0, 6, 12, 24, 48, 72],
                           labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
df["ticket_band"] = pd.cut(df["support_tickets"], [-1, 0, 2, 4, 100],
                           labels=["0", "1-2", "3-4", "5+"])

by_contract = rate_by("contract")
by_tenure = rate_by("tenure_band")
by_tickets = rate_by("ticket_band")
by_payment = rate_by("payment_method")
print("\nChurn by contract:\n", by_contract)
print("\nChurn by tenure band:\n", by_tenure)
print("\nChurn by ticket band:\n", by_tickets)

findings["by_contract"] = by_contract["mean"].to_dict()
findings["by_tenure"] = by_tenure["mean"].to_dict()
findings["by_tickets"] = by_tickets["mean"].to_dict()
findings["by_payment"] = by_payment["mean"].to_dict()

# ---- From-scratch logistic regression to rank standardized drivers ----
X = pd.get_dummies(
    df[["contract", "plan", "payment_method", "tenure_months", "support_tickets", "monthly_charges"]],
    columns=["contract", "plan", "payment_method"], drop_first=True,
).astype(float)
feature_names = list(X.columns)
Xv = X.values
mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
Xs = (Xv - mu) / sd
y = df["churned"].values.astype(float)

w = np.zeros(Xs.shape[1])
b = 0.0
lr = 0.3
for _ in range(4000):
    z = Xs @ w + b
    p = 1.0 / (1.0 + np.exp(-z))
    gw = Xs.T @ (p - y) / len(y)
    gb = float((p - y).mean())
    w -= lr * gw
    b -= lr * gb

p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
acc = float(((p > 0.5).astype(int) == y).mean())
# Rank-based AUC (Mann-Whitney), no sklearn needed.
order = p.argsort()
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(1, len(p) + 1)
n_pos, n_neg = y.sum(), (1 - y).sum()
auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
print("\nLogReg train accuracy: %.3f | AUC: %.3f" % (acc, auc))

drivers = sorted(zip(feature_names, w), key=lambda t: abs(t[1]), reverse=True)
findings["model"] = {"accuracy": round(acc, 3), "auc": round(auc, 3)}
findings["top_drivers"] = [{"feature": f, "coef": round(float(c), 3)} for f, c in drivers[:6]]
print("Top drivers (standardized):")
for f, c in drivers[:6]:
    print("   %+0.3f  %s" % (c, f))

# ---- Revenue-at-risk + targeted intervention ----
high = df[(df["contract"] == "month-to-month") & (df["support_tickets"] >= 3) & (df["tenure_months"] <= 12)]
seg_churn = round(100 * high["churned"].mean(), 1) if len(high) else 0.0
mrr_at_risk = round(float(high.loc[high["churned"] == 1, "monthly_charges"].sum()), 2)
arr_at_risk = round(mrr_at_risk * 12, 2)
findings["segment"] = {"name": "month-to-month, tenure<=12, tickets>=3",
                       "customers": int(len(high)), "churn_pct": seg_churn,
                       "mrr_at_risk": mrr_at_risk, "arr_at_risk": arr_at_risk}
print("\nHigh-risk segment: %d customers, %.1f%% churn, MRR-at-risk $%.0f (ARR $%.0f)"
      % (len(high), seg_churn, mrr_at_risk, arr_at_risk))


def bar(series, title, fname, color=ACCENT):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    series.plot(kind="bar", ax=ax, color=color, width=0.62)
    ax.set_title(title, color=INK, fontweight="bold", loc="left")
    ax.set_ylabel("churn rate (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS, fname), dpi=130)
    plt.close(fig)


bar(by_contract["mean"], "Churn rate by contract type", "churn_by_contract.png")
bar(by_tenure["mean"], "Churn rate by tenure band (months)", "churn_by_tenure.png")
bar(by_tickets["mean"], "Churn rate by support-ticket band", "churn_by_tickets.png")

fig, ax = plt.subplots(figsize=(5.6, 3.2))
names = [f for f, _ in drivers[:6]][::-1]
vals = [c for _, c in drivers[:6]][::-1]
ax.barh(names, vals, color=[ACCENT if v > 0 else "#6b7db5" for v in vals])
ax.axvline(0, color="#999", lw=0.8)
ax.set_title("Churn drivers (standardized logistic coefficients)", color=INK, fontweight="bold", loc="left")
plt.tight_layout()
fig.savefig(os.path.join(CHARTS, "churn_drivers.png"), dpi=130)
plt.close(fig)

with open(os.path.join(HERE, "findings.json"), "w") as f:
    json.dump(findings, f, indent=2)
print("\nSaved 4 charts to charts/ and findings.json")
