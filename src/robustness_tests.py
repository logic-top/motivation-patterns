# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FEAT_PATH = PROCESSED / "features_with_IMI_v1_with_info_v2.parquet"

print(f"Reading features from: {FEAT_PATH}")
df = pd.read_parquet(FEAT_PATH)
df["y_final"] = df["final_result"].map({"Pass": 1, "Distinction": 1}).fillna(0).astype(int)

desired = [
    "regularity_cv","last_minute_ratio","self_initiation_rate",
    "deep_session_share","social_evenness","route_entropy",
    "score_gain_rate","active_days","total_clicks"
]
present = [c for c in desired if c in df.columns]
missing = [c for c in desired if c not in df.columns]
if missing:
    print(f"[WARN] Skipping absent features: {missing}")

X = df[present].copy()
y = df["y_final"]

ok = X.notna().all(axis=1) & y.notna()
X, y = X[ok], y[ok]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

def eval_model(Xtr, Xte):
    m = GradientBoostingClassifier(random_state=42)
    m.fit(Xtr, y_train)
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(y_test, p), average_precision_score(y_test, p)

# 1) Базовый скор
base_roc, base_pr = eval_model(X_train, X_test)

# 2) Шумовая устойчивость (добавим N(0,σ) к числовым признакам, σ=0.05*std)
Xn = X_test.copy()
num_cols = Xn.columns.tolist()
for c in num_cols:
    std = Xn[c].std()
    if std > 0:
        Xn[c] = Xn[c] + np.random.RandomState(42).normal(0, 0.05*std, size=len(Xn))

rob_roc, rob_pr = eval_model(X_train, Xn)

# 3) Абляция признаков (drop-1)
ablation = []
for c in present:
    Xt = X_test.drop(columns=[c])
    Xtr = X_train.drop(columns=[c])
    roc, pr = eval_model(Xtr, Xt)
    ablation.append({"feature": c, "roc_auc": roc, "pr_auc": pr,
                     "delta_roc": base_roc - roc, "delta_pr": base_pr - pr})

out_dir = PROCESSED / "robustness"
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "base_roc_auc": [base_roc], "base_pr_auc": [base_pr],
    "robust_roc_auc": [rob_roc], "robust_pr_auc": [rob_pr]
}).to_csv(out_dir / "robustness_summary.csv", index=False)

pd.DataFrame(ablation).sort_values("delta_pr", ascending=False)\
    .to_csv(out_dir / "ablation_rank.csv", index=False)

print("Saved ->", out_dir / "robustness_summary.csv")
print("Saved ->", out_dir / "ablation_rank.csv")