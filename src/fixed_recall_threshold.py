# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="umap.umap_",
    lineno=1952
)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score
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
if missing := [c for c in desired if c not in df.columns]:
    print(f"[WARN] Skipping absent features: {missing}")

X = df[present].copy()
y = df["y_final"]
ok = X.notna().all(axis=1) & y.notna()
X, y = X[ok], y[ok]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

clf = GradientBoostingClassifier(random_state=42)
clf.fit(X_train, y_train)
p = clf.predict_proba(X_test)[:, 1]

# подбираем порог под фиксированный recall=0.80
prec, rec, thr = precision_recall_curve(y_test, p)
target = 0.80
# найдём ближайшую по модулю точку
idx = int(np.argmin(np.abs(rec - target)))
thr_star = float(thr[max(0, idx-1)]) if idx==len(thr) else float(thr[idx])

y_pred = (p >= thr_star).astype(int)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

out_dir = PROCESSED / "models_final" / "policy_recall80"
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame([{
    "model": "GBDT",
    "status": "ok",
    "thr": thr_star,
    "val_recall_target": target,
    "test_precision": float(prec[idx]),
    "test_f1": float(f1),
    "test_brier": float(np.mean((y_test - p) ** 2)),
}]).to_csv(out_dir / "metrics_recall80.csv", index=False)

pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"])\
    .to_csv(out_dir / "confusion_test.csv")
print("Confusion matrices saved ->", out_dir / "confusion_test.csv")
print("Saved ->", out_dir / "metrics_recall80.csv")