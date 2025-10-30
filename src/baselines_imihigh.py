# src/baselines_imihigh.py
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    brier_score_loss, confusion_matrix, f1_score
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import shap

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- загрузка признаков
def load_any(basename: str) -> pd.DataFrame:
    pq = PROCESSED / f"{basename}.parquet"
    cs = PROCESSED / f"{basename}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if cs.exists():
        return pd.read_csv(cs)
    raise FileNotFoundError(f"Not found: {basename}.parquet/csv in {PROCESSED}")

df = load_any("features_with_IMI_v1_with_info_v2")

# индикаторы (те же, что в IMI)
cols_pos = ["self_initiation_rate", "deep_session_share", "score_gain_rate",
            "social_evenness", "route_entropy"]
cols_inv = ["regularity_cv", "last_minute_ratio"]
X_cols = cols_pos + cols_inv

# фильтрация полных кейсов по X
X = df[X_cols].copy()
y_cont = df["IMI_v1"].copy()
id_student = df["id_student"].astype(int)

ok = X.notna().all(axis=1) & y_cont.notna()
X = X.loc[ok].reset_index(drop=True)
y_cont = y_cont.loc[ok].reset_index(drop=True)
id_student = id_student.loc[ok].reset_index(drop=True)

# метка: верхний квартиль IMI -> 1
q75 = y_cont.quantile(0.75)
y = (y_cont >= q75).astype(int)

print(f"Labeling rule: IMI_high = IMI_v1 >= {q75:.3f}")
print("Class balance:", dict(zip(*np.unique(y, return_counts=True))))

# сплит по студентам
unique_ids = id_student.unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.20, random_state=RANDOM_STATE, shuffle=True)
train_ids, val_ids = train_test_split(train_ids, test_size=0.20, random_state=RANDOM_STATE, shuffle=True)

def pick(ids):
    mask = id_student.isin(ids)
    return X.loc[mask].copy(), y.loc[mask].copy()

X_train, y_train = pick(train_ids)
X_val, y_val = pick(val_ids)
X_test, y_test = pick(test_ids)

# масштабирвание для MLP
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

@dataclass
class ModelReport:
    name: str
    roc_auc: float
    pr_auc: float
    brier: float
    thresh_youden: float
    f1_youden: float
    thresh_f1: float
    f1_best: float

def eval_model(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> ModelReport:
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # пороги
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    youden = np.argmax(tpr - fpr)
    thr_youden = thr_roc[youden]
    y_pred_y = (y_prob >= thr_youden).astype(int)
    f1_y = f1_score(y_true, y_pred_y)

    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    # f1 от кривой PR (thr_pr на 1 короче)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    i_best = np.nanargmax(f1s)
    thr_f1 = thr_pr[i_best]
    f1_best = float(f1s[i_best])

    # фигуры
    fig_dir = OUT_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC={roc:.3f}")
    plt.plot([0, 1], [0, 1], ls="--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {name}")
    plt.legend()
    plt.tight_layout(); plt.savefig(fig_dir / "roc.png", dpi=150); plt.close()

    # PR
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"PR-AUC={pr:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {name}")
    plt.legend()
    plt.tight_layout(); plt.savefig(fig_dir / "pr.png", dpi=150); plt.close()

    # калибровка (reliability)
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=15, strategy="quantile")
    plt.figure(figsize=(5, 4))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], ls="--", lw=1)
    plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title(f"Calibration — {name}")
    plt.tight_layout(); plt.savefig(fig_dir / "calibration.png", dpi=150); plt.close()

    # confusion @ best F1
    y_pred = (y_prob >= thr_f1).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"])
    cm_df.to_csv(fig_dir / "confusion_at_bestF1.csv")

    return ModelReport(name, float(roc), float(pr), float(brier),
                       float(thr_youden), float(f1_y),
                       float(thr_f1), float(f1_best))

reports = []

# ---- GBDT
gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
gb.fit(X_train, y_train)
y_val_prob_gb = gb.predict_proba(X_val)[:, 1]
rep_gb = eval_model("GBDT", y_val.to_numpy(), y_val_prob_gb)
reports.append(rep_gb)

# feature importance (GBDT)
imp = pd.Series(gb.feature_importances_, index=X_cols).sort_values(ascending=False)
imp.to_csv(OUT_DIR / "GBDT" / "feature_importance.csv")
plt.figure(figsize=(6, 4))
imp.head(10).iloc[::-1].plot(kind="barh")
plt.title("GBDT — top-10 importance")
plt.tight_layout(); plt.savefig(OUT_DIR / "GBDT" / "feature_importance_top10.png", dpi=150); plt.close()

# SHAP для GBDT (быстрый TreeExplainer)
try:
    expl = shap.TreeExplainer(gb)
    shap_vals = expl.shap_values(X_val)
    shap.summary_plot(shap_vals, X_val, feature_names=X_cols, show=False)
    plt.tight_layout(); plt.savefig(OUT_DIR / "GBDT" / "shap_summary.png", dpi=150); plt.close()
except Exception as e:
    print("SHAP warning:", e)

# ---- MLP (скромная архитектура + ранний стоп через валид. сплит)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                    alpha=1e-4, learning_rate_init=1e-3, batch_size=256,
                    max_iter=50, random_state=RANDOM_STATE, early_stopping=True,
                    n_iter_no_change=5, validation_fraction=0.1)
mlp.fit(X_train_s, y_train)
y_val_prob_mlp = mlp.predict_proba(X_val_s)[:, 1]
rep_mlp = eval_model("MLP", y_val.to_numpy(), y_val_prob_mlp)
reports.append(rep_mlp)

# --- сводная таблица метрик на валидации
rep_df = pd.DataFrame([r.__dict__ for r in reports])
rep_df.to_csv(OUT_DIR / "val_metrics.csv", index=False)
print("Saved validation metrics ->", OUT_DIR / "val_metrics.csv")
print(rep_df)

# --- финальная фиксация: оцениваем лучшую модель на тесте
best_name = rep_df.sort_values("pr_auc", ascending=False).iloc[0]["name"]
print("Best (by PR-AUC) on val:", best_name)

if best_name == "GBDT":
    best_model = gb
    Xte = X_test; yte = y_test; scaler_used = None
    thr = float(rep_df.loc[rep_df["name"]=="GBDT","thresh_f1"].iloc[0])
elif best_name == "MLP":
    best_model = mlp
    Xte = X_test_s; yte = y_test
    thr = float(rep_df.loc[rep_df["name"]=="MLP","thresh_f1"].iloc[0])
else:
    raise RuntimeError("Unknown best model")

yprob = best_model.predict_proba(Xte)[:, 1]
roc = roc_auc_score(yte, yprob)
pr = average_precision_score(yte, yprob)
brier = brier_score_loss(yte, yprob)
ypred = (yprob >= thr).astype(int)
cm = confusion_matrix(yte, ypred, labels=[0,1])
f1b = f1_score(yte, ypred)

test_df = pd.DataFrame([{
    "name": best_name, "roc_auc": roc, "pr_auc": pr, "brier": brier,
    "thr_bestF1": thr, "f1_at_bestF1": f1b
}])
test_df.to_csv(OUT_DIR / "test_metrics.csv", index=False)
pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(OUT_DIR / "test_confusion.csv", index=False)
print("Saved test metrics ->", OUT_DIR / "test_metrics.csv")
print(test_df)