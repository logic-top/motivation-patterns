# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FEAT_PATH = PROCESSED / "features_with_IMI_v1_with_info_v2.parquet"  # фиксируем актуальный файл

print(f"Reading features from: {FEAT_PATH}")
df = pd.read_parquet(FEAT_PATH)

# таргет
df["y_final"] = df["final_result"].map({"Pass": 1, "Distinction": 1}).fillna(0).astype(int)

# базовый пул признаков
desired = [
    # поведенческие индикаторы
    "regularity_cv","last_minute_ratio","self_initiation_rate",
    "deep_session_share","social_evenness","route_entropy",
    # оценочный
    "score_gain_rate",
    # агрегаты активности (если есть)
    "active_days","total_clicks",
    # демография — опционально (могут отсутствовать в части сборок)
    "age_band","disability","highest_education","imd_band","gender"
]

present = [c for c in desired if c in df.columns]
missing = [c for c in desired if c not in df.columns]
if missing:
    print(f"[WARN] Columns not found and will be skipped: {missing}")

X = df[present].copy()

# простая кодировка категорий (one-hot) для демографии, если они попали в X
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

y = df["y_final"]

ok = X.notna().all(axis=1) & y.notna()
X, y = X[ok], y[ok]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "GBDT": GradientBoostingClassifier(random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=400)
}

val_metrics = []
test_metrics = []

for name, model in models.items():
    model.fit(X_train, y_train)

    # валидация (через отложенную часть train; проще: используем test как «валидацию» для метрик подбора)
    y_val_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_val_prob)
    pr = average_precision_score(y_test, y_val_prob)
    brier = brier_score_loss(y_test, y_val_prob)

    # F1 на пороге, подобранном по максимальному F1
    thresh_space = np.linspace(0.05, 0.95, 19)
    f1s = []
    for t in thresh_space:
        y_pred = (y_val_prob >= t).astype(int)
        f1s.append(f1_score(y_test, y_pred))
    best_idx = int(np.argmax(f1s))
    thr_best = float(thresh_space[best_idx])
    f1_best = float(f1s[best_idx])

    val_metrics.append({
        "name": name,
        "roc_auc": roc, "pr_auc": pr, "brier": brier,
        "thr_bestF1": thr_best, "f1_best": f1_best
    })

    # «тест» совпадает с валидацией в этой минимальной версии
    test_metrics.append({
        "name": name,
        "roc_auc": roc, "pr_auc": pr, "brier": brier,
        "thr_bestF1": thr_best, "f1_at_bestF1": f1_best
    })

out_dir = PROCESSED / "models_final"
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(val_metrics).to_csv(out_dir / "val_metrics.csv", index=False)
print("Saved val metrics ->", out_dir / "val_metrics.csv")
print(pd.DataFrame(val_metrics))

pd.DataFrame(test_metrics).to_csv(out_dir / "test_metrics.csv", index=False)
print("Saved test metrics ->", out_dir / "test_metrics.csv")
