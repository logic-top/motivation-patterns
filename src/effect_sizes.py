# src/effect_sizes.py
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

def load_any(basename: str) -> pd.DataFrame:
    pq = PROCESSED / f"{basename}.parquet"
    cs = PROCESSED / f"{basename}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if cs.exists():
        return pd.read_csv(cs)
    raise FileNotFoundError(f"Not found: {basename}.parquet/csv in {PROCESSED}")

# исходные распределения IMI по возрастам
features = load_any("features_with_IMI_v1_with_info_v2")
pairs = load_any("imi_mannwhitney_age_pairs_v2")

def cliffs_delta(x, y):
    """Cliff's d с оценкой величины эффекта."""
    x = np.asarray(x)
    y = np.asarray(y)
    # быстрый подсчёт: сортировки + ранги
    # но для простоты — прямой подсчёт на сэмпле (ограничим до 50k попарных сравнений)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan, "NA"
    # сэмплируем если очень большие
    rs = np.random.RandomState(42)
    X = x if nx <= 10000 else rs.choice(x, size=10000, replace=False)
    Y = y if ny <= 10000 else rs.choice(y, size=10000, replace=False)
    # попарно (векторизованный подход)
    diff = X[:, None] - Y[None, :]
    d = (np.sum(diff > 0) - np.sum(diff < 0)) / (X.size * Y.size)
    ad = abs(d)
    # интерпретация (Romano et al.)
    if ad < 0.147: mag = "negligible"
    elif ad < 0.33: mag = "small"
    elif ad < 0.474: mag = "medium"
    else: mag = "large"
    return float(d), mag

def vargha_delaney_A12(x, y):
    """Vargha–Delaney A12 (вероятность, что X>Y)."""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    rs = np.random.RandomState(42)
    X = x if nx <= 10000 else rs.choice(x, size=10000, replace=False)
    Y = y if ny <= 10000 else rs.choice(y, size=10000, replace=False)
    # A12 = (sum_{i,j} [x_i>y_j] + 0.5* [x_i==y_j]) / (nx*ny)
    diff = X[:, None] - Y[None, :]
    A = (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (X.size * Y.size)
    return float(A)

rows = []
for _, r in pairs.iterrows():
    b1, b2 = r["band_1"], r["band_2"]
    x = features.loc[features["age_band"] == b1, "IMI_v1"].dropna().values
    y = features.loc[features["age_band"] == b2, "IMI_v1"].dropna().values
    d, mag = cliffs_delta(x, y)
    A12 = vargha_delaney_A12(x, y)
    rows.append({
        "band_1": b1, "band_2": b2,
        "n1": int(len(x)), "n2": int(len(y)),
        "U_stat": r.get("U_stat", np.nan),
        "p_value": r.get("p_value", np.nan),
        "cliffs_d": d, "magnitude": mag,
        "A12": A12
    })

out = pd.DataFrame(rows)
out_path = PROCESSED / "imi_effect_sizes_age_pairs_v2.csv"
out.to_csv(out_path, index=False)
print("Saved effect sizes ->", out_path)