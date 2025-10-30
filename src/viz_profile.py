# src/viz_profile.py
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import re

def safe_fname(s: str) -> str:
    # спец-замены для сравнений
    s = s.replace('<=', '_le_').replace('>=', '_ge_').replace('<', '_lt_').replace('>', '_gt_')
    # заменить всё, кроме безопасных символов, на "_"
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')

# --- paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = PROCESSED / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- load features (parquet -> csv fallback)
def load_any(basename: str) -> pd.DataFrame:
    pq = PROCESSED / f"{basename}.parquet"
    cs = PROCESSED / f"{basename}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if cs.exists():
        return pd.read_csv(cs)
    raise FileNotFoundError(f"Not found: {basename}.parquet/csv in {PROCESSED}")

df = load_any("features_with_IMI_v1_with_info_v2")

# --- indicators used in IMI
cols_pos = ["self_initiation_rate", "deep_session_share", "score_gain_rate",
            "social_evenness", "route_entropy"]
cols_inv = ["regularity_cv", "last_minute_ratio"]
imi_cols = cols_pos + cols_inv

# --- helper: safe median
def med(series):
    return float(np.nanmedian(series.values)) if len(series) else np.nan

# ========= 1) Таблица профилей кластеров =========
# Нужны метки кластеров
clusters_path = PROCESSED / "cluster_labels_by_age_band_v2.parquet"
if not clusters_path.exists():
    clusters_path = PROCESSED / "cluster_labels_by_age_band_v2.csv"

if clusters_path.exists():
    cl = pd.read_parquet(clusters_path) if clusters_path.suffix == ".parquet" else pd.read_csv(clusters_path)
    merged = df.merge(cl, on="id_student", how="inner")
else:
    # если кластеров нет — агрегируем без них
    merged = df.copy()
    merged["cluster_label"] = -1
    merged["age_band_val"] = merged["age_band"].fillna("NA")

# агрегаты: медианы индикаторов и IMI, размер кластера
profile = (merged
           .groupby(["age_band_val", "cluster_label"], dropna=False)
           .agg(**{c: (c, med) for c in imi_cols + ["IMI_v1"]} | {"n": ("id_student", "count")})
           .reset_index())
profile = profile.sort_values(["age_band_val", "cluster_label"])
out_csv = PROCESSED / "cluster_profiles_v2.csv"
profile.to_csv(out_csv, index=False)
print("Saved cluster profiles ->", out_csv)

# ========= 2) Boxplot IMI по возрастам =========
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="age_band", y="IMI_v1")
sns.stripplot(data=df.sample(min(3000, len(df)), random_state=42),
              x="age_band", y="IMI_v1", size=2, alpha=0.25, jitter=0.25, color="k")
plt.title("IMI by age_band")
plt.xlabel("age_band")
plt.ylabel("IMI_v1")
fig1 = FIG_DIR / "imi_boxplot_by_age.png"
plt.tight_layout(); plt.savefig(fig1, dpi=150)
plt.close()
print("Saved figure ->", fig1)

# ========= 3) Радар-диаграммы по кластерам (для каждой возрастной группы) =========
def radar_for_age_band(ab: str, prof_age: pd.DataFrame):
    # порядок осей: pos сначала, потом inv (в подписи пометим инверсные)
    axes = cols_pos + [c + " (inv)" for c in cols_inv]
    cols_in_order = cols_pos + cols_inv
    # нормируем каждую метрику на [0,1] по всем кластерам данной age_band
    M = prof_age[cols_in_order].copy()
    M = (M - M.min()) / (M.max() - M.min() + 1e-9)

    labels = axes
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # замкнуть
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    for _, row in M.iterrows():
        vals = row.values
        vals = np.concatenate([vals, [vals[0]]])
        ax.plot(angles, vals, linewidth=1)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    ax.set_title(f"Cluster radar — age_band={ab}")
    ax.grid(True)
    plt.tight_layout()
    out = FIG_DIR / f"radar_age_{safe_fname(str(ab))}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved figure ->", out)

if "age_band_val" in profile.columns and (profile["cluster_label"] >= 0).any():
    for ab, sub in profile[profile["cluster_label"] >= 0].groupby("age_band_val"):
        radar_for_age_band(ab, sub)

# ========= 4) 2D-проекция признаков (UMAP если установлен, иначе PCA) =========
proj_df = merged.dropna(subset=imi_cols).copy()
if len(proj_df) > 0:
    X = proj_df[imi_cols].values
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
        Z = reducer.fit_transform(StandardScaler().fit_transform(X))
        method = "UMAP"
    except Exception:
        Z = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
        method = "PCA"

    proj_df["z1"] = Z[:, 0]
    proj_df["z2"] = Z[:, 1]

    plt.figure(figsize=(7, 5))
    hue = "cluster_label" if "cluster_label" in proj_df.columns else None
    sns.scatterplot(data=proj_df.sample(min(8000, len(proj_df)), random_state=42),
                    x="z1", y="z2", hue=hue, size=None, alpha=0.35, linewidth=0)
    plt.title(f"{method} projection of IMI indicators")
    fig2 = FIG_DIR / f"proj_{method.lower()}_imi_indicators.png"
    plt.tight_layout(); plt.savefig(fig2, dpi=150)
    plt.close()
    print("Saved figure ->", fig2)

print("Done.")