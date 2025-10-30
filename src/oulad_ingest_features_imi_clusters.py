# src/oulad_ingest_features_imi_clusters.py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy, mannwhitneyu
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------
# Paths & folders
# --------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "OULAD"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Utils
# --------------------------
def regularity_cv_by_days(days: np.ndarray) -> float:
    """CV интервалов между активными днями (выше = нерегулярнее)."""
    days = np.unique(np.sort(days))
    if days.size < 2:
        return np.nan
    gaps = np.diff(days)
    m = gaps.mean()
    return float(gaps.std() / (m + 1e-9))

def route_entropy(activity_types: pd.Series) -> float:
    """Энтропия типов активностей (выше = исследовательность/вариативность)."""
    if activity_types.empty:
        return np.nan
    p = activity_types.value_counts(normalize=True)
    return float(entropy(p, base=2))

def pos_ratio_improvements(scores: pd.Series) -> float:
    """Доля положительных приростов между соседними оценочными событиями (прокси ретраев с улучшением)."""
    if scores.size < 2:
        return np.nan
    s = scores.sort_values().to_numpy()
    dif = np.diff(s)
    return float((dif > 0).mean())

def safe_minmax(x: pd.Series, invert: bool = False) -> pd.Series:
    """Min–max с защитой от NaN; invert=True инвертирует шкалу."""
    v = x.copy()
    if v.notna().sum() == 0:
        return pd.Series(np.nan, index=x.index)
    arr = v.values.astype(float)
    scaled = minmax_scale(arr, feature_range=(0, 1), copy=True)
    if invert:
        scaled = 1 - scaled
    return pd.Series(scaled, index=x.index)

# --------------------------
# 1) Import 7 CSV (RAW → INTERIM)
# --------------------------
assessments = pd.read_csv(DATA_DIR / "assessments.csv")
student_assessment = pd.read_csv(DATA_DIR / "studentAssessment.csv")
student_vle = pd.read_csv(DATA_DIR / "studentVle.csv")
vle = pd.read_csv(DATA_DIR / "vle.csv")
student_reg = pd.read_csv(DATA_DIR / "studentRegistration.csv")
courses = pd.read_csv(DATA_DIR / "courses.csv")
student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")

assessments.to_parquet(INTERIM_DIR / "assessments_raw.parquet", index=False)
student_assessment.to_parquet(INTERIM_DIR / "studentAssessment_raw.parquet", index=False)
student_vle.to_parquet(INTERIM_DIR / "studentVle_raw.parquet", index=False)
vle.to_parquet(INTERIM_DIR / "vle_raw.parquet", index=False)
student_reg.to_parquet(INTERIM_DIR / "studentRegistration_raw.parquet", index=False)
courses.to_parquet(INTERIM_DIR / "courses_raw.parquet", index=False)
student_info.to_parquet(INTERIM_DIR / "studentInfo_raw.parquet", index=False)

# --------------------------
# 2) Enrich studentVle activity by type
# --------------------------
sv = student_vle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")

# активные дни (клики по дню)
active_by_day = (sv.groupby(["id_student", "date"], as_index=False)["sum_click"]
                   .sum()
                   .rename(columns={"sum_click": "clicks"}))

# --------------------------
# 3) Indicators
# --------------------------
# 3.1 Regularity (CV), high = нерегулярно (внешняя) — позже инвертируем
regul = (active_by_day.groupby("id_student")["date"]
         .apply(lambda d: regularity_cv_by_days(d.to_numpy()))
         .rename("regularity_cv")
         .reset_index())

# 3.2 Last-minute ratio: доля дней near deadline (±2 дня от любого дедлайна; для пилота)
deadline_days = set(assessments["date"].unique().tolist())
def is_near_deadline(day: int, dl_set=deadline_days) -> bool:
    return any(abs(day - dd) <= 2 for dd in dl_set)

abd = active_by_day.copy()
abd["near_deadline"] = abd["date"].apply(is_near_deadline)
last_minute = (abd.groupby("id_student")["near_deadline"]
               .mean()
               .rename("last_minute_ratio")
               .reset_index())

# 3.3 Self-initiation: доля «неоценочных/теоретических» активностей (эвристика)
non_assess_types = {"oucontent", "resource", "url", "subpage", "ouelluminate"}
sv2 = sv.copy()
sv2["is_theory"] = sv2["activity_type"].isin(non_assess_types)
self_init = (sv2.groupby("id_student")["is_theory"]
             .mean()
             .rename("self_initiation_rate")
             .reset_index())

# 3.4 Depth (proxy): доля «нагруженных» дней (клики > медианы по студенту)
tmp = active_by_day.copy()
med = tmp.groupby("id_student")["clicks"].transform("median").replace(0, 1)
tmp["heavy_day"] = (tmp["clicks"] > med).astype(int)
deep_share = (tmp.groupby("id_student")["heavy_day"]
              .mean()
              .rename("deep_session_share")
              .reset_index())

# 3.5 Improvements: доля положительных приростов оценок (прокси ретраев с улучшением)
gain = (student_assessment.groupby("id_student")["score"]
        .apply(pos_ratio_improvements)
        .rename("score_gain_rate")
        .reset_index())

# 3.6 Social evenness: равномерная форумная активность (без всплеска в последний день)
forum = sv.assign(is_forum=sv["activity_type"].eq("forumng"))
social_even = (forum.groupby(["id_student", "date"])["is_forum"].max()
               .groupby("id_student").mean()
               .rename("social_evenness")
               .reset_index())

# 3.7 Entropy: вариативность маршрутов (типы активностей)
ent = (sv.groupby("id_student")["activity_type"]
       .apply(route_entropy)
       .rename("route_entropy")
       .reset_index())

# --------------------------
# 4) Assemble feature table
# --------------------------
parts = [regul, last_minute, self_init, deep_share, gain, social_even, ent]
feat = parts[0]
for t in parts[1:]:
    feat = feat.merge(t, on="id_student", how="outer")

# --------------------------
# 5) Normalize + IMI_v1
# --------------------------
cols_pos = ["self_initiation_rate", "deep_session_share", "score_gain_rate",
            "social_evenness", "route_entropy"]            # выше = внутренне
cols_inv = ["regularity_cv", "last_minute_ratio"]         # выше = внешне → инвертируем

norm = feat.copy()
for c in cols_pos:
    norm[c] = safe_minmax(norm[c], invert=False)
for c in cols_inv:
    norm[c] = safe_minmax(norm[c], invert=True)

imi_cols = cols_pos + cols_inv
norm["IMI_v1"] = norm[imi_cols].mean(axis=1)

# --------------------------
# 6) Attach studentInfo (for analysis only; НЕ в IMI)
# --------------------------
info_cols = ['id_student', 'age_band', 'disability', 'highest_education',
             'region', 'imd_band', 'gender', 'final_result']
for col in info_cols:
    if col not in student_info.columns:
        student_info[col] = np.nan
student_info_small = student_info[info_cols].copy()

# иногда бины возраста в OULAD разные; оставим категорию как есть
norm_with_info = norm.merge(student_info_small, on="id_student", how="left")

# сохранения
feat.to_parquet(PROCESSED_DIR / "features_raw.parquet", index=False)
norm.to_parquet(PROCESSED_DIR / "features_with_IMI_v1.parquet", index=False)
norm.to_csv(PROCESSED_DIR / "features_with_IMI_v1.csv", index=False)
norm_with_info.to_parquet(PROCESSED_DIR / "features_with_IMI_v1_with_info.parquet", index=False)
norm_with_info.to_csv(PROCESSED_DIR / "features_with_IMI_v1_with_info.csv", index=False)

print("Saved features to:", PROCESSED_DIR)

# --------------------------
# 7) Stratified clustering by age_band
# --------------------------
def kmeans_silhouette(X: pd.DataFrame, k: int, random_state: int = 42):
    if len(X) < k:
        return None, None, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels) if len(np.unique(labels)) > 1 else np.nan
    return km, labels, sil

cluster_reports = []
cluster_assignments = []  # (id_student, age_band, k, label)

for age_band, g in norm_with_info.groupby("age_band", dropna=False):
    g2 = g.dropna(subset=imi_cols).copy()
    if g2.empty or len(g2) < 10:
        continue
    X = g2[imi_cols]

    # попробуем k=3..6, выберем лучший по silhouette
    best = {"k": None, "sil": -np.inf, "labels": None}
    for k in range(3, 7):
        km, labels, sil = kmeans_silhouette(X, k)
        if km is None:
            continue
        if sil is not None and sil > best["sil"]:
            best.update({"k": k, "sil": sil, "labels": labels})

    if best["k"] is None:
        continue

    g2 = g2.assign(cluster_label=best["labels"], best_k=best["k"], sil_score=best["sil"], age_band_val=age_band)
    cluster_assignments.append(g2[["id_student", "age_band_val", "cluster_label", "best_k", "sil_score"]])

    cluster_reports.append({
        "age_band": age_band,
        "n": len(g2),
        "best_k": best["k"],
        "silhouette": round(best["sil"], 3),
        "IMI_mean": round(g2["IMI_v1"].mean(), 3),
        "IMI_std": round(g2["IMI_v1"].std(ddof=1), 3),
    })

if cluster_reports:
    cluster_reports_df = pd.DataFrame(cluster_reports).sort_values(by="age_band")
    cluster_reports_df.to_csv(PROCESSED_DIR / "cluster_reports_by_age_band.csv", index=False)
    print("\nCluster reports by age_band:\n", cluster_reports_df)
else:
    print("\nNo sufficient data for clustering by age_band.")

if cluster_assignments:
    clusters_df = pd.concat(cluster_assignments, ignore_index=True)
    clusters_df.to_csv(PROCESSED_DIR / "cluster_labels_by_age_band.csv", index=False)

# --------------------------
# 8) IMI comparisons across cohorts (Mann–Whitney U)
# --------------------------
# выберем самые частые две-три возрастные группы
age_counts = norm_with_info["age_band"].value_counts(dropna=False)
bands = age_counts.index.tolist()

def mw_test(a: pd.Series, b: pd.Series):
    a = a.dropna()
    b = b.dropna()
    if len(a) < 10 or len(b) < 10:
        return np.nan, np.nan
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(stat), float(p)

mw_rows = []
for i in range(len(bands)):
    for j in range(i + 1, len(bands)):
        b1, b2 = bands[i], bands[j]
        imi1 = norm_with_info.loc[norm_with_info["age_band"] == b1, "IMI_v1"]
        imi2 = norm_with_info.loc[norm_with_info["age_band"] == b2, "IMI_v1"]
        stat, p = mw_test(imi1, imi2)
        mw_rows.append({
            "band_1": b1,
            "band_2": b2,
            "n1": int(imi1.dropna().shape[0]),
            "n2": int(imi2.dropna().shape[0]),
            "U_stat": stat,
            "p_value": p
        })

mw_df = pd.DataFrame(mw_rows)
mw_df.to_csv(PROCESSED_DIR / "imi_mannwhitney_age_pairs.csv", index=False)
print("\nMann–Whitney IMI_v1 comparisons saved to:", PROCESSED_DIR)
