# -*- coding: utf-8 -*-
"""
OULAD -> признаки на уровне (id_student, code_module, code_presentation)

Версия без FutureWarning:
- Нормализация и IMI_v1 через groupby().transform()
- last_minute_ratio и self_initiation_rate вычисляются векторно (merge + булев флаг + agg)
- Нигде не используем groupby.apply, возвращающий нестабильные типы

Индикаторы:
- regularity_cv         : CV интервалов между активными днями
- last_minute_ratio     : доля кликов в окне [deadline-2; deadline]
- self_initiation_rate  : доля активных дней вне дедлайновых окон
- deep_session_share    : доля «глубоких» дней (клики >= median + 0.5*std)
- social_evenness       : равномерность активности по дням (1 - Gini)
- route_entropy         : энтропия распределения activity_type (VLE)
- score_gain_rate       : норм. средний балл по оценкам (0..1)
- IMI_v1                : агрегированная метрика (внутри курса), веса стартово = 1

Выход:
data/processed/features_with_IMI_v1_with_info_v2.parquet (.csv)
+ версия с меткой времени (бэкап)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# --------------------------- параметры ---------------------------
RANDOM_STATE = 42
DEADLINE_WIN_DAYS = 2          # окно для last_minute_ratio (в днях)
MIN_ACTIVE_DAYS = 5            # минимум активных дней, чтобы считать deep_session_share
MIN_ROWS_TO_SAVE = 100         # минимум строк для сохранения

# --------------------------- пути ---------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "OULAD"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# --------------------------- утилиты ---------------------------
def read_csv_norm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def gini_evenness_from_series(s: pd.Series) -> float:
    """Evenness = 1 - Gini по значениям серии (>=0)."""
    x = s.to_numpy(dtype=float)
    if x.size == 0:
        return np.nan
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(max(0.0, 1.0 - gini))

def entropy_of_activity(s: pd.Series) -> float:
    """Энтропия Шеннона по counts категорий в Series."""
    counts = s.value_counts().to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def coefvar_intervals_from_days(s_days: pd.Series) -> float:
    """CV интервалов между активными днями (по Series дней)."""
    d = np.sort(s_days.to_numpy(dtype=float))
    if d.size <= 2:
        return np.nan
    gaps = np.diff(np.unique(d))
    if gaps.size == 0:
        return np.nan
    mu = np.mean(gaps)
    sigma = np.std(gaps, ddof=1) if gaps.size > 1 else 0.0
    return float(sigma / mu) if mu > 0 else np.nan

def minmax_transform_within_group(df: pd.DataFrame, keys: list[str], col: str) -> pd.Series:
    """(x - min) / (max - min) по группам; константные группы -> 0."""
    g = df.groupby(keys)[col]
    vmin = g.transform('min')
    vmax = g.transform('max')
    span = (vmax - vmin).replace(0, np.nan)
    z = (df[col] - vmin) / span
    z = z.fillna(0.0)
    return z

# --------------------------- старт ---------------------------
print("Course keys in VLE after merge: YES (course-specific metrics ON)")

# --- загрузка исходников
vle = read_csv_norm(RAW / "vle.csv")                             # id_site, code_module, code_presentation, activity_type, ...
student_vle = read_csv_norm(RAW / "studentVle.csv")              # id_student, id_site, date, sum_click
assessments = read_csv_norm(RAW / "assessments.csv")             # id_assessment, code_module, code_presentation, date, ...
student_assessment = read_csv_norm(RAW / "studentAssessment.csv")# id_student, id_assessment, score, ...
student_info = read_csv_norm(RAW / "studentInfo.csv")            # id_student, code_module, code_presentation, final_result, ...

# --- проверка ключевых полей
def must_have(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns {missing}. Have: {df.columns.tolist()}")

must_have(vle, ["id_site","activity_type","code_module","code_presentation"], "vle")
must_have(student_vle, ["id_student","id_site","date","sum_click"], "studentVle")
must_have(assessments, ["id_assessment","date","code_module","code_presentation"], "assessments")
must_have(student_assessment, ["id_student","id_assessment"], "studentAssessment")

# --- удаляем возможные дубликаты курсо-ключей из studentVle, чтобы не плодить _x/_y
student_vle = student_vle.drop(columns=["code_module","code_presentation"], errors="ignore")

# --- merge кликов с VLE-ключами
sv = student_vle.merge(
    vle[["id_site","activity_type","code_module","code_presentation"]],
    on="id_site", how="left", suffixes=("", "_vle")
)

# --- приведение типов / базовая чистка
sv["date"] = pd.to_numeric(sv["date"], errors="coerce")
sv["sum_click"] = pd.to_numeric(sv["sum_click"], errors="coerce").fillna(0.0).astype(float)
sv = sv.dropna(subset=["id_student","date"])

# --- диагностика колонок после merge
print("vle columns:", vle.columns.tolist()[:10], "... total:", len(vle.columns))
print("sv columns :", sv.columns.tolist()[:15], "... total:", len(sv.columns))
print("sv head    :", sv.head(3).to_dict(orient="records"))

# --------------------------- дневные агрегаты (id_student×курс×дата) ---------------------------
sv_day = (sv.groupby(["id_student","code_module","code_presentation","date"], as_index=False)["sum_click"]
            .sum()
            .rename(columns={"sum_click":"clicks"}))

# --------------------------- дедлайны и связка с оценками ---------------------------
assess_min = assessments[["id_assessment","code_module","code_presentation","date"]].copy()
assess_min["date"] = pd.to_numeric(assess_min["date"], errors="coerce")

sa = student_assessment.merge(assess_min, on="id_assessment", how="left")

# --------------------------- агрегаты по студенту×курсу (без apply) ---------------------------
agg_days = sv_day.groupby(["id_student","code_module","code_presentation"])

sv_agg = agg_days.agg(
    active_days=("date", "nunique"),
    total_clicks=("clicks", "sum"),
    regularity_cv=("date", lambda s: coefvar_intervals_from_days(s)),
    deep_session_share=("clicks", lambda s: float(((s >= (np.median(s) + 0.5*np.std(s, ddof=1))).mean())
                                                 ) if len(s) >= MIN_ACTIVE_DAYS else np.nan),
    social_evenness=("clicks", lambda s: gini_evenness_from_series(s))
).reset_index()

# --------------------------- last_minute_ratio & self_initiation_rate (векторно, без apply) ----
# 1) Разворачиваем дедлайны "по одному на строку"
assess_days_expl = (assessments[["code_module","code_presentation","date"]]
                    .assign(adate=lambda d: pd.to_numeric(d["date"], errors="coerce").astype("Int64"))
                    .dropna(subset=["adate"])
                    .astype({"adate":"int64"})
                    [["code_module","code_presentation","adate"]])

# 2) Соединяем каждый активный день sv_day с ВСЕМИ дедлайнами курса
sv_match = sv_day.merge(assess_days_expl, on=["code_module","code_presentation"], how="left")

# 3) Флаг "last-minute" на строке (true, если попали в окно перед хотя бы одним дедлайном)
sv_match["is_lm_row"] = (
    (sv_match["adate"].notna()) &
    (sv_match["date"] >= sv_match["adate"] - DEADLINE_WIN_DAYS) &
    (sv_match["date"] <= sv_match["adate"])
)

# 4) Схлопываем обратно к уникальному (student×course×date): any по is_lm_row
sv_lm = (sv_match.groupby(["id_student","code_module","code_presentation","date"], as_index=False)
         .agg(clicks=("clicks","first"),
              is_lm=("is_lm_row","max")))

# 5) Считаем метрики на уровне (student×course)
sv_lm["lm_clicks"] = sv_lm["clicks"] * sv_lm["is_lm"].astype(int)
lm_agg = (sv_lm.groupby(["id_student","code_module","code_presentation"], as_index=False)
          .agg(total_clicks=("clicks","sum"),
               lm_clicks=("lm_clicks","sum"),
               self_initiation_rate=("is_lm", lambda s: float((~s).mean()))))

lm_agg["last_minute_ratio"] = np.where(
    lm_agg["total_clicks"] > 0,
    lm_agg["lm_clicks"] / lm_agg["total_clicks"],
    np.nan
)

lm_si = lm_agg[["id_student","code_module","code_presentation","last_minute_ratio","self_initiation_rate"]]

# --------------------------- route_entropy (через agg + lambda) ---------------------------
route_ent = (sv.groupby(["id_student","code_module","code_presentation"])
             .agg(route_entropy=("activity_type", lambda s: entropy_of_activity(s)))
             .reset_index())

# --------------------------- оценки/баллы в курсе ---------------------------
sa_course = (sa.groupby(["id_student","code_module","code_presentation"])
               .agg(score_mean=("score","mean"))
               .reset_index())
sa_course["score_gain_rate"] = (sa_course["score_mean"] / 100.0).clip(0, 1)

# --------------------------- сборка фич ---------------------------
features = (sv_agg
            .merge(lm_si, on=["id_student","code_module","code_presentation"], how="left")
            .merge(route_ent, on=["id_student","code_module","code_presentation"], how="left")
            .merge(sa_course[["id_student","code_module","code_presentation","score_gain_rate"]],
                   on=["id_student","code_module","code_presentation"], how="left"))

features = features[features["active_days"].fillna(0) > 0].reset_index(drop=True)

# --------------------------- IMI_v1 (через transform) ---------------------------
keys = ["code_module","code_presentation"]

pos_feats = ["self_initiation_rate","deep_session_share","score_gain_rate","social_evenness","route_entropy"]
neg_feats = ["regularity_cv","last_minute_ratio"]

for c in pos_feats + neg_feats:
    if c not in features.columns:
        features[c] = np.nan

# Z-преобразования внутри курса
for c in pos_feats + neg_feats:
    features[f"{c}_z"] = minmax_transform_within_group(features, keys, c)

# IMI_raw = сумма позитивных Z + сумма (1 - Z) для негативных
features["_imi_pos_sum"] = features[[f"{c}_z" for c in pos_feats]].sum(axis=1)
features["_imi_neg_sum"] = (1 - features[[f"{c}_z" for c in neg_feats]].sum(axis=1))
features["_IMI_raw"] = features["_imi_pos_sum"] + features["_imi_neg_sum"]

# Нормируем IMI_raw в [0,1] внутри курса
features["IMI_v1"] = minmax_transform_within_group(features, keys, "_IMI_raw")

# чистим служебные
features = features.drop(columns=["_imi_pos_sum","_imi_neg_sum","_IMI_raw"])

# --------------------------- демография/исход ---------------------------
info_keep = ["id_student","code_module","code_presentation","final_result","age_band",
             "disability","highest_education","region","imd_band","gender"]
for c in info_keep:
    if c not in student_info.columns:
        student_info[c] = np.nan
info = student_info[info_keep].copy()

features_final = features.merge(info, on=["id_student","code_module","code_presentation"], how="left")

# --------------------------- диагностика ---------------------------
print("DEBUG shapes:")
for name, d in [("vle", vle), ("student_vle", student_vle), ("sv_day", sv_day), ("features_final", features_final)]:
    try:
        print(f"  {name}: {d.shape}")
    except Exception:
        pass
print("Non-null rows in features_final:", len(features_final.dropna(how="all")))
print("Columns in features_final (first 20):", features_final.columns.tolist()[:20])

# --------------------------- предохранитель ---------------------------
if features_final is None or len(features_final) < MIN_ROWS_TO_SAVE:
    raise RuntimeError(f"features_final is empty or too small (rows={0 if features_final is None else len(features_final)}); save cancelled.")

# --------------------------- сохранение (версионирование) ---------------------------
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_base = PROCESSED / "features_with_IMI_v1_with_info_v2"

features_final.to_parquet(out_base.with_suffix(".parquet"), index=False)
features_final.to_csv(out_base.with_suffix(".csv"), index=False, encoding="utf-8")
features_final.to_parquet(PROCESSED / f"features_with_IMI_v1_with_info_v2_{stamp}.parquet", index=False)

print("Saved features to:", PROCESSED)
print("Rows:", len(features_final), "Cols:", features_final.shape[1])