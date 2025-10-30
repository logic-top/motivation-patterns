# -*- coding: utf-8 -*-
"""
Недельная динамика IMI (OULAD).

Артефакты:
- data/processed/imi_dyn/imi_by_week.csv
- data/processed/imi_dyn/converted_share.csv
- data/processed/imi_dyn/imi_week_trends.png
- data/processed/imi_dyn/converted_bar.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- ПАРАМЕТРЫ ----------
DEADLINE_WIN_DAYS = 2
EARLY_WEEKS = (1, 3)
LATE_WEEKS  = (7, 9)
DELTA_TAU   = 0.10
MIN_WEEKS_FOR_TREND = 4
TOP_COURSES_PLOTS = 6
USE_DEBUG_SV = False  # True — брать предварительно сохранённый temp_sv_debug.parquet

# ---------- ПУТИ ----------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "OULAD"
PROCESSED = ROOT / "data" / "processed"
OUTDIR = PROCESSED / "imi_dyn"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- УТИЛИТЫ ----------
def read_csv_norm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def gini_evenness_from_series(s: pd.Series) -> float:
    x = s.to_numpy(dtype=float)
    if x.size == 0 or np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(max(0.0, 1.0 - gini))

def coefvar_intervals_from_days(s_days: pd.Series) -> float:
    d = np.sort(s_days.to_numpy(dtype=float))
    if d.size <= 2:
        return np.nan
    gaps = np.diff(np.unique(d))
    if gaps.size == 0:
        return np.nan
    mu = float(np.mean(gaps))
    sigma = float(np.std(gaps, ddof=1)) if gaps.size > 1 else 0.0
    return float(sigma / mu) if mu > 0 else np.nan

def minmax_in_group(df: pd.DataFrame, keys: list[str], col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    g = df.groupby(keys, dropna=False)[col]
    vmin = g.transform("min")
    vmax = g.transform("max")
    span = (vmax - vmin).replace(0, np.nan)
    z = (df[col] - vmin) / span
    return z.fillna(0.0)

def coalesce_columns(df: pd.DataFrame, target: str, candidates: list[str], default_value) -> None:
    if target in df.columns:
        df[target] = df[target].fillna(default_value)
        return
    for c in candidates:
        if c in df.columns:
            df[target] = df[c].fillna(default_value)
            return
    df[target] = default_value

# ---------- ЗАГРУЗКА ----------
vle = read_csv_norm(RAW / "vle.csv")
student_vle = read_csv_norm(RAW / "studentVle.csv")
assessments = read_csv_norm(RAW / "assessments.csv")
student_assessment = read_csv_norm(RAW / "studentAssessment.csv")
student_info = read_csv_norm(RAW / "studentInfo.csv")

if USE_DEBUG_SV and (PROCESSED / "temp_sv_debug.parquet").exists():
    sv = pd.read_parquet(PROCESSED / "temp_sv_debug.parquet")
    print("[imi_dynamics] Using sv from temp_sv_debug.parquet")
else:
    must = {"id_site", "code_module", "code_presentation"}
    if not must.issubset(vle.columns):
        raise ValueError("vle.csv must contain: id_site, code_module, code_presentation")

    sv = student_vle.merge(
        vle[["id_site", "activity_type", "code_module", "code_presentation"]],
        on="id_site", how="left", validate="many_to_one", suffixes=("", "_vle")
    )
    # коалесценция ключей
    coalesce_columns(sv, "code_module", ["code_module","code_module_x","code_module_y","code_module_vle"], "UNK")
    coalesce_columns(sv, "code_presentation", ["code_presentation","code_presentation_x","code_presentation_y","code_presentation_vle"], "UNK")
    if "activity_type" not in sv.columns:
        coalesce_columns(sv, "activity_type", ["activity_type","activity_type_x","activity_type_y","activity_type_vle"], "unknown")

# базовая проверка и типы
core_cols = ["id_student","id_site","date","sum_click","activity_type","code_module","code_presentation"]
missing = [c for c in core_cols if c not in sv.columns]
if missing:
    raise KeyError(f"sv missing columns: {missing}")

sv["date"] = pd.to_numeric(sv["date"], errors="coerce")
sv["sum_click"] = pd.to_numeric(sv["sum_click"], errors="coerce").fillna(0.0).astype(float)
sv = sv.dropna(subset=["id_student", "date"])

sv["day_pos"] = np.maximum(0, sv["date"])
sv["week"] = (sv["day_pos"] // 7).astype(int)

# дневные агрегаты без apply
sv_day = (
    sv.groupby(["id_student","code_module","code_presentation","date","week"],
               as_index=False, dropna=False)
      .agg(clicks=("sum_click","sum"),
           activity_types=("activity_type", lambda s: list(s)))
)

# ---------- дедлайны ----------
ass_min = (assessments[["code_module","code_presentation","date"]]
           .assign(adate=lambda d: pd.to_numeric(d["date"], errors="coerce")))
ass_min = ass_min.dropna(subset=["adate"]).astype({"adate":"int64"})

sv_match = sv_day.merge(ass_min, on=["code_module","code_presentation"], how="left")

# КОАЛЕСЦЕНЦИЯ date (на случай появления date_x/date_y)
coalesce_columns(sv_match, "date", ["date","date_x","date_y"], np.nan)
if "date" not in sv_match.columns:
    raise KeyError("sv_match has no 'date' even after coalesce")
sv_match["date"] = pd.to_numeric(sv_match["date"], errors="coerce")

sv_match["is_last_minute"] = (
    (sv_match["adate"].notna()) &
    (sv_match["date"] >= sv_match["adate"] - DEADLINE_WIN_DAYS) &
    (sv_match["date"] <= sv_match["adate"])
)

# дневной срез с LM
sv_daily = (
    sv_match.groupby(["id_student","code_module","code_presentation","date","week"],
                     as_index=False, dropna=False)
            .agg(clicks=("clicks","first"),
                 lm=("is_last_minute","max"))
)

# ---------- недельные кумулятивы ----------
grp_keys = ["id_student","code_module","code_presentation"]
weekly_parts = []
for (sid, mod, pres), sub in sv_daily.groupby(grp_keys, sort=False, dropna=False):
    g = sub.sort_values("date").copy()
    weeks = np.sort(g["week"].unique())
    rows = []
    for w in weeks:
        g_up = g[g["week"] <= w]
        days = g_up["date"]
        clicks = g_up["clicks"]
        lm_flag = g_up["lm"].astype(int)

        reg_cv = coefvar_intervals_from_days(days)

        total_clicks = float(clicks.sum())
        lm_clicks = float((clicks * lm_flag).sum())
        lm_ratio = (lm_clicks / total_clicks) if total_clicks > 0 else np.nan

        self_init = float((1 - lm_flag).mean()) if len(lm_flag) > 0 else np.nan

        if len(clicks) >= 5:
            th = float(np.median(clicks) + 0.5 * np.std(clicks, ddof=1))
            deep_share = float((clicks >= th).mean())
        else:
            deep_share = np.nan

        even = gini_evenness_from_series(clicks)
        route_ent = 0.0

        rows.append({
            "id_student": sid,
            "code_module": mod,
            "code_presentation": pres,
            "week": int(w),
            "active_days": int(len(days)),
            "total_clicks": total_clicks,
            "regularity_cv": reg_cv,
            "last_minute_ratio": lm_ratio,
            "self_initiation_rate": self_init,
            "deep_session_share": deep_share,
            "social_evenness": even,
            "route_entropy": route_ent,
        })
    weekly_parts.append(pd.DataFrame(rows))

weekly = (pd.concat(weekly_parts, ignore_index=True)
          if weekly_parts else
          pd.DataFrame(columns=[
              "id_student","code_module","code_presentation","week",
              "active_days","total_clicks","regularity_cv","last_minute_ratio",
              "self_initiation_rate","deep_session_share","social_evenness","route_entropy"
          ]))

# ---------- IMI по неделям ----------
keys = ["code_module","code_presentation","week"]
pos = ["self_initiation_rate","deep_session_share","social_evenness","route_entropy"]
neg = ["regularity_cv","last_minute_ratio"]

for c in pos + neg:
    if c not in weekly.columns:
        weekly[c] = np.nan
for c in pos + neg:
    weekly[f"{c}_z"] = minmax_in_group(weekly, keys, c)

weekly["_imi_pos_sum"] = weekly[[f"{c}_z" for c in pos]].sum(axis=1)
weekly["_imi_neg_sum"] = (1 - weekly[[f"{c}_z" for c in neg]].sum(axis=1))
weekly["_IMI_raw"] = weekly["_imi_pos_sum"] + weekly["_imi_neg_sum"]
weekly["IMI_week"] = minmax_in_group(weekly, keys, "_IMI_raw")
weekly = weekly.drop(columns=["_imi_pos_sum","_imi_neg_sum","_IMI_raw"])

# ---------- агрегаты и сохранение ----------
imi_by_week = (weekly.groupby(["code_module","code_presentation","week"], as_index=False, dropna=False)
               .agg(IMI_mean=("IMI_week","mean"),
                    n=("IMI_week","size")))
imi_by_week.to_csv(OUTDIR / "imi_by_week.csv", index=False)

# «перешедшие» (ΔIMI ≥ τ)
early_mask = (weekly["week"] >= EARLY_WEEKS[0]) & (weekly["week"] <= EARLY_WEEKS[1])
late_mask  = (weekly["week"] >= LATE_WEEKS[0])  & (weekly["week"] <= LATE_WEEKS[1])

early = (weekly[early_mask]
         .groupby(grp_keys, as_index=False, dropna=False)
         .agg(IMI_early=("IMI_week","mean"), n_early=("IMI_week","size")))
late = (weekly[late_mask]
        .groupby(grp_keys, as_index=False, dropna=False)
        .agg(IMI_late=("IMI_week","mean"), n_late=("IMI_week","size")))

conv = early.merge(late, on=grp_keys, how="inner")
conv["delta_IMI"] = conv["IMI_late"] - conv["IMI_early"]
conv["converted"] = (conv["delta_IMI"] >= DELTA_TAU).astype(int)

info_keep = ["id_student","code_module","code_presentation","age_band","imd_band","gender"]
if set(info_keep).issubset(student_info.columns):
    conv = conv.merge(student_info[info_keep], on=grp_keys, how="left")

converted_share_course = (conv.groupby(["code_module","code_presentation"], as_index=False, dropna=False)
                          .agg(n=("converted","size"),
                               converted_share=("converted","mean")))
converted_share_course.to_csv(OUTDIR / "converted_share.csv", index=False)

# ---------- графики ----------
top_courses = (imi_by_week.groupby(["code_module","code_presentation"])["n"].sum()
               .sort_values(ascending=False).head(TOP_COURSES_PLOTS).index.tolist())

plt.figure(figsize=(10, 6))
for (mod, pres) in top_courses:
    cur = imi_by_week[(imi_by_week["code_module"] == mod) &
                      (imi_by_week["code_presentation"] == pres)]
    if cur["week"].nunique() < MIN_WEEKS_FOR_TREND:
        continue
    plt.plot(cur["week"], cur["IMI_mean"], marker="o", linewidth=1.5,
             label=f"{mod}-{pres} (N={int(cur['n'].sum())})")
plt.xlabel("Week")
plt.ylabel("IMI (mean)")
plt.title("IMI dynamics by week (top courses)")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "imi_week_trends.png", dpi=150)

cs_sorted = converted_share_course.sort_values("n", ascending=False).head(12)
plt.figure(figsize=(10, 5))
x = [f"{m}-{p}" for m, p in zip(cs_sorted["code_module"], cs_sorted["code_presentation"])]
plt.bar(x, cs_sorted["converted_share"])
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.ylabel(f"Converted share (ΔIMI ≥ {DELTA_TAU:.2f})")
plt.title("Share of 'converted' (early→late) by course")
plt.tight_layout()
plt.savefig(OUTDIR / "converted_bar.png", dpi=150)

print(f"[imi_dynamics] Saved -> {OUTDIR/'imi_by_week.csv'}")
print(f"[imi_dynamics] Saved -> {OUTDIR/'converted_share.csv'}")
print(f"[imi_dynamics] Saved -> {OUTDIR/'imi_week_trends.png'}")
print(f"[imi_dynamics] Saved -> {OUTDIR/'converted_bar.png'}")