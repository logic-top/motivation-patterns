# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "OULAD"
print("RAW =", RAW)

# 1) читаем исходники
vle = pd.read_csv(RAW / "vle.csv")
student_vle = pd.read_csv(RAW / "studentVle.csv")

# 2) аккуратный merge
sv = student_vle.merge(
    vle[["id_site", "activity_type", "code_module", "code_presentation"]],
    on="id_site", how="left", validate="many_to_one"
)

# 3) коалесценция возможных суффиксов (на всякий случай)
def coalesce(df, target, candidates, default):
    if target in df.columns:
        df[target] = df[target].fillna(default)
        return
    for c in candidates:
        if c in df.columns:
            df[target] = df[c].fillna(default)
            return
    df[target] = default

coalesce(sv, "code_module", ["code_module", "code_module_x", "code_module_y"], "UNK")
coalesce(sv, "code_presentation", ["code_presentation", "code_presentation_x", "code_presentation_y"], "UNK")
coalesce(sv, "activity_type", ["activity_type", "activity_type_x", "activity_type_y"], "unknown")

core_cols = ["id_student","id_site","date","sum_click","activity_type","code_module","code_presentation"]

print("\n--- sv columns (first 15) ---")
print(list(sv.columns)[:15], f"... total: {len(sv.columns)}")

print("\n--- sv[core_cols].head() ---")
print(sv[core_cols].head().to_dict(orient="records"))

# 4) на будущее — сохраним дамп для любого анализа
out = ROOT / "data" / "processed" / "temp_sv_debug.parquet"
sv.to_parquet(out, index=False)
print("\nSaved debug parquet ->", out)