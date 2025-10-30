# src/cohort_analysis.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "cohorts"
OUT.mkdir(parents=True, exist_ok=True)

def load_any(basename: str) -> pd.DataFrame:
    pq = PROCESSED / f"{basename}.parquet"
    cs = PROCESSED / f"{basename}.csv"
    if pq.exists(): return pd.read_parquet(pq)
    if cs.exists(): return pd.read_csv(cs)
    raise FileNotFoundError(f"Not found: {basename}.parquet/csv in {PROCESSED}")

df = load_any("features_with_IMI_v1_with_info_v2")

# таргет завершения
y_map = {"Pass": 1, "Distinction": 1, "Fail": 0, "Withdrawn": 0}
df["y_final"] = df["final_result"].map(y_map)

# страховка на случай отсутствующих колонок
for c in ["age_band", "code_module", "code_presentation"]:
    if c not in df.columns:
        df[c] = np.nan

# ===== 1) Сводка по возрастам =====
age_summary = (df.groupby("age_band", dropna=False)
                 .agg(n=("id_student", "count"),
                      completion_rate=("y_final", "mean"),
                      IMI_mean=("IMI_v1", "mean"),
                      IMI_std=("IMI_v1", "std"))
                 .reset_index())
age_summary.to_csv(OUT / "age_summary.csv", index=False)
print("Saved ->", OUT / "age_summary.csv")

# ===== 2) Сводка по курсам =====
course_summary = (df.groupby(["code_module", "code_presentation"], dropna=False)
                    .agg(n=("id_student", "count"),
                         completion_rate=("y_final", "mean"),
                         IMI_mean=("IMI_v1", "mean"))
                    .reset_index())
course_summary.to_csv(OUT / "course_summary.csv", index=False)
print("Saved ->", OUT / "course_summary.csv")

# ===== 3) Heatmap helper =====
def safe_heatmap(pivot_df: pd.DataFrame, title: str, out_path: Path, cmap: str):
    """Рисует heatmap только если есть >=1 ненулевое число; иначе — лог и сохранение CSV."""
    # удаляем строки/столбцы, где всё NaN
    clean = pivot_df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    # сохраняем очищенную матрицу (наглядно, что именно осталось)
    clean.to_csv(out_path.with_suffix(".csv"))
    if clean.size == 0 or np.all(np.isnan(clean.values)):
        print(f"Skip heatmap: '{title}' — пустая матрица (все NaN). Saved CSV for inspection:", out_path.with_suffix(".csv"))
        return
    plt.figure(figsize=(max(6, 0.6*clean.shape[1]+2), max(4, 0.5*clean.shape[0]+2)))
    sns.heatmap(clean, annot=False, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved ->", out_path)

# ===== 4) Тепловые карты по курсам =====
pivot_comp = course_summary.pivot(index="code_module",
                                  columns="code_presentation",
                                  values="completion_rate")
pivot_imi = course_summary.pivot(index="code_module",
                                 columns="code_presentation",
                                 values="IMI_mean")

safe_heatmap(pivot_comp, "Completion rate by course (module × presentation)",
             OUT / "heatmap_completion.png", cmap="viridis")
safe_heatmap(pivot_imi, "IMI mean by course (module × presentation)",
             OUT / "heatmap_imi.png", cmap="magma")