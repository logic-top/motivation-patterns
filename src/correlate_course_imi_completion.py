# src/correlate_course_imi_completion.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
COH = PROCESSED / "cohorts"
OUT_IMG = COH / "imi_vs_completion_scatter.png"
OUT_TXT = COH / "imi_completion_corr.txt"

cs = pd.read_csv(COH / "course_summary.csv")

# фильтр на валидные строки
df = cs.dropna(subset=["IMI_mean", "completion_rate"]).copy()
if len(df) < 3:
    raise SystemExit("Not enough course rows with both IMI_mean and completion_rate.")

# Spearman
rho = df["IMI_mean"].corr(df["completion_rate"], method="spearman")
n = len(df)

# Пытаемся построить LOWESS (если есть statsmodels), иначе просто scatter
plt.figure(figsize=(6.5, 5))
sns.scatterplot(data=df, x="IMI_mean", y="completion_rate", s=25, alpha=0.7)

lowess_ok = False
try:
    import statsmodels.api as sm
    low = sm.nonparametric.lowess(df["completion_rate"], df["IMI_mean"], frac=0.6, return_sorted=True)
    plt.plot(low[:,0], low[:,1])
    lowess_ok = True
except Exception:
    pass

plt.xlabel("IMI_mean (по курсу)")
plt.ylabel("Completion rate (по курсу)")
t = f"IMI vs Completion (Spearman ρ={rho:.3f}, n={n})"
if not lowess_ok:
    t += " — LOWESS unavailable"
plt.title(t)
plt.tight_layout()
plt.savefig(OUT_IMG, dpi=150)
plt.close()

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(f"Spearman rho: {rho:.6f}\n")
    f.write(f"n courses: {n}\n")
    f.write(f"LOWESS: {'yes' if lowess_ok else 'no'}\n")

print("Saved ->", OUT_IMG)
print("Saved ->", OUT_TXT)