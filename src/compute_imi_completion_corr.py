import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
p1 = BASE / "data" / "processed" / "cohorts" / "course_summary.csv"
p2 = BASE / "data" / "processed" / "course_summary.csv"
file_path = p1 if p1.exists() else p2
if not file_path.exists():
    raise FileNotFoundError(f"course_summary.csv not found at {p1} or {p2}")

df = pd.read_csv(file_path)
rho, p = spearmanr(df["IMI_mean"], df["completion_rate"])

out_dir = p1.parent if p1.exists() else (BASE / "data" / "processed" / "cohorts")
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "imi_completion_corr.txt"
out.write_text(f"Spearman rho = {rho:.3f}, p-value = {p:.2e}\n", encoding="utf-8")
print(f"✅ Saved: {out}")
