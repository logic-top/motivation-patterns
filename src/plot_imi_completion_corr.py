import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# База проекта = два уровня вверх от файла: .../motivation-patterns
BASE = Path(__file__).resolve().parents[1]

# CSV: пытаемся найти в двух вариантах
p1 = BASE / "data" / "processed" / "cohorts" / "course_summary.csv"
p2 = BASE / "data" / "processed" / "course_summary.csv"
file_path = p1 if p1.exists() else p2
if not file_path.exists():
    raise FileNotFoundError(f"course_summary.csv not found at {p1} or {p2}")

df = pd.read_csv(file_path)

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(6, 5))
x, y = df["IMI_mean"], df["completion_rate"]
sns.regplot(x=x, y=y, ci=95, scatter_kws={'s': 60, 'alpha': 0.8}, line_kws={'color': 'red'})

for _, row in df.iterrows():
    plt.text(row["IMI_mean"] + 0.002, row["completion_rate"] + 0.002, row["code_module"], fontsize=9)

plt.title("Связь IMI и завершения курса (по курсам OULAD)")
plt.xlabel("Средний IMI (по курсу)")
plt.ylabel("Доля завершивших (%)")

assets = BASE / "assets"
assets.mkdir(parents=True, exist_ok=True)
out = assets / "imi_completion_corr.png"
plt.tight_layout()
plt.savefig(out, dpi=300)
print(f"✅ Saved: {out}")
