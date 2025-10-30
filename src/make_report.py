# src/make_report.py
from pathlib import Path
import pandas as pd
import textwrap

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODF = PROC / "models_final"
ROBU = PROC / "robustness"
COH  = PROC / "cohorts"
REPO = PROC / "reports"
REPO.mkdir(parents=True, exist_ok=True)

def maybe_read_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None

def maybe_read_txt(p: Path):
    return p.read_text(encoding="utf-8") if p.exists() else None

val = maybe_read_csv(MODF / "val_metrics.csv")
test = maybe_read_csv(MODF / "test_metrics.csv")
pol = maybe_read_csv(MODF / "policy_recall80" / "metrics_recall80.csv")

rob_sum = maybe_read_csv(ROBU / "robustness_summary.csv")
abl = maybe_read_csv(ROBU / "ablation_rank.csv")

age = maybe_read_csv(COH / "age_summary.csv")
course = maybe_read_csv(COH / "course_summary.csv")

corr_txt = maybe_read_txt(COH / "imi_completion_corr.txt")

# собираем Markdown
md = []

md.append("# Отчёт по проекту: от мотивации к паттернам цифрового мышления\n")

# Метрики финальной задачи
md.append("## 1. Модели для `final_result` (завершение курса)\n")
if val is not None:
    md.append("**Валидация (лучшее по PR-AUC):**\n")
    md.append(val.to_markdown(index=False))
else:
    md.append("_val_metrics.csv не найден_\n")

if test is not None:
    md.append("\n**Тест (модель-победитель на валидации):**\n")
    md.append(test.to_markdown(index=False))
else:
    md.append("\n_test_metrics.csv не найден_\n")

# Политика recall≥0.8
md.append("\n## 2. Политика порога (recall ≥ 0.80)\n")
if pol is not None:
    md.append(pol.to_markdown(index=False))
    md.append("\nСм. матрицы ошибок: `models_final/policy_recall80/confusion_val.csv`, `.../confusion_test.csv`.\n")
else:
    md.append("_metrics_recall80.csv не найден_\n")

# Робастность
md.append("\n## 3. Робастность\n")
if rob_sum is not None:
    md.append("**Сценарии и PR-AUC на валидации:**\n")
    md.append(rob_sum.head(30).to_markdown(index=False))
else:
    md.append("_robustness_summary.csv не найден_\n")

if abl is not None:
    md.append("\n**Ablation — падение PR-AUC при удалении признака (топ-20):**\n")
    abl2 = abl.sort_values("delta_pr", ascending=False).head(20)
    md.append(abl2.to_markdown(index=False))
else:
    md.append("\n_ablation_rank.csv не найден_\n")

# Когорты
md.append("\n## 4. Когорты\n")
if age is not None:
    md.append("**Возрастные группы:**\n")
    md.append(age.to_markdown(index=False))
else:
    md.append("_age_summary.csv не найден_\n")

if course is not None:
    md.append("\n**Курсы (module × presentation):**\n")
    md.append(course.head(30).to_markdown(index=False))
    md.append("\nГрафики: `cohorts/heatmap_completion.png`, `cohorts/heatmap_imi.png`.\n")
else:
    md.append("\n_course_summary.csv не найден_\n")

# Корреляция IMI vs Completion
md.append("\n## 5. Корреляция IMI_mean и completion_rate по курсам\n")
if corr_txt is not None:
    md.append("```\n" + corr_txt + "```\n")
    md.append("График: `cohorts/imi_vs_completion_scatter.png`.\n")
else:
    md.append("_imi_completion_corr.txt не найден_\n")

# Рекомендации (шаблон, коротко)
md.append("\n## 6. Короткие рекомендации\n")
md.append("- Использовать порог из политики recall≥0.80 для раннего выявления рисковых студентов.\n"
          "- Сфокусироваться на курсах из квадранта Low IMI / Low completion — запланировать мягкие nudges.\n"
          "- По результатам ablation — работать с критичными признаками (увеличивать самоинициированную активность, снижать last-minute поведение и т. п.).\n")

(REPO / "summary.md").write_text("\n".join(md), encoding="utf-8")
print("Saved ->", REPO / "summary.md")