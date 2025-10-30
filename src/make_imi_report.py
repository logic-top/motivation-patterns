# -*- coding: utf-8 -*-
"""
Сборка отчёта в Markdown из артефактов пайплайна с русскими подписями/нумерацией
и расширенными интерпретациями.

Версия: 2025-10-09 (fixed: no babel, no manual figure numbering, FloatBarrier)
"""

from pathlib import Path
import pandas as pd

# ---------- 0) Флаги ----------
FORCE_MINIMAL_META = True   # перезаписать report.yaml на минимальный
USE_HEADER_TEX     = True   # подхватить header.tex из reports/

# ---------- 1) Пути ----------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = PROC / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------- 2) Кандидаты входных артефактов ----------
CANDIDATES = {
    "imi_by_week":      [PROC/"imi_dyn"/"imi_by_week.csv", PROC/"imi_by_week.csv"],
    "converted_share":  [PROC/"imi_dyn"/"converted_share.csv", PROC/"converted_share.csv"],
    "age_summary":      [PROC/"cohorts"/"age_summary.csv"],
    "course_summary":   [PROC/"cohorts"/"course_summary.csv"],
    "corr_txt":         [PROC/"cohorts"/"imi_completion_corr.txt", PROC/"imi_completion_corr.txt"],
    "fig_trends":       [PROC/"imi_dyn"/"imi_week_trends.png"],
    "fig_conv":         [PROC/"imi_dyn"/"converted_bar.png"],
    "fig_heat_imi":     [PROC/"cohorts"/"heatmap_imi.png"],
    "fig_heat_comp":    [PROC/"cohorts"/"heatmap_completion.png"],
    "fig_scatter":      [PROC/"cohorts"/"imi_vs_completion_scatter.png"],
}

def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

paths = {k: first_existing(v) for k, v in CANDIDATES.items()}

# Критичные файлы (без них отчёт не имеет смысла)
must_have = ("imi_by_week", "converted_share")
missing = [k for k in must_have if paths[k] is None]
if missing:
    raise SystemExit(
        "[ОШИБКА] Нет ключевых файлов: " + ", ".join(missing) +
        "\nЗапустите перед этим: src/imi_dynamics.py"
    )

print("[make_imi_report] ROOT =", ROOT)
for k, p in paths.items():
    print(f"[make_imi_report] {k:18s} -> {p if p else 'нет'}")

# ---------- 3) Чтение данных ----------
imi = pd.read_csv(paths["imi_by_week"])
conv = pd.read_csv(paths["converted_share"])
age = pd.read_csv(paths["age_summary"]) if paths["age_summary"] else None
course = pd.read_csv(paths["course_summary"]) if paths["course_summary"] else None
corr_text = paths["corr_txt"].read_text(encoding="utf-8") if paths["corr_txt"] else "нет данных"

# Базовые метрики для интерпретации
imi_start = float(imi["IMI_mean"].iloc[0])
imi_end   = float(imi["IMI_mean"].iloc[-1])
imi_rel   = (imi_end - imi_start) / max(1e-9, abs(imi_start))
conv_mean = float(conv["converted_share"].mean())

top_idx = conv["converted_share"].idxmax() if not conv.empty else None
top_row = conv.loc[top_idx] if top_idx is not None else None
top_course_str = None
if top_row is not None and {"code_module","code_presentation","converted_share"} <= set(top_row.index):
    top_course_str = f"{top_row['code_module']}-{top_row['code_presentation']} ({float(top_row['converted_share']):.1%})"

# ---------- 4) Относительные пути к рисункам (от корня проекта) ----------
def rel_from_root(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT).as_posix())

fig_trends_rel    = rel_from_root(paths["fig_trends"])    if paths["fig_trends"] else None
fig_conv_rel      = rel_from_root(paths["fig_conv"])      if paths["fig_conv"] else None
fig_heat_imi_rel  = rel_from_root(paths["fig_heat_imi"])  if paths["fig_heat_imi"] else None
fig_heat_comp_rel = rel_from_root(paths["fig_heat_comp"]) if paths["fig_heat_comp"] else None
fig_scatter_rel   = rel_from_root(paths["fig_scatter"])   if paths["fig_scatter"] else None

# ---------- 5) Вспомогательные вставки ----------
def fig(path: str, caption: str):
    """Вставка рисунка без ручной нумерации.
    Pandoc сам проставит «Рис. <номер>: <подпись>» (см. header.tex)."""
    return [f"![{caption}]({path}){{ width=95% }}", ""]

def table(title: str, df: pd.DataFrame, n=10):
    """Возвращает заголовок и Markdown-таблицу (первые n строк)."""
    lines = [f"**{title}**", ""]
    lines += [df.head(n).to_markdown(index=False), ""]
    return lines

def barrier():
    """Заставить LaTeX вывести все предыдущие float перед следующим заголовком."""
    return ["\\FloatBarrier", ""]

# ---------- 6) Сборка Markdown ----------
md_path = REPORTS / "imi_dynamics_report.md"
lines = []

# Титул
lines += [
    "# Отчёт: Динамика внутренней мотивации (IMI)",
    "",
    "*(Проект «От мотивации к паттернам цифрового мышления», ИТМО)*",
    ""
]

# Краткое резюме
lines += [
    "## 1. Краткое резюме",
    f"- Средний IMI вырос с **{imi_start:.3f}** до **{imi_end:.3f}** "
    f"(изменение {imi_rel:+.1%}).",
    f"- Средняя доля студентов, **перешедших во внутреннюю мотивацию**, составила **{conv_mean:.1%}**.",
]
if top_course_str:
    lines += [f"- Курс-лидер по доле «перешедших»: **{top_course_str}**."]
lines += [
    "- Карты когорт показывают межкурсовые различия по IMI и завершённости.",
    "- Корреляция IMI с завершением курса — положительная (см. график и текстовую сводку).",
    ""
]

# Методика
lines += [
    "## 2. Методика",
    "- **Данные**: OULAD (`studentVLE`, `vle`, `assessments`, `studentAssessment`, `studentInfo`).",
    "- **Признаки** (нормированы внутри курса): регулярность активности, «дедлайновость», самоинициатива, глубина сессий, социальная ровность, энтропия маршрутов, прирост результата на попытку.",
    "- **Индекс IMI_v1**: взвешенная комбинация нормированных индикаторов (веса валидируются).",
    "- **Порог «перешёл во внутреннюю мотивацию»**: верхний квантиль IMI (в текущем эксперименте ~0.636).",
    "- **Когортные срезы**: по возрасту и курсу.",
    ""
]

# Результаты
lines += [
    "## 3. Результаты",
]

lines += ["### 3.1 Динамика IMI по неделям"]
if fig_trends_rel:
    lines += fig(fig_trends_rel, "Средний IMI по неделям курса.")
else:
    lines += ["(Рисунок недоступен)", ""]
lines += barrier()

lines += ["### 3.2 Доля «перешедших» по курсам"]
if fig_conv_rel:
    lines += fig(fig_conv_rel, "Доля студентов, перешедших во внутреннюю мотивацию (по курсам).")
else:
    lines += ["(Рисунок недоступен)", ""]
lines += barrier()

lines += ["### 3.3 Когортные карты"]
if fig_heat_imi_rel:
    lines += fig(fig_heat_imi_rel, "Средний IMI: межкурсовые различия.")
if fig_heat_comp_rel:
    lines += fig(fig_heat_comp_rel, "Завершённость курса: межкурсовые различия.")
lines += barrier()

lines += ["### 3.4 Связь IMI и завершения"]
if fig_scatter_rel:
    lines += fig(fig_scatter_rel, "IMI vs. доля завершения (по курсам).")
lines += [f"*Текстовая сводка:* `{corr_text}`", ""]
lines += barrier()

# Таблицы
if age is not None and not age.empty:
    lines += table("Возрастные когорты: сводные метрики", age, n=10)
if course is not None and not course.empty:
    lines += table("Курсы: сводные метрики", course, n=10)

# Интерпретация
lines += [
    "## 4. Интерпретация",
    "- **Рост IMI** по неделям согласуется с гипотезой интериоризации мотивации: часть студентов «уходит» от дедлайнового поведения к инициативному.",
    "- **Доля «перешедших»** указывает на потенциал педагогических вмешательств (тренажёры, микро-обратная связь, вариативные траектории).",
    "- **Межкурсовые различия** по IMI и completion сигнализируют о роли дизайна курса: структура заданий, темп, качество материалов и форматы поддержки.",
    "- **Положительная связь IMI↔completion** — ожидаема: устойчивое внутреннее вовлечение повышает вероятность дойти до конца.",
    ""
]

# Ограничения
lines += [
    "## 5. Ограничения",
    "- OULAD — исторические данные; реальная аудитория может отличаться.",
    "- IMI_v1 — агрегатный индикатор: полезно валидировать веса и пороги на новых курсах.",
    "- Логи фиксируют поведение, а не мотив напрямую; интерпретация требует осторожности и поддержки опросами (SDT/mini-Why42).",
    ""
]

# Следующие шаги
lines += [
    "## 6. Следующие шаги",
    "1) Перенести расчёты в Moodle/ИТМО: собрать локальные логи и микро-анкеты.",
    "2) Уточнить веса IMI и пороги по курсам (калибровка, ECE/MCE).",
    "3) Проверить устойчивость эффектов на продвинутых выборках (магистратура, разные дисциплины).",
    "4) Подготовить статью: дизайн вмешательства → эффект на IMI и completion → XAI-интерпретации.",
    ""
]

# Воспроизводимость
lines += [
    "## 7. Воспроизводимость",
    "- Скрипты: `src/imi_dynamics.py`, `src/cohort_analysis.py`, `src/correlate_course_imi_completion.py`, `src/make_imi_report.py`.",
    "- Данные: `data/processed/*`.",
    ""
]

md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"[OK] Markdown-отчёт сохранён -> {md_path}")

# ---------- 7) report.yaml (минимальный, без babel/polyglossia) ----------
yaml = REPORTS / "report.yaml"
if FORCE_MINIMAL_META or (not yaml.exists()):
    yaml.write_text(
        'title: "Отчёт: Динамика внутренней мотивации (IMI)"\n'
        'author: "Проект «От мотивации к паттернам цифрового мышления», ИТМО"\n'
        'lang: ru-RU\n'
        'date: today\n'
        'geometry:\n'
        '  - margin=2.5cm\n'
        'fontsize: 11pt\n'
        'papersize: a4\n'
        'mainfont: "Times New Roman"\n'
        'monofont: "Consolas"\n',
        encoding="utf-8"
    )
    print(f"[INFO] report.yaml обновлён -> {yaml}")

# ---------- 8) header.tex (фикс плавающих фигур и подписей) ----------
if USE_HEADER_TEX:
    header = REPORTS / "header.tex"
    if not header.exists():
        header.write_text(r"""
% Фиксация положения рисунков и барьеры
\usepackage{float}
\usepackage[section]{placeins}
\floatplacement{figure}{H}

% Русские подписи к рисункам
\renewcommand{\figurename}{Рис.}

% Гиперссылки
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}
""".lstrip(), encoding="utf-8")
        print(f"[INFO] header.tex создан -> {header}")

# ---------- 9) Подсказка по сборке PDF ----------
print("\nКак собрать PDF (из КОРНЯ проекта):")
if USE_HEADER_TEX:
    print(r"""
$md = "data/processed/reports/imi_dynamics_report.md"
$out = "data/processed/reports/imi_dynamics_report.pdf"
$meta = "data/processed/reports/report.yaml"
$hdr = "data/processed/reports/header.tex"
pandoc $md `
  --from markdown `
  --pdf-engine=xelatex `
  --metadata-file=$meta `
  --include-in-header=$hdr `
  -V toc=true `
  -o $out
""")
else:
    print(r"""
$md = "data/processed/reports/imi_dynamics_report.md"
$out = "data/processed/reports/imi_dynamics_report.pdf"
$meta = "data/processed/reports/report.yaml"
pandoc $md `
  --from markdown `
  --pdf-engine=xelatex `
  --metadata-file=$meta `
  -V toc=true `
  -o $out
""")