# src/plot_shap_summary.py
import os, joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Пути
base = os.path.join("..")
X_path = os.path.join(base, "data", "processed", "X_test.csv")     # замените на свой файл фич
model_path = os.path.join(base, "data", "processed", "models_final", "gbdt_model.joblib")

# Загрузка
X = pd.read_csv(X_path)
model = joblib.load(model_path)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

os.makedirs(os.path.join(base, "assets"), exist_ok=True)
plt.figure()
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(base, "assets", "shap_summary.png"), dpi=300)
plt.close()
print("✅ Сохранено: assets/shap_summary.png")
