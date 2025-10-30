# --- Dockerfile ---
# Используем официальный легковесный Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt /app/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Запускаем генерацию отчёта при старте контейнера
CMD ["python", "src/make_report.py"]
