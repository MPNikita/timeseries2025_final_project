# Favorita Forecasting (Course Final Project)

Финальный проект по курсу **ФКН ИИ «Анализ временных рядов», 2025** на задаче Kaggle  
**Corporacion Favorita Grocery Sales Forecasting**.

Сразу смотреть сюда:
- итоговый PDF-отчет: [results/report.pdf](results/report.pdf)
- основной ноутбук с ходом работы: [favorita.ipynb](favorita.ipynb)

Полная сводка по проделанной работе (EDA, метрики, сравнение моделей, выводы) лежит в `results/report.pdf`.

Ссылка на артефакты (чекпоинты и сабмиты):  
https://drive.google.com/drive/folders/1sFmh4nrHkmsxtDWET7j-DKcNblRWzEQz?usp=sharing

## Коротко про подход

- Baselines: иерархические fallback/lookup.
- Градиентный бустинг: LightGBM и CatBoost.
- Deep Learning: TFT (Temporal Fusion Transformer) с fallback для невалидных серий.
- Валидация: rolling-origin и pseudo-public full-panel horizon (16 дней).

## Как запустить

1. Установить зависимости:

```bash
python -m venv .venv
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

2. Положить данные соревнования в рабочую директорию/`data`:

- `train.csv`
- `test.csv`
- `items.csv`
- `stores.csv`
- `transactions.csv`
- `oil.csv`
- `holidays_events.csv`

3. Открыть [favorita.ipynb](favorita.ipynb) и выполнить ячейки.

## Где что лежит (кратко)

- `results/report.pdf`: финальный отчет по HW3.
- `favorita.ipynb`: основной демонстрационный ноутбук и отчетные графики/таблицы.
- `src/favorita/*`: основная кодовая база (features, models, metrics, validation, io).
- `src/configs.py`: дефолтные параметры моделей.
- `submissions/`: готовые сабмиты и inference-результаты.
- `.cache/favorita/`: кеши подготовки данных и экспериментов.
- `artifacts_tft/`: TFT-чекпоинты и TensorBoard-логи.
- `trash/`: архив/устаревшие служебные файлы.
