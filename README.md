# Corporacion Favorita: Project Handoff

## 1. Что это за проект

Это рабочая папка по Kaggle-задаче **Corporacion Favorita Grocery Sales Forecasting**.

Ключевая специфика задачи:

- это большая panel time series, а не один временной ряд;
- `train.csv` не содержит строк с нулевыми продажами;
- `unit_sales < 0` означает возвраты;
- в `test.csv` есть новые `item_nbr`, которых нет в `train`;
- `transactions.csv` есть только на train-period, поэтому его нельзя использовать в финальной test-time модели без отдельного прогноза.


## 2. Что уже сделано

В проекте уже есть:

- полноценный EDA и выполненный ноутбук [`favorita.ipynb`](./favorita.ipynb);
- baseline-модели и старый `LightGBM` pipeline;
- новый основной `CatBoost` pipeline;
- recent-train cache для ускорения повторных экспериментов;
- pseudo-public full-panel validation;
- финальный `CatBoost` submission:
  [`submission_catboost_optuna_lb224.csv.gz`](./submission_catboost_optuna_lb224.csv.gz)

Исторический контекст:

- старый `LightGBM` offline holdout выглядел хорошо, но на Kaggle у пользователя дал около `0.74`;
- это показало, что observed-row validation была слишком оптимистичной;
- после этого основной акцент был перенесён на более честную validation-схему и `CatBoost`.


## 3. Основные файлы

### Данные

- [`train.csv`](./train.csv)
- [`test.csv`](./test.csv)
- [`items.csv`](./items.csv)
- [`stores.csv`](./stores.csv)
- [`transactions.csv`](./transactions.csv)
- [`oil.csv`](./oil.csv)
- [`holidays_events.csv`](./holidays_events.csv)
- [`sample_submission.csv`](./sample_submission.csv)

### Код

- [`favorita_eda_utils.py`](./favorita_eda_utils.py)
  - chunk-based обход `train.csv`;
  - кеширование EDA-агрегатов;
  - базовые таблицы и summaries.

- [`favorita_baselines.py`](./favorita_baselines.py)
  - `weighted_rmsle`;
  - `recent_mean_28d`;
  - `hierarchical_weekday_promo`.

- [`favorita_models.py`](./favorita_models.py)
  - старый `LightGBM` pipeline;
  - rolling-origin CV;
  - исторические aggregate priors;
  - сохранён для сравнения и как reference implementation.

- [`favorita_catboost.py`](./favorita_catboost.py)
  - основной текущий pipeline;
  - recent-train cache;
  - pseudo-public validation;
  - sampled `CatBoost` training;
  - `Optuna`;
  - финальный submission.

- [`build_favorita_notebook.py`](./build_favorita_notebook.py)
  - генератор ноутбука [`favorita.ipynb`](./favorita.ipynb).


## 4. Ключевые инженерные выводы

### 4.1 Почему observed-row holdout был плохим ориентиром

Старый holdout оценивал модель только на строках, которые уже присутствуют в `train.csv`.

Проблема:

- в `train.csv` отсутствуют нулевые продажи;
- Kaggle leaderboard оценивает полный `test.csv`, где много нулей;
- поэтому observed-row WRMSLE систематически завышал качество.

Практический эффект:

- `LightGBM` offline выглядел сильно лучше, чем потом показал leaderboard.


### 4.2 Что было сделано вместо этого

Добавлена pseudo-public validation:

- берутся реальные `store_nbr-item_nbr` пары из `test.csv`;
- строится полный panel horizon на `16` дней;
- actual внутри validation horizon:
  - наблюдавшиеся строки берутся из `train`;
  - отсутствующие строки считаются `0`;
- missing `onpromotion` для отсутствующих строк impute как `False`.

Это всё ещё приближение, но оно заметно честнее, чем observed-row holdout.


### 4.3 Почему в текущей боевой модели используется sampled fit

Даже на окне `224` дня observed fit содержит больше `23M` строк.

Для `CatBoost` сделана более практичная схема:

- статистические признаки считаются по полному observed fit window;
- сам raw training frame для `CatBoost` берётся как recent-biased sample;
- при необходимости подмешиваются sampled implicit-zero rows.

Так модель получает:

- полную историю для aggregate statistics;
- управляемый объём данных для обучения.


## 5. EDA и исторические результаты

### 5.1 EDA

В ноутбуке уже есть:

- обзор файлов и размеров;
- метаданные `stores` и `items`;
- daily sales;
- weekday/payday эффекты;
- promotions;
- связь с `transactions` и `oil`;
- holidays;
- earthquake window;
- срезы по `family` и `store`.

### 5.2 Старые baseline результаты

- `recent_mean_28d`: `0.568332`
- `hierarchical_weekday_promo`: `0.553391`

Это observed-row результаты, их нельзя сравнивать напрямую с текущей pseudo-public validation.

### 5.3 Старый LightGBM

Diagnostic observed-row holdout:

- `lightgbm_feature_model`: `0.514604`

Но этот score оказался неинформативным относительно leaderboard, поэтому `LightGBM` не считается текущей основной боевой стратегией.


## 6. Текущая основная стратегия: CatBoost

### 6.1 Feature set

Используются:

- календарные признаки:
  - `weekday`, `day`, `month`, `weekofyear`, `is_month_end`, `is_payday`;
- metadata:
  - `store_code`, `city_code`, `state_code`, `type_code`, `cluster`,
  - `item_code`, `family_code`, `class`, `perishable`;
- business features:
  - `onpromotion`, `oil`, holiday flags;
- recent statistics:
  - `si_recent28`, `si_recent56`, `siw_mean`, `siw_count_log`,
  - `fsw_mean`, `fw_mean`,
  - `item_recent_mean`, `store_recent_mean`;
- prior statistics:
  - `si_all_mean`, `si_all_count_log`,
  - `item_all_mean`, `item_all_count_log`,
  - `store_all_mean`,
  - `family_all_mean`,
  - `family_store_all_mean`, `family_store_all_count_log`;
- derived features:
  - `recent_ratio_28_56`,
  - `recent_trend_28_56`,
  - `item_vs_family_ratio`,
  - `family_store_vs_store_ratio`,
  - `history_strength`,
  - и другие из [`favorita_catboost.py`](./favorita_catboost.py).

### 6.2 Recent cache

Критичный артефакт:

- [`.cache/favorita/train_recent_from_20160801.pkl`](./.cache/favorita/train_recent_from_20160801.pkl)

Что это даёт:

- `train.csv` не перечитывается целиком для каждого fold;
- CatBoost experiments стали практически воспроизводимыми.

Размер кеша:

- около `703 MB`

### 6.3 Верифицированный контрольный результат

На последнем pseudo-public fold внутри train:

- `valid_start = 2017-07-31`
- `lookback = 224`
- `fit_observed_rows = 23,426,907`
- actual training rows для модели после sampling: `300,000`
- sampled implicit zeros: `80,000`

Проверенная конфигурация:

- `iterations = 120`
- `depth = 8`
- `learning_rate = 0.05`
- `border_count = 128`

Результат:

- raw `CatBoost` pseudo-public WRMSLE: `0.582715`

Это на текущий момент лучший вручную проверенный score в новом pipeline.


## 7. Optuna

В [`favorita_catboost.py`](./favorita_catboost.py) есть `run_catboost_optuna_search`.

Что уже запускалось:

- mini-search на `2` trial’ах;
- он сохранился в кеше:
  - `catboost_optuna_lb224_hz16_trials2.pkl`

Результат этого короткого поиска:

- best score: `0.608394`

То есть в коротком режиме `Optuna` пока **не обогнал** ручную конфигурацию `120 / depth 8 / lr 0.05`.

Вывод:

- `Optuna` pipeline рабочий;
- но его нужно запускать шире и дольше, а не на `2` trial’ах.


## 8. Финальная модель и submission

Текущий основной submission:

- [`submission_catboost_optuna_lb224.csv.gz`](./submission_catboost_optuna_lb224.csv.gz)

Как он был обучен:

- `lookback_days = 224`
- `fit_observed_rows = 23,692,343`
- sampled training rows: `300,000`
- sampled implicit zeros: `80,000`
- без `transactions`
- без принудительного fallback-blend в финальной выдаче:
  - `min_model_weight = 1.0`
  - `unseen_model_weight = 1.0`

Почему fallback выключен по умолчанию:

- в коротких проверках общий blend с hierarchical fallback ухудшал fold score;
- fallback-код оставлен в проекте, но пока не считается verified improvement.

Дополнительный артефакт:

- старый submission для сравнения:
  [`submission_lgbm_tscv_lb224_tx0.csv.gz`](./submission_lgbm_tscv_lb224_tx0.csv.gz)


## 9. Кеши

Основные кеши лежат в:

- [`.cache/favorita`](./.cache/favorita)

Наиболее важные:

- `train_eda_bundle.pkl`
- `train_recent_from_20160801.pkl`
- `prior_bundle_until_*.pkl`
- `baseline_validation_lb112_hz16.pkl`
- `lgbm_validation_lb112_hz16.pkl`
- `catboost_optuna_lb224_hz16_trials2.pkl`
- `catboost_final_lb224_fr300000_zr80000.pkl`


## 10. Как воспроизвести

### 10.1 Пересобрать ноутбук

```bash
python3 build_favorita_notebook.py
```

### 10.2 Выполнить ноутбук

```bash
python3 - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path("favorita.ipynb")
nb = nbformat.read(path, as_version=4)
client = NotebookClient(
    nb,
    timeout=3600,
    kernel_name="python3",
    resources={"metadata": {"path": str(path.parent.resolve())}},
)
client.execute()
nbformat.write(nb, path)
PY
```

### 10.3 Запустить проверенный single-fold CatBoost

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd
from favorita_models import build_rolling_origin_folds
from favorita_catboost import run_single_fold_catboost_experiment

folds = build_rolling_origin_folds(data_dir=Path("."), horizon_days=16, step_days=28, n_folds=4)
valid_start = pd.Timestamp(folds.iloc[-1]["valid_start"])

result = run_single_fold_catboost_experiment(
    valid_start=valid_start,
    lookback_days=224,
    fit_max_rows=300_000,
    eval_max_rows=120_000,
    zero_sample_size=80_000,
    zero_sample_days=28,
    model_params={
        "iterations": 120,
        "depth": 8,
        "learning_rate": 0.05,
        "border_count": 128,
        "verbose": False,
    },
    use_cache=True,
)
print(result["raw_model_score"])
PY
```

### 10.4 Переобучить финальную CatBoost модель

```bash
python3 - <<'PY'
from favorita_catboost import train_final_catboost_model

result = train_final_catboost_model(
    lookback_days=224,
    fit_max_rows=300_000,
    zero_sample_size=80_000,
    zero_sample_days=28,
    model_params={
        "iterations": 120,
        "depth": 8,
        "learning_rate": 0.05,
        "border_count": 128,
        "verbose": False,
    },
    postprocess_params={
        "history_scale": 3.0,
        "min_model_weight": 1.0,
        "unseen_model_weight": 1.0,
    },
    use_cache=True,
)
print(result["metadata"])
PY
```


## 11. Что ещё нужно сделать

Самые логичные следующие шаги:

1. Добить полный `run_catboost_time_series_cv` хотя бы для `lookback_grid=(168, 224)`.
2. Запустить более длинный `Optuna`, а не `2` trial’а.
3. Сделать отдельную pseudo-cold-start validation и тюнить fallback только на ней.
4. Проверить, помогает ли возврат `transactions` через отдельную auxiliary model.
5. Попробовать blend нескольких `CatBoost` конфигураций или нескольких lookback windows.


## 12. Если коллеге нужно понять главное за 30 секунд

Главное сейчас такое:

- старый `LightGBM` больше не считается основным ориентиром;
- основной код теперь в [`favorita_catboost.py`](./favorita_catboost.py);
- честная validation для этой задачи должна быть full-panel, а не observed-row;
- лучший вручную проверенный CatBoost fold score сейчас `0.582715`;
- текущий production submission:
  [`submission_catboost_optuna_lb224.csv.gz`](./submission_catboost_optuna_lb224.csv.gz)
