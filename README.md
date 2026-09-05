# Система оценки суммаризаций ЭМК (LLM-as-Judge)

Автоматическая оценка качества кратких суммаризаций электронных медицинских карт
(ЭМК) ансамблем LLM-судей поверх детерминированного слоя проверки фактов.
Итоговое решение (категория пригодности + предохранитель безопасности **E1**)
выносит код, а не «мнение» одной модели.

Ветка `npkc-vllm` подготовлена для запуска на мощностях **НПКЦ** через **vLLM**
(модели `qwen3.5-122b-fp8`, `qwen3.6-27b-fp8`). Модели крутятся на сервере
заказчика; этот код только отправляет запросы и собирает результаты.

## Быстрый старт (НПКЦ, vLLM)

```bash
git checkout npkc-vllm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # впишите адреса и имена моделей vLLM
python check_environment.py  # проверить связь с vLLM и модели
python run_pipeline.py       # запустить оценку (возобновляемо)
```

**Полная пошаговая инструкция для оператора — [docs/ЗАПУСК_НПКЦ.md](docs/ЗАПУСК_НПКЦ.md).**

## Документация
- [docs/ЗАПУСК_НПКЦ.md](docs/ЗАПУСК_НПКЦ.md) — как поднять vLLM, настроить и запустить (для оператора).
- [docs/АРХИТЕКТУРА.md](docs/АРХИТЕКТУРА.md) — устройство конвейера, таксономия A–E, роли и модели.
- [docs/REPORT.md](docs/REPORT.md) — подробный отчёт о пилоте: результаты и почему они такие.
- [docs/HANDOFF.md](docs/HANDOFF.md) — журнал инженерных правок.
- [.env.example](.env.example) — все параметры конфигурации с пояснениями.

## Структура репозитория

```
repo/
├── README.md                  ← этот файл
├── .env.example               ← шаблон конфигурации (скопировать в .env)
├── requirements.txt           ← зависимости
│
│  ── КОНФИГУРАЦИЯ И КЛИЕНТ LLM ──
├── config.py                  ← роли, профили моделей, параметры (через env)
├── llm_client.py              ← клиент LLM: бэкенды vLLM / Ollama, JudgePanel, JSON-восстановление
├── check_environment.py       ← проверка связи с бэкендом и моделями (первый шаг)
│
│  ── КОНВЕЙЕР ОЦЕНКИ ──
├── preprocessor.py            ← сегментация ЭМК/суммаризации на разделы
├── objective_layer.py         ← детерминированная сверка фактов (числа, полярность, сущности…)
├── gate.py                    ← pre-evaluation gate (отсев до судей)
├── judge.py                   ← LLM-as-Judge: промпты A–E, 3 раунда
├── aggregator.py              ← детерминированная итоговая категория (стоп-правило E1)
├── report.py                  ← формирование Excel-отчёта
├── run_pipeline.py            ← ГЛАВНАЯ ТОЧКА ВХОДА — оркестратор полного прогона
│
│  ── ВАЛИДАЦИЯ И АУДИТ ──
├── synthetic.py               ← регрессия на синтетическом наборе искажений
├── reference_metrics.py       ← ROUGE-L / BERTScore (только где есть эталон)
├── correlation.py             ← корреляция с экспертной разметкой (Spearman ρ, F1)
├── eval_patient.py            ← разбор/ранжирование суммаризаций одного пациента
├── audit.py                   ← аудит-лог (JSONL)
├── drift.py                   ← мониторинг дрейфа
├── prepare_data.py            ← сборка data/summaries.xlsx из raw_data/
│
├── docs/                      ← документация
├── tests/                     ← офлайн-тесты (без обращения к LLM)
├── data/summaries.xlsx        ← датасет: emr_id × model_id × тексты
├── checkpoints/  reports/  audit_log/   ← результаты прогонов (в git не попадают)
```

## Конфигурация и модели

Пайплайн обращается к моделям только через абстрактные роли (`judge_1/2/3`,
`aggregator`); соответствие «роль → модель» задаёт **профиль**. Всё настраивается
через переменные окружения (файл `.env`, см. `.env.example`) — Python править не нужно.

Профиль по умолчанию — **`npkc`**:

| Роль | Модель |
|---|---|
| judge_1, judge_2 | qwen3.5-122b-fp8 |
| judge_3 | qwen3.6-27b-fp8 |
| aggregator | qwen3.5-122b-fp8 |

Локальная разработка возможна на профилях `pilot*` через Ollama
(`EMR_LLM_BACKEND=ollama`).

## Тесты (офлайн, без LLM)

```bash
python tests/test_json_robustness.py       # устойчивость к «грязному» JSON
python tests/test_truncation.py            # обрыв ответа по лимиту токенов и телеметрия
python tests/test_aggregator_decisions.py  # решающий путь: сбой != клинический вердикт
python tests/test_gate_rules.py            # правила шлюза и вырожденный вход
python tests/test_e1_citation.py           # стоп-правило E1 требует проверяемую цитату
python tests/test_objective_grounding.py   # заземление судей и длинные ЭМК
python tests/test_vllm_client.py           # форма запросов vLLM/Ollama (с моком сети)
```

Плюс самопроверки без сети:

```bash
python run_pipeline.py --dry-run   # проводка judge -> aggregator -> audit -> report -> drift
python synthetic.py --self-check   # легенда, загрузка набора, rule-based детекция
python gate.py --self-check        # правила шлюза на реальных парах
```

## Точки входа

| Команда | Назначение |
|---|---|
| `python check_environment.py` | проверка связи с бэкендом и моделями |
| `python run_pipeline.py [--limit N] [--profile NAME] [--dry-run]` | полный прогон корпуса `data/summaries.xlsx` |
| `python synthetic.py --mode a --xlsx <файл>` | регрессия на синтетическом наборе искажений |
| `python eval_patient.py --patient Ж1 --xlsx <файл>` | ранжирование суммаризаций одного пациента |

Прогоны возобновляемы: промежуточные результаты сохраняются в `checkpoints/`,
повторный запуск продолжает с места остановки.
