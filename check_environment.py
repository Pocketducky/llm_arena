"""
check_environment.py — самопроверка окружения перед запуском пайплайна.

ПЕРВЫЙ ШАГ оператора на сервере НПКЦ: подтверждает, что бэкенд LLM доступен,
модели активного профиля обслуживаются, и модель реально отвечает валидным JSON.
Роли и активный профиль читаются из config.py, поэтому смена профиля/бэкенда не
требует правок этого скрипта.

Запуск:
    python check_environment.py             # активный профиль (config/EMR_PROFILE)
    python check_environment.py npkc        # явно проверить конкретный профиль
"""

from __future__ import annotations

import sys
import time

import requests

import config
from llm_client import LLMError, JudgePanel, ask_json, list_available_models

TEST_PROMPT = (
    'Ответь строго в формате JSON: {"capital_of_russia": "<город одним словом>"}'
)


def check_server() -> list[str] | None:
    """[1] Доступность бэкенда и список обслуживаемых моделей."""
    if config.LLM_BACKEND == "ollama":
        print("\n[1] Сервер Ollama")
        try:
            models = list_available_models()
            print(f"    OK Ollama на {config.OLLAMA_HOST}")
            print(f"    Установленные модели: {models}")
            return models
        except LLMError as e:
            print(f"    ОШИБКА: {e}")
            return None

    print("\n[1] Сервер(ы) vLLM")
    all_models: list[str] = []
    ok_any = False
    for endpoint in config.vllm_endpoints():
        try:
            models = list_available_models(endpoint=endpoint)
            ok_any = True
            print(f"    OK vLLM на {endpoint} — модели: {models}")
            all_models.extend(models)
        except LLMError as e:
            print(f"    ОШИБКА на {endpoint}: {e}")
    return list(dict.fromkeys(all_models)) if ok_any else None


def check_profile(profile_name: str | None, available: list[str]) -> JudgePanel:
    """[2] Сверка «роль → модель» с реально обслуживаемыми моделями."""
    panel = JudgePanel(profile_name)
    print(f"\n[2] Активный профиль: '{panel.profile.name}'  (бэкенд: {config.LLM_BACKEND})")
    print("    Роли -> модели:")
    for role, model_name in panel.profile.roles.items():
        present = any(model_name == m or model_name in m for m in available)
        print(f"      [{'OK ' if present else 'НЕТ'}] {role:<11} -> {model_name}")
    return panel


def check_roundtrip(panel: JudgePanel, available: list[str]) -> dict[str, bool]:
    """[3] Реальный тестовый JSON-запрос по каждой уникальной модели профиля."""
    print("\n[3] Тестовый JSON-запрос по каждой уникальной модели профиля")
    results: dict[str, bool] = {}
    for model_name in panel.profile.unique_models():
        if not any(model_name == m or model_name in m for m in available):
            print(f"    [{model_name}] пропуск — модель не обслуживается")
            results[model_name] = False
            continue
        print(f"    [{model_name}] запрос…", end=" ", flush=True)
        start = time.time()
        try:
            parsed = ask_json(model_name, TEST_PROMPT, desc="self-check",
                              max_attempts=2, num_predict=200)
            print(f"OK ({time.time() - start:.1f}с) -> {parsed}")
            results[model_name] = True
        except Exception as e:  # noqa: BLE001 — любая ошибка = модель не готова
            print(f"ОШИБКА ({time.time() - start:.1f}с): {e}")
            results[model_name] = False
    return results


def _fix_hint(model_name: str) -> str:
    if config.LLM_BACKEND == "ollama":
        return f"   ollama pull {model_name}"
    return (f"   модель '{model_name}' не обслуживается: проверьте, что vLLM запущен "
            f"с --served-model-name '{model_name}' и что имя совпадает с EMR_MODEL_* "
            f"(список — GET /v1/models)")


# ══════════════════════════════════════════════════════════════════
# PREFLIGHT: влезут ли промпты корпуса в контекст сервера
# ══════════════════════════════════════════════════════════════════
# Раньше этой проверки не было вовсе: num_ctx для vLLM не передаётся (окно
# задаёт сервер через --max-model-len), а единственной пробой был 200-токенный
# roundtrip. Переполнение вылезало как HTTP 400 посреди многочасового прогона —
# и, из-за дефекта агрегации, записывалось в отчёт как клиническое
# «Неприемлемо». Теперь несоответствие видно ДО старта.

# Постоянная часть промпта каждого раунда в символах — измерена на реальных
# прогонах (см. reports/eval_patient_zh1.log): размер промпта хорошо описывается
# как «константа раунда + длина исходника + длина суммаризации».
_PROMPT_OVERHEAD_CHARS = {"R1 (блок A)": 2480, "R1 (блок E)": 3589,
                          "R2 (cross-review)": 14531, "R3 (агрегация)": 28062}

# Символов на токен для русского медицинского текста (токенизатор Qwen).
# Берём консервативную оценку: недооценить число токенов опаснее, чем переоценить.
_CHARS_PER_TOKEN = 2.0


def vllm_context_limits() -> dict[str, int]:
    """max_model_len по каждой модели — из GET /v1/models самого сервера."""
    if config.LLM_BACKEND != "vllm":
        return {}
    out: dict[str, int] = {}
    for endpoint in config.vllm_endpoints():
        try:
            r = requests.get(f"{endpoint.rstrip('/')}/models",
                             headers={"Authorization": f"Bearer {config.VLLM_API_KEY}"},
                             timeout=5)
            r.raise_for_status()
            for m in r.json().get("data", []):
                if m.get("max_model_len"):
                    out[m["id"]] = int(m["max_model_len"])
        except Exception:   # noqa: BLE001 — preflight не должен падать сам
            continue
    return out


def largest_pair_chars(dataset: str = "data/summaries.xlsx") -> tuple[int, str]:
    """Самая «тяжёлая» пара корпуса: (символов исходник+суммаризация, её id)."""
    try:
        import pandas as pd
        df = pd.read_excel(dataset)
        sizes = df["source_text"].str.len() + df["summary_text"].str.len()
        i = int(sizes.idxmax())
        return int(sizes.max()), f"{df.at[i, 'emr_id']} / модель {df.at[i, 'model_id']}"
    except Exception as exc:   # noqa: BLE001
        print(f"    (датасет {dataset} не прочитан: {exc})")
        return 0, "—"


def check_context_budget(dataset: str = "data/summaries.xlsx") -> bool:
    """Проверяет, что самый большой промпт корпуса + бюджет ответа влезают в окно."""
    print("\n[4] Влезают ли промпты корпуса в контекст модели")
    pair_chars, pair_id = largest_pair_chars(dataset)
    if not pair_chars:
        print("    пропуск: датасет недоступен")
        return True

    limits = vllm_context_limits()
    budgets = {"R1 (блок A)": config.TOKENS_R1, "R1 (блок E)": config.TOKENS_R1_LARGE,
               "R2 (cross-review)": config.TOKENS_R2, "R3 (агрегация)": config.TOKENS_R3}
    print(f"    самая тяжёлая пара: {pair_id} — {pair_chars} симв. (исходник + суммаризация)")

    worst = 0
    for label, overhead in _PROMPT_OVERHEAD_CHARS.items():
        chars = overhead + pair_chars
        tokens = int(chars / _CHARS_PER_TOKEN)
        need = tokens + budgets[label]
        worst = max(worst, need)
        print(f"      {label:20} ~{chars:>7} симв. ≈ {tokens:>6} ток. "
              f"+ ответ {budgets[label]:>5} = {need:>6} ток.")

    if config.LLM_BACKEND == "ollama":
        limit = config.NUM_CTX
        ok = worst <= limit
        print(f"    окно Ollama (EMR_OLLAMA_NUM_CTX): {limit} ток. — "
              f"{'ОК' if ok else 'НЕ ХВАТАЕТ'}")
        if not ok:
            print(f"    ⚠ Ollama МОЛЧА срежет промпт слева — потеряется начало, где стоят "
                  f"инструкции и стоп-правило E1. Поднимите EMR_OLLAMA_NUM_CTX "
                  f"минимум до {worst}.")
        return ok

    if not limits:
        print("    max_model_len не сообщается сервером — проверить автоматически нечем.")
        print(f"    Убедитесь вручную, что --max-model-len >= {worst} токенов.")
        return True

    ok = True
    for model_name, limit in sorted(limits.items()):
        fits = worst <= limit
        ok = ok and fits
        print(f"    {model_name}: max_model_len={limit} — "
              f"{'ОК' if fits else 'НЕ ХВАТАЕТ'} (нужно ~{worst})")
        if not fits:
            print(f"      ⚠ vLLM вернёт HTTP 400 на самых длинных парах. Поднимите "
                  f"--max-model-len минимум до {worst} либо уменьшите "
                  f"EMR_MAX_TOKENS_R3/R2.")
    return ok


def main() -> int:
    profile_arg = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 62)
    print(f"  Проверка окружения — бэкенд '{config.LLM_BACKEND}' (роли — из config.py)")
    print("=" * 62)

    available = check_server()
    if available is None:
        backend = config.LLM_BACKEND
        hint = ("Запустите `ollama serve`" if backend == "ollama"
                else f"Проверьте, что vLLM запущен и адрес верный "
                     f"(EMR_VLLM_BASE_URL={config.VLLM_BASE_URL})")
        print(f"\nИтог: бэкенд '{backend}' недоступен. {hint} и повторите.")
        return 1

    try:
        panel = check_profile(profile_arg, available)
    except KeyError as e:
        print(f"\nОШИБКА выбора профиля: {e}")
        return 1

    results = check_roundtrip(panel, available)
    context_ok = check_context_budget()

    print("\n" + "=" * 62)
    print("ИТОГ:")
    for model_name, ok in results.items():
        print(f"  [{'OK ' if ok else 'НЕТ'}] {model_name}")
    print(f"  [{'OK ' if context_ok else 'НЕТ'}] контекст вмещает самые длинные промпты корпуса")

    all_ok = bool(results) and all(results.values()) and context_ok
    if all_ok:
        print(f"\nПрофиль '{panel.profile.name}' полностью готов к работе.")
    else:
        print("\nНе готовы модели:")
        for m, ok in results.items():
            if not ok:
                print(_fix_hint(m))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
