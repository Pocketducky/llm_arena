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

    print("\n" + "=" * 62)
    print("ИТОГ:")
    for model_name, ok in results.items():
        print(f"  [{'OK ' if ok else 'НЕТ'}] {model_name}")

    all_ok = bool(results) and all(results.values())
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
