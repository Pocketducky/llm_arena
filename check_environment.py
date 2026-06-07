"""
check_environment.py — самопроверка окружения перед запуском пайплайна.

В отличие от прежнего test_setup.py (хардкод трёх конкретных моделей),
этот скрипт читает роли и активный профиль из config.py — поэтому при
переключении ACTIVE_PROFILE ("pilot" → "pilot_diverse" → "target") не
требует никаких правок: просто проверяет ту конфигурацию, которая
выбрана сейчас.

Запуск:
    python check_environment.py             # активный профиль из config.py
    python check_environment.py target      # явно проверить другой профиль
"""

from __future__ import annotations

import sys
import time

import config
from ollama_client import (
    OllamaError,
    JudgePanel,
    ask_json,
    list_available_models,
)

TEST_PROMPT = (
    'Ответь строго в формате JSON: {"capital_of_russia": "<город одним словом>"}'
)


def check_server() -> list[str] | None:
    print("\n[1] Сервер Ollama")
    try:
        models = list_available_models()
        print(f"    OK Ollama запущена на {config.OLLAMA_HOST}")
        print(f"    Установленные модели: {models}")
        return models
    except OllamaError as e:
        print(f"    ОШИБКА: {e}")
        return None


def check_profile(profile_name: str | None, available: list[str]) -> JudgePanel:
    panel = JudgePanel(profile_name)
    print(f"\n[2] Активный профиль: '{panel.profile.name}'")
    print(f"    Роли -> модели:")
    for role, model_name in panel.profile.roles.items():
        present = model_name in available or any(model_name in m for m in available)
        mark = "OK " if present else "НЕТ"
        print(f"      [{mark}] {role:<11} -> {model_name}")
    return panel


def check_roundtrip(panel: JudgePanel, available: list[str]) -> dict[str, bool]:
    print(f"\n[3] Тестовый JSON-запрос по каждой уникальной модели профиля")
    results: dict[str, bool] = {}
    for model_name in panel.profile.unique_models():
        if not any(model_name == m or model_name in m for m in available):
            print(f"    [{model_name}] пропуск — модель не установлена "
                  f"(ollama pull {model_name})")
            results[model_name] = False
            continue

        print(f"    [{model_name}] запрос…", end=" ", flush=True)
        start = time.time()
        try:
            parsed = ask_json(model_name, TEST_PROMPT, desc="self-check",
                              max_attempts=2, num_predict=200)
            elapsed = time.time() - start
            print(f"OK ({elapsed:.1f}с) -> {parsed}")
            results[model_name] = True
        except (OllamaError, Exception) as e:
            elapsed = time.time() - start
            print(f"ОШИБКА ({elapsed:.1f}с): {e}")
            results[model_name] = False
    return results


def check_num_ctx(model_name: str) -> None:
    print(f"\n[4] Проверка num_ctx={config.NUM_CTX} (важно для длинных ЭМК)")
    from ollama_client import generate
    try:
        raw = generate(model_name, "Привет", force_json=False,
                       num_predict=5, num_ctx=config.NUM_CTX)
        print(f"    OK num_ctx принят моделью '{model_name}', ответ: {raw[:60]!r}")
    except OllamaError as e:
        print(f"    ОШИБКА: {e}")


def main() -> int:
    profile_arg = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("  Проверка окружения — llm_arena (роли определяются config.py)")
    print("=" * 60)

    available = check_server()
    if available is None:
        print("\nИтог: Ollama недоступна. Запустите `ollama serve` и повторите.")
        return 1

    try:
        panel = check_profile(profile_arg, available)
    except KeyError as e:
        print(f"\nОШИБКА выбора профиля: {e}")
        return 1

    results = check_roundtrip(panel, available)
    if results:
        first_ok = next((m for m, ok in results.items() if ok), None)
        if first_ok:
            check_num_ctx(first_ok)

    print("\n" + "=" * 60)
    print("ИТОГ:")
    all_ok = bool(results) and all(results.values())
    for model_name, ok in results.items():
        print(f"  [{'OK ' if ok else 'НЕТ'}] {model_name}")

    if all_ok:
        print(f"\nПрофиль '{panel.profile.name}' полностью готов к работе.")
        print("Переключение на другой профиль — правка ACTIVE_PROFILE в config.py")
        print("или: python check_environment.py <имя_профиля>")
    else:
        missing = [m for m, ok in results.items() if not ok]
        print("\nНе готовы модели:")
        for m in missing:
            print(f"   ollama pull {m}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
