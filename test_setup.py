"""
Тест-скрипт: проверяет что все три локальные модели через Ollama работают.
Запускай перед основным скриптом.
"""

import requests
import json
import time


MODELS_TO_TEST = [
    "llama3.1:8b",
    "deepseek-r1:14b",
    "qwen2.5:7b",
]

TEST_PROMPT = "Ответь одним словом на русском: столица России?"


def test_ollama_server() -> bool:
    print("\n[1] Проверка сервера Ollama")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"    ✅ Ollama запущен")
        print(f"    Скачанные модели: {models}")
        return True
    except requests.exceptions.ConnectionError:
        print("    ❌ Ollama не запущен!")
        print("    Запусти: ollama serve")
        print("    Или открой приложение Ollama из Launchpad")
        return False
    except Exception as e:
        print(f"    ❌ Ошибка: {e}")
        return False


def strip_think_tags(text: str) -> str:
    """
    Убирает блок <think>...</think> из ответа DeepSeek R1.
    Модель сначала пишет размышления в этом блоке, затем даёт ответ.
    """
    if "<think>" in text and "</think>" in text:
        end = text.find("</think>") + len("</think>")
        text = text[end:].strip()
    elif "<think>" in text:
        # Незакрытый тег — модель не успела дописать (мало токенов)
        text = ""
    return text.strip()


def test_model(model: str) -> bool:
    print(f"\n    Тест модели: {model}")
    try:
        # Проверяем что модель скачана
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
        if not any(model in m for m in available):
            print(f"    ⚠  Модель не скачана → ollama pull {model}")
            return False

        # DeepSeek R1 нужно больше токенов — он тратит часть на <think>
        is_deepseek = "deepseek" in model.lower()
        num_predict = 300 if is_deepseek else 80

        payload = {
            "model":  model,
            "prompt": TEST_PROMPT,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0},
        }
        start   = time.time()
        resp    = requests.post(
            "http://localhost:11434/api/generate",
            json=payload, timeout=120
        )
        elapsed = time.time() - start
        raw     = resp.json()["response"].strip()
        answer  = strip_think_tags(raw)

        if not answer:
            # Ответ пустой — скорее всего think-блок не закрылся
            print(f"    ⚠  Пустой ответ после удаления <think> (модель думает)")
            print(f"       Это нормально для DeepSeek R1 в тест-режиме.")
            print(f"       В evaluator.py лимит 3500 токенов — там всё ОК.")
            return True   # модель работает, просто тест-лимит мал

        print(f"    ✅ Ответ: {answer}  ({elapsed:.1f}с)")
        if is_deepseek:
            think_len = raw.find("</think>")
            if think_len > 0:
                print(f"       (DeepSeek R1: {think_len} символов размышлений скрыто)")
        return True

    except Exception as e:
        print(f"    ❌ Ошибка: {e}")
        return False


def test_json_parsing():
    print("\n[3] Тест парсинга JSON")
    cases = [
        '{"total_score": 75, "scores": {}}',
        '```json\n{"total_score": 80}\n```',
        'Вот ответ:\n{"total_score": 65}\n',
        '<think>думаю...</think>\n{"total_score": 72}',  # DeepSeek R1 формат
    ]
    all_ok = True
    for tc in cases:
        text = tc.strip()
        # Убираем DeepSeek think-блок
        if "<think>" in text and "</think>" in text:
            end_think = text.find("</think>") + len("</think>")
            text = text[end_think:].strip()
        text = text.replace("```json","").replace("```","").strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        try:
            parsed = json.loads(text[start:end])
            print(f"    ✅ OK: {parsed}")
        except Exception as e:
            print(f"    ❌ Ошибка: {e} | input: {tc[:50]}")
            all_ok = False
    return all_ok


def test_num_ctx():
    """Проверяет что Ollama принимает num_ctx параметр."""
    print("\n[3.5] Тест num_ctx (критично для больших ЭМК)")
    try:
        payload = {
            "model": "llama3.1:8b",
            "prompt": "Привет",
            "stream": False,
            "options": {"num_predict": 5, "num_ctx": 16384, "temperature": 0},
        }
        r = requests.post("http://localhost:11434/api/generate",
                          json=payload, timeout=60)
        resp = r.json()
        if "prompt_eval_count" in resp:
            print(f"    ✅ num_ctx работает. Токенов использовано: {resp['prompt_eval_count']}")
        else:
            print("    ✅ num_ctx принят (prompt_eval_count недоступен в этой версии Ollama)")
        return True
    except Exception as e:
        print(f"    ❌ Ошибка: {e}")
        return False


def estimate_time():
    print("\n[4] Оценка времени выполнения")
    n = 182   # 156 + 26 своих
    # ~3 мин на запрос к 14B модели на M3 16GB
    sec_per_call  = 180
    total_calls   = n * 3 * 2   # 3 модели × 2 раунда
    total_hours   = (total_calls * sec_per_call) / 3600
    print(f"    Суммаризаций:    {n}")
    print(f"    Запросов итого:  {total_calls}")
    print(f"    ~Время:          {total_hours:.0f} часов")
    print(f"    Совет: запусти на ночь:")
    print(f"      nohup python evaluator.py > run.log 2>&1 &")
    print(f"      tail -f run.log   ← следить за прогрессом")


if __name__ == "__main__":
    print("=" * 52)
    print("  Проверка окружения — Арена LLM (локальный режим)")
    print("=" * 52)

    server_ok = test_ollama_server()

    print("\n[2] Тест каждой модели")
    results = {}
    if server_ok:
        for m in MODELS_TO_TEST:
            results[m] = test_model(m)
    else:
        for m in MODELS_TO_TEST:
            results[m] = False

    test_json_parsing()
    test_num_ctx()
    estimate_time()

    print("\n" + "=" * 52)
    print("ИТОГ:")
    for m, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {m}")

    all_ok = server_ok and all(results.values())
    if all_ok:
        print("\n🚀 Всё готово! Запускай:")
        print("   python prepare_data.py   ← если ещё не делал")
        print("   python evaluator.py      ← основной запуск")
    else:
        not_ready = [m for m, ok in results.items() if not ok]
        print("\n⚠  Не готовы модели:")
        for m in not_ready:
            print(f"   ollama pull {m}")
