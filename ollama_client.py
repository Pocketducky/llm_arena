"""
ollama_client.py — переиспользуемый клиент для общения с локальными LLM
через Ollama, плюс «панель судей» (JudgePanel), абстрагирующая роли от
конкретных моделей с помощью config.py.

Этот модуль НЕ содержит ничего специфичного для медицинской оценки —
только инфраструктуру: HTTP-вызовы, восстановление JSON из «грязного»
ответа модели, повторные попытки, маппинг ролей на модели. Промпты,
схемы вывода и доменная логика живут в модулях более высокого уровня
(препроцессор, объективный слой, judge-раунды).

Большая часть низкоуровневой логики унаследована из предыдущей версии
evaluator.py (repair_json/extract_json/retry-цепочка показали себя
рабочими) — здесь она обобщена и отвязана от конкретных промптов.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable, Optional

import requests

import config

log = logging.getLogger("ollama_client")


class OllamaError(RuntimeError):
    """Ollama недоступна, модель не установлена или не вернула пригодный ответ."""


# ══════════════════════════════════════════════════════════════════
# СЕРВЕР И СПИСОК МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════

def list_available_models(timeout: float = 5.0) -> list[str]:
    """Возвращает имена моделей, установленных в локальной Ollama."""
    try:
        r = requests.get(config.OLLAMA_TAGS_URL, timeout=timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            "Ollama недоступна по адресу "
            f"{config.OLLAMA_HOST}. Запустите сервер: `ollama serve`."
        ) from e


def is_model_available(model_name: str, available: Optional[list[str]] = None) -> bool:
    """
    Проверяет, установлена ли модель локально.
    Сравнение «по вхождению», т.к. Ollama может возвращать теги с
    суффиксом квантизации (например, "qwen3:8b" в списке как "qwen3:8b").
    """
    available = available if available is not None else list_available_models()
    return any(model_name == m or model_name in m for m in available)


# ══════════════════════════════════════════════════════════════════
# НИЗКОУРОВНЕВЫЙ ВЫЗОВ МОДЕЛИ
# ══════════════════════════════════════════════════════════════════

def generate(
    model_name: str,
    prompt: str,
    *,
    force_json: bool = True,
    think: Optional[bool] = None,
    temperature: float = config.DEFAULT_TEMPERATURE,
    num_predict: int = config.DEFAULT_NUM_PREDICT,
    num_ctx: int = config.NUM_CTX,
    timeout: int = config.TIMEOUT_SECONDS,
    max_timeout_retries: int = 2,
) -> str:
    """
    Делает один запрос к Ollama и возвращает «сырой» текстовый ответ.

    force_json=True добавляет "format": "json" — Ollama тогда генерирует
    ТОЛЬКО валидный JSON на уровне токенов (модель физически не может
    выйти за рамки JSON-грамматики). Отключайте для retry-промптов,
    где модель может предпочесть свободный текст.

    think — явное управление режимом рассуждений «мыслящих» моделей
    (qwen3, DeepSeek-R1 и т.п.). По умолчанию (None) параметр не передаётся
    и Ollama использует своё поведение по умолчанию — для гибридных моделей
    семейства qwen3 это, как правило, ВКЛЮЧЁННОЕ «мышление» даже при
    format="json". Из-за этого скрытые рассуждения (<think>...</think>)
    съедают бюджет num_predict ДО того, как модель доходит до самого
    JSON-ответа — итог обрывается на середине (нет закрывающей скобки)
    или оказывается пустым. Диагностировано прямым вызовом Ollama: тот же
    промпт с think=False уложился в 592 токена (из 1024 бюджета) и вернул
    полностью корректный JSON с done_reason="stop"; со включённым по
    умолчанию «мышлением» бюджет тратился на рассуждения, и видимый ответ
    обрывался. Поэтому для чисто экстрактивных задач (см. вызовы из
    objective_layer.extract_semantic_entities/gate.filter_relevant_entities)
    передавайте think=False явно — рассуждения там не нужны по сути задачи,
    а их отключение одновременно чинит обрыв JSON и ускоряет вызов.
    Раунды LLM-as-Judge (judge.py, R1/R2/R3) ТОЖЕ передают think=False: на
    пилотном железе (qwen3:8b) включённое «мышление» регулярно обрывало большой
    структурированный отчёт (особенно полный отчёт A-E в R2) — надёжность JSON
    важнее теоретической пользы скрытых рассуждений; заземление (объективный
    слой, шлюз, отчёты коллег) уже в промпте.

    При таймауте делает до `max_timeout_retries` повторов с паузой.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    if force_json:
        payload["format"] = "json"
    if think is not None:
        payload["think"] = think

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_timeout_retries + 2):
        try:
            r = requests.post(config.OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            resp = r.json()
            used = resp.get("prompt_eval_count", "?")
            log.debug(
                "        %s: токенов %s/%s%s",
                model_name, used, num_ctx, " [JSON mode]" if force_json else "",
            )
            return resp["response"]
        except requests.exceptions.Timeout as e:
            last_exc = e
            log.warning("        таймаут (%s) попытка %d/%d",
                        model_name, attempt, max_timeout_retries + 1)
            if attempt <= max_timeout_retries:
                time.sleep(config.RETRY_SLEEP_SECONDS)
        except requests.exceptions.ConnectionError as e:
            raise OllamaError(
                f"Ollama недоступна по адресу {config.OLLAMA_HOST} "
                f"при обращении к модели '{model_name}'."
            ) from e

    raise OllamaError(f"Модель '{model_name}' не ответила за отведённое время "
                      f"после {max_timeout_retries + 1} попыток") from last_exc


# ══════════════════════════════════════════════════════════════════
# ВОССТАНОВЛЕНИЕ И ИЗВЛЕЧЕНИЕ JSON ИЗ «ГРЯЗНОГО» ОТВЕТА МОДЕЛИ
# ══════════════════════════════════════════════════════════════════

def repair_json(text: str) -> str:
    """
    Исправляет типичные ошибки JSON, которые допускают LLM:
      1. Trailing comma:              {"a":1,}        -> {"a":1}
      2. Одинарные кавычки:           {'a':'b'}       -> {"a":"b"}
      3. Python-литералы:             True/False/None -> true/false/null
      4. Ключи без кавычек:           {a:1}           -> {"a":1}
      5. Незакрытые скобки в конце
      6. Пропущенное двоеточие:       "key" "value"   -> "key": "value"
      7. Сдвоенные запятые:           {"a":1,,}       -> {"a":1}
    """
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    # Одинарные кавычки -> двойные (стараемся не трогать апострофы внутри строк)
    text = re.sub(r"(?<!\\)'([^']*)'", r'"\1"', text)

    # Trailing comma перед } или ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Ключи без кавычек: {score:5} -> {"score":5}
    text = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', text)

    # Сдвоенные запятые
    text = re.sub(r",{2,}", ",", text)

    # Пропущенное двоеточие между строкой-ключом и строкой-значением
    text = re.sub(r'("\s*)(\s+")', r"\1:\2", text)

    # Закрываем незакрытые скобки в конце
    opens = text.count("{") - text.count("}")
    opens2 = text.count("[") - text.count("]")
    if opens > 0:
        text = text.rstrip() + "}" * opens
    if opens2 > 0:
        text = text.rstrip() + "]" * opens2

    return text


_DECODER = json.JSONDecoder()


def _salvage_json_object(text: str) -> Optional[dict]:
    """
    ПОСЛЕДНИЙ РУБЕЖ восстановления. Когда объект целиком не парсится даже после
    repair_json (типично — модель ОБОРВАЛА ответ на середине), собираем словарь
    из тех ВЕРХНЕУРОВНЕВЫХ пар "ключ": значение, что успели прийти корректно и
    ЦЕЛИКОМ. Каждое значение (объект подкритерия, строка, список, bool) парсится
    по отдельности через json.raw_decode; первое же незавершённое значение
    обрывает сбор — но всё, что до него, сохраняется.

    Это то, что превращает «пришёл битый JSON → потеряли ВЕСЬ отчёт блока» в
    «пришёл битый JSON → сохранили все подкритерии, кроме недописанного».
    Возвращает dict (возможно частичный) или None, если не удалось собрать ничего.
    """
    start = text.find("{")
    if start < 0:
        return None
    out: dict = {}
    i, n = start + 1, len(text)
    while i < n:
        ch = text[i]
        if ch == "}":
            break
        if ch == '"':
            try:
                key, key_end = _DECODER.raw_decode(text, i)   # ключ-строка
            except ValueError:
                break
            j = key_end
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] != ":":
                break
            j += 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            try:
                value, val_end = _DECODER.raw_decode(text, j)   # значение целиком
            except ValueError:
                break            # значение оборвано — дальше собирать нечего
            out[str(key)] = value
            i = val_end
            continue
        i += 1
    return out or None


def extract_json(raw: str) -> dict:
    """
    Извлекает JSON-объект из ответа модели: убирает markdown-обёртку,
    <think>-блоки (DeepSeek-R1 и подобные reasoning-модели), текст вокруг
    JSON. При неудаче пробует автоматический ремонт и поиск первого
    «целостного» объекта.
    """
    text = raw.strip()

    if "<think>" in text:
        if "</think>" in text:
            text = text[text.find("</think>") + len("</think>"):].strip()
        else:
            # Незакрытый think-блок — отбрасываем тег, ищем первую { дальше
            text = text[text.find("<think>") + len("<think>"):]
            text = re.sub(r"^.*?(?=\{)", "", text, flags=re.DOTALL).strip()

    text = re.sub(r"```(?:json)?", "", text).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Фигурные скобки не найдены. Ответ: {raw[:300]!r}")

    json_str = text[start:end]

    for candidate in (json_str, repair_json(json_str)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Последний шанс: первый сбалансированный объект (модель иногда пишет 2 JSON подряд)
    depth, best_end = 0, -1
    for idx in range(start, min(end, len(text))):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                best_end = idx + 1
                break
    if best_end > start:
        try:
            return json.loads(repair_json(text[start:best_end]))
        except json.JSONDecodeError:
            pass

    # Последний рубеж: собрать объект из корректно пришедших верхнеуровневых
    # пар (спасает оборванные ответы — сохраняем всё, кроме недописанного поля).
    for candidate in (text, repair_json(text)):
        salvaged = _salvage_json_object(candidate)
        if salvaged:
            log.warning("      JSON собран салважем верхнеуровневых полей "
                        "(ответ был оборван); ключей извлечено: %d", len(salvaged))
            return salvaged

    raise ValueError(f"JSON не удалось распарсить даже после ремонта. "
                     f"Фрагмент: {json_str[:300]!r}")


# ══════════════════════════════════════════════════════════════════
# ЗАПРОС С ПОВТОРНЫМИ ПОПЫТКАМИ И СХЕМО-СПЕЦИФИЧНЫМИ ХУКАМИ
# ══════════════════════════════════════════════════════════════════

FallbackPromptFn = Callable[[int, str], Optional[str]]
ValidateFn = Callable[[dict, str], None]


def ask_json(
    model_name: str,
    prompt: str,
    *,
    desc: str = "",
    max_attempts: int = 3,
    fallback_prompt_fn: Optional[FallbackPromptFn] = None,
    validate_fn: Optional[ValidateFn] = None,
    think: Optional[bool] = None,
    temperature: float = config.DEFAULT_TEMPERATURE,
    num_predict: int = config.DEFAULT_NUM_PREDICT,
) -> dict:
    """
    Запрашивает у модели структурированный JSON-ответ с автоматическим
    восстановлением и повторными попытками.

    Стратегия попыток:
      1) основной промпт, format=json
      2) тот же промпт, БЕЗ format=json (иногда снимает «зависание» модели
         в шаблоне при жёстком JSON-режиме)
      3) если задан fallback_prompt_fn — альтернативный (упрощённый) промпт,
         иначе третья попытка повторяет (1) ещё раз

    fallback_prompt_fn(attempt, last_raw_response) -> str | None
        Может вернуть промпт-«скелет», подходящий под конкретную JSON-схему
        (см. пример в judge.py). Вернуть None — использовать исходный промпт.

    validate_fn(parsed_json, raw_response) -> None
        Может бросить ValueError, если результат выглядит как «заглушка»
        (например, все числовые поля равны нулю) — это запустит retry.
        Доменная валидация остаётся за вызывающим кодом, не за клиентом.

    think — пробрасывается в generate() как есть (см. там подробное
        объяснение диагностированной проблемы «мышление съедает num_predict
        и обрывает JSON»). Передавайте think=False для чисто экстрактивных
        промптов (см. objective_layer.extract_semantic_entities).
    """
    log.info("      %s %s | %d симв.", model_name, desc, len(prompt))
    last_raw = ""

    for attempt in range(1, max_attempts + 1):
        force_json = True
        current_prompt = prompt

        if attempt == 2:
            log.warning("      %s retry %d (без format=json)…", model_name, attempt)
            force_json = False
        elif attempt >= 3:
            alt = fallback_prompt_fn(attempt, last_raw) if fallback_prompt_fn else None
            if alt is not None:
                log.warning("      %s retry %d (альтернативный промпт)…", model_name, attempt)
                current_prompt = alt

        try:
            raw = generate(model_name, current_prompt,
                           force_json=force_json, think=think,
                           temperature=temperature, num_predict=num_predict)
            last_raw = raw
            log.debug("      RAW попытка %d: %.300s", attempt, raw)

            parsed = extract_json(raw)
            if validate_fn is not None:
                validate_fn(parsed, raw)   # может бросить ValueError -> retry

            time.sleep(config.REQUEST_PAUSE_SECONDS)
            if attempt > 1:
                log.info("      %s ✅ получен на попытке %d", model_name, attempt)
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            log.warning("      %s попытка %d: %.120s", model_name, attempt, str(e))
            if attempt < max_attempts:
                time.sleep(3)

    raise OllamaError(f"Модель '{model_name}' не вернула пригодный JSON "
                      f"после {max_attempts} попыток ({desc})")


# ══════════════════════════════════════════════════════════════════
# ПАНЕЛЬ СУДЕЙ — связывает абстрактные роли с моделями через config.py
# ══════════════════════════════════════════════════════════════════

class JudgePanel:
    """
    Связывает абстрактные роли ("judge_1", "judge_2", "judge_3", "aggregator")
    с конкретными моделями Ollama согласно выбранному профилю из config.py.

    Весь код пайплайна оценки обращается ИСКЛЮЧИТЕЛЬНО к ролям через эту
    панель — благодаря этому переключение профиля (например, "pilot" → "target",
    когда появится GPU-инфраструктура с DeepSeek-R1 70B и Qwen3-Next-80B)
    не требует изменений в коде препроцессора, объективного слоя, промптов
    или агрегации.

    Пример:
        panel = JudgePanel()                 # активный профиль из config.ACTIVE_PROFILE
        panel = JudgePanel(profile="target")  # явный выбор профиля

        result = panel.ask_json("judge_1", prompt, desc="блок A — точность")
    """

    def __init__(self, profile: Optional[str] = None):
        self.profile = config.get_profile(profile)

    def __repr__(self) -> str:
        return f"JudgePanel(profile={self.profile.name!r}, roles={self.profile.roles})"

    def model_for(self, role: str) -> str:
        return self.profile.model_for(role)

    def ask_json(self, role: str, prompt: str, **kwargs) -> dict:
        return ask_json(self.model_for(role), prompt, **kwargs)

    def availability_report(self) -> dict[str, dict]:
        """
        Сверяет модели текущего профиля со списком, реально установленным
        в Ollama. Возвращает {роль: {"model": ..., "available": bool}}.
        """
        try:
            available = list_available_models()
        except OllamaError:
            return {role: {"model": self.model_for(role), "available": None}
                    for role in self.profile.roles}

        return {
            role: {
                "model": model_name,
                "available": is_model_available(model_name, available),
            }
            for role, model_name in self.profile.roles.items()
        }
