"""
llm_client.py — переиспользуемый клиент для общения с LLM, плюс «панель судей»
(JudgePanel), абстрагирующая роли от конкретных моделей с помощью config.py.

Поддерживает ДВА бэкенда (выбор — config.LLM_BACKEND):
  • "vllm"   — продакшн НПКЦ: OpenAI-совместимый API vLLM (/v1/chat/completions);
  • "ollama" — локальная разработка: нативный API Ollama (/api/generate).
Весь остальной код пайплайна о бэкенде НЕ знает: он обращается к моделям только
через JudgePanel.ask_json(role, ...).

Модуль НЕ содержит ничего специфичного для медицинской оценки — только
инфраструктуру: HTTP-вызовы, восстановление JSON из «грязного» ответа модели,
повторные попытки, маппинг ролей на модели. Промпты, схемы вывода и доменная
логика живут в модулях более высокого уровня (препроцессор, объективный слой,
judge-раунды).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import requests

import config

log = logging.getLogger("llm_client")


class LLMError(RuntimeError):
    """LLM-бэкенд недоступен, модель не обслуживается или не вернула пригодный ответ.

    `retryable` разделяет два принципиально разных случая:
      • False — сервер недоступен / модель не обслуживается. Повторять бессмысленно,
        ошибка должна всплыть немедленно (иначе на мёртвом сервере мы сделаем
        3 попытки ask_json x 3 попытки generate = 9 бесполезных ожиданий таймаута);
      • True  — таймаут или HTTP 400 (типично: промпт не влез в max_model_len).
        Здесь имеет смысл дойти до УПРОЩЁННОГО fallback-промпта, который короче
        основного. Раньше LLMError всегда обходил лестницу повторов ask_json
        (там ловились только JSONDecodeError/ValueError), и fallback-промпт был
        недостижим ровно в тех случаях, ради которых его писали.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class TruncatedResponse(ValueError):
    """Модель упёрлась в лимит длины ответа и оборвала генерацию на середине.

    Наследуется от ValueError, чтобы попасть в уже существующую ветку повторов
    ask_json. Несёт частичный текст: он нужен для диагностики и для лога, но
    НЕ используется как валидный результат — раньше именно такой оборванный
    ответ уходил в салваж и молча превращался в неполный отчёт судьи.
    """

    def __init__(self, message: str, *, partial: str = "", model: str = "", limit: int = 0):
        super().__init__(message)
        self.partial = partial
        self.model = model
        self.limit = limit


# Обратная совместимость: прежнее имя исключения (код ловит `OllamaError`).
OllamaError = LLMError


# ══════════════════════════════════════════════════════════════════
# ТЕЛЕМЕТРИЯ ВЫЗОВОВ
# ══════════════════════════════════════════════════════════════════
# Раньше всё это существовало только как строки в stdout: по итоговому Excel
# нельзя было отличить «судьи нашли проблемы» от «мы не смогли разобрать ответ
# судьи». Счётчики собираются по паре (collect_telemetry) и попадают на лист
# «Диагностика прогона».

@dataclass
class CallStats:
    """Счётчики LLM-вызовов за один скоуп (обычно — одна пара ЭМК/суммаризация)."""
    calls: int = 0                  # успешных HTTP-ответов от модели
    attempts: int = 0               # попыток ask_json (включая повторные)
    truncated: int = 0              # ответов с finish_reason == "length"
    timeouts: int = 0
    http_errors: int = 0
    conn_errors: int = 0
    salvaged: int = 0               # раз, когда JSON собран салважем
    salvaged_keys: int = 0
    repaired: int = 0               # раз, когда JSON разобран только после repair_json
    truncation_repaired: int = 0    # из них — с дописыванием незакрытых скобок (=обрыв)
    fallback_used: int = 0          # попыток с упрощённым промптом (без заземления)
    budget_raised: int = 0          # раз, когда бюджет токенов поднимался после обрыва
    json_failures: int = 0          # ответов, не прошедших парсинг/валидацию
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    notes: list = field(default_factory=list)   # человекочитаемые события

    def note(self, text: str) -> None:
        if len(self.notes) < 200:        # страховка от разрастания на длинных прогонах
            self.notes.append(text)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "notes"}
        d["notes"] = list(self.notes)
        return d


# Хранилище — thread-local: run_pipeline обрабатывает пары параллельно
# (config.CONCURRENCY), и счётчики не должны перемешиваться между потоками.
_TLS = threading.local()


def _stats() -> Optional[CallStats]:
    return getattr(_TLS, "stats", None)


@contextmanager
def collect_telemetry() -> Iterator[CallStats]:
    """Собрать статистику всех LLM-вызовов внутри блока `with`."""
    prev = getattr(_TLS, "stats", None)
    st = CallStats()
    _TLS.stats = st
    try:
        yield st
    finally:
        _TLS.stats = prev


# ══════════════════════════════════════════════════════════════════
# СЕРВЕР И СПИСОК МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════

def list_available_models(timeout: float = 5.0, *, endpoint: Optional[str] = None) -> list[str]:
    """Имена моделей, обслуживаемых активным бэкендом. Для vLLM можно указать
    конкретный endpoint (иначе — config.VLLM_BASE_URL)."""
    if config.LLM_BACKEND == "ollama":
        return _list_models_ollama(timeout)
    return _list_models_vllm(endpoint or config.VLLM_BASE_URL, timeout)


def _list_models_ollama(timeout: float) -> list[str]:
    try:
        r = requests.get(config.OLLAMA_TAGS_URL, timeout=timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.ConnectionError as e:
        raise LLMError(f"Ollama недоступна по адресу {config.OLLAMA_HOST}. "
                       "Запустите сервер: `ollama serve`.") from e


def _list_models_vllm(endpoint: str, timeout: float) -> list[str]:
    try:
        headers = {"Authorization": f"Bearer {config.VLLM_API_KEY}"}
        r = requests.get(f"{endpoint.rstrip('/')}/models", headers=headers, timeout=timeout)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except requests.exceptions.ConnectionError as e:
        raise LLMError(f"vLLM недоступен по адресу {endpoint}. Проверьте, что сервер "
                       "запущен и адрес верный (EMR_VLLM_BASE_URL).") from e


def is_model_available(model_name: str, available: Optional[list[str]] = None) -> bool:
    """
    Проверяет, обслуживается ли модель бэкендом. Сравнение «по вхождению»,
    т.к. бэкенды могут возвращать имена с суффиксами (квантизация/путь).
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
    if config.LLM_BACKEND == "ollama":
        return _generate_ollama(
            model_name, prompt, force_json=force_json, think=think,
            temperature=temperature, num_predict=num_predict, num_ctx=num_ctx,
            timeout=timeout, max_timeout_retries=max_timeout_retries)
    return _generate_vllm(
        model_name, prompt, force_json=force_json, think=think,
        temperature=temperature, max_tokens=num_predict,
        timeout=timeout, max_timeout_retries=max_timeout_retries)


def _generate_ollama(model_name, prompt, *, force_json, think, temperature,
                     num_predict, num_ctx, timeout, max_timeout_retries) -> str:
    """Нативный вызов Ollama (/api/generate)."""
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
    st = _stats()
    for attempt in range(1, max_timeout_retries + 2):
        t0 = time.monotonic()
        try:
            r = requests.post(config.OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            resp = r.json()
            text = resp.get("response") or ""
            if st is not None:
                st.calls += 1
                st.seconds += time.monotonic() - t0
                st.prompt_tokens += int(resp.get("prompt_eval_count") or 0)
                st.completion_tokens += int(resp.get("eval_count") or 0)
            log.debug("        %s: токенов %s/%s%s", model_name,
                      resp.get("prompt_eval_count", "?"), num_ctx,
                      " [JSON mode]" if force_json else "")
            # Признак обрыва Ollama отдаёт в done_reason ("stop" | "length").
            # Раньше он просто игнорировался, и обрыв обнаруживался лишь косвенно —
            # когда JSON не парсился (а после дописывания скобок в repair_json
            # мог и не обнаружиться вовсе).
            if str(resp.get("done_reason") or "").lower() == "length":
                if st is not None:
                    st.truncated += 1
                    st.note(f"обрыв по лимиту: {model_name}, num_predict={num_predict}")
                raise TruncatedResponse(
                    f"ответ оборван по лимиту num_predict={num_predict}",
                    partial=text, model=model_name, limit=num_predict)
            return text
        except requests.exceptions.Timeout as e:
            last_exc = e
            if st is not None:
                st.timeouts += 1
                st.seconds += time.monotonic() - t0
            log.warning("        таймаут (%s) попытка %d/%d",
                        model_name, attempt, max_timeout_retries + 1)
            if attempt <= max_timeout_retries:
                time.sleep(config.RETRY_SLEEP_SECONDS)
        except requests.exceptions.ConnectionError as e:
            if st is not None:
                st.conn_errors += 1
            raise LLMError(f"Ollama недоступна по адресу {config.OLLAMA_HOST} "
                           f"при обращении к модели '{model_name}'.",
                           retryable=False) from e

    raise LLMError(f"Модель '{model_name}' не ответила за отведённое время "
                   f"после {max_timeout_retries + 1} попыток",
                   retryable=True) from last_exc


def _generate_vllm(model_name, prompt, *, force_json, think, temperature,
                   max_tokens, timeout, max_timeout_retries) -> str:
    """OpenAI-совместимый вызов vLLM (/v1/chat/completions).

    force_json → response_format={"type":"json_object"} (guided JSON у vLLM —
    аналог Ollama format="json"). Управление «мышлением» Qwen3 — через
    chat_template_kwargs.enable_thinking: по умолчанию (config.VLLM_ENABLE_THINKING
    выключен) размышления ОТКЛЮЧЕНЫ для надёжности JSON; при включённом флаге
    решает параметр think вызова."""
    endpoint = config.vllm_endpoint_for(model_name)
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload: dict = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if config.SEED is not None:
        payload["seed"] = config.SEED      # воспроизводимость прогона
    if force_json:
        payload["response_format"] = {"type": "json_object"}
    if config.VLLM_ENABLE_THINKING:
        if think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {config.VLLM_API_KEY}",
               "Content-Type": "application/json"}

    last_exc: Optional[Exception] = None
    st = _stats()
    for attempt in range(1, max_timeout_retries + 2):
        t0 = time.monotonic()
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            content = ((choice.get("message") or {}).get("content")) or ""
            usage = data.get("usage") or {}
            if st is not None:
                st.calls += 1
                st.seconds += time.monotonic() - t0
                st.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                st.completion_tokens += int(usage.get("completion_tokens") or 0)
            log.debug("        %s: vLLM ok (токенов %s)%s", model_name,
                      usage.get("total_tokens", "?"),
                      " [JSON mode]" if force_json else "")
            # finish_reason == "length" — модель упёрлась в max_tokens. Раньше
            # это поле не читалось вовсе, и оборванный ответ шёл в парсер как
            # обычный: repair_json дописывал закрывающие скобки, и часть обрывов
            # становилась «валидным» JSON с молча потерянными подкритериями.
            if str(choice.get("finish_reason") or "").lower() == "length":
                if st is not None:
                    st.truncated += 1
                    st.note(f"обрыв по лимиту: {model_name}, max_tokens={max_tokens}")
                raise TruncatedResponse(
                    f"ответ оборван по лимиту max_tokens={max_tokens}",
                    partial=content, model=model_name, limit=max_tokens)
            return content
        except requests.exceptions.Timeout as e:
            last_exc = e
            if st is not None:
                st.timeouts += 1
                st.seconds += time.monotonic() - t0
            log.warning("        таймаут (%s) попытка %d/%d",
                        model_name, attempt, max_timeout_retries + 1)
            if attempt <= max_timeout_retries:
                time.sleep(config.RETRY_SLEEP_SECONDS)
        except requests.exceptions.ConnectionError as e:
            if st is not None:
                st.conn_errors += 1
            raise LLMError(f"vLLM недоступен по адресу {endpoint} при обращении "
                           f"к модели '{model_name}'.", retryable=False) from e
        except requests.exceptions.HTTPError as e:
            body = getattr(getattr(e, "response", None), "text", "") or ""
            status = getattr(getattr(e, "response", None), "status_code", None)
            if st is not None:
                st.http_errors += 1
                st.note(f"HTTP {status} от {model_name}: {body[:120]}")
            # 400 у vLLM — почти всегда «промпт + max_tokens > max_model_len».
            # Такой запрос стоит повторить УКОРОЧЕННЫМ промптом (fallback), а не
            # ронять всю пару, поэтому помечаем ошибку как повторяемую.
            retryable = status == 400
            hint = ("Похоже на переполнение контекста: промпт + max_tokens больше "
                    "--max-model-len. " if retryable else
                    "Сверьте имя модели с /v1/models (--served-model-name). ")
            raise LLMError(
                f"vLLM вернул ошибку для модели '{model_name}' ({endpoint}): {e}. "
                f"Ответ: {body[:300]}. {hint}",
                retryable=retryable,
            ) from e

    raise LLMError(f"Модель '{model_name}' (vLLM) не ответила за отведённое время "
                   f"после {max_timeout_retries + 1} попыток",
                   retryable=True) from last_exc


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

    # Незакрытые скобки — прямой признак того, что модель оборвала ответ.
    # repair_json дописывает их в конец, и такой JSON начинает парситься как
    # валидный, МОЛЧА потеряв недописанные поля. Раньше это не фиксировалось
    # нигде: по отчёту нельзя было отличить «судья так решил» от «мы не дочитали
    # ответ судьи». Считаем оба случая отдельно.
    unbalanced = json_str.count("{") > json_str.count("}")

    for repaired_pass, candidate in enumerate((json_str, repair_json(json_str))):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if repaired_pass:
            st = _stats()
            if st is not None:
                st.repaired += 1
                if unbalanced:
                    st.truncation_repaired += 1
                    st.note("ответ был оборван: JSON достроен дописыванием скобок, "
                            "недописанные поля потеряны")
            if unbalanced:
                log.warning("      JSON достроен после обрыва (дописаны скобки) — "
                            "часть полей ответа потеряна")
        return parsed

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
            st = _stats()
            if st is not None:
                st.salvaged += 1
                st.salvaged_keys += len(salvaged)
                st.note(f"салваж: восстановлено {len(salvaged)} верхнеуровневых полей")
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
    num_ctx: int = config.NUM_CTX,
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
    st = _stats()
    budget = num_predict
    was_truncated = False       # предыдущая попытка оборвалась по лимиту длины

    for attempt in range(1, max_attempts + 1):
        force_json = True
        current_prompt = prompt
        if st is not None:
            st.attempts += 1

        if attempt == 2 and not was_truncated:
            # Снятие format=json помогает, когда модель «залипла» в JSON-грамматике.
            # При ОБРЫВЕ это не при чём — там виноват бюджет, и режим JSON нужно
            # сохранить, иначе к обрыву добавится ещё и свободный текст.
            log.warning("      %s retry %d (без format=json)…", model_name, attempt)
            force_json = False
        elif attempt >= 3:
            alt = fallback_prompt_fn(attempt, last_raw) if fallback_prompt_fn else None
            if alt is not None:
                log.warning("      %s retry %d (упрощённый промпт, без заземления)…",
                            model_name, attempt)
                current_prompt = alt
                if st is not None:
                    st.fallback_used += 1
                    st.note(f"попытка {attempt}: упрощённый промпт без заземления ({desc})")

        try:
            raw = generate(model_name, current_prompt,
                           force_json=force_json, think=think,
                           temperature=temperature, num_predict=budget,
                           num_ctx=num_ctx)
            last_raw = raw
            was_truncated = False
            log.debug("      RAW попытка %d: %.300s", attempt, raw)

            parsed = extract_json(raw)
            if validate_fn is not None:
                validate_fn(parsed, raw)   # может бросить ValueError -> retry

            time.sleep(config.REQUEST_PAUSE_SECONDS)
            if attempt > 1:
                log.info("      %s ✅ получен на попытке %d", model_name, attempt)
            return parsed

        except TruncatedResponse as e:
            # Повторять тот же запрос с тем же бюджетом бессмысленно — он оборвётся
            # ровно так же. Удваиваем бюджет (до потолка) и пробуем снова.
            last_raw = e.partial
            was_truncated = True
            new_budget = min(budget * 2, config.MAX_TOKENS_CEILING)
            if new_budget > budget:
                log.warning("      %s попытка %d: ответ оборван (лимит %d) — "
                            "поднимаю бюджет до %d", model_name, attempt, budget, new_budget)
                if st is not None:
                    st.budget_raised += 1
                budget = new_budget
            else:
                log.warning("      %s попытка %d: ответ оборван на потолке бюджета %d",
                            model_name, attempt, budget)
            if attempt < max_attempts:
                time.sleep(3)

        except (json.JSONDecodeError, ValueError) as e:
            if st is not None:
                st.json_failures += 1
            log.warning("      %s попытка %d: %.120s", model_name, attempt, str(e))
            if attempt < max_attempts:
                time.sleep(3)

        except LLMError as e:
            # Недоступный сервер / неверное имя модели — повторять нечего,
            # ошибка должна всплыть сразу. Таймаут и HTTP 400 (переполнение
            # контекста) — повторяем: следующая попытка может уйти на укороченный
            # fallback-промпт, который влезет.
            if not getattr(e, "retryable", False):
                raise
            log.warning("      %s попытка %d: %.140s", model_name, attempt, str(e))
            if attempt >= max_attempts:
                raise
            time.sleep(3)

    tail = " (ответ обрывался по лимиту длины)" if was_truncated else ""
    raise LLMError(f"Модель '{model_name}' не вернула пригодный JSON "
                   f"после {max_attempts} попыток ({desc}){tail}", retryable=False)


# ══════════════════════════════════════════════════════════════════
# ПАНЕЛЬ СУДЕЙ — связывает абстрактные роли с моделями через config.py
# ══════════════════════════════════════════════════════════════════

class JudgePanel:
    """
    Связывает абстрактные роли ("judge_1", "judge_2", "judge_3", "aggregator")
    с конкретными моделями согласно выбранному профилю из config.py.

    Весь код пайплайна оценки обращается ИСКЛЮЧИТЕЛЬНО к ролям через эту
    панель — благодаря этому смена профиля или бэкенда (vLLM в НПКЦ ↔ Ollama
    локально) не требует изменений в коде препроцессора, объективного слоя,
    промптов или агрегации.

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
        Сверяет модели текущего профиля со списком, реально обслуживаемым
        активным бэкендом. Возвращает {роль: {"model": ..., "available": bool}}.
        """
        try:
            available = list_available_models()
        except LLMError:
            return {role: {"model": self.model_for(role), "available": None}
                    for role in self.profile.roles}

        return {
            role: {
                "model": model_name,
                "available": is_model_available(model_name, available),
            }
            for role, model_name in self.profile.roles.items()
        }
