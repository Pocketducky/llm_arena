"""
test_truncation.py — офлайн-тест обработки ОБОРВАННЫХ ответов модели (без сети).

Регрессия на диагностированный дефект пилота НПКЦ: единый бюджет генерации
(1024-1536 токенов) применялся и к ответу из двух булевых полей, и к финальной
агрегации R3 по шести полным отчётам. Ответ обрывался, признак обрыва
(`finish_reason`/`done_reason`) не читался вовсе, repair_json дописывал
закрывающие скобки — и часть обрывов становилась «валидным» JSON с молча
потерянными подкритериями. Дальше потерянный подкритерий превращался в
клиническое «Неприемлемо».

Проверяем:
  1. finish_reason="length" (vLLM) -> TruncatedResponse с частичным текстом;
  2. done_reason="length" (Ollama) -> то же самое;
  3. ask_json при обрыве УДВАИВАЕТ бюджет, а не повторяет тот же запрос;
  4. ask_json не отдаёт оборванный ответ как результат, даже если он парсится;
  5. LLMError с retryable=False (сервер недоступен) всплывает немедленно;
  6. HTTP 400 (переполнение контекста) помечается retryable и доходит до
     упрощённого fallback-промпта;
  7. телеметрия считает обрывы, ремонты, салважи и подъёмы бюджета;
  8. валидация блока и полного отчёта R2 отвергает недосчитанные ответы.

Запуск:  python tests/test_truncation.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests as _real_requests

import config
import llm_client


class FakeResp:
    def __init__(self, data: dict, *, raise_http: bool = False,
                 text: str = "", status: int | None = None):
        self._data = data
        self._raise = raise_http
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self._raise:
            raise _real_requests.exceptions.HTTPError("HTTP error", response=self)

    def json(self):
        return self._data


class ScriptedRequests:
    """Отдаёт заготовленные ответы по очереди, запоминая все запросы."""

    def __init__(self, responses):
        self.exceptions = _real_requests.exceptions
        self.responses = list(responses)
        self.posts: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json})
        resp = self.responses[min(len(self.posts) - 1, len(self.responses) - 1)]
        if isinstance(resp, Exception):
            raise resp
        return resp

    def get(self, url, headers=None, timeout=None):
        return self.responses[0]


class _Swap:
    def __init__(self, requests_stub, **cfg):
        self.stub = requests_stub
        self.cfg = cfg
        self._saved_req = None
        self._saved_cfg: dict = {}

    def __enter__(self):
        self._saved_req = llm_client.requests
        llm_client.requests = self.stub
        # паузы между запросами в тесте не нужны
        self.cfg.setdefault("REQUEST_PAUSE_SECONDS", 0)
        self.cfg.setdefault("RETRY_SLEEP_SECONDS", 0)
        for k, v in self.cfg.items():
            self._saved_cfg[k] = getattr(config, k)
            setattr(config, k, v)
        return self.stub

    def __exit__(self, *exc):
        llm_client.requests = self._saved_req
        for k, v in self._saved_cfg.items():
            setattr(config, k, v)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _vllm(content: str, finish: str = "stop") -> FakeResp:
    return FakeResp({"choices": [{"message": {"content": content},
                                  "finish_reason": finish}],
                     "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                               "total_tokens": 150}})


def _ollama(response: str, done: str = "stop") -> FakeResp:
    return FakeResp({"response": response, "done_reason": done,
                     "prompt_eval_count": 100, "eval_count": 50})


# ── Тесты ─────────────────────────────────────────────────────────

def test_vllm_length_raises_truncated():
    print("[1] vLLM: finish_reason='length' -> TruncatedResponse")
    partial = '{"A1": {"pass": true, "comment": "ок"}, "A2": {"pass": fal'
    with _Swap(ScriptedRequests([_vllm(partial, finish="length")]),
               LLM_BACKEND="vllm", VLLM_MODEL_ENDPOINTS={}):
        try:
            llm_client.generate("m", "p", num_predict=1536)
            raise AssertionError("ожидался TruncatedResponse")
        except llm_client.TruncatedResponse as e:
            assert e.partial == partial, e.partial
            assert e.limit == 1536, e.limit
    _ok("обрыв распознан, частичный текст и лимит сохранены")


def test_ollama_length_raises_truncated():
    print("[2] Ollama: done_reason='length' -> TruncatedResponse")
    with _Swap(ScriptedRequests([_ollama('{"A1": {"pass": tr', done="length")]),
               LLM_BACKEND="ollama"):
        try:
            llm_client.generate("qwen3:8b", "p", num_predict=1024)
            raise AssertionError("ожидался TruncatedResponse")
        except llm_client.TruncatedResponse as e:
            assert e.limit == 1024, e.limit
    _ok("обрыв Ollama распознан")


def test_ask_json_doubles_budget():
    print("[3] ask_json: обрыв -> удвоение бюджета, а не повтор того же запроса")
    good = '{"A1": {"pass": true}, "A2": {"pass": false}}'
    script = [_vllm("{...oборвано", finish="length"),
              _vllm("{...oборвано", finish="length"),
              _vllm(good)]
    with _Swap(ScriptedRequests(script), LLM_BACKEND="vllm",
               VLLM_MODEL_ENDPOINTS={}, MAX_TOKENS_CEILING=16384) as stub:
        with llm_client.collect_telemetry() as st:
            out = llm_client.ask_json("m", "p", desc="блок A", max_attempts=3,
                                      num_predict=2048)
    budgets = [p["json"]["max_tokens"] for p in stub.posts]
    assert budgets == [2048, 4096, 8192], budgets
    assert out["A1"]["pass"] is True and out["A2"]["pass"] is False, out
    assert st.truncated == 2 and st.budget_raised == 2, st.as_dict()
    _ok(f"бюджет рос {budgets}, результат получен на 3-й попытке")


def test_budget_respects_ceiling():
    print("[4] ask_json: подъём бюджета не превышает потолок")
    script = [_vllm("{", finish="length")] * 4
    with _Swap(ScriptedRequests(script), LLM_BACKEND="vllm",
               VLLM_MODEL_ENDPOINTS={}, MAX_TOKENS_CEILING=4096) as stub:
        try:
            llm_client.ask_json("m", "p", max_attempts=3, num_predict=2048)
            raise AssertionError("ожидался LLMError после исчерпания попыток")
        except llm_client.LLMError as e:
            assert "обрыв" in str(e), str(e)
    budgets = [p["json"]["max_tokens"] for p in stub.posts]
    assert budgets == [2048, 4096, 4096], budgets
    _ok("потолок соблюдён; оборванный ответ НЕ выдан за результат")


def test_truncated_never_returned_even_if_parsable():
    print("[5] ask_json: оборванный, но синтаксически валидный ответ не принимается")
    # Обрыв ровно на границе объекта: json.loads такой текст разберёт,
    # хотя подкритерии A2/A3 модель дописать не успела.
    looks_valid = '{"A1": {"pass": true, "comment": "ок"}}'
    script = [_vllm(looks_valid, finish="length"), _vllm(looks_valid, finish="stop")]
    with _Swap(ScriptedRequests(script), LLM_BACKEND="vllm", VLLM_MODEL_ENDPOINTS={}) as stub:
        out = llm_client.ask_json("m", "p", max_attempts=2, num_predict=1024)
    assert len(stub.posts) == 2, "обрыв обязан был вызвать повтор"
    assert out == {"A1": {"pass": True, "comment": "ок"}}
    _ok("обрыв не замаскировался под валидный JSON — сделан повтор")


def test_connection_error_fails_fast():
    print("[6] Недоступный сервер: ошибка всплывает сразу, без повторов")
    err = _real_requests.exceptions.ConnectionError("connection refused")
    with _Swap(ScriptedRequests([err]), LLM_BACKEND="vllm", VLLM_MODEL_ENDPOINTS={}) as stub:
        try:
            llm_client.ask_json("m", "p", max_attempts=3)
            raise AssertionError("ожидался LLMError")
        except llm_client.LLMError as e:
            assert e.retryable is False, "недоступность сервера не повторяема"
    assert len(stub.posts) == 1, f"сделано {len(stub.posts)} запросов вместо 1"
    _ok("один запрос вместо девяти (3 ask_json x 3 generate)")


def test_http_400_reaches_fallback_prompt():
    print("[7] HTTP 400 (переполнение контекста) -> повтор и упрощённый промпт")
    bad = FakeResp({}, raise_http=True, text='{"error":"maximum context length"}', status=400)
    ok = _vllm('{"A1": {"pass": true}}')
    seen: list[int] = []

    def fallback(attempt: int, last_raw: str):
        seen.append(attempt)
        return "короткий промпт"

    with _Swap(ScriptedRequests([bad, bad, ok]), LLM_BACKEND="vllm",
               VLLM_MODEL_ENDPOINTS={}) as stub:
        out = llm_client.ask_json("m", "длинный промпт", max_attempts=3,
                                  fallback_prompt_fn=fallback)
    assert out["A1"]["pass"] is True, out
    assert seen == [3], f"fallback вызван на попытках {seen}"
    assert stub.posts[-1]["json"]["messages"][0]["content"] == "короткий промпт"
    _ok("400 признан повторяемым, 3-я попытка ушла на укороченный промпт")


def test_telemetry_counts_repair_and_salvage():
    print("[8] Телеметрия: восстановление оборванного JSON видно в счётчиках")

    # (а) типичный обрыв: repair_json дописывает скобки, JSON становится
    #     «валидным», а недописанные поля молча теряются. Раньше это не
    #     фиксировалось нигде — именно так обрыв маскировался под решение судьи.
    with llm_client.collect_telemetry() as st:
        out = llm_client.extract_json(
            '{"A1": {"pass": true}, "A2": {"pass": false, "comment": "не закрыт')
    assert out == {"A1": {"pass": True}}, out
    assert st.repaired == 1 and st.truncation_repaired == 1, st.as_dict()
    assert any("оборван" in n for n in st.notes), st.notes
    _ok("обрыв, «починенный» дописыванием скобок, помечен (A2 потерян — и это видно)")

    # (б) последний рубеж: салваж верхнеуровневых полей.
    with llm_client.collect_telemetry() as st:
        out = llm_client.extract_json('{"A1": {"pass": true}, "A2": nan, "A3": {"pass": false}}')
    assert st.salvaged == 1 and st.salvaged_keys == 1, st.as_dict()
    assert any("салваж" in n for n in st.notes), st.notes
    _ok("салваж зафиксирован в телеметрии")


def test_block_validation_rejects_truncated_report():
    print("[9] Валидация блока отвергает недосчитанный отчёт (обрыв -> повтор)")
    import judge
    validate = judge._validate_block("A")
    try:
        validate({"A1": {"pass": True}}, "raw")
        raise AssertionError("ожидался ValueError: A2/A3 отсутствуют")
    except ValueError as e:
        assert "A2" in str(e), str(e)
    _ok("неполный блок A не проходит валидацию — уйдёт в повтор с бо́льшим бюджетом")

    # Раунд 2 раньше вызывался с validate_fn=None: любой огрызок принимался молча.
    try:
        judge._validate_full_report({"A": {"A1": {"pass": True}}}, "raw")
        raise AssertionError("ожидался ValueError: отчёт R2 неполон")
    except ValueError as e:
        assert "неполный отчёт R2" in str(e), str(e)
    _ok("неполный отчёт R2 больше не принимается молча")


def main():
    print("=" * 62)
    print("Офлайн-тест обработки оборванных ответов и телеметрии")
    print("=" * 62)
    test_vllm_length_raises_truncated()
    test_ollama_length_raises_truncated()
    test_ask_json_doubles_budget()
    test_budget_respects_ceiling()
    test_truncated_never_returned_even_if_parsable()
    test_connection_error_fails_fast()
    test_http_400_reaches_fallback_prompt()
    test_telemetry_counts_repair_and_salvage()
    test_block_validation_rejects_truncated_report()
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅")


if __name__ == "__main__":
    main()
