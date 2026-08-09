"""
test_vllm_client.py — офлайн-тест бэкенда llm_client БЕЗ реального сервера.

`requests` подменяется фейком: проверяем форму запроса и разбор ответа, не
обращаясь к сети. Покрывает:
  1. vLLM: URL /chat/completions, тело (messages, max_tokens, response_format,
     chat_template_kwargs.enable_thinking), разбор choices[0].message.content;
  2. роутинг «модель → endpoint» (разные порты для разных моделей);
  3. маппинг HTTP-ошибки vLLM в LLMError (типичный кейс — неверное имя модели);
  4. список моделей vLLM (GET /v1/models → data[].id);
  5. регрессия Ollama-бэкенда (нативный payload /api/generate).

Запуск:  python tests/test_vllm_client.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests as _real_requests

import config
import llm_client


# ── Фейки сети ────────────────────────────────────────────────────
class FakeResp:
    def __init__(self, data: dict, *, raise_http: bool = False, text: str = ""):
        self._data = data
        self._raise = raise_http
        self.text = text

    def raise_for_status(self):
        if self._raise:
            raise _real_requests.exceptions.HTTPError("500 Server Error", response=self)

    def json(self):
        return self._data


class FakeRequests:
    """Подмена модуля requests: запоминает последний вызов, отдаёт заготовку."""
    def __init__(self, resp: FakeResp):
        self.exceptions = _real_requests.exceptions
        self.resp = resp
        self.last_post: dict | None = None
        self.last_get: dict | None = None

    def post(self, url, json=None, headers=None, timeout=None):
        self.last_post = {"url": url, "json": json, "headers": headers}
        return self.resp

    def get(self, url, headers=None, timeout=None):
        self.last_get = {"url": url, "headers": headers}
        return self.resp


class _Swap:
    """Контекст: временно подменить llm_client.requests и поля config."""
    def __init__(self, resp: FakeResp, **cfg):
        self.fake = FakeRequests(resp)
        self.cfg = cfg
        self._saved_req = None
        self._saved_cfg: dict = {}

    def __enter__(self):
        self._saved_req = llm_client.requests
        llm_client.requests = self.fake
        for k, v in self.cfg.items():
            self._saved_cfg[k] = getattr(config, k)
            setattr(config, k, v)
        return self.fake

    def __exit__(self, *exc):
        llm_client.requests = self._saved_req
        for k, v in self._saved_cfg.items():
            setattr(config, k, v)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


_CHAT_OK = FakeResp({"choices": [{"message": {"content": '{"A1": {"pass": true}}'}}],
                     "usage": {"total_tokens": 42}})


# ── Тесты ─────────────────────────────────────────────────────────
def test_vllm_payload_and_parse():
    print("[1] vLLM: форма запроса и разбор ответа")
    with _Swap(_CHAT_OK, LLM_BACKEND="vllm", VLLM_BASE_URL="http://host:8000/v1",
               VLLM_MODEL_ENDPOINTS={}, VLLM_ENABLE_THINKING=False, VLLM_API_KEY="EMPTY") as fake:
        out = llm_client.generate("qwen3.5-122b-fp8", "оцени", force_json=True,
                                  think=False, num_predict=1536, temperature=0.1)
    assert out == '{"A1": {"pass": true}}', out
    p = fake.last_post
    assert p["url"] == "http://host:8000/v1/chat/completions", p["url"]
    body = p["json"]
    assert body["model"] == "qwen3.5-122b-fp8"
    assert body["messages"] == [{"role": "user", "content": "оцени"}]
    assert body["max_tokens"] == 1536 and body["temperature"] == 0.1
    assert body["response_format"] == {"type": "json_object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["headers"]["Authorization"] == "Bearer EMPTY"
    _ok("chat/completions, messages, max_tokens, JSON-mode, enable_thinking=False, авторизация")


def test_vllm_no_json_mode_when_disabled():
    print("[2] vLLM: без force_json нет response_format")
    with _Swap(_CHAT_OK, LLM_BACKEND="vllm", VLLM_MODEL_ENDPOINTS={},
               VLLM_ENABLE_THINKING=False) as fake:
        llm_client.generate("m", "p", force_json=False, think=None, num_predict=200)
    assert "response_format" not in fake.last_post["json"]
    _ok("response_format отсутствует при force_json=False")


def test_vllm_routing():
    print("[3] vLLM: роутинг модель → endpoint (разные порты)")
    endpoints = {"qwen3.5-122b-fp8": "http://a:8000/v1", "qwen3.6-27b-fp8": "http://b:8001/v1"}
    with _Swap(_CHAT_OK, LLM_BACKEND="vllm", VLLM_BASE_URL="http://a:8000/v1",
               VLLM_MODEL_ENDPOINTS=endpoints) as fake:
        llm_client.generate("qwen3.6-27b-fp8", "p", force_json=True, num_predict=100)
    assert fake.last_post["url"] == "http://b:8001/v1/chat/completions", fake.last_post["url"]
    _ok("средняя модель ушла на свой endpoint :8001")


def test_vllm_http_error_maps_to_llmerror():
    print("[4] vLLM: HTTP-ошибка → LLMError с подсказкой про имя модели")
    bad = FakeResp({}, raise_http=True, text='{"error":"model not found"}')
    raised = False
    with _Swap(bad, LLM_BACKEND="vllm", VLLM_MODEL_ENDPOINTS={}):
        try:
            llm_client.generate("wrong-name", "p", force_json=True, num_predict=100)
        except llm_client.LLMError as e:
            raised = True
            assert "served-model-name" in str(e) or "/v1/models" in str(e), str(e)
    assert raised, "ожидался LLMError"
    _ok("404/500 vLLM корректно превращается в LLMError")


def test_vllm_list_models():
    print("[5] vLLM: список моделей (GET /v1/models)")
    resp = FakeResp({"data": [{"id": "qwen3.5-122b-fp8"}, {"id": "qwen3.6-27b-fp8"}]})
    with _Swap(resp, LLM_BACKEND="vllm", VLLM_BASE_URL="http://host:8000/v1") as fake:
        models = llm_client.list_available_models(endpoint="http://host:8000/v1")
    assert models == ["qwen3.5-122b-fp8", "qwen3.6-27b-fp8"], models
    assert fake.last_get["url"] == "http://host:8000/v1/models"
    _ok("data[].id → список моделей")


def test_ollama_regression():
    print("[6] Ollama: нативный payload /api/generate (регрессия)")
    resp = FakeResp({"response": "{}", "prompt_eval_count": 10})
    with _Swap(resp, LLM_BACKEND="ollama") as fake:
        llm_client.generate("qwen3:8b", "p", force_json=True, think=False,
                            num_predict=512, num_ctx=16384)
    body = fake.last_post["json"]
    assert fake.last_post["url"].endswith("/api/generate")
    assert body["prompt"] == "p" and body["format"] == "json"
    assert body["options"]["num_predict"] == 512 and body["options"]["num_ctx"] == 16384
    assert body["think"] is False
    _ok("Ollama-путь строит прежний payload (prompt/options/format/think)")


def main():
    print("=" * 62)
    print("Офлайн-тест бэкенда llm_client (vLLM + Ollama), без сети")
    print("=" * 62)
    test_vllm_payload_and_parse()
    test_vllm_no_json_mode_when_disabled()
    test_vllm_routing()
    test_vllm_http_error_maps_to_llmerror()
    test_vllm_list_models()
    test_ollama_regression()
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅")


if __name__ == "__main__":
    main()
