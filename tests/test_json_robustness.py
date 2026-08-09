"""
test_json_robustness.py — регрессионный тест на баг «неверный JSON → оценки
обнулялись». Проверяет три уровня защиты, ВСЕ без обращения к Ollama:

  1. llm_client.extract_json / _salvage_json_object — восстановление
     «грязного» и оборванного JSON (markdown-обёртка, незакрытый <think>,
     одинарные кавычки, обрыв на середине объекта).
  2. judge.score_round1 — пер-блочная изоляция: сбой ОДНОГО блока не уничтожает
     отчёт судьи; проваленный блок заменяется на sentinel (pass=None), остальные
     сохраняют реальные оценки.
  3. eval_patient._block_pass_majority — sentinel-блок трактуется как «нет данных»
     («н/д»), а НЕ как фальшивый «0/N»; блок с реальными голосами не обнуляется
     из-за сбоя у одного судьи.

Запуск:  python tests/test_json_robustness.py   (или из tests/: python test_json_robustness.py)
"""

from __future__ import annotations

import os
import sys

# Тесты лежат в tests/, а модули приложения — в корне репозитория.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_client
import judge
from judge import PARSE_ERROR_KEY, TAXONOMY
from eval_patient import _block_pass_majority


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


# ══════════════════════════════════════════════════════════════════
# 1. Восстановление JSON
# ══════════════════════════════════════════════════════════════════

def test_extract_json_recovery() -> None:
    print("[1] extract_json — восстановление грязного/оборванного JSON")

    # markdown-обёртка
    r = llm_client.extract_json('```json\n{"A1": {"pass": true, "comment": "ok"}}\n```')
    assert r["A1"]["pass"] is True, r
    _ok("markdown-обёртка снята")

    # незакрытый <think> перед JSON
    r = llm_client.extract_json('<think> рассуждаю тут долго... {"A1": {"pass": false}}')
    assert r["A1"]["pass"] is False, r
    _ok("незакрытый <think> отброшен")

    # одинарные кавычки + trailing comma
    r = llm_client.extract_json("{'A1': {'pass': true}, 'A2': {'pass': false},}")
    assert r["A1"]["pass"] is True and r["A2"]["pass"] is False, r
    _ok("одинарные кавычки и хвостовая запятая отремонтированы")

    # ОБРЫВ на середине (модель не дописала ответ) — префикс обязан сохраниться
    truncated = ('{"A1": {"pass": true, "comment": "верно"}, '
                 '"A2": {"pass": false, "comment": "число искажено"}, '
                 '"A3": {"pass": tr')
    r = llm_client.extract_json(truncated)
    assert r.get("A1", {}).get("pass") is True, r
    assert r.get("A2", {}).get("pass") is False, r
    _ok("оборванный ответ: A1/A2 восстановлены (недописанный A3 отброшен)")


def test_salvage_multiple_fields() -> None:
    print("[2] _salvage_json_object — сбор верхнеуровневых полей до обрыва")
    # Два целых объекта-подкритерия, затем битое значение: собрать оба целых.
    raw = '{"A1": {"pass": true}, "A2": {"pass": false}, "A3": {"pass": zzz'
    salv = llm_client._salvage_json_object(raw)
    assert salv is not None, "салваж вернул None"
    assert salv.get("A1", {}).get("pass") is True, salv
    assert salv.get("A2", {}).get("pass") is False, salv
    _ok("A1 и A2 собраны из ответа с битым A3")


# ══════════════════════════════════════════════════════════════════
# 3. Пер-блочная изоляция в score_round1
# ══════════════════════════════════════════════════════════════════

def test_score_round1_isolation() -> None:
    print("[3] score_round1 — сбой ОДНОГО блока не обнуляет весь отчёт судьи")

    original = judge.score_block

    def fake_score_block(panel, role, block, ctx):
        if block == "C":
            raise llm_client.LLMError("смоделированный сбой JSON в блоке C")
        return {c: {"pass": True, "comment": "ok"} for c in TAXONOMY[block]}

    judge.score_block = fake_score_block
    try:
        report = judge.score_round1(panel=None, role="judge_1", ctx=None)
    finally:
        judge.score_block = original

    # Все 5 блоков присутствуют
    assert set(report) == set(TAXONOMY), report.keys()
    # C — sentinel: помечен _parse_error, подкритерии pass=None (НЕ False!)
    assert report["C"].get(PARSE_ERROR_KEY) is True, report["C"]
    assert report["C"]["C1"]["pass"] is None, report["C"]["C1"]
    # Остальные блоки СОХРАНИЛИ реальные оценки (не обнулены)
    assert report["A"]["A1"]["pass"] is True, report["A"]
    assert report["E"]["E1"]["pass"] is True, report["E"]
    _ok("блок C → sentinel (pass=None); A/B/D/E сохранили реальные оценки")


# ══════════════════════════════════════════════════════════════════
# 4. Мажоритарный подсчёт: sentinel = «н/д», не «0/N»
# ══════════════════════════════════════════════════════════════════

def test_block_pass_not_zeroed() -> None:
    print("[4] _block_pass_majority — sentinel не превращается в фальшивый «0/N»")

    def real_report():
        return {b: {c: {"pass": True, "comment": ""} for c in TAXONOMY[b]} for b in TAXONOMY}

    # Три судьи, у всех блок C — sentinel (сбой), блок A — реальные pass.
    reports = [real_report() for _ in range(3)]
    for rep in reports:
        rep["C"] = judge._sentinel_block("C", "сбой")

    assert _block_pass_majority(reports, "A", TAXONOMY["A"]) == "3/3", "блок A обнулён!"
    assert _block_pass_majority(reports, "C", TAXONOMY["C"]) == "н/д", "sentinel не помечен н/д"
    _ok("A=3/3 (не обнулён), C=н/д (сбой помечен явно, не 0/2)")

    # Сбой лишь у ОДНОГО из трёх судей по блоку A — большинство сохраняет оценку.
    mixed = [real_report(), real_report(), real_report()]
    mixed[0]["A"] = judge._sentinel_block("A", "сбой у одного судьи")
    assert _block_pass_majority(mixed, "A", TAXONOMY["A"]) == "3/3", mixed[0]["A"]
    _ok("сбой у 1 из 3 судей по блоку A → мажоритарно всё равно 3/3")


def main() -> None:
    print("=" * 66)
    print("РЕГРЕССИЯ: «неверный JSON → оценки обнулялись» — проверка фикса")
    print("=" * 66)
    test_extract_json_recovery()
    test_salvage_multiple_fields()
    test_score_round1_isolation()
    test_block_pass_not_zeroed()
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅")


if __name__ == "__main__":
    main()
