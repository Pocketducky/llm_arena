"""
test_gate_rules.py — офлайн-тест правил pre-evaluation gate (без LLM).

Шлюз в v1 имел три класса дефектов, подтверждённых измерением:

  1. ДЕГРАДИРОВАЛ «ОТКРЫТО». Проверок на пустоту, длину и язык не было вовсе.
     При scope="radiologist" правила по числам и предикатам не выполняются, и
     если извлечение сущностей вернуло пустоту, у шлюза не оставалось НИ ОДНОГО
     действующего правила: пустая строка и английский текст получали pass.
  2. МЁРТВОЕ ПРАВИЛО. Стем "обнаруж" (7 символов) сравнивался с ключами,
     обрезанными до 6 (objective_layer._predicate_counts), поэтому пятая часть
     правила критических предикатов не срабатывала никогда.
  3. НЕВАЛИДИРУЕМЫЙ scope. Опечатка вроде "radiologists" молча включала строгий
     порог entity_recall по всем шести категориям БЕЗ фильтра релевантности —
     ровно тот системный ложный отказ, ради предотвращения которого scope вводился.

Плюс мина, которую важно не «починить»: schema_incomplete СЧИТАЕТСЯ, но
намеренно не попадает в причины (0 из 5 разделов распознаётся на 380 из 380
коротких нарративных суммаризаций — включение дало бы 100 % отказ).

Запуск:  python tests/test_gate_rules.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gate
import objective_layer

SRC = ("Пациент направлен на КТ органов брюшной полости. При обследовании "
       "обнаружено образование печени. Диагностирована гипертоническая болезнь. "
       "Выявлены изменения в лёгких. Жалобы на боль в правом подреберье.")

GOOD = ("Пациент направлен на КТ ОБП. Обнаружено образование печени, выявлены "
        "изменения в лёгких. Диагностирована гипертоническая болезнь. "
        "Жалобы на боль в правом подреберье.")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def test_degenerate_inputs_are_closed():
    print("[1] Вырожденный вход отсекается при ЛЮБОМ scope")
    cases = {
        "": "empty_summary",
        "   \n\t ": "empty_summary",
        "ок": "empty_summary",
        "The patient is stable and reports no complaints at this time.": "wrong_language",
    }
    for text, expected_code in cases.items():
        for scope in (None, "radiologist"):
            d = gate.evaluate_gate(SRC, text, scope=scope, panel=None)
            assert d.status == "reject", f"{text!r} scope={scope}: {d.status}"
            assert d.is_degenerate(), f"{text!r}: не помечен как вырожденный вход"
            assert expected_code in d.reason_codes(), (text, d.reason_codes())
    _ok("пустой, пробельный, слишком короткий и англоязычный текст закрыты")


def test_normal_summary_is_not_degenerate():
    print("[2] Нормальная суммаризация не помечается вырожденной")
    for scope in (None, "radiologist"):
        d = gate.evaluate_gate(SRC, GOOD, scope=scope, panel=None)
        assert not d.is_degenerate(), (scope, d.reason_codes())
    _ok("осмысленный русский текст проходит проверку входа")


def test_obnaruzh_predicate_rule_is_alive():
    print("[3] Правило predicate_coverage_low:обнаруж больше не мертво")
    keys = gate._predicate_counts(objective_layer.extract_polarity_facts(SRC))
    assert "обнару" in keys, f"нормализация изменилась: {keys}"
    assert "обнаруж" not in keys, "ключ внезапно стал полным — сверьте PREDICATE_KEY_LEN"

    empty = ("Пациент направлен на компьютерную томографию брюшной полости "
             "по направлению лечащего врача поликлиники.")
    d = gate.evaluate_gate(SRC, empty, scope=None, panel=None)
    assert "predicate_coverage_low:обнаруж" in d.reason_codes(), d.reason_codes()
    _ok("стем длиной 7 символов сопоставляется с ключом длиной 6")


def test_scope_validation():
    print("[4] Неизвестный scope не проходит молча")
    gate.validate_scope(None)
    gate.validate_scope("radiologist")
    for bad in ("radiologists", "Radiologist", "cardiologist", "radiolog"):
        try:
            gate.validate_scope(bad)
            raise AssertionError(f"scope {bad!r} принят молча")
        except SystemExit as e:
            assert "Неизвестный scope" in str(e), str(e)
    _ok("опечатка и неподдерживаемая специальность падают с объяснением")


def test_schema_reason_stays_informational():
    print("[5] schema_incomplete НЕ добавляется в причины (мина)")
    narrative = ("Больная поступила с жалобами на боль в правом подреберье, "
                 "тошноту и слабость. Направлена на КТ органов брюшной полости "
                 "для уточнения характера образования печени.")
    reason = gate._schema_reason(narrative)
    assert reason is not None, "на нарративном тексте разделы схемы не распознаются — ожидаемо"
    assert reason.severity == "reject", reason.severity

    d = gate.evaluate_gate(SRC, narrative, scope="radiologist", panel=None)
    assert "schema_incomplete" not in d.reason_codes(), (
        "schema_incomplete попал в причины: на нарративном корпусе это даёт "
        "100 % отказ — см. комментарий в gate.evaluate_gate")
    assert d.coverage.get("schema_note"), "сигнал должен сохраняться в coverage для отчёта"
    _ok("сигнал остаётся информационным и виден в coverage")


def test_coverage_is_populated():
    print("[6] coverage заполняется и годится для отчёта")
    d = gate.evaluate_gate(SRC, GOOD, scope="radiologist", panel=None)
    for key in ("declared_scope", "numeric_counts_source", "numeric_counts_candidate",
                "predicate_counts_source", "predicate_counts_candidate"):
        assert key in d.coverage, f"нет ключа {key}: {sorted(d.coverage)}"
    assert d.coverage["declared_scope"] == "radiologist"
    _ok("покрытие шлюза доступно вызывающему коду (в v1 терялось в judge.py)")


def main():
    print("=" * 62)
    print("Офлайн-тест правил pre-evaluation gate")
    print("=" * 62)
    test_degenerate_inputs_are_closed()
    test_normal_summary_is_not_degenerate()
    test_obnaruzh_predicate_rule_is_alive()
    test_scope_validation()
    test_schema_reason_stays_informational()
    test_coverage_is_populated()
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅")


if __name__ == "__main__":
    main()
