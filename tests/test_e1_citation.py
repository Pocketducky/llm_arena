"""
test_e1_citation.py — офлайн-тест стоп-правила E1 (без LLM).

В v1 правило было чистой дизъюнкцией: флаг ЛЮБОГО из четырёх источников
(3 судьи после merge R1/R2 + агрегатор R3) необратимо фиксировал «Неприемлемо».
`aggregator_flagged` при этом — невалидированный булев флаг ОДНОГО LLM-вызова:
если R3 галлюцинировал e1_triggered=true, вердикт выносился с источником
«агрегатор без указания судей».

Результат на пилоте измерим: E1 сработал в 38 случаях из 38, включая эталонную
суммаризацию (строка 1) и benign-искажения уровня опечатки, пунктуации и
синонима. Предохранитель, срабатывающий всегда, не несёт информации.

Теперь голос судьи засчитывается, только если он привёл конкретный фрагмент
суммаризации как опасный И этот фрагмент в ней действительно есть — проверяет
КОД, а не LLM. Флаг агрегатора в одиночку лишь помечает расхождение.

Запуск:  python tests/test_e1_citation.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aggregator
import config
import judge

SUMMARY = ("Пациентка госпитализирована в отделение кардиореанимации. "
           "Назначен нитроглицерин сублингвально при болях за грудиной. "
           "Гемоглобин 126 г/л, креатинфосфокиназа повышена.")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _report(*, e1_pass=True, danger=None) -> dict:
    rep = {b: {c: {"pass": True, "comment": "ok"} for c in judge.TAXONOMY[b]}
           for b in judge.TAXONOMY}
    rep["E"]["E1"] = {"pass": e1_pass, "comment": "проверено",
                      "danger_examples": list(danger or [])}
    return rep


def test_citation_matcher():
    print("[1] Проверка цитаты: код сверяет фрагмент с текстом суммаризации")
    cases = [
        ("нитроглицерин сублингвально", True, "дословная цитата"),
        ("госпитализирована в кардиореанимацию", True, "иной падеж и порядок слов"),
        ("креатинфосфокиназы повышение", True, "словоформа"),
        ("назначен инсулин 40 единиц внутривенно", False, "выдуманное назначение"),
        ("пациентке отменены все препараты и рекомендована выписка", False, "выдуманный фрагмент"),
        ("", False, "пустая цитата"),
    ]
    for quote, expected, label in cases:
        got = judge._citation_supported(quote, SUMMARY)
        assert got is expected, f"{label}: {quote!r} -> {got}, ожидалось {expected}"
    _ok("падежи и перестановки принимаются, выдуманные фрагменты отвергаются")


def test_judge_without_citation_is_not_counted():
    print("[2] Судья поднял E1 без цитаты — голос не засчитан")
    r1 = {"judge_1": _report(e1_pass=False, danger=[])}
    sig = judge._collect_e1_signals(r1, {}, {}, SUMMARY)
    assert sig["raised_by_judges"] == [], sig["raised_by_judges"]
    assert sig["raised_by_judges_raw"] == ["judge_1"]
    assert sig["raised_without_citation"] == ["judge_1"]
    out = aggregator.finalize({"gate": {"status": "pass"}, "r1": r1, "r2": {}, "r3": {},
                               "e1_signals": sig})
    assert out["category"] != aggregator.CATEGORY_REJECT, out["category"]
    _ok("флаг без доказательства не фиксирует «Неприемлемо», но виден в диагностике")


def test_judge_with_fabricated_citation_is_not_counted():
    print("[3] Цитата, которой нет в суммаризации, не засчитывается")
    r1 = {"judge_1": _report(e1_pass=False, danger=["назначен инсулин 40 единиц внутривенно"])}
    sig = judge._collect_e1_signals(r1, {}, {}, SUMMARY)
    assert sig["raised_by_judges"] == [], sig["raised_by_judges"]
    assert sig["unverifiable_citations"]["judge_1"], sig["unverifiable_citations"]
    _ok("галлюцинированная «опасность» отсеяна кодом")


def test_judge_with_real_citation_triggers():
    print("[4] Подтверждённая цитата фиксирует «Неприемлемо»")
    r1 = {"judge_1": _report(e1_pass=False, danger=["нитроглицерин сублингвально"])}
    sig = judge._collect_e1_signals(r1, {}, {}, SUMMARY)
    assert sig["raised_by_judges"] == ["judge_1"], sig["raised_by_judges"]
    assert sig["citations"]["judge_1"] == ["нитроглицерин сублингвально"]
    out = aggregator.finalize({"gate": {"status": "pass"}, "r1": r1, "r2": {}, "r3": {},
                               "e1_signals": sig})
    assert out["category"] == aggregator.CATEGORY_REJECT, out["category"]
    assert out["e1_triggered"] is True
    _ok("чувствительность предохранителя сохранена")


def test_aggregator_alone_cannot_trigger():
    print("[5] Флаг агрегатора без единого судьи — расхождение, а не вердикт")
    r1 = {"judge_1": _report(), "judge_2": _report(), "judge_3": _report()}
    r3 = {"e1_triggered": True, "e1_triggered_by": [], "category": "Неприемлемо"}
    sig = judge._collect_e1_signals(r1, {}, r3, SUMMARY)
    assert sig["aggregator_flagged"] is False, "одиночный флаг R3 не должен засчитываться"
    assert sig["disputed_by_aggregator"] is True
    assert sig["aggregator_raw_flag"] is True, "сырой флаг обязан сохраниться для отчёта"
    out = aggregator.finalize({"gate": {"status": "pass"}, "r1": r1, "r2": {}, "r3": r3,
                               "e1_signals": sig})
    assert out["category"] != aggregator.CATEGORY_REJECT, out["category"]
    assert any("не подтвердил" in p for p in out["decision_path"]), out["decision_path"]
    _ok("невалидированный булев одного LLM-вызова больше не решает исход")


def test_aggregator_plus_judge_triggers():
    print("[6] Флаг агрегатора + голос судьи = два независимых источника")
    r1 = {"judge_1": _report(e1_pass=False, danger=[]), "judge_2": _report()}
    r3 = {"e1_triggered": True, "e1_triggered_by": ["judge_1"]}
    sig = judge._collect_e1_signals(r1, {}, r3, SUMMARY)
    assert sig["aggregator_flagged"] is True, sig
    out = aggregator.finalize({"gate": {"status": "pass"}, "r1": r1, "r2": {}, "r3": r3,
                               "e1_signals": sig})
    assert out["category"] == aggregator.CATEGORY_REJECT, out["category"]
    _ok("совпадение сигналов срабатывает даже без дословной цитаты")


def test_r2_overrides_r1():
    print("[7] Позиция судьи берётся из R2 (пересмотренной), а не из R1")
    r1 = {"judge_1": _report(e1_pass=False, danger=["нитроглицерин сублингвально"])}
    r2 = {"judge_1": _report(e1_pass=True)}       # после ревью судья снял флаг
    sig = judge._collect_e1_signals(r1, r2, {}, SUMMARY)
    assert sig["raised_by_judges"] == [], sig["raised_by_judges"]
    _ok("снятый в R2 флаг не воскресает из R1")


def test_legacy_mode_switch():
    print("[8] EMR_E1_REQUIRE_CITATION=0 возвращает прежнее поведение (для A/B)")
    saved = config.E1_REQUIRE_CITATION
    try:
        config.E1_REQUIRE_CITATION = False
        r1 = {"judge_1": _report(e1_pass=False, danger=[])}
        sig = judge._collect_e1_signals(r1, {}, {"e1_triggered": True}, SUMMARY)
        assert sig["raised_by_judges"] == ["judge_1"], sig
        assert sig["aggregator_flagged"] is True
        assert sig["require_citation"] is False
    finally:
        config.E1_REQUIRE_CITATION = saved
    _ok("переключатель работает — можно сравнить два режима на синтетике")


def main():
    print("=" * 62)
    print("Офлайн-тест стоп-правила E1 (проверяемая цитата)")
    print("=" * 62)
    assert config.E1_REQUIRE_CITATION, "тест рассчитан на включённую проверку цитаты"
    test_citation_matcher()
    test_judge_without_citation_is_not_counted()
    test_judge_with_fabricated_citation_is_not_counted()
    test_judge_with_real_citation_triggers()
    test_aggregator_alone_cannot_trigger()
    test_aggregator_plus_judge_triggers()
    test_r2_overrides_r1()
    test_legacy_mode_switch()
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅")


if __name__ == "__main__":
    main()
