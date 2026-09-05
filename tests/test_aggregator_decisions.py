"""
test_aggregator_decisions.py — офлайн-тест РЕШАЮЩЕГО пути (Блок 5), без LLM.

Мотивация: регрессионный набор проекта покрывал путь ОТОБРАЖЕНИЯ
(eval_patient._block_pass_majority -> «н/д»), но не путь, который реально
присваивает категорию. Из-за этого в v1 незамеченными жили три дефекта,
подтверждённые исполнением кода:

  1. judge._err(...) возвращал category="ошибка", а finalize перезаписывал её
     на «Неприемлемо: системный характер проблем». Обрыв TCP, OOM на сервере
     vLLM или HTTP 400 из-за длинного промпта попадали в аудит-лог как
     клинический отказ.
  2. `status = "ok" if not failed and not undetermined else "issues"`
     приравнивал «нет данных» к «провалено»: три оборванных ответа модели
     давали «Неприемлемо» наравне с тремя реально проваленными блоками.
  3. Ветка шага 3 была написана только под gate_status == "rework". Значения
     "reject", None и любые иные молча попадали в ветку «шлюз = pass»: пара,
     отклонённая шлюзом, при отработавших судьях получала «Готово к
     клиническому применению», а decision_path утверждал «шлюз = pass».

Запуск:  python tests/test_aggregator_decisions.py
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aggregator
import config
import judge


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _pass_report() -> dict:
    return {b: {c: {"pass": True, "comment": "ok"} for c in judge.TAXONOMY[b]}
            for b in judge.TAXONOMY}


def _with_sentinel(report: dict, blocks: tuple[str, ...]) -> dict:
    """Отчёт, где перечисленные блоки — заглушки «нет данных» (сбой JSON)."""
    out = copy.deepcopy(report)
    for b in blocks:
        out[b] = {c: {"pass": None, "comment": "нет данных: сбой JSON"}
                  for c in judge.TAXONOMY[b]}
        out[b][judge.PARSE_ERROR_KEY] = True
    return out


def _with_failed(report: dict, blocks: tuple[str, ...]) -> dict:
    out = copy.deepcopy(report)
    for b in blocks:
        first = judge.TAXONOMY[b][0]
        out[b][first] = {"pass": False, "comment": "нарушено"}
    return out


def _evaluation(*, gate=None, reports=None, e1=None, category=None) -> dict:
    ev = {
        "gate": {"status": gate} if gate is not None else {},
        "r1": reports or {}, "r2": {}, "r3": {},
        "e1_signals": e1 or {"raised_by_judges": [], "aggregator_flagged": False,
                             "aggregator_named": []},
    }
    if category:
        ev["category"] = category
    return ev


def _three(report: dict) -> dict:
    return {"judge_1": report, "judge_2": report, "judge_3": report}


# ── 1. Инфраструктурный сбой ──────────────────────────────────────

def test_exception_is_not_a_clinical_verdict():
    print("[1] Исключение прогона НЕ становится клиническим вердиктом")
    err = judge._err("EMR_01", "2", "Прогон прерван исключением: HTTPError 400")
    assert err["category"] == aggregator.CATEGORY_ERROR, err["category"]
    out = aggregator.finalize(err)
    assert out["category"] == aggregator.CATEGORY_ERROR, (
        f"finalize перезаписал «ошибка» на {out['category']!r} — это и был дефект v1")
    assert out["e1_triggered"] is False
    assert "не оценивалась" in out["decision_path"][0].lower(), out["decision_path"]
    _ok("category='ошибка' сохранена, E1 не выставлен, трасса объясняет причину")


# ── 2. «Нет данных» != «провалено» ────────────────────────────────

def test_nodata_blocks_do_not_become_reject():
    print("[2] Оборванные ответы судей -> «Оценка неполна», а не «Неприемлемо»")
    sent3 = _with_sentinel(_pass_report(), ("A", "B", "C"))
    out = aggregator.finalize(_evaluation(gate="pass", reports=_three(sent3)))
    assert out["category"] == aggregator.CATEGORY_INCOMPLETE, out["category"]
    assert set(out["nodata_blocks"]) == {"A", "B", "C"}, out["nodata_blocks"]
    assert out["blocks"]["A"]["status"] == "no_data"
    assert out["blocks"]["A"]["parse_error_roles"] == ["judge_1", "judge_2", "judge_3"], \
        out["blocks"]["A"]["parse_error_roles"]
    assert "перепрогн" in out["verdict"], out["verdict"]
    _ok("3 блока без данных -> «Оценка неполна» (в v1 было «Неприемлемо»)")

    # Настоящие провалы по-прежнему дают «Неприемлемо».
    failed3 = _with_failed(_pass_report(), ("A", "B", "C"))
    out2 = aggregator.finalize(_evaluation(gate="pass", reports=_three(failed3)))
    assert out2["category"] == aggregator.CATEGORY_REJECT, out2["category"]
    _ok("3 РЕАЛЬНО проваленных блока -> «Неприемлемо» (чувствительность сохранена)")


def test_nodata_within_threshold_is_tolerated():
    print("[3] Один блок без данных — в пределах порога, но не «Готово»")
    one = _with_sentinel(_pass_report(), ("D",))
    out = aggregator.finalize(_evaluation(gate="pass", reports=_three(one)))
    assert out["category"] == aggregator.CATEGORY_EDIT, out["category"]
    assert out["nodata_blocks"] == ["D"], out["nodata_blocks"]
    _ok("непроверенный блок не позволяет объявить «Готово», но и не даёт отказа")


# ── 3. Шлюз — сигнал, а не фильтр ─────────────────────────────────

def test_gate_reject_never_yields_ready():
    print("[4] gate=reject при отработавших судьях -> не «Готово»")
    out = aggregator.finalize(_evaluation(gate="reject", reports=_three(_pass_report())))
    assert out["category"] == aggregator.CATEGORY_EDIT, (
        f"получено {out['category']!r} — в v1 здесь было «Готово к клиническому применению»")
    joined = " ".join(out["decision_path"])
    assert "шлюз = pass" not in joined, f"трасса лжёт о статусе шлюза: {joined}"
    assert "отклонил" in joined, joined
    _ok("потолок опущен до «Требует редактирования», трасса не лжёт")


def test_gate_reject_plus_failures_is_reject():
    print("[5] gate=reject + провалы у судей -> «Неприемлемо» (совпадение сигналов)")
    failed2 = _with_failed(_pass_report(), ("A", "B"))
    out = aggregator.finalize(_evaluation(gate="reject", reports=_three(failed2)))
    assert out["category"] == aggregator.CATEGORY_REJECT, out["category"]
    _ok("два независимых сигнала сходятся -> отказ")


def test_unknown_gate_status_is_explicit():
    print("[6] Неизвестный/отсутствующий статус шлюза отражается в трассе честно")
    for gate, marker in ((None, "не запускался"), ("странно", "неизвестный статус")):
        out = aggregator.finalize(_evaluation(gate=gate, reports=_three(_pass_report())))
        joined = " ".join(out["decision_path"])
        assert marker in joined, f"gate={gate!r}: {joined}"
        assert out["category"] == aggregator.CATEGORY_READY, out["category"]
    _ok("None и мусорный статус больше не выдаются за «шлюз = pass»")


# ── 4. E1 ─────────────────────────────────────────────────────────

def test_e1_still_stops_everything():
    print("[7] Подтверждённый E1 по-прежнему фиксирует «Неприемлемо» немедленно")
    e1 = {"raised_by_judges": ["judge_2"], "aggregator_flagged": False, "aggregator_named": []}
    out = aggregator.finalize(_evaluation(gate="pass", reports=_three(_pass_report()), e1=e1))
    assert out["category"] == aggregator.CATEGORY_REJECT, out["category"]
    assert out["e1_triggered"] is True and out["e1_sources"] == ["judge_2"]
    _ok("предохранитель безопасности не ослаблен")


def test_disputed_e1_is_visible_in_path():
    print("[8] Неподтверждённый цитатой E1 агрегатора отражён в трассе")
    e1 = {"raised_by_judges": [], "aggregator_flagged": False, "aggregator_named": [],
          "disputed_by_aggregator": True}
    out = aggregator.finalize(_evaluation(gate="pass", reports=_three(_pass_report()), e1=e1))
    assert out["e1_triggered"] is False
    assert any("не подтвердил" in p for p in out["decision_path"]), out["decision_path"]
    _ok("расхождение агрегатор/судьи видно в трассе решения")


# ── 5. Устойчивость ───────────────────────────────────────────────

def test_none_block_does_not_crash():
    print("[9] Блок со значением null не роняет прогон")
    broken = copy.deepcopy(_pass_report())
    broken["C"] = None                      # модель прислала null
    out = aggregator.finalize(_evaluation(gate="pass", reports=_three(broken)))
    assert out["blocks"]["C"]["status"] == "no_data", out["blocks"]["C"]
    _ok("AttributeError не возникает, блок честно помечен «нет данных»")


def main():
    print("=" * 62)
    print("Офлайн-тест решающего пути агрегатора (Блок 5)")
    print("=" * 62)
    assert config.MAX_NODATA_BLOCKS == 1, "тест рассчитан на порог 1"
    test_exception_is_not_a_clinical_verdict()
    test_nodata_blocks_do_not_become_reject()
    test_nodata_within_threshold_is_tolerated()
    test_gate_reject_never_yields_ready()
    test_gate_reject_plus_failures_is_reject()
    test_unknown_gate_status_is_explicit()
    test_e1_still_stops_everything()
    test_disputed_e1_is_visible_in_path()
    test_none_block_does_not_crash()
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅")


if __name__ == "__main__":
    main()
