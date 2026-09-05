"""
test_objective_grounding.py — офлайн-тест заземления судей и извлечения
сущностей из длинных ЭМК (без LLM).

Два измеренных дефекта v1:

1. ШУМ В ПРОМПТЕ БЛОКА E. В прогоне eval_patient_zh1 у ВСЕХ 17 кандидатов
   16 «жёстких находок» совпадали побайтово, и все были вида «сущность не
   найдена» — то есть законные пропуски при сжатии 3.4:1 (сводка ~543 симв.
   против источника ~1857). Реальная находка присутствовала лишь в 3 строках
   из 17: сигнал/шум ≈ 1:17. Весь список уходил в промпт блока E с указанием
   «каждое ОБЯЗАТЕЛЬНО разбери: опасность (->E1) или пропуск (->E2)», то есть
   судье выдавалось 17 одинаковых готовых аргументов за E2.pass=false.
   Заземление, идентичное для эталона и для антонимической инверсии, не может
   различать кандидатов — только смещать всех.

2. ОБРЕЗКА ИСХОДНИКА. `text[:6000]` стояло и в промпте, и в ключе кэша. На
   прод-корпусе (26 ЭМК, 13004-32527 симв., медиана 21107) обрезались ВСЕ 26
   карт, в среднем терялось 72 % текста. При scope="radiologist" entity_recall —
   единственное действующее правило шлюза, и его знаменатель считался по первой
   трети карты.

Запуск:  python tests/test_objective_grounding.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import gate
import judge
import objective_layer


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _report(*, missing=(), numeric_mismatches=()):
    return objective_layer.ObjectiveComparisonReport(
        numeric={"mismatches": list(numeric_mismatches), "unit_mismatches": [],
                 "total_in_a": 0, "matched": 0,
                 "mismatch_count": len(numeric_mismatches), "unit_mismatch_count": 0},
        polarity={"flips": [], "total_in_a": 0, "matched": 0, "flip_count": 0},
        entities={"symptoms": {"missing_in_b": list(missing), "in_a": len(missing),
                               "in_b": 0, "recall": 0.0, "extra_in_b": []}},
        causality=None, interpretation=None)


def test_findings_are_split_by_kind():
    print("[1] hard_findings разделяет реальные ошибки и компрессию")
    rep = _report(missing=["миома матки", "общий анализ мочи", "дизурические явления"])
    assert rep.hard_findings("real") == [], rep.hard_findings("real")
    assert len(rep.hard_findings("compression")) == 3
    assert len(rep.hard_findings("all")) == 3
    assert rep.n_compression() == 3
    try:
        rep.hard_findings("что-то")
        raise AssertionError("ожидался ValueError на неизвестном kind")
    except ValueError:
        pass
    _ok("непокрытые сущности больше не смешаны с искажёнными фактами")


def test_block_e_grounding_excludes_compression():
    print("[2] Промпт блока E не заполняется пропусками сжатия")
    rep = _report(missing=["миома матки", "общий анализ мочи", "дизурические явления"])
    decision = gate.GateDecision(status="pass", reasons=[],
                                 coverage={"declared_scope": "radiologist"})
    hard, gate_summary = judge._ground_block_e(rep, decision)

    for noise in ("миома матки", "общий анализ мочи", "дизурические явления"):
        assert noise not in hard, f"«{noise}» попало в заземление блока E: {hard}"
    assert "не вошло в суммаризацию: 3" in hard, hard
    assert "ОЖИДАЕМО" in hard, "компрессия должна быть явно названа ожидаемой"
    assert "НЕ предрешает" in gate_summary, "вердикт шлюза не должен подаваться как приговор"
    _ok("пропуски показаны справочно и одной строкой, а не 17 «расхождениями»")


def test_block_e_grounding_keeps_real_errors():
    print("[3] Реальные расхождения по-прежнему доходят до судьи")
    class _F:
        def __init__(self, raw): self.raw, self.context = raw, "контекст фрагмента"
    class _MM:
        def __init__(self): self.fact_a, self.fact_b = _F("56 лет"), _F("46 лет")
    rep = _report(missing=["миома матки"], numeric_mismatches=[_MM()])
    hard, _ = judge._ground_block_e(rep, gate.GateDecision(status="pass", reasons=[], coverage={}))
    assert "56 лет" in hard and "46 лет" in hard, hard
    assert "миома матки" not in hard, hard
    _ok("искажённое число видно, шум отфильтрован")


def test_long_source_is_chunked_not_truncated():
    print("[4] Длинная ЭМК режется на куски, а не обрезается на 6000 символах")
    text = ". ".join(f"Фрагмент номер {i} с клиническими сведениями пациента" for i in range(700))
    assert len(text) > config.ENTITY_CHUNK_CHARS, "тестовый текст должен быть длинным"

    chunks = objective_layer._split_for_extraction(text)
    assert len(chunks) > 1, f"текст {len(text)} симв. не разбит: {len(chunks)} кусок"
    assert all(len(c) <= config.ENTITY_CHUNK_CHARS for c in chunks), \
        [len(c) for c in chunks]
    # Конец текста обязан попасть хотя бы в один кусок — раньше он терялся молча.
    assert text[-80:] in chunks[-1], "хвост исходника потерян при нарезке"
    covered = sum(len(c) for c in chunks)
    assert covered >= len(text), f"покрыто {covered} из {len(text)} символов"
    _ok(f"{len(text)} симв. -> {len(chunks)} кусков, потерь нет")


def test_short_source_is_single_chunk():
    print("[5] Короткий текст обрабатывается одним вызовом, как раньше")
    short = "Пациент направлен на КТ органов брюшной полости с жалобами на боль."
    assert objective_layer._split_for_extraction(short) == [short]
    _ok("лишних LLM-вызовов на коротких текстах не появилось")


def test_merge_deduplicates_overlap():
    print("[6] Слияние кусков снимает дубликаты со стыков")
    merged = objective_layer._merge_entity_lists([
        {"diagnoses": ["Гипертоническая болезнь", "ИБС"]},
        {"diagnoses": ["гипертоническая болезнь", "Миома матки"]},
    ])
    assert merged["diagnoses"] == ["Гипертоническая болезнь", "ИБС", "Миома матки"], \
        merged["diagnoses"]
    _ok("перекрытие кусков не раздувает список сущностей")


def test_cache_key_uses_full_text():
    print("[7] Ключ кэша учитывает весь текст, а не первые 6000 символов")
    base = "Одинаковое начало карты. " * 300      # > 6000 символов
    a, b = base + "Диагноз: гипертония.", base + "Диагноз: сахарный диабет."
    assert len(base) > 6000, len(base)
    ka = objective_layer._split_for_extraction(a)
    kb = objective_layer._split_for_extraction(b)
    assert ka != kb, "две разные карты с общим началом дают одинаковую нарезку"
    _ok("карты с одинаковым началом больше не склеиваются в кэше")


def main():
    print("=" * 62)
    print("Офлайн-тест заземления судей и извлечения сущностей")
    print("=" * 62)
    test_findings_are_split_by_kind()
    test_block_e_grounding_excludes_compression()
    test_block_e_grounding_keeps_real_errors()
    test_long_source_is_chunked_not_truncated()
    test_short_source_is_single_chunk()
    test_merge_deduplicates_overlap()
    test_cache_key_uses_full_text()
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅")


if __name__ == "__main__":
    main()
