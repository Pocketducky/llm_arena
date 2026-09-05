"""
aggregator.py — Блок 5: детерминированная агрегация и итоговое решение.

Полностью заменяет старую формулу `final_score = positive + penalties` и
логику выбора арбитра «у кого балл не ноль» из прежней монолитной версии — то
есть саму идею, что финальный вердикт выносит модель, а код лишь форматирует её
ответ. Здесь — наоборот: единственный источник итоговой категории — этот
прозрачный, воспроизводимый код поверх СТРУКТУРИРОВАННЫХ данных Блоков 1-4
(объективный слой, шлюз, бинарные подкритерии трёх раундов судей).

LLM (агрегатор раунда 3, judge.py) по-прежнему синтезирует свою версию
категории/вердикта — она сохраняется в выводе под именем `llm_category`/
`llm_verdict` для аудита и сравнения (Блок 6/7 могут измерять, насколько
часто и в чём код расходится с самооценкой модели), но НЕ является финальной.
Финальные `category`/`verdict` всегда вычисляются здесь.

Порядок решения (см. План, Блок 5):
  1. Стоп-правило E1 — ПЕРВЫМ и безусловно. Если флаг поднял хоть один
     источник (любой судья — в исходном R1 или в уточнённом R2-отчёте —
     либо сам агрегатор в R3), категория «Неприемлемо» фиксируется
     немедленно; шаги 2-3 не пересматривают и не отменяют это решение.
  2. Раздельные статус/балл по каждому блоку таксономии A-E — мажоритарное
     голосование судей по их ФИНАЛЬНОЙ позиции (R2, если он состоялся —
     это «уточнённый» отчёт после содержательного cross-peer-review,
     иначе R1; тот же принцип `{**r1, **r2}`, что и в judge._collect_e1_signals).
  3. Итоговая категория — явная таблица решающих правил поверх (1), (2)
     и решения шлюза (Блок 3): «Готово к клиническому применению» только
     если шлюз = pass и ни в одном блоке нет замечаний; «Неприемлемо» —
     если замечания системные (≥3 блоков из 5) или шлюз потребовал
     доработки при множественных проблемах; иначе — «Требует редактирования».
"""

from __future__ import annotations

from typing import Optional

import config
import judge

# Версия таблицы решающих правил — Блок 7 («версии… порогов» в audit-логе).
# Поднимается вручную при изменении самой логики агрегации (не порогов
# внутри gate.py/objective_layer.py — те снимаются audit.py отдельным
# срезом, см. _thresholds_snapshot там же).
DECISION_TABLE_VERSION = "aggregator-decision-table-v2"

CATEGORY_READY  = "Готово к клиническому применению"
CATEGORY_EDIT   = "Требует редактирования"
CATEGORY_REJECT = "Неприемлемо"

# СЛУЖЕБНЫЕ статусы — принципиально НЕ клинические вердикты.
#
# Диагностика пилота: судья, чей ответ оборвался по лимиту токенов, давал
# подкритерии со значением None («нет данных»). Прежняя строка
#     status = "ok" if not failed and not undetermined else "issues"
# приравнивала «нет данных» к «провален», три таких блока давали
# len(issues) >= 3 и категорию «Неприемлемо». Так артефакт бюджета токенов
# становился клиническим отказом, неотличимым в отчёте от реальных претензий.
#
# Ещё хуже был путь исключений: judge._err честно возвращал category="ошибка",
# но finalize её ПЕРЕЗАПИСЫВАЛ — обрыв TCP, OOM или HTTP 400 попадали в
# аудит-лог как «Неприемлемо: системный характер проблем».
CATEGORY_INCOMPLETE = "Оценка неполна"
CATEGORY_ERROR      = "ошибка"

CATEGORIES = (CATEGORY_READY, CATEGORY_EDIT, CATEGORY_REJECT)
# Полный набор значений, которые может принять поле category в отчёте.
ALL_CATEGORIES = (*CATEGORIES, CATEGORY_INCOMPLETE, CATEGORY_ERROR)
# Служебные статусы: их нельзя смешивать с клиническими при подсчёте статистики.
SERVICE_CATEGORIES = (CATEGORY_INCOMPLETE, CATEGORY_ERROR)


# ══════════════════════════════════════════════════════════════════
# ШАГ 2: мажоритарное голосование судей по бинарным подкритериям
# ══════════════════════════════════════════════════════════════════

def _final_reports(r1: dict, r2: dict) -> dict[str, dict]:
    """Финальная позиция каждого судьи: его R2-отчёт («уточнённый» —
    результат содержательного cross-peer-review), если он состоялся,
    иначе исходный R1. Намеренно тот же merge `{**r1, **r2}`, которым
    в judge._collect_e1_signals уже выражено «R2 — более авторитетное,
    пересмотренное мнение судьи о самом себе»."""
    merged = {**r1, **r2}
    return {role: rep for role, rep in merged.items() if isinstance(rep, dict)}


def _subcriterion_vote(reports: dict[str, dict], block: str, code: str) -> Optional[bool]:
    """Большинство голосов валидных судей по одному подкритерию.
    None — если ни один судья не вернул по нему валидный ответ
    (структура отчётов гарантирована _validate_block/_validate_full_report
    в judge.py, так что это вырожденный случай — но явная обработка лучше
    тихого «pass по умолчанию»)."""
    votes = []
    for rep in reports.values():
        # rep[block] может быть None (модель прислала null) — .get() на None
        # уронил бы многочасовой прогон с AttributeError.
        blk = rep.get(block)
        if not isinstance(blk, dict):
            continue
        sub = blk.get(code)
        if isinstance(sub, dict) and isinstance(sub.get("pass"), bool):
            votes.append(sub["pass"])
    if not votes:
        return None
    return sum(votes) > len(votes) / 2


def _block_verdict(reports: dict[str, dict], block: str) -> dict:
    """Раздельный статус/балл по одному блоку таксономии (A-E).

      • score  — доля подкритериев, прошедших мажоритарную проверку (0-100);
                 None, если ни по одному нет данных;
      • status — ТРИ значения, а не два:
            "ok"      — каждый подкритерий подтверждён большинством голосов;
            "issues"  — есть подкритерии, ПРОВАЛЕННЫЕ большинством;
            "no_data" — провалов нет, но часть подкритериев осталась без
                        валидных оценок (сбой JSON, таймаут, обрыв ответа).

    Разделение «issues» и «no_data» — суть исправления. Раньше оба случая
    сливались в «issues», и три оборванных ответа модели давали клиническое
    «Неприемлемо» наравне с тремя реально проваленными блоками. «Нет данных» —
    это отсутствие измерения, а не отрицательный результат измерения; оно
    обязано вести к «оценка неполна, нужен перепрогон», а не к вердикту.

    Признак `parse_error` берётся из judge.PARSE_ERROR_KEY, который
    judge._sentinel_block проставлял, но не читал НИКТО (вопреки docstring).
    """
    codes = judge.TAXONOMY[block]
    votes = {code: _subcriterion_vote(reports, block, code) for code in codes}
    failed       = [c for c, v in votes.items() if v is False]
    undetermined = [c for c, v in votes.items() if v is None]
    decided      = [v for v in votes.items() if v[1] is not None]
    score = round(100 * sum(1 for _, v in decided if v) / len(decided), 1) if decided else None

    if failed:
        status = "issues"
    elif undetermined:
        status = "no_data"
    else:
        status = "ok"

    # Кто из судей вообще не смог отдать разбираемый отчёт по этому блоку.
    parse_errors = sorted(
        role for role, rep in reports.items()
        if isinstance(rep.get(block), dict) and rep[block].get(judge.PARSE_ERROR_KEY)
    )
    return {
        "score": score, "status": status, "votes": votes,
        "failed_subcriteria": failed, "undetermined_subcriteria": undetermined,
        "parse_error_roles": parse_errors,
    }


# ══════════════════════════════════════════════════════════════════
# ШАГ 1 + 3: стоп-правило E1 и итоговая таблица решающих правил
# ══════════════════════════════════════════════════════════════════

def _e1_sources(e1_signals: dict) -> list[str]:
    return sorted(set(e1_signals.get("raised_by_judges", ()))
                  | set(e1_signals.get("aggregator_named", ())))


def _decide(gate_status: Optional[str], e1_signals: dict,
            blocks: dict[str, dict]) -> tuple[str, bool, list[str], list[str]]:
    """Возвращает (категория, e1_triggered, e1_sources, decision_path).

    Порядок: (1) стоп-правило E1, (2) полнота самой оценки, (3) таблица правил
    поверх голосов судей и решения шлюза.

    Что изменилось против v1:
      • «нет данных» больше не приравнивается к «провалено» — блоки без валидных
        оценок ведут в служебную категорию «Оценка неполна», а не в «Неприемлемо»;
      • шлюз стал СИГНАЛОМ, а не фильтром: судьи запускаются всегда, а reject
        лишь опускает потолок категории. Раньше ветка была написана только под
        gate_status == "rework", поэтому "reject", None и любое иное значение
        молча попадали в ветку «шлюз = pass» — пара, отклонённая шлюзом, при
        отработавших судьях получала «Готово к клиническому применению», а
        decision_path утверждал «шлюз = pass». Проверено исполнением на v1.
    """
    path: list[str] = []

    # ── Шаг 1 — стоп-правило E1, ПЕРВЫМ и безусловно ─────────────
    e1_triggered = bool(e1_signals.get("raised_by_judges")) or bool(e1_signals.get("aggregator_flagged"))
    sources = _e1_sources(e1_signals)
    if e1_triggered:
        path.append(f"Шаг 1 — СТОП-ПРАВИЛО E1 сработало (источники: {', '.join(sources) or 'агрегатор без указания судей'}); "
                    f"категория «{CATEGORY_REJECT}» фиксируется немедленно, шаги 2-3 не пересматривают это решение")
        return CATEGORY_REJECT, True, sources, path
    disputed = e1_signals.get("disputed_by_aggregator")
    if disputed:
        path.append("Шаг 1 — агрегатор R3 поднял E1, но НИ ОДИН судья не подтвердил его "
                    "проверяемой цитатой из суммаризации; флаг не засчитан (см. лист «Критические»)")
    else:
        path.append("Шаг 1 — стоп-правило E1 не сработало ни у одного источника (судьи R1/R2, агрегатор R3)")

    # ── Шаг 2 — полнота самой оценки ─────────────────────────────
    failed_blocks = [b for b, v in blocks.items() if v["status"] == "issues"]
    nodata_blocks = [b for b, v in blocks.items() if v["status"] == "no_data"]
    if len(nodata_blocks) > config.MAX_NODATA_BLOCKS:
        path.append(f"Шаг 2 — оценка НЕПОЛНА: по {len(nodata_blocks)} блок(ам) "
                    f"({', '.join(nodata_blocks)}) нет валидных оценок судей "
                    f"(порог — не более {config.MAX_NODATA_BLOCKS}). Это сбой разбора "
                    f"ответа модели, а не утверждение о качестве суммаризации: клинический "
                    f"вердикт не выносится, пара требует перепрогона")
        return CATEGORY_INCOMPLETE, False, [], path
    if nodata_blocks:
        path.append(f"Шаг 2 — блок(и) {', '.join(nodata_blocks)} без данных (в пределах "
                    f"допустимого порога {config.MAX_NODATA_BLOCKS}); учитываются только "
                    f"блоки с валидными голосами")
    path.append(f"Шаг 2 — мажоритарное голосование судей по блокам A-E: "
                f"{'провалов нет' if not failed_blocks else 'провалы в ' + ', '.join(failed_blocks)}")

    # ── Шаг 3 — итоговая категория ───────────────────────────────
    # Шлюз не отсекает пары (решение НПКЦ: судьи идут всегда), но его вердикт
    # ограничивает потолок категории.
    if gate_status in ("reject", "rework"):
        gate_word = "отклонил" if gate_status == "reject" else "потребовал доработки"
        if len(failed_blocks) >= 2:
            path.append(f"Шаг 3 — шлюз (Блок 3) {gate_word}, и судьи независимо подтверждают "
                        f"провалы ещё в {len(failed_blocks)} блок(ах) → совпадение сигналов "
                        f"→ «{CATEGORY_REJECT}»")
            return CATEGORY_REJECT, False, [], path
        path.append(f"Шаг 3 — шлюз (Блок 3) {gate_word} → готовой категория быть не может "
                    f"→ «{CATEGORY_EDIT}»")
        return CATEGORY_EDIT, False, [], path

    if gate_status is None:
        path.append("Шаг 3 — шлюз не запускался (его вердикт не учитывается)")
    elif gate_status != "pass":
        # Неизвестное значение: раньше молча трактовалось как «pass».
        path.append(f"Шаг 3 — шлюз вернул неизвестный статус {gate_status!r}; "
                    f"трактуется как «не ограничивает», решение — по голосам судей")

    if len(failed_blocks) >= 3:
        path.append(f"Шаг 3 — провалы затрагивают {len(failed_blocks)} из 5 блоков таксономии "
                    f"→ системный характер проблем → «{CATEGORY_REJECT}»")
        return CATEGORY_REJECT, False, [], path

    if not failed_blocks:
        if nodata_blocks:
            path.append(f"Шаг 3 — провалов нет, но блок(и) {', '.join(nodata_blocks)} "
                        f"не проверены → «{CATEGORY_EDIT}» (готовой категория быть не может "
                        f"без полной проверки)")
            return CATEGORY_EDIT, False, [], path
        gate_note = "шлюз = pass" if gate_status == "pass" else "шлюз не ограничивает"
        path.append(f"Шаг 3 — {gate_note}, провалов по блокам A-E нет → «{CATEGORY_READY}»")
        return CATEGORY_READY, False, [], path

    path.append(f"Шаг 3 — точечные провалы в {len(failed_blocks)} блок(ах) из 5 → «{CATEGORY_EDIT}»")
    return CATEGORY_EDIT, False, [], path


def _compose_verdict(category: str, blocks: dict[str, dict], e1_sources: list[str]) -> str:
    """Краткое, полностью выводимое из (1)-(3) обоснование — без LLM."""
    if category == CATEGORY_REJECT and e1_sources:
        return (f"«{category}»: сработало стоп-правило E1 — обнаружена клинически опасная ошибка "
                f"(источник(и): {', '.join(e1_sources)}). Дальнейшие оси A-E не пересматривают это решение.")

    nodata = {b: v for b, v in blocks.items() if v["status"] == "no_data"}
    if category == CATEGORY_INCOMPLETE:
        parts = [f"{b} ({', '.join(v['undetermined_subcriteria'])})" for b, v in nodata.items()]
        return (f"«{category}»: судьи не вернули разбираемых оценок по блокам — "
                + "; ".join(parts)
                + ". Это сбой разбора ответа модели, а НЕ утверждение о качестве суммаризации; "
                  "пару нужно перепрогнать.")

    failed = {b: v for b, v in blocks.items() if v["status"] == "issues"}
    if not failed and not nodata:
        return (f"«{category}»: по всем блокам таксономии (A-E) большинство судей подтвердили "
                f"прохождение каждого подкритерия; стоп-правило E1 не сработало.")

    parts = [f"{b} ({', '.join(v['failed_subcriteria'])} — не подтверждены большинством судей)"
             for b, v in failed.items()]
    parts += [f"{b} ({', '.join(v['undetermined_subcriteria'])} — нет валидных оценок судей)"
              for b, v in nodata.items()]
    return f"«{category}»: замечания по блокам — " + "; ".join(parts) + "."


# ══════════════════════════════════════════════════════════════════
# ПУБЛИЧНАЯ ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════

def finalize(evaluation: dict) -> dict:
    """Принимает сырой результат judge.evaluate_summary(...) и выносит
    ОКОНЧАТЕЛЬНОЕ, детерминированное решение поверх его структурированных
    данных. Возвращает новый dict (не мутирует вход):
      • category/verdict      — ФИНАЛЬНЫЕ, авторитетные (вычислены здесь)
      • llm_category/llm_verdict — самооценка агрегатора R3 (judge.py),
                                    сохранена для аудита/сравнения, не финальна
      • blocks                — раздельные score/status/голоса по A-E
      • decision_path         — пошаговая, человекочитаемая трасса решения
      • e1_triggered/e1_sources — итог проверки стоп-правила (шаг 1)
    """
    gate = evaluation.get("gate", {}) or {}
    gate_status = gate.get("status")
    r1, r2, r3 = evaluation.get("r1", {}), evaluation.get("r2", {}), evaluation.get("r3", {}) or {}

    # ── Пара вообще не оценивалась: прогон упал на исключении ─────
    # judge._err честно ставит category="ошибка". Раньше finalize её
    # ПЕРЕЗАПИСЫВАЛ: gate={} => gate_status=None => короткий путь не срабатывал
    # => пустые r1/r2/r3 давали 5 блоков «нет данных» => len(issues)>=3 =>
    # «Неприемлемо: системный характер проблем». То есть обрыв TCP, OOM на
    # сервере vLLM или HTTP 400 из-за длинного промпта записывались в аудит-лог
    # как КЛИНИЧЕСКИЙ ОТКАЗ. Проверено исполнением на v1.
    if evaluation.get("category") == CATEGORY_ERROR:
        return {
            **evaluation,
            "verdict": evaluation.get("verdict", "") or "Пара не оценена: сбой прогона.",
            "llm_category": None, "llm_verdict": None,
            "blocks": {}, "e1_triggered": False, "e1_sources": [],
            "decision_path": [
                "Пара НЕ оценивалась: прогон прерван исключением. Это отказ "
                "инфраструктуры, а не вердикт о качестве суммаризации — "
                "клиническая категория не присваивается, пара идёт в перепрогон.",
                evaluation.get("verdict", "")[:300],
            ],
        }

    # Короткого пути «отклонена шлюзом» больше нет: по решению НПКЦ шлюз —
    # только сигнал, судьи запускаются всегда (см. judge.evaluate_summary).
    # Его вердикт учитывается в _decide как ограничитель потолка категории.
    reports = _final_reports(r1, r2)
    blocks = {block: _block_verdict(reports, block) for block in judge.TAXONOMY}
    category, e1_triggered, e1_sources, decision_path = _decide(
        gate_status, evaluation.get("e1_signals", {}) or {}, blocks)
    verdict = _compose_verdict(category, blocks, e1_sources)

    return {
        **evaluation,
        "category": category, "verdict": verdict,
        "llm_category": r3.get("category"), "llm_verdict": r3.get("verdict"),
        "blocks": blocks, "e1_triggered": e1_triggered, "e1_sources": e1_sources,
        "decision_path": decision_path,
        "nodata_blocks": [b for b, v in blocks.items() if v["status"] == "no_data"],
        "failed_blocks": [b for b, v in blocks.items() if v["status"] == "issues"],
    }
