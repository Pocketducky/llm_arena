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

import judge

# Версия таблицы решающих правил — Блок 7 («версии… порогов» в audit-логе).
# Поднимается вручную при изменении самой логики агрегации (не порогов
# внутри gate.py/objective_layer.py — те снимаются audit.py отдельным
# срезом, см. _thresholds_snapshot там же).
DECISION_TABLE_VERSION = "aggregator-decision-table-v1"

CATEGORY_READY  = "Готово к клиническому применению"
CATEGORY_EDIT   = "Требует редактирования"
CATEGORY_REJECT = "Неприемлемо"

CATEGORIES = (CATEGORY_READY, CATEGORY_EDIT, CATEGORY_REJECT)


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
    votes = [rep[block][code]["pass"]
             for rep in reports.values()
             if isinstance(rep.get(block, {}).get(code), dict)
             and isinstance(rep[block][code].get("pass"), bool)]
    if not votes:
        return None
    return sum(votes) > len(votes) / 2


def _block_verdict(reports: dict[str, dict], block: str) -> dict:
    """Раздельный статус/балл по одному блоку таксономии (A-E):
      • score  — доля подкритериев, прошедших мажоритарную проверку (0-100);
                 None, если ни по одному нет данных
      • status — "ok", если КАЖДЫЙ подкритерий прошёл большинством голосов;
                 "issues" в любом другом случае (провален ИЛИ нет данных —
                 отсутствие положительного подтверждения не равно проверке,
                 тот же принцип, что и в шлюзе: при сомнении — на доработку)
    """
    codes = judge.TAXONOMY[block]
    votes = {code: _subcriterion_vote(reports, block, code) for code in codes}
    failed       = [c for c, v in votes.items() if v is False]
    undetermined = [c for c, v in votes.items() if v is None]
    decided      = [v for v in votes.items() if v[1] is not None]
    score = round(100 * sum(1 for _, v in decided if v) / len(decided), 1) if decided else None
    status = "ok" if not failed and not undetermined else "issues"
    return {
        "score": score, "status": status, "votes": votes,
        "failed_subcriteria": failed, "undetermined_subcriteria": undetermined,
    }


# ══════════════════════════════════════════════════════════════════
# ШАГ 1 + 3: стоп-правило E1 и итоговая таблица решающих правил
# ══════════════════════════════════════════════════════════════════

def _e1_sources(e1_signals: dict) -> list[str]:
    return sorted(set(e1_signals.get("raised_by_judges", ()))
                  | set(e1_signals.get("aggregator_named", ())))


def _decide(gate_status: Optional[str], e1_signals: dict,
            blocks: dict[str, dict]) -> tuple[str, bool, list[str], list[str]]:
    """Возвращает (категория, e1_triggered, e1_sources, decision_path)."""
    path: list[str] = []

    # ── Шаг 1 — стоп-правило E1, ПЕРВЫМ и безусловно ─────────────
    e1_triggered = bool(e1_signals.get("raised_by_judges")) or bool(e1_signals.get("aggregator_flagged"))
    sources = _e1_sources(e1_signals)
    if e1_triggered:
        path.append(f"Шаг 1 — СТОП-ПРАВИЛО E1 сработало (источники: {', '.join(sources) or 'агрегатор без указания судей'}); "
                    f"категория «{CATEGORY_REJECT}» фиксируется немедленно, шаги 2-3 не пересматривают это решение")
        return CATEGORY_REJECT, True, sources, path
    path.append("Шаг 1 — стоп-правило E1 не сработало ни у одного источника (судьи R1/R2, агрегатор R3)")

    # ── Шаг 2 — уже посчитан вызывающей стороной, фиксируем сводку ─
    issues = [b for b, v in blocks.items() if v["status"] != "ok"]
    path.append(f"Шаг 2 — мажоритарное голосование судей по блокам A-E: "
                f"{'замечаний нет' if not issues else 'замечания в ' + ', '.join(issues)}")

    # ── Шаг 3 — итоговая категория по явной таблице правил ────────
    if gate_status == "rework":
        if len(issues) >= 2:
            path.append(f"Шаг 3 — шлюз (Блок 3) вернул «доработка», и судьи независимо подтверждают "
                        f"проблемы ещё в {len(issues)} блок(ах) → совпадение сигналов → «{CATEGORY_REJECT}»")
            return CATEGORY_REJECT, False, [], path
        path.append(f"Шаг 3 — шлюз (Блок 3) вернул «доработка» → готовой категория быть не может → «{CATEGORY_EDIT}»")
        return CATEGORY_EDIT, False, [], path

    if not issues:
        path.append(f"Шаг 3 — шлюз = pass, замечаний по блокам A-E нет → «{CATEGORY_READY}»")
        return CATEGORY_READY, False, [], path

    if len(issues) >= 3:
        path.append(f"Шаг 3 — замечания затрагивают {len(issues)} из 5 блоков таксономии "
                    f"→ системный характер проблем → «{CATEGORY_REJECT}»")
        return CATEGORY_REJECT, False, [], path

    path.append(f"Шаг 3 — точечные замечания в {len(issues)} блок(ах) из 5 → «{CATEGORY_EDIT}»")
    return CATEGORY_EDIT, False, [], path


def _compose_verdict(category: str, blocks: dict[str, dict], e1_sources: list[str]) -> str:
    """Краткое, полностью выводимое из (1)-(3) обоснование — без LLM:
    то, что обычно «на глаз» формулировал арбитр, здесь читается прямо
    из решающих правил и голосов судей."""
    if category == CATEGORY_REJECT and e1_sources:
        return (f"«{category}»: сработало стоп-правило E1 — обнаружена клинически опасная ошибка "
                f"(источник(и): {', '.join(e1_sources)}). Дальнейшие оси A-E не пересматривают это решение.")

    issues = {b: v for b, v in blocks.items() if v["status"] != "ok"}
    if not issues:
        return (f"«{category}»: по всем блокам таксономии (A-E) большинство судей подтвердили "
                f"прохождение каждого подкритерия; стоп-правило E1 не сработало.")

    parts = []
    for b, v in issues.items():
        names = v["failed_subcriteria"] or v["undetermined_subcriteria"]
        kind = "не подтверждены большинством судей" if v["failed_subcriteria"] else "нет валидных оценок судей"
        parts.append(f"{b} ({', '.join(names)} — {kind})")
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

    # Короткий путь — пара отклонена шлюзом (Блок 3) ДО запуска судей:
    # judge.evaluate_summary уже вернул «Неприемлемо» с прозрачным обоснованием,
    # пересчитывать нечего (r1/r2/r3 пусты по построению).
    if gate_status == "reject" and not r1:
        return {
            **evaluation,
            "category": evaluation.get("category", CATEGORY_REJECT),
            "verdict": evaluation.get("verdict", ""),
            "llm_category": None, "llm_verdict": None,
            "blocks": {}, "e1_triggered": True, "e1_sources": [],
            "decision_path": ["Отклонена шлюзом (Блок 3) до запуска LLM-as-Judge — "
                              "агрегация Блока 5 не требуется, решение уже окончательно"],
        }

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
    }
