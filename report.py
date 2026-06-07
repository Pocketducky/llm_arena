"""
report.py — Блок 5 (продолжение): формирование Excel-отчёта под новую модель
данных — раздельные оси таксономии A-E + детерминированная категория из
aggregator.py, вместо единого `final_score`/`criteria`/`quality`
(старая структура — evaluator.save_results, evaluator.py:929-1086).

Сохранена удобная структура листов прежнего отчёта (сводная/детали/
статистика/критические ошибки), но содержание каждого подчинено новой
модели:
  • «Сводная»     — итоговая категория (код, не самооценка LLM), решение
                    шлюза, статус каждого блока A-E, флаг E1 и его источники,
                    категория по версии LLM-агрегатора — для прямого сравнения
  • «Детали по блокам» — по каждой паре/судье/блоку: какие именно подкритерии
                    не прошли мажоритарную проверку (а не агрегированный балл
                    «на глаз», как раньше)
  • «Статистика»  — по моделям: распределение по 3 категориям, частота
                    срабатывания E1, решения шлюза, согласие код↔LLM
                    (метрика для Блока 6/7 — насколько нужен детерминированный
                    слой поверх самооценки модели)
  • «Критические ошибки» — все срабатывания E1 (с источниками и конкретными
                    примерами из danger_examples) + все блоки со статусом
                    "issues" (с именно теми подкритериями, что не прошли)

Принимает результаты, уже прошедшие aggregator.finalize() — то есть с
полями category/verdict/blocks/e1_triggered/e1_sources/llm_category/decision_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import aggregator
import judge

CATEGORY_COLORS = {
    aggregator.CATEGORY_READY:  "C6EFCE",
    aggregator.CATEGORY_EDIT:   "FFEB9C",
    aggregator.CATEGORY_REJECT: "FFC7CE",
    "ошибка": "D9D9D9",
}

BLOCK_STATUS_COLORS = {"ok": "C6EFCE", "issues": "FFC7CE", None: "D9D9D9"}

_BLOCKS = list(judge.TAXONOMY)   # ("A","B","C","D","E") — порядок таксономии


# ══════════════════════════════════════════════════════════════════
# СТИЛИ (минимальный набор — самодостаточный, без зависимости от evaluator.py,
# который план полностью выводит из эксплуатации)
# ══════════════════════════════════════════════════════════════════

def _brd():
    s = Side(style="thin")
    return Border(left=s, right=s, top=s, bottom=s)


def _h(ws, r, c, v, bg="1F4E79", fc="FFFFFF"):
    cell = ws.cell(row=r, column=c, value=v)
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.font = Font(bold=True, color=fc, size=9)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = _brd()


def _d(ws, r, c, v, bold=False, bg=None, red=False):
    cell = ws.cell(row=r, column=c, value=v)
    cell.font = Font(bold=bold, color="C00000" if red else "000000", size=9)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = _brd()
    if bg:
        cell.fill = PatternFill("solid", fgColor=bg)


# ══════════════════════════════════════════════════════════════════
# Лист 1 — Сводная таблица
# ══════════════════════════════════════════════════════════════════

def _sheet_summary(wb: Workbook, results: list[dict]):
    ws = wb.active
    ws.title = "Сводная таблица"
    hdrs = (["ЭМК", "Модель", "Категория\n(итог)", "🚨 E1", "Источники E1", "Шлюз (Блок 3)",
             "Объективный слой\n(Блок 2)"]
            + [f"Блок {b}" for b in _BLOCKS]
            + ["Категория\nпо LLM", "Совпадает\nс LLM?", "Вердикт"])
    for ci, h in enumerate(hdrs, 1):
        _h(ws, 1, ci, h)
    ws.row_dimensions[1].height = 48
    ws.freeze_panes = "A2"

    for ridx, r in enumerate(results, 2):
        category = r.get("category", "ошибка")
        blocks = r.get("blocks", {}) or {}
        gate = r.get("gate", {}) or {}
        objective = r.get("objective")
        e1 = bool(r.get("e1_triggered"))
        sources = ", ".join(r.get("e1_sources", []) or []) or "—"
        llm_cat = r.get("llm_category") or "—"
        agree = (category == r.get("llm_category")) if r.get("llm_category") else None

        row = ([r["emr_id"], r["model_id"], category,
                "🚨 ДА" if e1 else "нет", sources,
                gate.get("status", "—"), _objective_cell_text(objective)]
               + [_block_cell_text(blocks.get(b)) for b in _BLOCKS]
               + [llm_cat,
                  ("да" if agree else ("нет" if agree is False else "—")),
                  r.get("verdict", "")[:200]])

        bgs = ([None, None, CATEGORY_COLORS.get(category, "FFFFFF"),
                "FFC7CE" if e1 else None, None, None,
                ("FFC7CE" if _objective_has_findings(objective) else None)]
               + [BLOCK_STATUS_COLORS.get((blocks.get(b) or {}).get("status")) for b in _BLOCKS]
               + [None, ("FFC7CE" if agree is False else None), None])

        for ci, (val, bg) in enumerate(zip(row, bgs), 1):
            _d(ws, ridx, ci, val, bold=(ci == 3), bg=bg)
        ws.row_dimensions[ridx].height = 30

    widths = [10, 9, 16, 7, 16, 12, 26] + [9] * len(_BLOCKS) + [16, 9, 50]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _block_cell_text(block: Optional[dict]) -> str:
    if not block:
        return "—"
    score = block.get("score")
    status = "✓" if block["status"] == "ok" else "✗"
    failed = block.get("failed_subcriteria") or []
    label = f"{status} {score:.0f}%" if score is not None else f"{status} —"
    return f"{label} ({', '.join(failed)})" if failed else label


# ── Объективный слой (Блок 2/7) — компактное отображение «рядом с LLM» ──
# Принимает уже сериализованное резюме (objective_layer.
# ObjectiveComparisonReport.to_summary(), прокинутое через judge.
# evaluate_summary -> aggregator.finalize в поле "objective"). None
# означает «не построено» — честно (пара отклонена шлюзом ДО этого шага,
# см. комментарий в judge.evaluate_summary про экономию GPU-времени),
# а не «забыли посчитать» — поэтому здесь не ошибка, а отдельное «—».

def _objective_cell_text(objective: Optional[dict]) -> str:
    if not objective:
        return "— (шлюз отклонил до сверки)"
    num = objective.get("numeric") or {}
    pol = objective.get("polarity") or {}
    parts = []

    mm, um, total_num = num.get("mismatch_count", 0), num.get("unit_mismatch_count", 0), num.get("total_in_a", 0)
    if mm or um:
        bits = []
        if mm:
            bits.append(f"{mm} знач.")
        if um:
            bits.append(f"{um} ед.изм.")
        parts.append(f"числа: {' + '.join(bits)} расхожд. (из {total_num})")

    fl, total_pol = pol.get("flip_count", 0), pol.get("total_in_a", 0)
    if fl:
        parts.append(f"полярность: {fl} инверс. (из {total_pol})")

    ents = objective.get("entities")
    if ents:
        f1s = [rep["f1"] for rep in ents.values()]
        if f1s:
            parts.append(f"сущности: F1≈{sum(f1s) / len(f1s):.2f}")

    return "; ".join(parts) if parts else "✓ расхождений не найдено"


def _objective_has_findings(objective: Optional[dict]) -> bool:
    """«Жёсткие» автоматические находки — то, что обязано бросаться в глаза
    рядом с оценкой судей (числовые/полярные расхождения, см. Блок 2)."""
    if not objective:
        return False
    num = objective.get("numeric") or {}
    pol = objective.get("polarity") or {}
    return bool(num.get("mismatch_count") or num.get("unit_mismatch_count") or pol.get("flip_count"))


# ══════════════════════════════════════════════════════════════════
# Лист 2 — Детали по блокам и судьям (что именно не прошло и почему)
# ══════════════════════════════════════════════════════════════════

def _sheet_block_details(wb: Workbook, results: list[dict]):
    ws = wb.create_sheet("Детали по блокам")
    hdrs = ["ЭМК", "Модель", "Блок", "Балл, %", "Статус",
            "Не прошли (мажоритарно)", "Без данных", "Комментарии судей (по непрошедшим)"]
    for ci, h in enumerate(hdrs, 1):
        _h(ws, 1, ci, h)
    ws.freeze_panes = "A2"

    ridx = 2
    for r in results:
        blocks = r.get("blocks", {}) or {}
        reports = aggregator._final_reports(r.get("r1", {}), r.get("r2", {}))
        for b in _BLOCKS:
            v = blocks.get(b)
            if v is None:
                continue
            failed = v.get("failed_subcriteria") or []
            comments = _collect_comments(reports, b, failed)
            row = [r["emr_id"], r["model_id"], b,
                   f"{v['score']:.0f}" if v.get("score") is not None else "—",
                   "ok" if v["status"] == "ok" else "issues",
                   ", ".join(failed) or "—",
                   ", ".join(v.get("undetermined_subcriteria") or []) or "—",
                   comments[:300]]
            bg = BLOCK_STATUS_COLORS.get(v["status"])
            for ci, val in enumerate(row, 1):
                _d(ws, ridx, ci, val, bg=(bg if ci == 5 else None))
            ridx += 1

    for i, w in enumerate([10, 9, 7, 8, 8, 28, 16, 60], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _collect_comments(reports: dict[str, dict], block: str, failed_codes: list[str]) -> str:
    """Конкретные обоснования судей по подкритериям, не прошедшим мажоритарную
    проверку — то самое содержательное «почему», которое старый compact()
    обрезал перед обменом между судьями (см. judge._format_full_report);
    здесь оно доходит до читателя отчёта в явном виде."""
    parts = []
    for code in failed_codes:
        for role, rep in reports.items():
            sub = rep.get(block, {}).get(code)
            if isinstance(sub, dict) and sub.get("pass") is False and sub.get("comment"):
                parts.append(f"{code}/{role}: {sub['comment']}")
    return "; ".join(parts)


# ══════════════════════════════════════════════════════════════════
# Лист 3 — Статистика по моделям
# ══════════════════════════════════════════════════════════════════

def _sheet_statistics(wb: Workbook, results: list[dict]):
    ws = wb.create_sheet("Статистика")
    hdrs = ["Модель", "N", aggregator.CATEGORY_READY, aggregator.CATEGORY_EDIT,
            aggregator.CATEGORY_REJECT, "🚨 E1, %", "Шлюз: reject", "Шлюз: rework",
            "Согласие код↔LLM, %",
            "Объективный слой (Блок 2) — доля пар с расхождениями:",
            "  числа/ед.изм., %", "  полярность, %"]
    for ci, h in enumerate(hdrs, 1):
        _h(ws, 1, ci, h)
    # Заголовок-«зонтик» над двумя последними колонками — наглядно
    # группирует объективные метрики отдельно от категорий/шлюза/согласия.
    ws.merge_cells(start_row=1, start_column=10, end_row=1, end_column=12)

    df = pd.DataFrame([{
        "model_id": r["model_id"],
        "category": r.get("category", "ошибка"),
        "e1": bool(r.get("e1_triggered")),
        "gate": (r.get("gate") or {}).get("status"),
        "agree": (r.get("category") == r.get("llm_category")) if r.get("llm_category") else None,
        # Доля пар, где объективный слой нашёл хоть одно числовое/полярное
        # расхождение — None для записей без objective (gate reject ДО
        # сверки): не учитываем их ни в числителе, ни в знаменателе доли,
        # т.к. для них вопрос «было ли расхождение» неприменим, а не «нет».
        "obj_numeric_finding": _obj_flag(r.get("objective"), "numeric"),
        "obj_polarity_finding": _obj_flag(r.get("objective"), "polarity"),
    } for r in results])

    if not df.empty:
        for ridx, (mid, g) in enumerate(df.groupby("model_id"), 2):
            cc = g["category"].value_counts()
            agree = g["agree"].dropna()
            num_flags = g["obj_numeric_finding"].dropna()
            pol_flags = g["obj_polarity_finding"].dropna()
            row = [mid, len(g),
                   int(cc.get(aggregator.CATEGORY_READY, 0)),
                   int(cc.get(aggregator.CATEGORY_EDIT, 0)),
                   int(cc.get(aggregator.CATEGORY_REJECT, 0)),
                   round(100 * g["e1"].mean(), 1),
                   int((g["gate"] == "reject").sum()),
                   int((g["gate"] == "rework").sum()),
                   round(100 * agree.mean(), 1) if len(agree) else None,
                   None,  # «зонтик» — пустая ячейка под объединённым заголовком
                   round(100 * num_flags.mean(), 1) if len(num_flags) else None,
                   round(100 * pol_flags.mean(), 1) if len(pol_flags) else None]
            for ci, v in enumerate(row, 1):
                bg = ("FFC7CE" if ci in (5,) and isinstance(v, (int, float)) and v > 0 else
                      ("FFC7CE" if ci == 5 and isinstance(v, (int, float)) and v > 1.0 else
                       ("FFC7CE" if ci in (11, 12) and isinstance(v, (int, float)) and v > 0 else None)))
                _d(ws, ridx, ci, v, bg=bg)

    for i, w in enumerate([14, 6, 16, 16, 16, 9, 14, 14, 16, 4, 16, 14], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _obj_flag(objective: Optional[dict], section: str) -> Optional[bool]:
    """True/False — «нашёл ли объективный слой расхождение в данном
    разрезе (numeric/polarity)»; None — резюме недоступно (пара отклонена
    шлюзом до сверки, см. _objective_cell_text). None исключается из
    усреднения в _sheet_statistics — доля считается по применимым парам."""
    if not objective:
        return None
    blk = objective.get(section) or {}
    if section == "numeric":
        return bool(blk.get("mismatch_count") or blk.get("unit_mismatch_count"))
    if section == "polarity":
        return bool(blk.get("flip_count"))
    return None


# ══════════════════════════════════════════════════════════════════
# Лист 4 — Критические ошибки (E1 + проблемные блоки)
# ══════════════════════════════════════════════════════════════════

def _sheet_critical(wb: Workbook, results: list[dict]):
    ws = wb.create_sheet("🚨 Критические")
    hdrs = ["ЭМК", "Модель", "Тип", "Блок/подкритерий", "Описание", "Категория"]
    for ci, h in enumerate(hdrs, 1):
        _h(ws, 1, ci, h, bg="C00000")

    ridx = 2
    for r in results:
        category = r.get("category", "ошибка")
        if r.get("e1_triggered"):
            sources = ", ".join(r.get("e1_sources", []) or []) or "—"
            examples = _collect_e1_examples(aggregator._final_reports(r.get("r1", {}), r.get("r2", {})))
            row = [r["emr_id"], r["model_id"], "🚨 E1 — стоп-правило", f"E1 (источники: {sources})",
                   (examples or r.get("verdict", ""))[:250], category]
            for ci, v in enumerate(row, 1):
                _d(ws, ridx, ci, v, bg="FFC7CE", bold=True)
            ridx += 1

        blocks = r.get("blocks", {}) or {}
        reports = aggregator._final_reports(r.get("r1", {}), r.get("r2", {}))
        for b, v in blocks.items():
            if v.get("status") != "issues":
                continue
            for code in (v.get("failed_subcriteria") or []):
                comment = next((reports[role][b][code]["comment"]
                                for role in reports
                                if reports[role].get(b, {}).get(code, {}).get("pass") is False
                                and reports[role][b][code].get("comment")), "")
                row = [r["emr_id"], r["model_id"], "Замечание по таксономии",
                       f"{b}/{code} — {judge.SUBCRITERIA_LABELS_RU.get(code, '')}",
                       comment[:250], category]
                for ci, val in enumerate(row, 1):
                    _d(ws, ridx, ci, val, bg="FFEB9C")
                ridx += 1

    for i, w in enumerate([10, 12, 22, 36, 70, 18], 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _collect_e1_examples(reports: dict[str, dict]) -> str:
    parts = []
    for role, rep in reports.items():
        e1 = rep.get("E", {}).get("E1")
        if isinstance(e1, dict) and e1.get("pass") is False:
            ex = "; ".join(str(x) for x in (e1.get("danger_examples") or []))
            comment = e1.get("comment", "")
            parts.append(f"{role}: {comment}" + (f" [{ex}]" if ex else ""))
    return " | ".join(parts)


# ══════════════════════════════════════════════════════════════════
# ПУБЛИЧНАЯ ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════

def save_results(results: list[dict], path: str):
    """results — список словарей, каждый уже прошедший aggregator.finalize()
    (т.е. содержит category/verdict/blocks/e1_triggered/e1_sources/
    llm_category/gate/r1/r2 — детерминированные финальные поля Блока 5,
    а не самооценку LLM-агрегатора из Блока 4)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    _sheet_summary(wb, results)
    _sheet_block_details(wb, results)
    _sheet_statistics(wb, results)
    _sheet_critical(wb, results)
    wb.save(path)
