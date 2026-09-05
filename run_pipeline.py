"""
run_pipeline.py — Блок 7: оркестратор полного end-to-end прогона новой системы.

Назначение (план, Блок 7, критерий проверки): «полный end-to-end прогон
новой системы на текущем датасете (26 ЭМК × 6 моделей) с сохранением
аудит-лога и итогового отчёта».

Связывает воедино все блоки 0-7:
  загрузка пар (data/summaries.xlsx)
        │
        ▼
  judge.evaluate_summary   — Блоки 1-4 (препроцессор, объективный слой,
        │                     шлюз, LLM-as-Judge); чекпоинты на диск
        ▼
  aggregator.finalize      — Блок 5 (детерминированное решение)
        │
        ├──▶ audit.AuditLogger.log_evaluation   — Блок 7 (audit-лог JSONL,
        │                                          версии промптов/моделей/
        │                                          порогов, хэши вместо текста)
        │
        ▼ (после всех пар)
  report.save_results      — Блок 5/7 (Excel: сводная + объективный слой
        │                     рядом с LLM-оценками + статистика + критические)
        ▼
  drift.compute_snapshot   — Блок 7 (снимок распределений для мониторинга
                              дрейфа; сохраняется в audit_log/drift_snapshots/)

ВАЖНО — честно о масштабе полного прогона: 26 ЭМК × 6 моделей = 156 пар,
каждая — это gate (несколько LLM-вызовов извлечения сущностей) + 3 раунда
LLM-as-Judge (раунд 1: 3 судьи × 5 блоков = 15 вызовов; раунд 2: ещё ~3;
раунд 3: 1) — порядка 20+ обращений к Ollama НА ОДНУ ПАРУ. На пилотном
железе (один qwen3:8b на все роли, без параллелизма GPU) это те же часы
на сотни вызовов, что и в Блоке 6 сделали невозможным полный 3-раундовый
прогон на 370 синтетических парах за разумное время сессии. Поэтому здесь:
  • оркестратор написан и обкатан на ВСЮ цепочку (а не на заглушках —
    см. _self_check_dry_run, прогоняющий стаб-evaluation через все слои);
  • демонстрационный прогон --limit делает РЕАЛЬНЫЕ вызовы LLM на
    небольшом срезе пар — доказательство, что проводка «judge ->
    aggregator -> audit -> report -> drift» работает на живых данных
    end-to-end (не только в теории на стабах);
  • полный прогон всех 156 пар — отдельная, многочасовая задача,
    которую целесообразно ставить на ночь/на отдельном железе, и
    которая уже полностью поддерживается этим же скриптом без правок
    кода: `python run_pipeline.py` (без --limit) с уже встроенными
    чекпоинтами judge.py (переживают перезапуск процесса).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

import aggregator
import audit
import config
import gate
import drift
import judge
import report
from llm_client import JudgePanel

log = logging.getLogger("run_pipeline")

DATA_PATH = Path("data") / "summaries.xlsx"
REPORTS_DIR = Path("reports")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str).fillna("")
    required = {"emr_id", "model_id", "source_text", "summary_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В {path} не хватает колонок: {sorted(missing)}")
    return df


def run(*, limit: Optional[int] = None, profile: Optional[str] = None,
        run_id: Optional[str] = None, use_checkpoints: bool = True,
        scope: Optional[str] = config.DATASET_SCOPE) -> dict:
    """
    Полный (или усечённый --limit) прогон. Возвращает сводку:
    {n_pairs, results, audit_path, report_path, snapshot_path}.

    scope — декларация ЗАДАЧИ суммаризации в этом датасете (см. подробное
            объяснение в config.DATASET_SCOPE и judge._SCOPE_DESCRIPTIONS).
            По умолчанию берётся из config.DATASET_SCOPE ("radiologist" —
            нынешний датасет содержит ИМЕННО целевые выжимки «для
            рентгенолога», а не полные суммаризации ЭМК). Передайте
            scope=None, только если оцениваете ДРУГОЙ датасет с полными
            суммаризациями — иначе шлюз будет сравнивать целевую выжимку
            со ВСЕЙ картой и систематически отклонять корректные ответы
            как «неполные» (см. gate.evaluate_gate, комментарий про
            «системный ложный reject на целевых выжимках»).
    """
    df = load_dataset()
    if limit is not None:
        df = df.head(limit)
    log.info("Загружено пар: %d (emr_id × model_id), профиль=%s, scope=%s",
             len(df), profile or config.ACTIVE_PROFILE, scope)

    panel = JudgePanel(profile)
    logger = audit.AuditLogger(run_id=run_id)
    log.info("Audit-лог: %s", logger.path)

    results: list[dict] = []
    t0 = time.monotonic()
    for i, row in enumerate(df.itertuples(index=False), 1):
        emr_id, model_id = row.emr_id, row.model_id
        source_text, summary_text = row.source_text, row.summary_text

        try:
            cached = judge.load_checkpoint(emr_id, model_id) if use_checkpoints else None
            # Принимаем чекпоинт ТОЛЬКО если: (а) он уже содержит "objective"
            # (запись формата ДО Блока 7 честно пересчитывается заново — иначе
            # в audit-логе/отчёте появится молчаливый пробел в новой колонке
            # для части пар, хуже, чем потратить время на пересчёт), И
            # (б) он посчитан с ТЕМ ЖЕ scope, что и текущий прогон — иначе
            # словим ровно ту стэйл-проблему, что обнаружилась эмпирически:
            # запись "EMR_10__2", посчитанная со scope=None (gate=reject —
            # ложный отказ из-за сравнения целевой выжимки со ВСЕЙ картой),
            # молча подставлялась бы дальше даже после исправления scope.
            cached_ok = (cached is not None and "objective" in cached
                         and cached.get("scope") == scope)
            if cached_ok:
                log.info("[%d/%d] %s / %s — из чекпоинта (Блок 4+7 формат, scope=%s)",
                         i, len(df), emr_id, model_id, scope)
                evaluation = cached
            else:
                if cached is not None and not cached_ok:
                    log.info("[%d/%d] %s / %s — чекпоинт устарел (scope изменился: %s -> %s), пересчёт...",
                             i, len(df), emr_id, model_id, cached.get("scope"), scope)
                else:
                    log.info("[%d/%d] %s / %s — оценка...", i, len(df), emr_id, model_id)
                evaluation = judge.evaluate_summary(source_text, summary_text, emr_id, model_id,
                                                     scope=scope, panel=panel)
                if use_checkpoints:
                    judge.save_checkpoint(evaluation)
        except Exception as exc:
            # Известное ограничение пилотного железа (qwen3:8b — единственная
            # доступная модель на все роли): иногда модель не возвращает
            # пригодный JSON даже после авторемонта/retry в llm_client —
            # LLMError всплывает из objective_layer.extract_semantic_entities
            # (вызывается ИЗ ШЛЮЗА — gate.evaluate_gate — для семантических
            # сущностей), не пойманный там по дизайну Блока 3 (предполагалось,
            # что 2 retry'я внутри ask_json достаточно). Один сбойный ответ
            # модели не должен останавливать многочасовой прогон на 156 парах —
            # фиксируем как «ошибка» и идём дальше; реальная причина видна
            # и в audit-логе (verdict), и здесь, в логе прогона.
            log.error("[%d/%d] %s / %s — СБОЙ: %s — пара пропущена, прогон продолжается",
                      i, len(df), emr_id, model_id, exc)
            evaluation = judge._err(emr_id, model_id, f"Прогон прерван исключением: {exc}")

        try:
            finalized = aggregator.finalize(evaluation)
        except Exception as agg_exc:   # noqa: BLE001 — агрегация не должна ронять прогон
            log.exception("[%d/%d] %s / %s — сбой агрегации: %s",
                          i, len(df), emr_id, model_id, agg_exc)
            finalized = {**judge._err(emr_id, model_id,
                                      f"Сбой детерминированной агрегации: {agg_exc}"),
                         "blocks": {}, "decision_path": [f"Агрегация упала: {agg_exc}"],
                         "e1_triggered": False, "e1_sources": []}
        results.append(finalized)

        logger.log_evaluation(emr_id=emr_id, model_id=model_id, evaluation=finalized,
                              source_text=source_text, summary_text=summary_text,
                              profile_name=profile, scope=scope)

        log.info("    -> %s (шлюз=%s, объективный=%s)", finalized.get("category"),
                 (finalized.get("gate") or {}).get("status"),
                 "есть" if finalized.get("objective") else "—")

    elapsed = time.monotonic() - t0
    log.info("Готово: %d пар за %.1f сек (%.1f сек/пара)", len(results), elapsed,
             elapsed / len(results) if results else 0.0)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{logger.run_id}.xlsx"
    report.save_results(results, str(report_path))
    log.info("Отчёт сохранён: %s", report_path)

    snapshot_path = None
    if results:
        snap = drift.compute_snapshot(audit.load_entries(logger.path), run_id=logger.run_id, profile=profile)
        snapshot_path = snap.save()
        log.info("Снимок дрейфа сохранён: %s", snapshot_path)
        log.info("\n%s", snap.render())

    return {"n_pairs": len(results), "results": results, "audit_path": logger.path,
            "report_path": report_path, "snapshot_path": snapshot_path}


# ══════════════════════════════════════════════════════════════════
# СУХОЙ ПРОГОН — проверяет проводку judge-результат -> aggregator ->
# audit -> report -> drift БЕЗ единого вызова LLM (на стаб-evaluation,
# повторяющих реальную форму evaluate_summary для pass/reject путей).
# Цель — отделить «правильно ли соединены модули» (проверяется здесь,
# мгновенно и детерминированно) от «что говорит конкретная LLM на
# конкретных данных» (проверяется --limit-прогоном на живых вызовах).
# ══════════════════════════════════════════════════════════════════

def _stub_evaluation(emr_id: str, model_id: str, *, kind: str) -> dict:
    if kind == "reject":
        return {
            "emr_id": emr_id, "model_id": model_id, "category": "Неприемлемо",
            "verdict": "Отклонена pre-evaluation gate (Блок 3) до основной оценки: missing_critical_entity",
            "e1_signals": {"raised_by_judges": [], "aggregator_flagged": False,
                           "aggregator_named": [], "aggregator_category": None, "consistent": True},
            "gate": {"status": "reject", "reasons": ["missing_critical_entity"]},
            "objective": None, "r1": {}, "r2": {}, "r3": {},
        }
    block_ok = {f"{b}{n}": {"pass": True, "comment": ""} for b, k in
                (("A", 3), ("B", 5), ("C", 2), ("D", 3), ("E", 3)) for n in range(1, k + 1)}
    rep = {"A": {**{k: v for k, v in block_ok.items() if k.startswith("A")}, "hallucinations": [], "wrong_values": []},
           "B": {k: {**v, "missing": []} for k, v in block_ok.items() if k.startswith("B")},
           "C": {**{k: v for k, v in block_ok.items() if k.startswith("C")}},
           "D": {k: v for k, v in block_ok.items() if k.startswith("D")},
           "E": {k: ({**v, "danger_examples": []} if k == "E1" else v) for k, v in block_ok.items() if k.startswith("E")}}
    r1 = {"judge_1": rep, "judge_2": rep, "judge_3": rep}
    return {
        "emr_id": emr_id, "model_id": model_id, "category": "Готово к клиническому применению",
        "verdict": "Сводка точна и полна.", "summary_by_block": {b: "ок" for b in "ABCDE"},
        "e1_signals": {"raised_by_judges": [], "aggregator_flagged": False,
                       "aggregator_named": [], "aggregator_category": None, "consistent": True},
        "gate": {"status": "pass", "reasons": []},
        "objective": {
            "numeric": {"total_in_a": 8, "matched": 8, "mismatch_count": 0, "unit_mismatch_count": 0,
                        "mismatches": [], "unit_mismatches": []},
            "polarity": {"total_in_a": 3, "matched": 3, "flip_count": 0, "flips": []},
            "entities": None,
        },
        "r1": r1, "r2": r1, "r3": {"category": "Готово к клиническому применению", "verdict": "ок", "summary_by_block": {}},
    }


def _self_check_dry_run():
    """
    Сухой прогон проводки (без LLM): берём по одной паре каждого вида
    исхода (pass / gate-reject), пропускаем через aggregator.finalize ->
    audit.AuditLogger -> report.save_results -> drift.compute_snapshot
    и проверяем, что результат на каждом стыке выглядит так, как должен.
    """
    import tempfile
    import shutil

    tmpdir = Path(tempfile.mkdtemp(prefix="run_pipeline_selfcheck_"))
    try:
        finalized = [
            aggregator.finalize(_stub_evaluation("EMR_DRY1", "stub-model-A", kind="pass")),
            aggregator.finalize(_stub_evaluation("EMR_DRY2", "stub-model-A", kind="reject")),
            aggregator.finalize(_stub_evaluation("EMR_DRY1", "stub-model-B", kind="pass")),
        ]

        logger = audit.AuditLogger(run_id="dry-run-selfcheck", directory=tmpdir / "audit_log")
        for f in finalized:
            logger.log_evaluation(emr_id=f["emr_id"], model_id=f["model_id"], evaluation=f,
                                  source_text=f"исходник {f['emr_id']}", summary_text=f"сводка {f['model_id']}")
        assert logger.count == 3
        entries = audit.load_entries(logger.path)
        assert len(entries) == 3
        assert entries[0]["objective"]["numeric"]["matched"] == 8
        assert entries[1]["objective"] is None

        report_path = tmpdir / "dry_run_report.xlsx"
        report.save_results(finalized, str(report_path))
        assert report_path.exists()
        import openpyxl
        wb = openpyxl.load_workbook(report_path)
        assert "Сводная таблица" in wb.sheetnames
        ws = wb["Сводная таблица"]
        hdr = [c.value for c in ws[1]]
        assert "Объективный слой\n(Блок 2)" in hdr
        col = hdr.index("Объективный слой\n(Блок 2)") + 1
        cell_vals = [ws.cell(row=r, column=col).value for r in (2, 3, 4)]
        assert any("расхождений не найдено" in (v or "") for v in cell_vals)
        assert any("шлюз отклонил" in (v or "") for v in cell_vals)

        snap = drift.compute_snapshot(entries, run_id="dry-run-selfcheck")
        assert snap.n == 3
        assert snap.category_rates.get("Неприемлемо", 0) == round(1 / 3, 4)
        snap_path = snap.save(directory=tmpdir / "snapshots")
        assert snap_path.exists()

        print("OK: судья-стаб -> aggregator.finalize -> audit -> report -> drift — проводка цела")
        print(f"  audit: {logger.path.name} ({logger.count} записей)")
        print(f"  report: {report_path.name} (лист 'Сводная таблица', колонка "
              f"'Объективный слой' на месте)")
        print(f"  drift snapshot: {snap_path.name} (n={snap.n}, "
              f"reject_rate={snap.category_rates.get('Неприемлемо', 0):.1%})")
        print("Самопроверка run_pipeline.py (сухой прогон проводки) — OK")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None,
                   help="оценить только первые N пар датасета (демонстрационный прогон; "
                        "без флага — ВСЕ 156 пар, часы на пилотном железе)")
    p.add_argument("--profile", default=None, help="имя профиля моделей (по умолчанию — config.ACTIVE_PROFILE)")
    p.add_argument("--run-id", default=None, help="идентификатор прогона (по умолчанию — метка времени)")
    p.add_argument("--no-checkpoints", action="store_true", help="игнорировать сохранённые чекпоинты judge.py")
    p.add_argument("--scope", default=None,
                   help="декларация задачи суммаризации (по умолчанию — config.DATASET_SCOPE, "
                        "сейчас 'radiologist' — целевые выжимки для рентгенолога). "
                        "Передайте 'none', если оцениваете датасет с ПОЛНЫМИ суммаризациями ЭМК "
                        "(иначе шлюз будет систематически отклонять корректные целевые выжимки "
                        "как «неполные» — см. config.DATASET_SCOPE)")
    p.add_argument("--dry-run", action="store_true",
                   help="не вызывать LLM вовсе — прогнать самопроверку проводки на стаб-данных")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    args = _build_argparser().parse_args()

    if args.dry_run:
        _self_check_dry_run()
        sys.exit(0)

    if args.scope is None:
        scope = config.DATASET_SCOPE
    elif args.scope.strip().lower() in ("none", "null", "-", ""):
        scope = None
    else:
        scope = args.scope
    gate.validate_scope(scope)   # опечатка в --scope включила бы строгий порог без фильтра

    summary = run(limit=args.limit, profile=args.profile, run_id=args.run_id,
                  use_checkpoints=not args.no_checkpoints, scope=scope)
    print()
    print(f"Готово: {summary['n_pairs']} пар")
    print(f"  audit-лог: {summary['audit_path']}")
    print(f"  отчёт:     {summary['report_path']}")
    print(f"  снимок:    {summary['snapshot_path']}")
