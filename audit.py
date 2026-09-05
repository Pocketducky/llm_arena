"""
audit.py — Блок 7: структурированный audit-лог (JSONL).

Назначение (план, Блок 7): «структурированный audit-лог (JSONL):
вход/выход каждого модуля, версии промптов/моделей/порогов, решения
и их обоснования — для воспроизводимости и будущей регуляторной
отчётности».

Принципы:
  • JSONL — одна запись (JSON-объект) на строку: устойчив к частичной
    записи при сбое посередине прогона, читается потоково, тривиально
    грепается/фильтруется без специальных инструментов.
  • Сырой текст ЭМК/суммаризации в лог НЕ попадает — только sha256-хэши:
    этого достаточно, чтобы доказать «оценка проводилась именно для ЭТОЙ
    пары текстов» (воспроизводимость, сверка), не тиражируя при этом
    персональные медицинские данные по дисковым логам.
  • «Версии» — не абстрактный номер сборки, а СТРУКТУРНЫЙ снимок:
    профиль + маппинг роль→модель (config.get_profile), версии
    промптов/таксономии/решающей таблицы (judge.PROMPT_SET_VERSION,
    judge.TAXONOMY_VERSION, aggregator.DECISION_TABLE_VERSION) и
    фактические значения порогов шлюза/объективного слоя на момент
    прогона (gate.MIN_*). То есть не «релиз 1.4», а «вот буквально чем
    было оценено» — пригодно для прямого сравнения двух прогонов и для
    регуляторной реконструкции «почему было принято именно это решение».
"""

from __future__ import annotations

import hashlib
import json
import threading
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
import judge
import aggregator
import gate

log = logging.getLogger("audit")

AUDIT_LOG_DIR = Path("audit_log")


def _hash_text(text: Optional[str]) -> Optional[str]:
    """sha256 текста — НЕ сам текст: идентификатор содержимого без его раскрытия."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _thresholds_snapshot() -> dict:
    """
    Срез действующих порогов шлюза/объективного слоя на момент прогона —
    это и есть «версии порогов» из формулировки плана. Не версия-номер
    (пороги правятся точечно, не релизными циклами), а фактические
    значения: достаточно, чтобы постфактум понять, ПОЧЕМУ шлюз решил
    именно так, и заметить, если порог впоследствии изменится.
    """
    import config
    return {
        "gate.MIN_SCHEMA_SECTIONS": gate.MIN_SCHEMA_SECTIONS,
        "gate.MIN_ENTITY_RECALL": gate.MIN_ENTITY_RECALL,
        "gate.CRITICAL_NUMERIC_CATEGORIES": list(gate.CRITICAL_NUMERIC_CATEGORIES),
        "gate.CRITICAL_PREDICATES": list(gate.CRITICAL_PREDICATES),
        # Раньше здесь не было CRITICAL_ENTITY_CATEGORIES — то есть константа,
        # решающая исход шлюза при scope=None, не версионировалась вовсе.
        "gate.CRITICAL_ENTITY_CATEGORIES": list(gate.CRITICAL_ENTITY_CATEGORIES),
        # Параметры, влияющие на вердикт не меньше порогов шлюза.
        "config.E1_REQUIRE_CITATION": config.E1_REQUIRE_CITATION,
        "config.MAX_NODATA_BLOCKS": config.MAX_NODATA_BLOCKS,
        "config.TEMPERATURE": config.DEFAULT_TEMPERATURE,
        "config.SEED": config.SEED,
        "config.TOKENS": {"r1": config.TOKENS_R1, "r1_large": config.TOKENS_R1_LARGE,
                          "r2": config.TOKENS_R2, "r3": config.TOKENS_R3,
                          "entities": config.TOKENS_ENTITIES,
                          "ceiling": config.MAX_TOKENS_CEILING},
        "config.ENTITY_MAX_CHARS": config.ENTITY_MAX_CHARS,
    }


def versions_snapshot(profile_name: Optional[str] = None) -> dict:
    """
    Полный срез «чем оценено»: активный профиль (роль -> модель),
    версии промптов/таксономии/решающей таблицы и действующие пороги.
    Это и есть «версии промптов/моделей/порогов» из плана — собранные
    в одном месте, пригодные для прямого diff между двумя прогонами.
    """
    profile = config.get_profile(profile_name)
    return {
        "profile": profile.name,
        "roles": dict(profile.roles),
        "prompt_set_version": judge.PROMPT_SET_VERSION,
        "taxonomy_version": judge.TAXONOMY_VERSION,
        "decision_table_version": aggregator.DECISION_TABLE_VERSION,
        "thresholds": _thresholds_snapshot(),
    }


@dataclass
class AuditEntry:
    """
    Одна запись audit-лога — «вход/выход» одного вызова evaluate_summary
    (judge.py) для пары (ЭМК, суммаризация модели). Сериализуется в одну
    строку JSONL через `to_json()`.
    """
    run_id: str
    timestamp: str
    emr_id: str
    model_id: str
    scope: Optional[str]
    source_hash: Optional[str]
    summary_hash: Optional[str]
    versions: dict
    gate: dict
    objective: Optional[dict]
    e1_signals: dict
    category: str
    verdict: str
    llm_category: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_evaluation(cls, *, run_id: str, emr_id: str, model_id: str, evaluation: dict,
                        source_text: Optional[str] = None, summary_text: Optional[str] = None,
                        scope: Optional[str] = None, profile_name: Optional[str] = None) -> "AuditEntry":
        """
        Строит запись из словаря, который возвращает evaluate_summary().
        Хэширование текстов — по явно переданным аргументам: вызывающий
        код сам решает, передавать ли их (например, при пакетном прогоне
        по чекпоинтам исходный текст уже может быть недоступен «под рукой»,
        и это не должно ломать запись аудита — хэши тогда будут None).
        """
        r3 = evaluation.get("r3") or {}
        return cls(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            emr_id=emr_id, model_id=model_id,
            # scope критичен для реконструкции решения (от него зависит НАБОР
            # правил шлюза, а не только пороги). run_pipeline его не передавал,
            # и во всех записях аудита стояло scope: null — подстраховываемся
            # значением из самого результата оценки.
            scope=scope if scope is not None else evaluation.get("scope"),
            source_hash=_hash_text(source_text),
            summary_hash=_hash_text(summary_text),
            versions=versions_snapshot(profile_name),
            gate=evaluation.get("gate", {}) or {},
            objective=evaluation.get("objective"),
            e1_signals=evaluation.get("e1_signals", {}) or {},
            category=evaluation.get("category", "—"),
            verdict=evaluation.get("verdict", ""),
            llm_category=r3.get("category"),
        )


class AuditLogger:
    """
    Аппендер JSONL audit-лога. Один файл на запуск (run_id зашит в имя
    файла) — так параллельные/повторные прогоны не перемешивают записи
    без блокировок; «склейка» истории при необходимости — `cat *.jsonl`.
    """

    def __init__(self, run_id: Optional[str] = None, *, directory: Path = AUDIT_LOG_DIR):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{self.run_id}.jsonl"
        self._n = 0
        self._lock = threading.Lock()

    def log(self, entry: AuditEntry) -> None:
        # Блокировка нужна с появлением параллельного прогона (config.CONCURRENCY):
        # без неё строки JSONL из разных потоков могли бы перемешаться внутри строки.
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(entry.to_json() + "\n")
            self._n += 1

    def log_evaluation(self, *, emr_id: str, model_id: str, evaluation: dict,
                       source_text: Optional[str] = None, summary_text: Optional[str] = None,
                       scope: Optional[str] = None, profile_name: Optional[str] = None) -> AuditEntry:
        """Удобный путь: собрать запись из результата evaluate_summary и сразу дописать в лог."""
        entry = AuditEntry.from_evaluation(
            run_id=self.run_id, emr_id=emr_id, model_id=model_id, evaluation=evaluation,
            source_text=source_text, summary_text=summary_text, scope=scope,
            profile_name=profile_name)
        self.log(entry)
        return entry

    @property
    def count(self) -> int:
        return self._n


def load_entries(path) -> list[dict]:
    """Чтение JSONL audit-лога обратно в список словарей — вход для drift.py и ручного анализа."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ══════════════════════════════════════════════════════════════════
# САМОПРОВЕРКА
# ══════════════════════════════════════════════════════════════════

def _self_check():
    import tempfile
    import shutil

    tmpdir = Path(tempfile.mkdtemp(prefix="audit_selfcheck_"))
    try:
        logger = AuditLogger(run_id="selfcheck-run", directory=tmpdir)

        stub_eval_pass = {
            "category": "Готово к клиническому применению",
            "verdict": "Сводка точна и полна.",
            "gate": {"status": "pass", "reasons": []},
            "objective": {
                "numeric": {"total_in_a": 5, "matched": 5, "mismatch_count": 0,
                            "unit_mismatch_count": 0, "mismatches": [], "unit_mismatches": []},
                "polarity": {"total_in_a": 2, "matched": 2, "flip_count": 0, "flips": []},
                "entities": None,
            },
            "e1_signals": {"raised_by_judges": [], "aggregator_flagged": False,
                           "aggregator_named": [], "aggregator_category": None, "consistent": True},
            "r1": {}, "r2": {}, "r3": {"category": "Готово к клиническому применению"},
        }
        stub_eval_reject = {
            "category": "Неприемлемо",
            "verdict": "Отклонена pre-evaluation gate (Блок 3).",
            "gate": {"status": "reject", "reasons": ["missing_critical_entity"]},
            "objective": None,
            "e1_signals": {"raised_by_judges": [], "aggregator_flagged": False,
                           "aggregator_named": [], "aggregator_category": None, "consistent": True},
            "r1": {}, "r2": {}, "r3": {},
        }

        logger.log_evaluation(emr_id="EMR_01", model_id="qwen3:8b", evaluation=stub_eval_pass,
                              source_text="Исходный текст ЭМК…", summary_text="Суммаризация…")
        logger.log_evaluation(emr_id="EMR_02", model_id="qwen3:8b", evaluation=stub_eval_reject,
                              source_text="Другой исходник…", summary_text="Урезанная суммаризация…")

        assert logger.count == 2
        assert logger.path.exists()

        loaded = load_entries(logger.path)
        assert len(loaded) == 2, loaded
        assert loaded[0]["emr_id"] == "EMR_01"
        assert loaded[0]["category"] == "Готово к клиническому применению"
        assert loaded[0]["source_hash"] == _hash_text("Исходный текст ЭМК…")
        assert loaded[0]["objective"]["numeric"]["matched"] == 5
        assert loaded[1]["gate"]["status"] == "reject"
        assert loaded[1]["objective"] is None        # reject-путь честно не строит obj_report
        assert loaded[0]["versions"]["profile"]
        assert loaded[0]["versions"]["prompt_set_version"] == judge.PROMPT_SET_VERSION
        assert loaded[0]["versions"]["taxonomy_version"] == judge.TAXONOMY_VERSION
        assert loaded[0]["versions"]["decision_table_version"] == aggregator.DECISION_TABLE_VERSION
        assert loaded[0]["versions"]["thresholds"]["gate.MIN_ENTITY_RECALL"] == gate.MIN_ENTITY_RECALL
        # хэши детерминированы и различны для разных текстов
        assert loaded[0]["source_hash"] != loaded[1]["source_hash"]
        assert _hash_text("то же") == _hash_text("то же")
        # сырой текст в лог не попал
        dump = json.dumps(loaded, ensure_ascii=False)
        assert "Исходный текст ЭМК" not in dump and "Другой исходник" not in dump

        print("OK: AuditEntry/AuditLogger — запись, JSONL-формат, версии, хэширование без сырого текста")
        print(f"  файл: {logger.path.name}, записей: {logger.count}")
        print("  пример versions-среза:")
        print(json.dumps(loaded[0]["versions"], ensure_ascii=False, indent=2))
        print()
        print("Самопроверка audit.py — OK")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _self_check()
