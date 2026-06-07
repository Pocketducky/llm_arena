"""
drift.py — Блок 7: скелет мониторинга дрейфа.

Назначение (план, Блок 7): «скелет drift-мониторинга: сохранение
распределений баллов по времени + пороговые алерты (наполняется по
мере накопления продуктивных прогонов)».

Это ИНФРАСТРУКТУРА — построенная и самопротестированная на синтетических
снимках уже сейчас (тот же принцип, что и в Блоке 6 для correlation.py:
«инфраструктура готова, наполняется данными по мере накопления продуктивных
прогонов»). Реальные алерты появятся, когда накопится 2+ прогонов одного и
того же профиля на сопоставимых данных — раньше сравнивать дрейф попросту
не с чем.

Откуда берутся данные:
  • снимок строится из записей audit-лога (audit.load_entries) —
    то есть drift-мониторинг работает НАД тем же JSONL, не требуя
    отдельного хранилища: один источник правды для аудита и дрейфа;
  • снимок = распределение итоговых категорий + распределение
    объективных метрик (доля числовых/полярных расхождений, частота
    срабатывания шлюза) для одного run_id/профиля/среза по времени.

Что считается «дрейфом» здесь: значимое изменение этих распределений
между двумя снимками одного профиля — например, резкий рост доли
«Неприемлемо» или доли срабатываний шлюза при том же наборе моделей
может означать (а) деградацию промптов/моделей, (б) изменение состава
входных данных, (в) баг в пайплайне. Различить эти причины автоматика
не может — алерт лишь указывает «здесь что-то изменилось, посмотрите».
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import audit

log = logging.getLogger("drift")

DRIFT_DIR = Path("audit_log") / "drift_snapshots"

# Категории, как их возвращает aggregator (см. aggregator.CATEGORIES) —
# дублируем здесь строками, а не импортом enum'а, т.к. снимок должен
# уметь читать и записи, сделанные более старыми версиями кода.
CATEGORY_ORDER = ("Готово к клиническому применению", "Требует редактирования",
                  "Неприемлемо", "ошибка")

# ── Пороги алертов по умолчанию — НАЧАЛЬНОЕ приближение ──────────────
# Обоснование выбора цифр: «жёсткая» категория («Неприемлемо») и
# отклонения шлюза — это именно то, рост доли чего сигнализирует о
# регрессии безопасности (см. философию E1 — «пропуск опасной карты
# дороже ложной тревоги»), поэтому порог на них ниже (чувствительнее),
# чем на общий сдвиг распределения категорий. Цифры — первое разумное
# приближение для калибровки на реальных повторных прогонах; план прямо
# говорит, что скелет «наполняется по мере накопления данных» — корректировка
# этих порогов и есть часть такого наполнения.
DEFAULT_THRESHOLDS = {
    "reject_rate_delta": 0.10,     # +10 п.п. доли «Неприемлемо» между прогонами
    "gate_reject_rate_delta": 0.10,  # +10 п.п. доли отклонений шлюза
    "category_distribution_delta": 0.15,  # суммарный |Δдоли| по всем категориям
}


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


@dataclass
class DriftSnapshot:
    """
    Снимок распределений на момент одного прогона (run_id).

    n                   — число оценённых пар (ЭМК × модель)
    category_counts     — {категория: количество} по итоговой `category`
    category_rates      — то же в долях от n
    gate_status_counts  — {pass/rework/reject: количество}
    gate_status_rates   — то же в долях
    objective_means     — средние по объективным метрикам там, где
                          objective != None (доля пар с числовыми/
                          полярными расхождениями, средний entity-F1,
                          если сущности извлекались)
    by_model            — то же распределение category_rates в разрезе
                          model_id — драйфы часто специфичны для модели
    """
    run_id: str
    timestamp: str
    profile: Optional[str]
    n: int
    category_counts: dict[str, int]
    category_rates: dict[str, float]
    gate_status_counts: dict[str, int]
    gate_status_rates: dict[str, float]
    objective_means: dict[str, float]
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "timestamp": self.timestamp, "profile": self.profile,
            "n": self.n,
            "category_counts": self.category_counts, "category_rates": self.category_rates,
            "gate_status_counts": self.gate_status_counts, "gate_status_rates": self.gate_status_rates,
            "objective_means": self.objective_means, "by_model": self.by_model,
        }

    def save(self, *, directory: Path = DRIFT_DIR) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.snapshot.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "DriftSnapshot":
        return cls(run_id=d["run_id"], timestamp=d["timestamp"], profile=d.get("profile"),
                   n=d["n"], category_counts=d["category_counts"], category_rates=d["category_rates"],
                   gate_status_counts=d["gate_status_counts"], gate_status_rates=d["gate_status_rates"],
                   objective_means=d["objective_means"], by_model=d.get("by_model", {}))

    def render(self) -> str:
        lines = [f"Снимок дрейфа: {self.run_id}  ({self.timestamp}, профиль={self.profile or '—'})",
                 f"  Пар оценено: {self.n}", "  Категории:"]
        for cat in CATEGORY_ORDER:
            if cat in self.category_counts:
                lines.append(f"    {cat:<32} {self.category_counts[cat]:>4}  "
                             f"({self.category_rates[cat]:.1%})")
        lines.append("  Статусы шлюза:")
        for status, cnt in sorted(self.gate_status_counts.items()):
            lines.append(f"    {status:<10} {cnt:>4}  ({self.gate_status_rates[status]:.1%})")
        if self.objective_means:
            lines.append("  Объективный слой (средние по парам с objective != None):")
            for k, v in self.objective_means.items():
                lines.append(f"    {k:<32} {v:.3f}")
        return "\n".join(lines)


def compute_snapshot(entries: list[dict], *, run_id: Optional[str] = None,
                      profile: Optional[str] = None) -> DriftSnapshot:
    """
    Строит снимок из списка записей audit-лога (формат audit.AuditEntry,
    как возвращает audit.load_entries). Если run_id/profile не заданы —
    берутся из первой записи (предполагается, что снимок строится по
    записям ОДНОГО прогона — иначе сравнение между снимками теряет смысл).
    """
    if not entries:
        raise ValueError("compute_snapshot: пустой список записей — снимок строить не из чего")

    run_id = run_id or entries[0].get("run_id") or "unknown-run"
    profile = profile or (entries[0].get("versions") or {}).get("profile")
    n = len(entries)

    category_counts = Counter(e.get("category", "—") for e in entries)
    gate_counts = Counter((e.get("gate") or {}).get("status", "—") for e in entries)

    category_rates = {k: round(_safe_div(v, n), 4) for k, v in category_counts.items()}
    gate_rates = {k: round(_safe_div(v, n), 4) for k, v in gate_counts.items()}

    # Объективные средние — только по записям, где objective != None
    # (gate reject честно его не строит, см. judge.evaluate_summary)
    obj_entries = [e["objective"] for e in entries if e.get("objective")]
    objective_means: dict[str, float] = {}
    if obj_entries:
        def _rate_with(key_outer, key_count, key_total):
            vals = []
            for o in obj_entries:
                blk = o.get(key_outer) or {}
                total = blk.get(key_total, 0)
                if total:
                    vals.append(_safe_div(blk.get(key_count, 0), total))
            return sum(vals) / len(vals) if vals else 0.0

        objective_means["доля_числовых_расхождений"] = round(
            _rate_with("numeric", "mismatch_count", "total_in_a")
            + _rate_with("numeric", "unit_mismatch_count", "total_in_a"), 4)
        objective_means["доля_полярных_инверсий"] = round(_rate_with("polarity", "flip_count", "total_in_a"), 4)

        f1s = []
        for o in obj_entries:
            ent = o.get("entities")
            if ent:
                f1s.extend(rep["f1"] for rep in ent.values())
        if f1s:
            objective_means["средний_entity_F1"] = round(sum(f1s) / len(f1s), 4)

        objective_means["доля_пар_с_objective"] = round(_safe_div(len(obj_entries), n), 4)

    by_model: dict[str, dict[str, float]] = {}
    models = sorted({e.get("model_id", "—") for e in entries})
    for model in models:
        sub = [e for e in entries if e.get("model_id") == model]
        sub_counts = Counter(e.get("category", "—") for e in sub)
        by_model[model] = {k: round(_safe_div(v, len(sub)), 4) for k, v in sub_counts.items()}

    return DriftSnapshot(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        profile=profile, n=n,
        category_counts=dict(category_counts), category_rates=category_rates,
        gate_status_counts=dict(gate_counts), gate_status_rates=gate_rates,
        objective_means=objective_means, by_model=by_model,
    )


def snapshot_from_audit_log(path, **kw) -> DriftSnapshot:
    """Удобный путь «от файла JSONL аудита сразу к снимку»."""
    return compute_snapshot(audit.load_entries(path), **kw)


def load_snapshots(directory: Path = DRIFT_DIR) -> list[DriftSnapshot]:
    """Загружает все сохранённые снимки из каталога, отсортированные по timestamp."""
    directory = Path(directory)
    if not directory.exists():
        return []
    snaps = []
    for p in sorted(directory.glob("*.snapshot.json")):
        snaps.append(DriftSnapshot.from_dict(json.loads(p.read_text(encoding="utf-8"))))
    snaps.sort(key=lambda s: s.timestamp)
    return snaps


@dataclass
class DriftAlert:
    """Один сигнал «между снимками baseline -> current показатель сдвинулся сильнее порога»."""
    metric: str
    baseline_value: float
    current_value: float
    delta: float
    threshold: float
    severity: str   # "внимание" | "тревога" — пока единая шкала; калибруется по данным
    note: str = ""

    def render(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        return (f"  [{self.severity.upper()}] {self.metric}: "
                f"{self.baseline_value:.1%} → {self.current_value:.1%}  "
                f"(Δ={sign}{self.delta:.1%}, порог={self.threshold:.1%})"
                + (f"  — {self.note}" if self.note else ""))


def compare_snapshots(baseline: DriftSnapshot, current: DriftSnapshot,
                       *, thresholds: Optional[dict] = None) -> list[DriftAlert]:
    """
    Сравнивает два снимка ОДНОГО профиля и возвращает список алертов —
    метрик, сдвинувшихся сильнее порога. Пустой список — «дрейфа не
    обнаружено» (в пределах текущих, первоначально откалиброванных порогов).

    Намеренно НЕ пытается определить причину сдвига (промпты/модели/
    данные/баг) — это решение за человеком; алерт лишь указывает,
    где искать.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    alerts: list[DriftAlert] = []

    if baseline.profile and current.profile and baseline.profile != current.profile:
        alerts.append(DriftAlert(
            metric="profile", baseline_value=0.0, current_value=0.0, delta=0.0, threshold=0.0,
            severity="внимание",
            note=f"профиль изменился: '{baseline.profile}' → '{current.profile}' — "
                 f"сравнение распределений между разными профилями некорректно по построению"))
        return alerts

    b_reject = baseline.category_rates.get("Неприемлемо", 0.0)
    c_reject = current.category_rates.get("Неприемлемо", 0.0)
    d_reject = c_reject - b_reject
    if abs(d_reject) >= th["reject_rate_delta"]:
        alerts.append(DriftAlert(
            metric="доля категории «Неприемлемо»", baseline_value=b_reject, current_value=c_reject,
            delta=d_reject, threshold=th["reject_rate_delta"],
            severity="тревога" if d_reject > 0 else "внимание",
            note="рост доли отклонённых пар — кандидат на регрессию (промпты/модель/данные)"
                 if d_reject > 0 else "доля отклонённых пар заметно снизилась"))

    b_greject = baseline.gate_status_rates.get("reject", 0.0)
    c_greject = current.gate_status_rates.get("reject", 0.0)
    d_greject = c_greject - b_greject
    if abs(d_greject) >= th["gate_reject_rate_delta"]:
        alerts.append(DriftAlert(
            metric="доля отклонений шлюза (Блок 3)", baseline_value=b_greject, current_value=c_greject,
            delta=d_greject, threshold=th["gate_reject_rate_delta"],
            severity="тревога" if d_greject > 0 else "внимание",
            note="шлюз стал отклонять заметно чаще/реже — проверить пороги объективного слоя "
                 "и состав входных данных"))

    cats = set(baseline.category_rates) | set(current.category_rates)
    total_delta = sum(abs(current.category_rates.get(c, 0.0) - baseline.category_rates.get(c, 0.0))
                      for c in cats)
    if total_delta >= th["category_distribution_delta"]:
        alerts.append(DriftAlert(
            metric="суммарный сдвиг распределения категорий",
            baseline_value=0.0, current_value=total_delta, delta=total_delta,
            threshold=th["category_distribution_delta"], severity="внимание",
            note="совокупный |Δдоли| по всем категориям превысил порог — "
                 "посмотреть распределение по моделям (by_model) для локализации"))

    return alerts


def render_alerts(alerts: list[DriftAlert], *, baseline: DriftSnapshot, current: DriftSnapshot) -> str:
    lines = [f"Сравнение снимков дрейфа: {baseline.run_id} (baseline) → {current.run_id} (current)"]
    if not alerts:
        lines.append("  Алертов нет — отклонения в пределах текущих порогов "
                     f"({json.dumps(DEFAULT_THRESHOLDS, ensure_ascii=False)}).")
    else:
        lines.append(f"  Найдено алертов: {len(alerts)}")
        for a in alerts:
            lines.append(a.render())
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# САМОПРОВЕРКА — на синтетических снимках (реальных продуктивных
# прогонов «во времени» пока нет, см. докстринг модуля)
# ══════════════════════════════════════════════════════════════════

def _stub_entries(run_id: str, *, n_reject: int, n_edit: int, n_ready: int,
                  gate_reject: int, profile: str = "pilot") -> list[dict]:
    entries = []
    plan = ([("Неприемлемо", "reject" if i < gate_reject else "pass")
             for i in range(n_reject)]
            + [("Требует редактирования", "pass")] * n_edit
            + [("Готово к клиническому применению", "pass")] * n_ready)
    for i, (cat, gstatus) in enumerate(plan):
        objective = None
        if gstatus != "reject":
            objective = {
                "numeric": {"total_in_a": 10, "matched": 9, "mismatch_count": 1 if cat != "Готово к клиническому применению" else 0,
                            "unit_mismatch_count": 0, "mismatches": [], "unit_mismatches": []},
                "polarity": {"total_in_a": 4, "matched": 4, "flip_count": 0, "flips": []},
                "entities": None,
            }
        entries.append({
            "run_id": run_id, "emr_id": f"EMR_{i:02d}", "model_id": "qwen3:8b",
            "category": cat, "gate": {"status": gstatus, "reasons": []},
            "objective": objective,
            "versions": {"profile": profile}, "e1_signals": {},
        })
    return entries


def _self_check():
    # Снимок A («baseline») — спокойное распределение
    entries_a = _stub_entries("run-A", n_reject=2, n_edit=6, n_ready=12, gate_reject=1)
    snap_a = compute_snapshot(entries_a, run_id="run-A")
    assert snap_a.n == 20
    assert snap_a.category_counts["Неприемлемо"] == 2
    assert abs(snap_a.category_rates["Готово к клиническому применению"] - 12 / 20) < 1e-9
    assert "доля_числовых_расхождений" in snap_a.objective_means

    # Снимок B («current») — заметный рост доли «Неприемлемо» и reject шлюза:
    # имитация регрессии (например, после смены модели роли или промпта)
    entries_b = _stub_entries("run-B", n_reject=8, n_edit=6, n_ready=6, gate_reject=6)
    snap_b = compute_snapshot(entries_b, run_id="run-B")
    assert snap_b.category_counts["Неприемлемо"] == 8

    alerts = compare_snapshots(snap_a, snap_b)
    assert alerts, "ожидался хотя бы один алерт при выраженном сдвиге распределения"
    metrics = {a.metric for a in alerts}
    assert "доля категории «Неприемлемо»" in metrics
    assert "доля отклонений шлюза (Блок 3)" in metrics
    assert all(a.severity == "тревога" for a in alerts if "Неприемлемо" in a.metric or "шлюза" in a.metric)

    # Снимок C — почти идентичный A: алертов быть не должно
    entries_c = _stub_entries("run-C", n_reject=3, n_edit=6, n_ready=11, gate_reject=2)
    snap_c = compute_snapshot(entries_c, run_id="run-C")
    alerts_quiet = compare_snapshots(snap_a, snap_c)
    assert not alerts_quiet, f"не ожидали алертов на близких распределениях, получили: {alerts_quiet}"

    # Смена профиля — отдельный, не метрический алерт
    entries_d = _stub_entries("run-D", n_reject=2, n_edit=6, n_ready=12, gate_reject=1, profile="target")
    snap_d = compute_snapshot(entries_d, run_id="run-D")
    alerts_profile = compare_snapshots(snap_a, snap_d)
    assert len(alerts_profile) == 1 and alerts_profile[0].metric == "profile"

    # save/load round-trip
    import tempfile, shutil
    tmpdir = Path(tempfile.mkdtemp(prefix="drift_selfcheck_"))
    try:
        p1 = snap_a.save(directory=tmpdir)
        p2 = snap_b.save(directory=tmpdir)
        assert p1.exists() and p2.exists()
        loaded = load_snapshots(tmpdir)
        assert len(loaded) == 2
        assert {s.run_id for s in loaded} == {"run-A", "run-B"}
        # повторное сравнение по загруженным с диска снимкам даёт те же алерты
        loaded_by_id = {s.run_id: s for s in loaded}
        alerts2 = compare_snapshots(loaded_by_id["run-A"], loaded_by_id["run-B"])
        assert {a.metric for a in alerts2} == metrics
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("=" * 70)
    print(snap_a.render())
    print()
    print(snap_b.render())
    print()
    print(render_alerts(alerts, baseline=snap_a, current=snap_b))
    print()
    print(render_alerts(alerts_quiet, baseline=snap_a, current=snap_c))
    print("=" * 70)
    print("Самопроверка drift.py — OK (инфраструктура готова; реальные алерты "
          "появятся по мере накопления продуктивных прогонов — см. докстринг)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _self_check()
