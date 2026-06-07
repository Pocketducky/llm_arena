"""
Блок 6 — валидационный контур, часть 2: корреляция авто-оценки пайплайна
с экспертной разметкой.

План (Блок 6): «Модуль корреляции: расчёт Spearman ρ и F1 (бинарная
классификация «приемлемо/неприемлемо») между авто-оценкой и экспертной
разметкой — инфраструктура готовится сейчас, наполняется данными по мере
поступления экспертных оценок (Этап 1 дизайна)».

На момент написания (2026-06) экспертная разметка ЕЩЁ НЕ ПОЛУЧЕНА — модуль
полностью готов к запуску (загрузчик + расчёты + отчёт), протестирован на
заглушках (`_self_check`); как только появится файл с оценками экспертов,
сценарий использования:

    experts = load_expert_annotations("materials/экспертная_разметка.xlsx")
    autos   = [AutoRecord.from_evaluation(case_id, finalize(...)) for ...]
    report  = correlate(autos, experts)
    print(report.render())

ИСТОРИЯ ОГРАНИЧЕНИЯ ОКРУЖЕНИЯ (зафиксировано при проектировании, с тех пор
устранено): на момент написания `scipy` не был установлен
(`ModuleNotFoundError: No module named 'scipy'`), а
`pandas.Series.corr(method="spearman")` ВНУТРИ САМ дёргает
`scipy.stats.spearmanr` (`pandas/core/nanops.py:1670`) — то есть ни прямой,
ни «обходной через pandas» путь не работал. По разрешению пользователя
`scipy` установлен (`pip install scipy`), поэтому `_spearman` теперь
ИСПОЛЬЗУЕТ `scipy.stats.spearmanr` напрямую — это даёт ещё и p-value
(требует CDF t-распределения из `scipy.stats`, раньше было недоступно).
Ручная реализация (ранговое преобразование + Пирсон от рангов — точное,
не приближённое определение Spearman ρ: ρ = Pearson(rank(x), rank(y)))
оставлена как `_spearman_manual` и используется как ПРОЗРАЧНЫЙ fallback,
если scipy вдруг недоступен в другом окружении (например, при переносе
кода) — числа обоих путей идентичны (см. сверку в `_self_check`), отличие
только в p-value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except ImportError:                                            # pragma: no cover
    _scipy_spearmanr = None

log = logging.getLogger("correlation")

# ══════════════════════════════════════════════════════════════════
# ОБЩАЯ ШКАЛА КАТЕГОРИЙ
# ══════════════════════════════════════════════════════════════════

# Порядок «от лучшей к худшей» — основа и для рангового сопоставления
# (Spearman), и для бинаризации «приемлемо/неприемлемо». Совпадает с
# CATEGORIES из aggregator.py (сознательно не импортируем оттуда напрямую —
# модуль корреляции должен уметь сравнивать и с историческими/внешними
# источниками авто-оценки, использующими ту же таксономию по названию).
_CATEGORY_RANK = {
    "Готово к клиническому применению": 0,
    "Требует редактирования": 1,
    "Неприемлемо": 2,
}


def to_binary(category: Optional[str], *, strict: bool = False) -> Optional[bool]:
    """
    Бинаризация категории в «приемлемо / неприемлемо» (план, Блок 6: «F1
    (бинарная классификация «приемлемо/неприемлемо»)»).

    strict=False (по умолчанию) — приемлемо = «Готово...» ИЛИ «Требует
        редактирования» (карта пригодна к использованию, пусть и после
        правки); неприемлемо — только фундаментальный провал. Это прочтение
        соответствует тому, как категория «Неприемлемо» выделена в дизайне
        (стоп-правило E1, «фундаментальные пропуски критической информации»).
    strict=True — приемлемо = ТОЛЬКО «Готово к клиническому применению»;
        более жёсткое прочтение — пригодится, если экспертная шкала
        предполагает именно его (имеет смысл сравнить оба варианта на
        малой выборке, когда она появится).

    Возвращает None для нераспознанной категории (опечатка в разметке,
    непредвиденное значение) — такие записи не участвуют в расчёте F1 и не
    искажают его молчаливой подменой на "False".
    """
    if category is None:
        return None
    rank = _CATEGORY_RANK.get(category.strip())
    if rank is None:
        return None
    return (rank == 0) if strict else (rank <= 1)


# ══════════════════════════════════════════════════════════════════
# ЗАПИСИ: ЭКСПЕРТ И АВТО-ОЦЕНКА
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExpertRecord:
    """Одна запись экспертной разметки для одной пары (карта, суммаризация)."""
    case_id: str
    expert_category: Optional[str] = None     # категория эксперта (если шкала — категориальная)
    expert_score: Optional[float] = None      # числовой балл эксперта (если шкала — числовая)
    note: Optional[str] = None


@dataclass
class AutoRecord:
    """Одна запись авто-оценки пайплайна — для сопоставления с экспертом."""
    case_id: str
    auto_category: Optional[str] = None
    auto_score: Optional[float] = None

    @classmethod
    def from_evaluation(cls, case_id: str, evaluation: dict) -> "AutoRecord":
        """Строит запись из результата `aggregator.finalize(...)` —
        удобная точка входа для скрипта прогона корреляции на реальных
        данных (берёт детерминированную, авторитетную `category`, см.
        Блок 5 — а не `llm_category`, которая лишь предложение модели)."""
        return cls(case_id=case_id, auto_category=evaluation.get("category"))


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "case_id": ("case_id", "id", "карта", "карточка", "пара", "идентификатор", "файл"),
    "category": ("category", "категория", "вердикт", "verdict", "класс"),
    "score": ("score", "балл", "оценка", "expert_score", "балл_эксперта", "итоговый_балл"),
    "note": ("note", "комментарий", "примечание", "comment", "обоснование"),
}


def _resolve_columns(columns) -> dict[str, str]:
    """Сопоставляет реальные заголовки колонок с ожидаемыми ролями по
    алиасам — СОВПАДЕНИЕ ПО ПОДСТРОКЕ (не точное), потому что реальные
    заголовки могут быть составными («Категория эксперта», «Итоговая
    оценка (балл)») — формат файла экспертов на момент написания ещё
    не зафиксирован, и жёсткая точная сверка была бы хрупкой."""
    norm = [(str(c).strip().lower(), c) for c in columns]
    out: dict[str, str] = {}
    used: set[str] = set()
    for key, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            match = next((orig for low, orig in norm
                          if alias in low and orig not in used), None)
            if match is not None:
                out[key] = match
                used.add(match)
                break
    return out


def _clean_str(value) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _clean_float(value) -> Optional[float]:
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    return None


def load_expert_annotations(path: str, *, sheet: Optional[str] = None) -> list[ExpertRecord]:
    """
    Загружает экспертную разметку из Excel (.xlsx/.xls) или CSV.

    Колонки распознаются гибко, по алиасам (см. `_COLUMN_ALIASES`) — формат
    итогового файла экспертов на момент написания ещё не зафиксирован,
    поэтому жёстко завязываться на конкретные имена было бы преждевременно.
    Обязательна только колонка идентификатора пары (`case_id`/`id`/`карта`/…
    — должна соответствовать `case_id`, под которым прогон сохраняет
    собственные результаты, иначе сопоставление в `correlate` ничего не найдёт).
    Колонки категории/балла — любая из них или обе (используются по-разному:
    категория — для F1 и (если нет числового балла) для Spearman; балл —
    для Spearman напрямую).
    """
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    colmap = _resolve_columns(df.columns)
    if "case_id" not in colmap:
        raise ValueError(
            f"В файле {path} не найдена колонка-идентификатор пары "
            f"(ожидались алиасы: {_COLUMN_ALIASES['case_id']}). "
            f"Колонки в файле: {list(df.columns)}")
    if "category" not in colmap and "score" not in colmap:
        raise ValueError(
            f"В файле {path} не найдена ни колонка категории, ни колонка балла — "
            f"экспертную оценку не из чего извлечь. Колонки в файле: {list(df.columns)}")

    records: list[ExpertRecord] = []
    for _, row in df.iterrows():
        case_id = _clean_str(row[colmap["case_id"]]) or _clean_str(str(row[colmap["case_id"]]))
        if not case_id:
            continue
        records.append(ExpertRecord(
            case_id=case_id,
            expert_category=_clean_str(row[colmap["category"]]) if "category" in colmap else None,
            expert_score=_clean_float(row[colmap["score"]]) if "score" in colmap else None,
            note=_clean_str(row[colmap["note"]]) if "note" in colmap else None,
        ))
    log.info("correlation: загружено %d экспертных записей из %s", len(records), path)
    return records


# ══════════════════════════════════════════════════════════════════
# SPEARMAN ρ — БЕЗ scipy (см. докстринг модуля)
# ══════════════════════════════════════════════════════════════════

def _spearman_manual(xs: list[float], ys: list[float]) -> Optional[float]:
    """
    ρ = коэффициент корреляции Пирсона от РАНГОВ — точное определение
    Spearman ρ (не аппроксимация); ранги берутся через `Series.rank()`
    со средними рангами на связках, что идентично `scipy.stats.rankdata`
    с `method="average"` (поведение по умолчанию и у scipy, и у pandas).
    Используется как прозрачный fallback и для сверки с scipy в `_self_check`
    (см. докстринг модуля — раньше это был ЕДИНСТВЕННЫЙ доступный путь).
    """
    sx, sy = pd.Series(xs, dtype="float64"), pd.Series(ys, dtype="float64")
    return float(sx.rank().corr(sy.rank()))   # Pearson(rank(x), rank(y)) ≡ Spearman ρ


def _spearman(xs: list[float], ys: list[float]) -> tuple[Optional[float], Optional[float], str]:
    """
    Возвращает (ρ, p-value, пояснение).

    ρ = None, если наблюдений недостаточно (n < 3) либо одна из выборок
    константна на пересечении (корреляция не определена). p-value
    вычисляется только если доступен scipy (см. докстринг модуля про
    ранее обнаруженное и с тех пор устранённое отсутствие зависимости) —
    иначе возвращается None с пояснением и используется ручной fallback,
    дающий идентичное ρ, но без p-value.
    """
    n = len(xs)
    if n < 3:
        return None, None, f"недостаточно наблюдений для ρ (n={n} < 3)"
    sx, sy = pd.Series(xs, dtype="float64"), pd.Series(ys, dtype="float64")
    if sx.nunique() < 2 or sy.nunique() < 2:
        return None, None, "одна из выборок константна на пересечении — ρ не определена"

    if _scipy_spearmanr is not None:
        result = _scipy_spearmanr(xs, ys)
        rho, pvalue = float(result.statistic), float(result.pvalue)
        return (round(rho, 4), round(pvalue, 4),
                f"ρ = scipy.stats.spearmanr, n={n}, p-value={round(pvalue, 4)}")

    rho = _spearman_manual(xs, ys)
    return (round(rho, 4), None,
            f"ρ = Pearson(ранги обеих выборок) [scipy недоступен — ручной fallback], "
            f"n={n}; p-value недоступен без scipy")


# ══════════════════════════════════════════════════════════════════
# F1 ПО БИНАРНОЙ КЛАССИФИКАЦИИ «ПРИЕМЛЕМО / НЕПРИЕМЛЕМО»
# ══════════════════════════════════════════════════════════════════

def _binary_metrics(auto_bin: list[bool], expert_bin: list[bool]) -> Optional[dict]:
    """
    Считает precision/recall/F1/accuracy для бинарной классификации.

    Положительный класс — «НЕПРИЕМЛЕМО» (to_binary -> False). Это
    осознанный выбор, согласованный с философией stop-rule E1 в дизайне:
    нас прежде всего интересует, ловит ли автоматика именно те карты,
    которые эксперт счёл неприемлемыми — пропуск такой карты (FN здесь)
    дороже ложной тревоги (FP). Полная матрица ошибок приводится целиком —
    пересчитать с обратным положительным классом из неё тривиально.
    """
    pairs = list(zip(auto_bin, expert_bin))
    if not pairs:
        return None
    tp = sum(1 for a, e in pairs if not a and not e)   # оба: «неприемлемо» — поймали верно
    fp = sum(1 for a, e in pairs if not a and e)       # авто: неприемлемо, эксперт: приемлемо — ложная тревога
    fn = sum(1 for a, e in pairs if a and not e)       # авто: приемлемо, эксперт: неприемлемо — ОПАСНЫЙ ПРОПУСК
    tn = sum(1 for a, e in pairs if a and e)           # оба: «приемлемо»
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0
          else None)
    accuracy = (tp + tn) / len(pairs)
    return {
        "n": len(pairs),
        "positive_class": "Неприемлемо",
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "accuracy": round(accuracy, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# ══════════════════════════════════════════════════════════════════
# ОТЧЁТ И ГЛАВНАЯ ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════

@dataclass
class CorrelationReport:
    n_matched: int
    n_spearman: int
    spearman_rho: Optional[float]
    spearman_pvalue: Optional[float]
    spearman_note: str
    binary: Optional[dict]
    matched_ids: list[str] = field(default_factory=list)
    unmatched_auto: list[str] = field(default_factory=list)
    unmatched_expert: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "=== Корреляция авто-оценки с экспертной разметкой ===",
            f"Сопоставлено пар по идентификатору: {self.n_matched}"
            f"  (не нашлось пары для {len(self.unmatched_auto)} авто-записей "
            f"и {len(self.unmatched_expert)} экспертных)",
            "",
            f"Spearman ρ: {self.spearman_rho if self.spearman_rho is not None else 'н/д'}"
            + (f"  (p-value={self.spearman_pvalue})" if self.spearman_pvalue is not None else "")
            + f"  [{self.spearman_note}]",
            "",
        ]
        if self.binary:
            b = self.binary
            lines += [
                f"Бинарная классификация «приемлемо/неприемлемо» "
                f"(положительный класс — «{b['positive_class']}», n={b['n']}):",
                f"  precision={b['precision']}  recall={b['recall']}  "
                f"F1={b['f1']}  accuracy={b['accuracy']}",
                f"  матрица ошибок: TP={b['confusion']['tp']} FP={b['confusion']['fp']} "
                f"FN={b['confusion']['fn']} TN={b['confusion']['tn']}",
                "  (TP/TN — совпадение с экспертом; FP — ложная тревога автоматики; "
                "FN — ОПАСНЫЙ пропуск: эксперт счёл карту неприемлемой, автоматика — нет)",
            ]
        else:
            lines.append("Бинарная классификация: недостаточно данных "
                         "(нет пар с распознанными категориями с обеих сторон).")
        if self.unmatched_auto:
            lines.append(f"\nАвто-записи без пары у эксперта: {self.unmatched_auto[:10]}"
                         + (" …" if len(self.unmatched_auto) > 10 else ""))
        if self.unmatched_expert:
            lines.append(f"Экспертные записи без пары в авто-результатах: {self.unmatched_expert[:10]}"
                         + (" …" if len(self.unmatched_expert) > 10 else ""))
        if self.n_matched == 0:
            lines.append("\n(Экспертная разметка пока не поступила/не сопоставилась — "
                         "это ОЖИДАЕМО на текущем этапе: план относит наполнение "
                         "данными к моменту получения хотя бы небольшой экспертной "
                         "выборки. Инфраструктура готова и проверена на заглушках.)")
        return "\n".join(lines)


def correlate(auto_records: list[AutoRecord], expert_records: list[ExpertRecord], *,
              strict: bool = False) -> CorrelationReport:
    """
    Главная точка входа: сопоставляет авто-оценки и экспертную разметку по
    `case_id`, считает Spearman ρ (порядковое согласие по тяжести вердикта)
    и F1/precision/recall (согласие по бинарному «приемлемо/неприемлемо»).

    Для Spearman порядковое значение берётся:
      • у авто-записи — из `auto_category` через `_CATEGORY_RANK` (если нет
        категории — из `auto_score` напрямую);
      • у экспертной записи — из `expert_score`, если он задан, иначе —
        из `expert_category` через ту же шкалу `_CATEGORY_RANK`.
    ВНИМАНИЕ: если эксперт даёт числовой балл по СВОЕЙ шкале, направление
    должно совпадать с конвенцией `_CATEGORY_RANK` («меньше = лучше») —
    иначе знак ρ нужно интерпретировать в обратную сторону (величина |ρ|
    от направления не зависит и останется корректной мерой согласия).
    """
    expert_by_id = {r.case_id: r for r in expert_records}
    auto_by_id = {r.case_id: r for r in auto_records}
    common_ids = sorted(set(auto_by_id) & set(expert_by_id))
    unmatched_auto = sorted(set(auto_by_id) - set(expert_by_id))
    unmatched_expert = sorted(set(expert_by_id) - set(auto_by_id))

    auto_vals: list[float] = []
    expert_vals: list[float] = []
    auto_bin: list[bool] = []
    expert_bin: list[bool] = []

    for cid in common_ids:
        a, e = auto_by_id[cid], expert_by_id[cid]

        a_rank = _CATEGORY_RANK.get(a.auto_category) if a.auto_category else None
        if a_rank is None:
            a_rank = a.auto_score
        e_val = e.expert_score
        if e_val is None:
            e_val = _CATEGORY_RANK.get(e.expert_category) if e.expert_category else None
        if a_rank is not None and e_val is not None:
            auto_vals.append(float(a_rank))
            expert_vals.append(float(e_val))

        a_b = to_binary(a.auto_category, strict=strict)
        e_b = to_binary(e.expert_category, strict=strict)
        if a_b is not None and e_b is not None:
            auto_bin.append(a_b)
            expert_bin.append(e_b)

    rho, pvalue, rho_note = _spearman(auto_vals, expert_vals)
    binary = _binary_metrics(auto_bin, expert_bin)

    return CorrelationReport(
        n_matched=len(common_ids), n_spearman=len(auto_vals),
        spearman_rho=rho, spearman_pvalue=pvalue, spearman_note=rho_note, binary=binary,
        matched_ids=common_ids, unmatched_auto=unmatched_auto, unmatched_expert=unmatched_expert,
    )


# ══════════════════════════════════════════════════════════════════
# САМОПРОВЕРКА
# ══════════════════════════════════════════════════════════════════

def _self_check() -> None:
    import os
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Spearman ρ — теперь через scipy.stats.spearmanr (установлен по
    #    разрешению пользователя), плюс сверка с ручной реализацией
    #    (_spearman_manual — оставлена как прозрачный fallback, см.
    #    докстринг модуля про историю обнаруженного ограничения).
    print(f"--- Spearman ρ (scipy {'ДОСТУПЕН' if _scipy_spearmanr else 'НЕДОСТУПЕН — используется ручной fallback'}) ---")
    perfect_rho, perfect_p, _ = _spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert perfect_rho == 1.0, perfect_rho
    inverse_rho, inverse_p, _ = _spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert inverse_rho == -1.0, inverse_rho
    xs_known = [86, 97, 99, 100, 101, 103, 106, 110, 112, 113]
    ys_known = [0, 20, 28, 27, 50, 29, 7, 17, 6, 12]
    rho_known, p_known, _ = _spearman(xs_known, ys_known)
    print(f"  возрастающий <-> возрастающий: ρ={perfect_rho}, p={perfect_p} (ожидали ρ=1.0)")
    print(f"  возрастающий <-> убывающий:    ρ={inverse_rho}, p={inverse_p} (ожидали ρ=-1.0)")
    print(f"  учебный пример: ρ={rho_known}, p={p_known}")
    none_rho, none_p, note = _spearman([1, 1, 1], [1, 2, 3])
    assert none_rho is None and none_p is None
    print(f"  константная выборка -> {none_rho} [{note}]")

    # Сверка scipy <-> ручной реализации — числа ОБЯЗАНЫ совпасть (см.
    # докстринг _spearman_manual: это два пути к одной и той же формуле).
    if _scipy_spearmanr is not None:
        manual_rho = round(_spearman_manual(xs_known, ys_known), 4)
        assert abs(manual_rho - rho_known) < 1e-9, (manual_rho, rho_known)
        print(f"  сверка scipy<->ручной фоллбэк: scipy ρ={rho_known}  ручной ρ={manual_rho}  "
              f"(совпадают => ручная реализация — точная, не аппроксимация)")

    # 2. to_binary
    print("\n--- бинаризация категорий ---")
    assert to_binary("Готово к клиническому применению") is True
    assert to_binary("Требует редактирования") is True
    assert to_binary("Неприемлемо") is False
    assert to_binary("Требует редактирования", strict=True) is False
    assert to_binary("какая-то ерунда") is None
    assert to_binary(None) is None
    print("  OK — нестрогая/строгая бинаризация и обработка нераспознанных категорий")

    # 3. Полный сценарий correlate() на заглушках (имитация: 8 «карт»,
    #    разное согласие авто/эксперта — включая «опасный пропуск»).
    autos = [
        AutoRecord("card_01", auto_category="Готово к клиническому применению"),
        AutoRecord("card_02", auto_category="Требует редактирования"),
        AutoRecord("card_03", auto_category="Неприемлемо"),
        AutoRecord("card_04", auto_category="Готово к клиническому применению"),
        AutoRecord("card_05", auto_category="Требует редактирования"),
        AutoRecord("card_06", auto_category="Неприемлемо"),
        AutoRecord("card_07", auto_category="Готово к клиническому применению"),  # «опасный пропуск» ниже
        AutoRecord("card_08", auto_category="Требует редактирования"),
        AutoRecord("card_09_без_пары", auto_category="Готово к клиническому применению"),
    ]
    experts = [
        ExpertRecord("card_01", expert_category="Готово к клиническому применению"),
        ExpertRecord("card_02", expert_category="Требует редактирования"),
        ExpertRecord("card_03", expert_category="Неприемлемо"),
        ExpertRecord("card_04", expert_category="Требует редактирования"),   # лёгкое расхождение
        ExpertRecord("card_05", expert_category="Требует редактирования"),
        ExpertRecord("card_06", expert_category="Неприемлемо"),
        ExpertRecord("card_07", expert_category="Неприемлемо"),              # !! опасный пропуск авто
        ExpertRecord("card_08", expert_category="Готово к клиническому применению"),
        ExpertRecord("card_10_без_пары", expert_category="Неприемлемо"),
    ]
    report = correlate(autos, experts)
    print("\n--- correlate() на заглушках (реальной разметки эксперта пока нет) ---")
    print(report.render())

    assert report.n_matched == 8
    assert report.unmatched_auto == ["card_09_без_пары"]
    assert report.unmatched_expert == ["card_10_без_пары"]
    assert report.spearman_rho is not None
    assert report.binary is not None
    # card_07: авто=«Готово» (rank 0 -> приемлемо), эксперт=«Неприемлемо» (неприемлемо)
    # — это ровно тот самый «опасный пропуск», FN в матрице с положительным классом «Неприемлемо».
    assert report.binary["confusion"]["fn"] >= 1, "тестовый «опасный пропуск» не обнаружен в FN"
    print(f"\n  OK — сопоставлено {report.n_matched}/9 авто-записей, "
          f"обнаружен заложенный «опасный пропуск» (FN={report.binary['confusion']['fn']})")

    # 4. load_expert_annotations — на временном CSV с гибкими названиями колонок.
    print("\n--- load_expert_annotations (гибкое распознавание колонок, временный CSV) ---")
    tmp_path = os.path.join(tempfile.gettempdir(), "_correlation_selfcheck_experts.csv")
    pd.DataFrame({
        "Карта": ["card_01", "card_02", "card_03"],
        "Категория эксперта": ["Готово к клиническому применению",
                               "Требует редактирования", "Неприемлемо"],
        "Комментарий": ["ок", "мелкие правки", "пропущено отрицание"],
    }).to_csv(tmp_path, index=False, encoding="utf-8")
    try:
        loaded = load_expert_annotations(tmp_path)
        assert len(loaded) == 3
        assert loaded[0].case_id == "card_01"
        assert loaded[0].expert_category == "Готово к клиническому применению"
        assert loaded[2].note == "пропущено отрицание"
        print(f"  OK — загружено {len(loaded)} записей, колонки распознаны по алиасам "
              f"(«Карта»->case_id, «Категория эксперта»->category, «Комментарий»->note)")
    finally:
        os.remove(tmp_path)

    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ — модуль готов; "
          "наполнение реальными данными — по получении экспертной разметки (план, Этап 1).")


if __name__ == "__main__":
    _self_check()
