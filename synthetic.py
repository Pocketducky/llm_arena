"""
Блок 6 — валидационный контур, часть 1: регрессия на синтетическом
stress-датасете управляемых искажений.

Источник данных: materials/Синтетический_НД_для_тестирования_Сеченовский_университет.xlsx
Структура (установлена инспекцией — см. сопроводительный отчёт):
  • лист "Лист1", 10 клинических случаев (колонки Ж1..Ж5, М1..М5);
  • для каждого случая: строка 0 — длинный исходный нарратив ЭМК
    (~1800-2900 симв., телеграфный стиль), строки 1..38 — последовательность
    КОМПАКТНЫХ суммаризаций (~500-1100 симв.), каждая получена из соседней
    одним-несколькими контролируемыми редактированиями (число, отрицание,
    подмена сущности/слова, опечатка, перестановка, вставка постороннего).
  • разметки типа искажения в файле НЕТ (ни комментариев к ячейкам, ни
    объединённых ячеек) — тип определяется ТОЛЬКО через diff соседних строк,
    как и предписывает план.

Сопоставление «база ↔ искажённый вариант»: берём ПОСЛЕДОВАТЕЛЬНЫЕ пары
(строка i, строка i+1) при i = 1..37 — это ряды одного стиля (суммаризация)
с ровно одним диффом контролируемого вида. Пара (строка 0, строка 1) НЕ
включается: это переход «нарратив ЭМК -> первая суммаризация» (полный
перефраз другого жанра/длины), а не контролируемое искажение — её diff
не классифицируется этим модулем.

Важное архитектурное ограничение, зафиксированное при проектировании:
diff НЕ способен надёжно отличить опасную подмену клинической сущности
(«кардиореанимацию» -> «нейрореанимацию») от безобидного синонимического
перефраза — для этого нужны семантические знания (LLM-сверка сущностей,
Блоки 2-3). Поэтому неоднозначные содержательные лексические замены
маршрутизируются в консервативный тип "лексическая_замена" и считаются
КРИТИЧЕСКИМИ (лучше лишний раз отправить на ручной разбор, чем тихо
пропустить настоящую подмену сущности).
"""

from __future__ import annotations

import glob
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

import gate
import objective_layer

log = logging.getLogger("synthetic")

SHEET_NAME = "Лист1"
ROW_LABEL_COL = "Unnamed: 0"
BASE_NARRATIVE_ROW = 0           # строка 0 — длинный исходный нарратив (исключается из пар)


# ══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ПАР «БАЗА ↔ ИСКАЖЁННЫЙ ВАРИАНТ»
# ══════════════════════════════════════════════════════════════════

@dataclass
class SyntheticPair:
    """Одна пара «база -> искажённый вариант» из синтетического набора."""
    pair_id: str        # например "Ж1#12->13"
    column: str         # клинический случай (Ж1, М3, ...)
    row_a: int          # индекс строки базового варианта
    row_b: int          # индекс строки искажённого варианта
    text_a: str
    text_b: str


def _find_xlsx() -> str:
    candidates = (glob.glob("materials/*.xlsx")
                  + glob.glob("../materials/*.xlsx")
                  + glob.glob("**/Синтетич*.xlsx", recursive=True))
    candidates = [c for c in candidates if "~$" not in c]   # игнорируем временные файлы Excel
    if not candidates:
        raise FileNotFoundError(
            "Не найден синтетический датасет (materials/*.xlsx). "
            "Передайте путь явным аргументом xlsx_path.")
    return candidates[0]


def load_pairs(xlsx_path: Optional[str] = None, *, sheet: str = SHEET_NAME) -> list[SyntheticPair]:
    """
    Загружает синтетический stress-датасет и строит список последовательных
    пар «база -> искажённый вариант» для всех клинических случаев.

    Пары строятся ТОЛЬКО между соседними «суммаризационными» строками
    (1..38) — строка 0 (длинный нарратив ЭМК) пропускается (см. докстринг
    модуля: переход 0->1 — это суммаризация, а не контролируемое искажение).
    """
    path = xlsx_path or _find_xlsx()
    df = pd.read_excel(path, sheet_name=sheet)

    case_columns = [c for c in df.columns if c != ROW_LABEL_COL]
    pairs: list[SyntheticPair] = []
    for col in case_columns:
        texts = df[col]
        n = len(texts)
        for i in range(BASE_NARRATIVE_ROW + 1, n - 1):
            text_a, text_b = texts.iloc[i], texts.iloc[i + 1]
            if not isinstance(text_a, str) or not isinstance(text_b, str):
                continue
            if not text_a.strip() or not text_b.strip():
                continue
            pairs.append(SyntheticPair(
                pair_id=f"{col}#{i}->{i + 1}",
                column=str(col), row_a=i, row_b=i + 1,
                text_a=text_a, text_b=text_b,
            ))
    log.info("synthetic: загружено %d пар из %d клинических случаев (%s)",
             len(pairs), len(case_columns), path)
    return pairs


# ══════════════════════════════════════════════════════════════════
# КЛАССИФИКАЦИЯ ТИПА ИСКАЖЕНИЯ ЧЕРЕЗ DIFF
# ══════════════════════════════════════════════════════════════════

# Таксономия — по плану («число/отрицание/сущность/порядок/опечатка»),
# дополненная двумя типами, которые показал реальный diff-анализ датасета
# (вставка постороннего контента — вплоть до курьёзного «Сорта кофе в
# зернах и их характеристики.» — и сведение «сущность» в более широкий,
# честный тип "лексическая_замена" — см. докстринг модуля).
DISTORTION_TYPES = (
    "число",                # числовое значение или единица измерения изменены
    "отрицание",            # вставлена/удалена частица отрицания
    "лексическая_замена",   # содержательная замена слова/сущности (консервативно — критично)
    "вставка",              # вставлен посторонний фрагмент (галлюцинация-подобное)
    "опечатка",             # опечатка в одном слове (высокое посимвольное сходство)
    "перестановка",         # порядок слов / пунктуация / безобидный перефраз
)

# «Жёсткие» искажения — обязаны ловиться (высокий recall — критерий плана).
CRITICAL_TYPES = frozenset({"число", "отрицание", "лексическая_замена", "вставка"})
# «Безобидные» искажения — не должны давать ложных срабатываний (низкий FPR).
BENIGN_TYPES = frozenset({"опечатка", "перестановка"})

assert CRITICAL_TYPES | BENIGN_TYPES == set(DISTORTION_TYPES)
assert not (CRITICAL_TYPES & BENIGN_TYPES)

_NEGATION_PARTICLES = {"не", "нет", "ни", "без", "никогда", "никакой", "никаких"}
_DIGIT_RE = re.compile(r"\d")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_PUNCT_ONLY_RE = re.compile(r"^[\s.,;:!?()«»\"'\-–—]*$")
_STRIP_CHARS = ".,;:!?()«»\"'-–— "

# Корни/формы слов-единиц измерения — нужны, чтобы ловить искажения вида
# «40 минут» -> «40 часов», «пачка сигарет» -> «пачка пачек» (план называет
# именно эту разновидность числового искажения «опасной — то же число,
# другая единица»). Сами по себе слова «часов»/«минут» не содержат цифр,
# поэтому без отдельной проверки они уходили бы в лексическую замену.
_UNIT_ROOTS = ("минут", "час", "лет", "год", "дн", "недел", "месяц",
               "сигарет", "пачк", "литр", "миллилитр", "грамм", "килограмм")
_UNIT_ABBR_RE = re.compile(r"^(мг|мл|см|кг|г|л|ед)([./][а-яё]+)?\.?,?$", re.IGNORECASE)


def _norm(word: str) -> str:
    """Нормализация токена для ВЫРАВНИВАНИЯ диффа — убирает пунктуацию по
    краям и регистр, чтобы «стационар.» и «стационар» совпали как один и
    тот же токен (иначе диф «расползается» на соседние, не изменённые
    по сути слова — наблюдалось на вставке постороннего текста)."""
    return word.strip(_STRIP_CHARS).lower()


def _looks_like_unit(word: str) -> bool:
    w = _norm(word)
    return bool(w) and (any(w.startswith(root) for root in _UNIT_ROOTS)
                        or bool(_UNIT_ABBR_RE.match(w)))


def _is_typo(word_a: str, word_b: str) -> bool:
    """Опечатка = ОДНА компактная зона редактирования (вставка/удаление/
    замена ≤ 3 символов) внутри достаточно длинного слова — «терапии»->
    «терапие», «госпитализирована»->«госпиталлизирована». Содержательная
    замена обычно меняет более протяжённый фрагмент (морфему/корень,
    «кардио...»->«нейро...») — отличаем по ширине затронутой зоны, а не
    по общему посимвольному сходству (оно может быть высоким и при
    замене корня в длинном слове)."""
    wa, wb = _norm(word_a), _norm(word_b)
    if wa == wb:
        return False
    sm = SequenceMatcher(None, wa, wb, autojunk=False)
    diffs = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if len(diffs) != 1:
        return False
    _, i1, i2, j1, j2 = diffs[0]
    span = max(i2 - i1, j2 - j1)
    return span <= 3 and min(len(wa), len(wb)) >= 6

# Приоритет при объединении нескольких изменённых фрагментов в один вердикт
# по паре: «жёсткие» типы важнее «безобидных» — иначе случайное безобидное
# совпадение в одном фрагменте могло бы замаскировать критическое искажение
# в другом (датасет местами содержит несколько правок за один шаг).
_TYPE_PRIORITY = {t: i for i, t in enumerate(
    ("число", "отрицание", "вставка", "лексическая_замена", "опечатка", "перестановка"))}


@dataclass
class DistortionLabel:
    """Результат классификации искажения для одной пары текстов."""
    primary: str                 # доминирующий (по приоритету) тип — итоговый вердикт по паре
    types: tuple[str, ...]       # все обнаруженные типы (диф может содержать > 1 правки)
    chunks: list[str] = field(default_factory=list)   # человекочитаемые описания фрагментов diff

    @property
    def is_critical(self) -> bool:
        return self.primary in CRITICAL_TYPES


def _word_tokens(text: str) -> list[str]:
    return text.split()


def _classify_chunk(words_a: list[str], words_b: list[str]) -> tuple[str, str]:
    """
    Классифицирует ОДИН изменённый фрагмент диффа (a -> b, любая из сторон
    может быть пустой — вставка/удаление). Возвращает (тип, описание).
    """
    joined_a, joined_b = " ".join(words_a), " ".join(words_b)
    desc = f"«{joined_a}» -> «{joined_b}»"
    set_a = {_norm(w) for w in words_a}
    set_b = {_norm(w) for w in words_b}

    # 1. Числа — изменилось значение и/или единица измерения (канонический
    #    пример из плана: «56 лет» -> «46 лет», «40 минут» -> «40 часов»,
    #    «пачка сигарет» -> «пачка пачек» — то же число, другая, опасно
    #    правдоподобная единица).
    if _DIGIT_RE.search(joined_a) or _DIGIT_RE.search(joined_b):
        return "число", desc
    if (len(words_a) == 1 and len(words_b) == 1
            and _looks_like_unit(words_a[0]) and _looks_like_unit(words_b[0])
            and set_a != set_b):
        return "число", desc

    # 2. Отрицание — частица «не/нет/ни/...» появляется только с одной
    #    стороны правки (вставлена либо удалена).
    if (set_a & _NEGATION_PARTICLES) != (set_b & _NEGATION_PARTICLES):
        return "отрицание", desc

    # 3. Вставка постороннего — одна сторона пустая, другая содержит
    #    длинный фрагмент (целое предложение/оборот, не одно-два слова).
    if not words_a or not words_b:
        longer = words_b if not words_a else words_a
        if len(longer) >= 3:
            return "вставка", desc
        if all(_PUNCT_ONLY_RE.match(w) for w in longer):
            return "перестановка", desc
        return "лексическая_замена", desc

    # 4. Опечатка — компактная зона редактирования внутри слова, см. _is_typo
    #    («терапии»->«терапие», «госпитализирована»->«госпиталлизирована»;
    #    НЕ «кардиореанимацию»->«нейрореанимацию» — там меняется целая
    #    морфема, а не 1-3 символа).
    if len(words_a) == 1 and len(words_b) == 1 and _is_typo(words_a[0], words_b[0]):
        return "опечатка", desc

    # 5. Перестановка / пунктуация / безобидный перефраз — фрагменты состоят
    #    из тех же слов в другом порядке, либо различаются только пунктуацией.
    #    ВАЖНО: рокировки вида «положительный» <-> «отрицательный» содержательно
    #    МЕНЯЮТ клинический факт — проверка равенства множеств нормализованных
    #    токенов их не «обезвреживает» (множества разные), такие случаи
    #    провалятся сюда и попадут в «лексическую замену» ниже, что и требуется.
    if set_a == set_b and set_a:
        return "перестановка", desc
    if _norm(joined_a) == _norm(joined_b):
        return "перестановка", desc

    # 6. Всё прочее — содержательная лексическая замена. Сюда же — подмена
    #    клинических сущностей/отделений/диагнозов (диффом «сущность» от
    #    «синоним» НЕ отличить, см. докстринг модуля — относим в этот,
    #    более широкий и честный, критический тип).
    return "лексическая_замена", desc


def classify_distortion(text_a: str, text_b: str) -> DistortionLabel:
    """
    Классифицирует тип искажения между «базовым» (text_a) и «искажённым»
    (text_b) вариантами на основе word-level диффа (difflib.SequenceMatcher).

    Возвращает DistortionLabel с полным списком обнаруженных типов
    (`types`) и доминирующим типом по приоритету критичности (`primary`,
    см. _TYPE_PRIORITY) — нужен и тот и другой: `primary` даёт единое
    решение «какой это шаг» (как того требует план — «выявляем тип
    правки»), `types` сохраняет диагностику для случаев, когда в одном
    шаге одновременно произошло несколько правок (в датасете встречается).
    """
    words_a, words_b = _word_tokens(text_a), _word_tokens(text_b)
    # Выравниваем по НОРМАЛИЗОВАННЫМ токенам (без пунктуации/регистра) —
    # иначе различие в одном знаке препинания («стационар.» vs «стационар»)
    # ломает совмещение и «размазывает» один локальный диф на соседние,
    # по сути не изменившиеся слова (наблюдалось на вставках предложений).
    norm_a, norm_b = [_norm(w) for w in words_a], [_norm(w) for w in words_b]
    matcher = SequenceMatcher(None, norm_a, norm_b, autojunk=False)

    raw_chunks: list[tuple[list[str], list[str]]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        raw_chunks.append((words_a[i1:i2], words_b[j1:j2]))

    if not raw_chunks:
        # Тексты совпали по словам (различия только в пробелах/регистре) —
        # формально это «перестановка»: безобидное форматирование.
        return DistortionLabel(primary="перестановка", types=("перестановка",), chunks=[])

    # Глобальная проверка на чистую рокировку: если совокупность УДАЛЁННЫХ
    # и совокупность ДОБАВЛЕННЫХ слов (без учёта порядка/пунктуации/регистра)
    # совпадают как мультимножества — это перестановка одних и тех же слов,
    # которую diff иногда режет на несколько локальных вставок/удалений
    # вместо одной смысловой замены позиции (наблюдалось на «слабость,
    # одышку» -> «одышку, слабость»: алгоритм выдал «вставка "одышку,"» +
    # «удаление "одышку"» по отдельности — оба безобидны ТОЛЬКО вместе).
    removed = Counter(_norm(w) for chunk_a, _ in raw_chunks for w in chunk_a if _norm(w))
    added = Counter(_norm(w) for _, chunk_b in raw_chunks for w in chunk_b if _norm(w))
    if removed and removed == added:
        chunks = [f"[перестановка] «{' '.join(ca)}» -> «{' '.join(cb)}»" for ca, cb in raw_chunks]
        return DistortionLabel(primary="перестановка", types=("перестановка",), chunks=chunks)

    found: list[str] = []
    chunks = []
    for chunk_a, chunk_b in raw_chunks:
        ctype, desc = _classify_chunk(chunk_a, chunk_b)
        found.append(ctype)
        chunks.append(f"[{ctype}] {desc}")

    types = tuple(sorted(set(found), key=lambda t: _TYPE_PRIORITY[t]))
    primary = types[0]
    return DistortionLabel(primary=primary, types=types, chunks=chunks)


# ══════════════════════════════════════════════════════════════════
# РЕГРЕССИОННЫЙ ПРОГОН: ловит ли объективный слой / шлюз искажение?
# ══════════════════════════════════════════════════════════════════

@dataclass
class RegressionCase:
    """Результат прогона пайплайна (быстрых rule-based слоёв) на одной паре."""
    pair: SyntheticPair
    label: DistortionLabel
    obj_report: "objective_layer.ObjectiveComparisonReport"
    gate_decision: "gate.GateDecision"

    # «Поймано» = хотя бы один из быстрых rule-based сигналов сработал.
    # Используем только rule-based часть (panel=None в обоих вызовах) —
    # сознательно, по той же причине, по которой Блоки 2-4 принимают
    # ограничение качества qwen3:8b: полный 3-раундовый LLM-as-Judge
    # прогон на ~370 парах при текущем железе неподъёмен по времени, а
    # сами правила (числа/полярность/структура/покрытие) — это именно
    # то, что синтетический набор целенаправленно проверяет (план,
    # Блок 2: «целенаправленно искажает... rule-based diff даёт высокую
    # точность без LLM»).
    caught_numeric: bool
    caught_unit: bool
    caught_polarity: bool
    caught_gate: bool

    @property
    def caught(self) -> bool:
        return self.caught_numeric or self.caught_unit or self.caught_polarity or self.caught_gate

    def signal_sources(self) -> list[str]:
        out = []
        if self.caught_numeric:
            out.append("числа")
        if self.caught_unit:
            out.append("единицы_измерения")
        if self.caught_polarity:
            out.append("полярность")
        if self.caught_gate:
            out.append(f"шлюз({self.gate_decision.status})")
        return out


def _run_one(pair: SyntheticPair, *, panel=None) -> RegressionCase:
    label = classify_distortion(pair.text_a, pair.text_b)

    obj_report = objective_layer.compare_texts(pair.text_a, pair.text_b, panel=panel)
    gate_decision = gate.evaluate_gate(pair.text_a, pair.text_b, scope=None, panel=panel)

    caught_numeric = obj_report.numeric.get("mismatch_count", 0) > 0
    caught_unit = obj_report.numeric.get("unit_mismatch_count", 0) > 0
    caught_polarity = obj_report.polarity.get("flip_count", 0) > 0
    caught_gate = gate_decision.status != "pass"

    return RegressionCase(
        pair=pair, label=label, obj_report=obj_report, gate_decision=gate_decision,
        caught_numeric=caught_numeric, caught_unit=caught_unit,
        caught_polarity=caught_polarity, caught_gate=caught_gate,
    )


def run_regression(pairs: list[SyntheticPair], *, panel=None) -> list[RegressionCase]:
    """
    Прогоняет быстрые rule-based слои (объективный слой Блока 2 без панели +
    шлюз Блока 3 без панели) на списке синтетических пар.

    `panel` — опционально: JudgePanel, если хотите включить LLM-извлечение
    сущностей (тогда регрессия станет медленнее, но содержательнее — те же
    компромиссы, что у `evaluate_gate`/`compare_texts`, см. их докстринги).
    По умолчанию None — детерминированный, быстрый и воспроизводимый прогон,
    пригодный как gate перед каждым изменением промптов/правил/весов
    (именно такую роль ему отводит план Блока 6).
    """
    cases = [_run_one(p, panel=panel) for p in pairs]
    log.info("synthetic: регрессия завершена — %d пар обработано", len(cases))
    return cases


# ══════════════════════════════════════════════════════════════════
# СВОДНЫЙ ОТЧЁТ ПО ТИПАМ ИСКАЖЕНИЙ
# ══════════════════════════════════════════════════════════════════

@dataclass
class TypeStats:
    distortion_type: str
    is_critical: bool
    n: int
    caught: int
    rate: float                  # критичные -> recall (поймать ОБЯЗАНЫ); безобидные -> FPR (НЕ обязаны ловить)
    by_signal: dict[str, int] = field(default_factory=dict)   # сколько раз сработал каждый источник сигнала

    @property
    def rate_label(self) -> str:
        return "recall" if self.is_critical else "false-positive rate"


@dataclass
class RegressionSummary:
    total_pairs: int
    by_type: dict[str, TypeStats]
    overall_critical_recall: Optional[float]
    overall_benign_fpr: Optional[float]

    def render(self) -> str:
        """Человекочитаемый отчёт «количественные показатели по каждому типу
        искажения» — формальный критерий приёмки Блока 6 по плану."""
        lines = [
            "=== Регрессия пайплайна на синтетическом stress-датасете ===",
            f"Всего пар «база -> искажённый вариант»: {self.total_pairs}",
            "",
        ]
        crit = [s for s in self.by_type.values() if s.is_critical]
        ben = [s for s in self.by_type.values() if not s.is_critical]

        lines.append("--- Критические искажения (цель: высокий RECALL — обязаны ловиться) ---")
        for s in sorted(crit, key=lambda x: -x.n):
            lines.append(f"  {s.distortion_type:<20} n={s.n:<4} "
                         f"recall={s.rate:.1%}  ({s.caught}/{s.n} поймано)  "
                         f"источники: {s.by_signal}")
        if self.overall_critical_recall is not None:
            lines.append(f"  ИТОГО recall по критическим типам: {self.overall_critical_recall:.1%}")
        lines.append("")

        lines.append("--- Безобидные искажения (цель: низкий FALSE-POSITIVE RATE — не должны ловиться) ---")
        for s in sorted(ben, key=lambda x: -x.n):
            lines.append(f"  {s.distortion_type:<20} n={s.n:<4} "
                         f"FPR={s.rate:.1%}  ({s.caught}/{s.n} ложно поймано)  "
                         f"источники: {s.by_signal}")
        if self.overall_benign_fpr is not None:
            lines.append(f"  ИТОГО FPR по безобидным типам: {self.overall_benign_fpr:.1%}")
        lines.append("")
        lines.append("Примечание: прогон выполнен на быстрых rule-based слоях "
                     "(объективный слой Блок 2 + шлюз Блок 3, без LLM-панели — "
                     "то же ограничение qwen3:8b, что принято для Блоков 2-4: "
                     "полный 3-раундовый LLM-as-Judge на ~370 парах неподъёмен "
                     "по времени на текущем железе). Тип «лексическая_замена» "
                     "сознательно объединяет подмену сущности и содержательный "
                     "синонимический перефраз — diff не различает их без "
                     "семантических знаний (см. докстринг модуля); это значит, "
                     "что часть «лексическая_замена» в реальности безобидна, и "
                     "приведённый recall по этому типу — нижняя оценка ловимости "
                     "именно ОПАСНЫХ подмен (полная проверка требует LLM-сверки "
                     "сущностей, Блоки 2-3).")
        return "\n".join(lines)


def summarize(cases: list[RegressionCase]) -> RegressionSummary:
    by_type: dict[str, TypeStats] = {}
    for dtype in DISTORTION_TYPES:
        subset = [c for c in cases if c.label.primary == dtype]
        if not subset:
            continue
        caught = sum(1 for c in subset if c.caught)
        by_signal = {
            "числа": sum(1 for c in subset if c.caught_numeric),
            "единицы_измерения": sum(1 for c in subset if c.caught_unit),
            "полярность": sum(1 for c in subset if c.caught_polarity),
            "шлюз": sum(1 for c in subset if c.caught_gate),
        }
        by_type[dtype] = TypeStats(
            distortion_type=dtype, is_critical=(dtype in CRITICAL_TYPES),
            n=len(subset), caught=caught,
            rate=round(caught / len(subset), 4),
            by_signal=by_signal,
        )

    crit_cases = [c for c in cases if c.label.is_critical]
    ben_cases = [c for c in cases if not c.label.is_critical]
    overall_recall = (round(sum(1 for c in crit_cases if c.caught) / len(crit_cases), 4)
                      if crit_cases else None)
    overall_fpr = (round(sum(1 for c in ben_cases if c.caught) / len(ben_cases), 4)
                   if ben_cases else None)

    return RegressionSummary(
        total_pairs=len(cases), by_type=by_type,
        overall_critical_recall=overall_recall, overall_benign_fpr=overall_fpr,
    )


# ══════════════════════════════════════════════════════════════════
# САМОПРОВЕРКА
# ══════════════════════════════════════════════════════════════════

def _self_check() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Классификатор — на синтетических примерах, отражающих реальные
    #    паттерны датасета (см. синтетический отчёт diff-анализа).
    samples: list[tuple[str, str, str]] = [
        ("число",
         "Пациент 56 лет поступил с жалобами.",
         "Пациент 46 лет поступил с жалобами."),
        ("число",
         "Боль купирована за 40 минут морфином.",
         "Боль купирована за 40 часов морфином."),
        ("отрицание",
         "Госпитализирована в кардиореанимацию, тест на тропонин положительный.",
         "Госпитализирована в кардиореанимацию, тест на тропонин не положительный."),
        ("лексическая_замена",
         "Пациент переведён в кардиореанимацию для наблюдения.",
         "Пациент переведён в нейрореанимацию для наблюдения."),
        ("вставка",
         "Госпитализирована в стационар для дальнейшего наблюдения.",
         "Госпитализирована в стационар. Сорта кофе в зернах и их характеристики. Для дальнейшего наблюдения."),
        ("опечатка",
         "Пациентка госпитализирована в отделение терапии.",
         "Пациентка госпиталлизирована в отделение терапии."),
        ("перестановка",
         "Жалобы на слабость, одышку и головокружение.",
         "Жалобы на одышку, слабость и головокружение."),
    ]
    print("--- классификатор искажений ---")
    n_ok = 0
    for expected, a, b in samples:
        label = classify_distortion(a, b)
        ok = label.primary == expected
        n_ok += ok
        mark = "OK " if ok else "ERR"
        print(f"  [{mark}] ожидали={expected:<20} получили={label.primary:<20} "
              f"types={label.types} chunks={label.chunks}")
    assert n_ok == len(samples), f"классификатор ошибся на {len(samples) - n_ok} из {len(samples)} примеров"

    # 2. Загрузка и регрессия — на реальном датасете (если найден).
    try:
        pairs = load_pairs()
    except FileNotFoundError as exc:
        print(f"\n(пропуск прогона на датасете: {exc})")
        print("\nВСЕ ПРОВЕРКИ КЛАССИФИКАТОРА ПРОЙДЕНЫ")
        return

    print(f"\n--- загружено {len(pairs)} пар из синтетического датасета ---")
    assert pairs, "пары не загрузились"
    assert len({p.column for p in pairs}) == 10, "ожидалось 10 клинических случаев"
    # Каждый случай: строки 1..38 -> 37 пар (i = 1..37).
    by_col: dict[str, int] = {}
    for p in pairs:
        by_col[p.column] = by_col.get(p.column, 0) + 1
    assert all(n == 37 for n in by_col.values()), f"неожиданное число пар по случаям: {by_col}"

    cases = run_regression(pairs)
    summary = summarize(cases)
    print()
    print(summary.render())

    assert summary.total_pairs == len(pairs)
    assert summary.by_type, "сводка пуста"
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ (классификатор + регрессия на датасете)")


if __name__ == "__main__":
    _self_check()
