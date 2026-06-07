"""
preprocessor.py — универсальная сегментация ЭМК и суммаризации на
канонические смысловые разделы.

Заменяет хрупкий `split_source` из старого evaluator.py (искал ПЕРВОЕ
вхождение нескольких маркеров, заточенных под один тип карты — КТ ОБП).

Проблема прежнего подхода: реальные ЭМК в датасете — это конкатенация
НЕСКОЛЬКИХ дневниковых записей/осмотров за разные даты, каждая со своим
набором стандартных заголовков («Жалобы», «Анамнез заболевания», «Общий
осмотр», «Результаты лабораторных исследований», ...). Поиск «первого»
маркера в таком документе обрезает текст произвольно и нерегулярно —
для одной карты «лаборатория» может начаться на 20% документа, для другой
на 80%, а на 26 разных типах карт (терапия, онкология, офтальмология,
лучевая диагностика и т.д.) фиксированный список маркеров вообще не сработает.

ПОДХОД ЗДЕСЬ:
  1. Заранее заданный словарь КАНОНИЧЕСКИХ категорий (соответствуют
     блокам B1–B5 из дизайна исследования + несколько вспомогательных:
     диагноз, рекомендации/лечение, объективный осмотр, заключение).
  2. Для каждой категории — список регулярных выражений-«заголовков»,
     собранный по факту встречаемости в реальном датасете (см. разведочный
     анализ headers в источнике/суммаризациях).
  3. Документ обходится построчно; строка, узнанная как заголовок одной
     из категорий, ОТКРЫВАЕТ новый раздел этой категории. Все остальные
     строки (включая «суб-заголовки» вида «Сердце», «Кожные покровы» —
     их в словаре нет специально) остаются телом текущего раздела.
     Так раздел не дробится на десятки micro-секций, и при этом учитываются
     ВСЕ повторения раздела одной категории по всем осмотрам документа
     (а не только первое).
  4. Для суммаризации заголовки распознаются иначе (markdown `##`,
     `**жирный текст:**`, нумерованные пункты схемы «1. Жалобы…») —
     и классифицируются по ключевым словам в самом заголовке, т.к. модели
     генерации формулируют их свободно («Анализ данных пациента»,
     «Резюме для врача-рентгенолога», «Сопутствующие заболевания», ...).

Результат — `DocumentStructure`: упорядоченный список найденных разделов
+ группировка по канонической категории. Это и есть «заземление» для
объективного слоя (Блок 2) и для параметризации промптов судей (Блок 4) —
вместо «оцени весь текст целиком».
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ══════════════════════════════════════════════════════════════════
# КАНОНИЧЕСКИЕ КАТЕГОРИИ
# ══════════════════════════════════════════════════════════════════
# Первые пять — прямое соответствие блоку B (полнота) из дизайна
# исследования и «пяти разделам схемы» (D1). Остальные — вспомогательные,
# нужны объективному слою и судьям, но не участвуют в проверке D1.

COMPLAINTS        = "complaints"          # B1 — жалобы
HISTORY_ILLNESS   = "history_illness"     # B2 — анамнез заболевания
HISTORY_LIFE      = "history_life"        # B3 — анамнез жизни (привычки, сопутствующие, операции, аллергии, семейный)
LABS              = "labs"                # B4 — лабораторные данные
INSTRUMENTAL      = "instrumental"        # B5 — инструментальные/лучевые исследования
DIAGNOSIS         = "diagnosis"           # доп. — диагнозы и осложнения
OBJECTIVE_EXAM    = "objective_exam"      # доп. — объективный статус/осмотр
RECOMMENDATIONS   = "recommendations"     # доп. — рекомендации/лечение/терапия
CONCLUSION        = "conclusion"          # доп. — заключение/эпикриз/итог
OTHER             = "other"               # нераспознанное

# Каноническая схема суммаризации (для D1 — порядок и наличие 5 разделов)
SCHEMA_CATEGORIES: tuple[str, ...] = (
    COMPLAINTS, HISTORY_ILLNESS, HISTORY_LIFE, LABS, INSTRUMENTAL,
)

CATEGORY_LABELS_RU: dict[str, str] = {
    COMPLAINTS:      "Жалобы",
    HISTORY_ILLNESS: "Анамнез заболевания",
    HISTORY_LIFE:    "Анамнез жизни",
    LABS:            "Лабораторные данные",
    INSTRUMENTAL:    "Инструментальные исследования",
    DIAGNOSIS:       "Диагноз",
    OBJECTIVE_EXAM:  "Объективный осмотр",
    RECOMMENDATIONS: "Рекомендации/лечение",
    CONCLUSION:      "Заключение",
    OTHER:           "Прочее/нераспознанное",
}


# ══════════════════════════════════════════════════════════════════
# СЛОВАРЬ ЗАГОЛОВКОВ ИСХОДНОЙ ЭМК
# ══════════════════════════════════════════════════════════════════
# Каждый паттерн проверяется against нормализованной строки целиком
# (см. _normalize_line). Порядок важен: проверяем сверху вниз, первое
# совпадение побеждает — поэтому более специфичные паттерны идут раньше
# более общих (напр. «диагноз при поступлении» раньше общего «диагноз»).

_EMR_HEADER_PATTERNS: list[tuple[str, re.Pattern]] = [
    (COMPLAINTS, re.compile(
        r"^жалоб(ы|а)( (при поступлении|пациента|на момент осмотра).*)?$")),

    (HISTORY_LIFE, re.compile(
        r"^(анамнез жизни.*|сопутствующ(ее|ие) заболевани[ея].*|"
        r"фоновое заболевание.*|вредные привычки.*|операции.*|"
        r"оперативные вмешательства.*|хирургические вмешательства.*|"
        r"аллергоанамнез.*|аллергологический анамнез.*|"
        r"семейный анамнез.*|эпидемиологический анамнез.*|"
        r"перенес[её]нные (заболевания|операции).*)$")),

    (HISTORY_ILLNESS, re.compile(
        r"^(анамнез заболевания.*|анамнез пациента.*|анамнез($| .*))$")),

    (LABS, re.compile(
        r"^(результаты лабораторных исследований.*|лабораторные исследования.*|"
        r"лабораторные данные.*|клинический анализ крови.*|"
        r"общий анализ (крови|мочи).*|биохимическое исследование.*|"
        r"биохимический анализ.*|коагулограмма.*|"
        r"гормональн(ый|ое) (анализ|исследование).*)$")),

    (INSTRUMENTAL, re.compile(
        r"^(компьютерная томография.*|магнитно-резонансная томография.*|"
        r"ультразвуковое исследование.*|узи .*|рентген(ография)?.*|"
        r"лучевая диагностика.*|инструментальные исследования.*|"
        r"инструментальные данные.*|описание результатов.*|"
        r"эндоскопия.*|экг.*|эхокг.*|"
        r"информация об оборудовании.*|доза облучения.*|"
        r"протокол расшифровки.*|острота зрения.*|вгд (od|os).*)$")),

    (DIAGNOSIS, re.compile(
        r"^(основной диагноз.*|диагноз при (поступлении|выписке).*|"
        r"предварительный диагноз.*|заключительный диагноз.*|"
        r"диагноз($|:.*|\s.*)|осложнение основного (заболевания|диагноза).*|"
        r"клиническая группа.*|статус диагноза.*|стадия tnm.*|"
        r"метод подтверждения диагноза.*|признак основной опухоли.*|"
        r"тип протокола.*)$")),

    (OBJECTIVE_EXAM, re.compile(
        r"^(общий осмотр.*|объективный статус.*|объективное обследование.*|"
        r"общее состояние.*|состояние при (поступлении|выписке).*|"
        r"местный статус.*|нервно-психический статус.*|"
        r"локальный статус.*|status (praesens|localis).*)$")),

    (RECOMMENDATIONS, re.compile(
        r"^(рекомендации.*|необходимые дообследования.*|лекарственная терапия.*|"
        r"назначения.*|план лечения.*|лечение.*|диета.*|трудоспособность.*)$")),

    (CONCLUSION, re.compile(
        r"^(заключение.*|выписной эпикриз.*|исход и результат госпитализации.*|"
        r"комментарий.*|резюме.*)$")),
]

# Строки, которые ВЫГЛЯДЯТ как заголовки (короткие, с заглавной буквы),
# но являются служебными метаданными визита, а не содержательным разделом —
# не открывают новую секцию и не должны попадать в тело текущей секции
# заметным «шумом». Помечаем их отдельно и пропускаем.
_EMR_NOISE_LINE = re.compile(
    r"^(выполнено|провел врач-.*|провёл врач-.*|выполнил врач.*|"
    r"осмотр (терапевта|пульмонолога|онколога|хирурга|кардиолога|"
    r"невролога|офтальмолога|эндокринолога|рентгенолога|уролога|"
    r"гастроэнтеролога|гинеколога|инфекциониста)|"
    r"консультация .*|оборудование.*|"
    r"данные о пациенте и времени его пребывания в больнице.*|"
    r"цель приёма.*|цель приема.*)$"
)

# Маркер начала новой дневниковой записи/визита: дата вида `"10 апр 2025`
# (часто в кавычках). Используется только для трассировки/метаданных —
# само по себе не управляет сегментацией по категориям.
_VISIT_DATE_RE = re.compile(
    r'^"*\s*(\d{1,2}\s+[а-яё]{3,8}\.?\s+\d{4})\s*$', re.IGNORECASE)

# ── Резервный режим: «нарративные» ЭМК без построчных заголовков ──
#
# Часть карт (в частности, синтетический stress-датасет Сеченовского
# университета) оформлена не «телеграфным» стилем построчных заголовков,
# а связным текстом, где разделы обозначены ВСТРОЕННЫМИ метками внутри
# абзаца: "Сопутствующие заболевания – сахарный диабет…",
# "Вредные привычки: курит по 10 сигарет…", "Общий анализ крови: гемоглобин…".
# Для такого формата построчный детектор заголовков почти ничего не находит
# (см. _is_headerlike — метка не на отдельной строке). Резервный сканер
# ищет фразы вида «Метка(:|–|—|-) текст…» в произвольном месте текста.
_INLINE_LABEL_RE = re.compile(
    r"(?:^|(?<=[.\n]))\s*"
    r"(?P<label>[А-ЯЁ][а-яёА-ЯЁ ]{2,55}?)"
    r"\s*[:–—-]\s+(?=\S)",
    re.MULTILINE,
)


def _segment_by_inline_labels(text: str) -> tuple[str, list[Section]]:
    """
    Резервная сегментация для «нарративных» ЭМК: ищет встроенные метки
    разделов прямо в потоке текста и режет документ по их границам.
    Возвращает (preamble, sections). Возвращает ([], "") если меток
    с распознаваемой категорией не нашлось — тогда вызывающий код
    остаётся на построчном результате.
    """
    matches = []
    for m in _INLINE_LABEL_RE.finditer(text):
        label = m.group("label").strip()
        category = _classify_summary_heading(label)
        if category != OTHER:
            matches.append((m.start(), m.end(), label, category))

    if not matches:
        return "", []

    preamble = text[:matches[0][0]].strip()
    sections: list[Section] = []
    for idx, (_start, end, label, category) in enumerate(matches):
        body_end = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
        body = text[end:body_end].strip()
        sections.append(Section(category=category, title=label, text=body, order=idx))
    return preamble, sections


# ══════════════════════════════════════════════════════════════════
# СЛОВАРЬ КЛЮЧЕВЫХ СЛОВ ДЛЯ ЗАГОЛОВКОВ СУММАРИЗАЦИИ
# ══════════════════════════════════════════════════════════════════
# Модели формулируют заголовки свободно («Резюме для врача-рентгенолога»,
# «Анализ данных пациента», «1. Жалобы, ставшие причиной выполнения КТ…»).
# Поэтому классифицируем по наличию ключевых слов, в порядке приоритета —
# первое совпадение побеждает (специфичные раньше общих).

_SUMMARY_KEYWORD_RULES: list[tuple[str, re.Pattern]] = [
    (COMPLAINTS,      re.compile(r"жалоб")),
    (HISTORY_LIFE,    re.compile(r"анамнез[а-я]*\s+жизни|сопутствующ|"
                                 r"вредны[ех]\s+привычк|перенес[её]нн|"
                                 r"аллергологическ|аллергоанамнез|"
                                 r"операци|семейный анамнез")),
    (HISTORY_ILLNESS, re.compile(r"анамнез[а-я]*\s+заболевания|анамнез($|[^а-я])")),
    (LABS,            re.compile(r"лаборатор|анализ[а-я]*\s+(крови|мочи)|"
                                 r"биохими")),
    (INSTRUMENTAL,    re.compile(r"инструментальн|\bкт\b|\bмрт\b|\bузи\b|"
                                 r"рентген|эндоскоп|\bэкг\b|\bэхокг\b|"
                                 r"лучев[а-я]*\s+диагностик|"
                                 r"объективн[а-я]*\s+(статус|обследовани|осмотр)")),
    (DIAGNOSIS,       re.compile(r"диагноз|осложнени")),
    (RECOMMENDATIONS, re.compile(r"рекомендаци|лечени|терапи|назначени|особые указания")),
    (CONCLUSION,      re.compile(r"заключени|резюме|итог|вывод|прогноз|"
                                 r"исход и результат")),
]


# ══════════════════════════════════════════════════════════════════
# СТРУКТУРЫ ДАННЫХ
# ══════════════════════════════════════════════════════════════════

@dataclass
class Section:
    """Один распознанный раздел документа."""
    category: str            # каноническая категория (см. константы выше)
    title: str               # заголовок, как он встретился в тексте
    text: str                # тело раздела (накопленный текст до следующего заголовка)
    order: int               # порядковый номер появления в документе (0-based)
    visit_date: str | None = None   # для ЭМК — дата визита, к которому относится раздел


@dataclass
class DocumentStructure:
    """
    Результат сегментации документа (ЭМК или суммаризации).

    sections     — все найденные разделы по порядку появления
    by_category  — те же разделы, сгруппированные по категории
    preamble     — текст до первого распознанного заголовка (часто шапка/метаданные)
    """
    raw_text: str
    sections: list[Section] = field(default_factory=list)
    preamble: str = ""

    @property
    def by_category(self) -> dict[str, list[Section]]:
        out: dict[str, list[Section]] = {}
        for s in self.sections:
            out.setdefault(s.category, []).append(s)
        return out

    def text_for(self, category: str) -> str:
        """Объединённый текст всех разделов категории (через разделитель)."""
        parts = [s.text for s in self.sections if s.category == category and s.text.strip()]
        return "\n\n".join(parts)

    def categories_present(self) -> set[str]:
        return {s.category for s in self.sections if s.text.strip()}

    def category_order(self) -> list[str]:
        """Категории в порядке ПЕРВОГО появления — для проверки D1 (порядок схемы)."""
        seen: list[str] = []
        for s in self.sections:
            if s.category not in seen and s.text.strip():
                seen.append(s.category)
        return seen

    def coverage_summary(self) -> dict[str, int]:
        """{категория: суммарная длина текста в символах} — быстрый обзор покрытия."""
        out: dict[str, int] = {cat: 0 for cat in CATEGORY_LABELS_RU}
        for s in self.sections:
            out[s.category] = out.get(s.category, 0) + len(s.text)
        out["__preamble__"] = len(self.preamble)
        return out


# ══════════════════════════════════════════════════════════════════
# НОРМАЛИЗАЦИЯ И РАСПОЗНАВАНИЕ ЗАГОЛОВКОВ
# ══════════════════════════════════════════════════════════════════

def _normalize_line(line: str) -> str:
    """Готовит строку к сопоставлению с паттернами заголовков: нижний регистр,
    убраны кавычки/дублирующиеся пробелы/завершающее двоеточие."""
    s = line.strip().strip('"').strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(":").rstrip()
    return s.lower()


def _is_headerlike(line: str) -> bool:
    """Эвристика «похоже на заголовок»: короткая строка без точки в конце,
    начинающаяся с заглавной буквы (для ЭМК с «телеграфным» форматированием
    каждый заголовок — на отдельной строке)."""
    s = line.strip().strip('"').strip()
    return bool(s) and len(s) <= 80 and not s.endswith(".") and s[:1].isupper()


def _match_emr_header(line: str) -> str | None:
    """Возвращает каноническую категорию, если строка — заголовок раздела ЭМК,
    иначе None (строка остаётся телом текущего раздела)."""
    if not _is_headerlike(line):
        return None
    norm = _normalize_line(line)
    if not norm:
        return None
    if _EMR_NOISE_LINE.match(norm):
        return "__noise__"
    for category, pattern in _EMR_HEADER_PATTERNS:
        if pattern.match(norm):
            return category
    return None


def _match_visit_date(line: str) -> str | None:
    norm = line.strip()
    m = _VISIT_DATE_RE.match(norm)
    return m.group(1) if m else None


# ══════════════════════════════════════════════════════════════════
# СЕГМЕНТАЦИЯ ИСХОДНОЙ ЭМК
# ══════════════════════════════════════════════════════════════════

def segment_emr(text: str) -> DocumentStructure:
    """
    Делит исходный текст ЭМК на канонические разделы.

    Документ — конкатенация дневниковых записей за разные даты, каждая
    со своими «Жалобы / Анамнез заболевания / Общий осмотр / Результаты
    лабораторных исследований / ...». Все вхождения раздела одной
    категории по всем визитам объединяются (см. DocumentStructure.text_for) —
    это даёт объективному слою и судьям ПОЛНУЮ картину, а не последний/первый
    случайно найденный фрагмент.

    Строки внутри раздела, которые сами выглядят как заголовки, но не входят
    в словарь (напр. «Сердце», «Кожные покровы», «Острота зрения OD» —
    под-пункты физикального осмотра), не переключают категорию и остаются
    частью текущего раздела — поэтому документ не дробится на сотни микро-секций.
    """
    lines = text.split("\n")

    sections: list[Section] = []
    preamble_lines: list[str] = []
    current_category: str | None = None
    current_title = ""
    current_lines: list[str] = []
    current_visit_date: str | None = None
    order = 0

    def flush():
        nonlocal current_category, current_title, current_lines, order
        if current_category is not None:
            body = "\n".join(current_lines).strip()
            sections.append(Section(
                category=current_category, title=current_title,
                text=body, order=order, visit_date=current_visit_date,
            ))
            order += 1
        current_lines = []

    for raw_line in lines:
        visit_date = _match_visit_date(raw_line)
        if visit_date:
            current_visit_date = visit_date
            continue   # дата сама по себе не формирует раздел

        matched = _match_emr_header(raw_line)
        if matched == "__noise__":
            continue   # служебная строка визита — отбрасываем
        if matched is not None:
            flush()
            current_category = matched
            current_title = raw_line.strip().strip('"')
            current_lines = []
            continue

        if current_category is None:
            preamble_lines.append(raw_line)
        else:
            current_lines.append(raw_line)

    flush()

    # Резервный режим: построчный детектор почти ничего не нашёл — скорее
    # всего, документ «нарративный» (см. _segment_by_inline_labels). Порог
    # 30% — эмпирический: реальные телеграфные ЭМК распознаются на 99.7%+,
    # а нарративные тексты без построчных заголовков — на 0% этим методом,
    # так что ложных переключений на типичных данных не возникает.
    recognized_len = sum(len(s.text) for s in sections)
    if recognized_len < 0.3 * max(len(text), 1):
        fb_preamble, fb_sections = _segment_by_inline_labels(text)
        if sum(len(s.text) for s in fb_sections) > recognized_len:
            return DocumentStructure(raw_text=text, sections=fb_sections, preamble=fb_preamble)

    return DocumentStructure(
        raw_text=text,
        sections=sections,
        preamble="\n".join(preamble_lines).strip(),
    )


# ══════════════════════════════════════════════════════════════════
# СЕГМЕНТАЦИЯ СУММАРИЗАЦИИ
# ══════════════════════════════════════════════════════════════════

# Заголовок суммаризации на ОТДЕЛЬНОЙ строке: markdown (## ...),
# жирный лейбл (**Текст:** / **Текст**), пункт схемы "1. Жалобы ..."
# либо «голый» лейбл с двоеточием без какого-либо форматирования
# ("Анамнез жизни пациента:" на отдельной строке).
_SUMMARY_HEADER_RE = re.compile(
    r"^\s*(?:#{1,4}\s*(?P<md>.+?)\s*#*\s*"
    r"|\*\*(?P<bold>[^*]{2,110}?)\*\*[.:\s]*$"
    r"|(?P<numbered>\d{1,2}[.)]\s*[А-ЯЁ][^\n]{1,120}?)\s*:?\s*"
    r"|(?P<plain>[А-ЯЁ][А-Яа-яёЁ ,()/-]{2,90}):\s*)$"
)

# Заголовок ВСТРОЕННЫЙ — модель пишет лейбл и содержимое в одну строку:
# "**Жалобы:** Пациент жалуется на боли в животе…",
# "Жалобы, ставшие причиной выполнения КТ…: Боли в животе…".
# Группа `rest` — начало тела раздела, которое нельзя терять.
_SUMMARY_INLINE_HEADER_RE = re.compile(
    r"^\s*(?:\*\*(?P<bold_label>[^*]{2,110}?)\*\*\s*:?\s+(?P<bold_rest>\S.+)"
    r"|(?P<plain_label>[А-ЯЁ][а-яёА-ЯЁ ,()/-]{2,90}):\s+(?P<plain_rest>\S.+))\s*$"
)

# Маркеры маркированных списков, которыми модели иногда оборачивают заголовки
# разделов: "* **Жалобы:** ...", "-   **Анамнез:** ...", "•  Диагноз: ...".
_BULLET_PREFIX_RE = re.compile(r"^\s*[*\-•]\s+")


def _classify_summary_heading(title: str) -> str:
    """Ключевые слова определяют каноническую категорию заголовка суммаризации."""
    norm = title.strip().strip(":*# ").lower()
    norm = re.sub(r"^\d{1,2}[.)]\s*", "", norm)   # снимаем нумерацию "1. "
    for category, pattern in _SUMMARY_KEYWORD_RULES:
        if pattern.search(norm):
            return category
    return OTHER


def _match_summary_header(line: str) -> tuple[str, str, str | None] | None:
    """
    Возвращает (заголовок, категория, встроенный_остаток_тела_или_None) либо None.

    Сначала проверяется форма «заголовок на отдельной строке» (приоритет —
    более надёжный сигнал), затем «заголовок + текст в той же строке»
    (`**Жалобы:** Пациент жалуется…») — частый паттерн у некоторых моделей,
    который нельзя отбрасывать, иначе содержимое раздела будет потеряно.
    """
    stripped = line.strip()
    # Снимаем маркер списка ("* ", "- ", "• "), под которым модели иногда
    # прячут заголовок раздела — но не трогаем "**жирный**" в начале строки
    # (после маркера списка он почти всегда начинается с **).
    delisted = _BULLET_PREFIX_RE.sub("", stripped, count=1)
    if delisted != stripped and delisted.startswith("**"):
        stripped = delisted

    m = _SUMMARY_HEADER_RE.match(stripped)
    if m:
        title = next(g for g in (m.group("md"), m.group("bold"),
                                 m.group("numbered"), m.group("plain")) if g)
        title = title.strip()
        if title and len(title) <= 120:
            category = _classify_summary_heading(title)
            # «Голый» лейбл без markdown-разметки — самый слабый сигнал
            # (легко спутать с обычным предложением вида "Возраст: 64 года").
            # Принимаем только если ключевые слова однозначно опознали раздел.
            if m.group("plain") and category == OTHER:
                return None
            return title, category, None

    m = _SUMMARY_INLINE_HEADER_RE.match(stripped)
    if m:
        label = m.group("bold_label") or m.group("plain_label")
        rest = m.group("bold_rest") or m.group("plain_rest")
        label = label.strip()
        category = _classify_summary_heading(label)
        # Категория должна реально определиться по лейблу — иначе слишком
        # высок риск принять обычное предложение вида "Боль: тупая, ноющая"
        # за заголовок раздела.
        if category != OTHER and len(label) <= 60:
            return label, category, rest.strip()

    return None


def segment_summary(text: str) -> DocumentStructure:
    """
    Делит текст суммаризации на канонические разделы.

    Модели форматируют вывод свободно: где-то это нумерованная схема
    "1. Жалобы… 2. Анамнез заболевания…", где-то — markdown с `##`/`**bold**`
    заголовками произвольной формулировки ("Резюме для врача-рентгенолога",
    "Анализ данных пациента", "Сопутствующие заболевания"…). Поэтому, в
    отличие от ЭМК (где работает словарь точных заголовков), здесь заголовок
    распознаётся по ФОРМЕ (markdown/жирный/нумерованный пункт), а категория —
    по ключевым словам внутри него.
    """
    lines = text.split("\n")

    sections: list[Section] = []
    preamble_lines: list[str] = []
    current_category: str | None = None
    current_title = ""
    current_lines: list[str] = []
    order = 0

    def flush():
        nonlocal current_category, current_title, current_lines, order
        if current_category is not None:
            body = "\n".join(current_lines).strip()
            sections.append(Section(
                category=current_category, title=current_title,
                text=body, order=order,
            ))
            order += 1
        current_lines = []

    for raw_line in lines:
        matched = _match_summary_header(raw_line)
        if matched is not None:
            flush()
            current_title, current_category, inline_rest = matched
            current_lines = [inline_rest] if inline_rest else []
            continue

        if current_category is None:
            preamble_lines.append(raw_line)
        else:
            current_lines.append(raw_line)

    flush()

    return DocumentStructure(
        raw_text=text,
        sections=sections,
        preamble="\n".join(preamble_lines).strip(),
    )


# ══════════════════════════════════════════════════════════════════
# ПРОВЕРКА D1 — соответствие суммаризации схеме из пяти разделов
# ══════════════════════════════════════════════════════════════════

def schema_compliance(summary_structure: DocumentStructure) -> dict:
    """
    D1: «наличие и корректная последовательность всех пяти разделов промпта»
    (B1 Жалобы → B2 Анамнез заболевания → B3 Анамнез жизни → B4 Лабораторные
    данные → B5 Инструментальные исследования).

    Возвращает словарь с отдельными бинарными признаками — пригоден и для
    отчёта объективного слоя, и как «заземление» для промпта судьи блока D
    (вместо того, чтобы судья сам решал, что считать «правильной схемой»).
    """
    present_order = [c for c in summary_structure.category_order() if c in SCHEMA_CATEGORIES]
    present_set = set(present_order)

    missing = [c for c in SCHEMA_CATEGORIES if c not in present_set]

    # Корректность порядка: подпоследовательность найденных разделов схемы
    # должна совпадать с каноническим порядком (без перестановок).
    expected_order = [c for c in SCHEMA_CATEGORIES if c in present_set]
    order_correct = present_order == expected_order

    return {
        "present_categories": present_order,
        "missing_categories": missing,
        "all_present":        not missing,
        "order_correct":      order_correct,
        "compliant":          not missing and order_correct,
    }


# ══════════════════════════════════════════════════════════════════
# САМОПРОВЕРКА: прогон по всему датасету и отчёт о покрытии
# ══════════════════════════════════════════════════════════════════

def _self_check():
    import pandas as pd
    from collections import Counter

    df = pd.read_excel("data/summaries.xlsx")
    emr_ids = sorted(df["emr_id"].unique())

    print(f"Загружено {len(df)} строк, {len(emr_ids)} уникальных ЭМК\n")

    print("=" * 70)
    print("ЭМК (исходники): покрытие категорий и доля 'other/preamble'")
    print("=" * 70)
    emr_other_ratio = []
    for emr_id in emr_ids:
        src = df[df["emr_id"] == emr_id]["source_text"].iloc[0]
        struct = segment_emr(src)
        cov = struct.coverage_summary()
        total = len(src) or 1
        unclassified = cov.get(OTHER, 0) + cov["__preamble__"]
        ratio = unclassified / total
        emr_other_ratio.append(ratio)
        present = struct.categories_present()
        missing_core = [c for c in SCHEMA_CATEGORIES if c not in present]
        flag = "  <-- НЕТ: " + ", ".join(CATEGORY_LABELS_RU[c] for c in missing_core) if missing_core else ""
        print(f"  {emr_id}: всего {total:6d} симв | нераспознано {ratio*100:5.1f}%"
              f" | разделов {len(struct.sections):3d}{flag}")

    print(f"\n  Среднее по корпусу нераспознанного текста (other+preamble):"
          f" {sum(emr_other_ratio)/len(emr_other_ratio)*100:.1f}%")

    print()
    print("=" * 70)
    print("Суммаризации: соответствие схеме D1 (Блок B1-B5) — выборка по моделям")
    print("=" * 70)
    compliance_counter = Counter()
    missing_counter = Counter()
    n = 0
    for _, row in df.iterrows():
        struct = segment_summary(row["summary_text"])
        comp = schema_compliance(struct)
        n += 1
        if comp["compliant"]:
            compliance_counter["compliant"] += 1
        elif comp["all_present"] and not comp["order_correct"]:
            compliance_counter["wrong_order"] += 1
        else:
            compliance_counter["missing_sections"] += 1
        for c in comp["missing_categories"]:
            missing_counter[c] += 1

    print(f"  Всего суммаризаций: {n}")
    for k, v in compliance_counter.most_common():
        print(f"    {k:<20} {v:4d}  ({v/n*100:.1f}%)")
    print(f"\n  Чаще всего отсутствуют разделы:")
    for cat, cnt in missing_counter.most_common():
        print(f"    {CATEGORY_LABELS_RU[cat]:<35} отсутствует в {cnt} из {n} ({cnt/n*100:.1f}%)")

    print()
    print("=" * 70)
    print("Пример структуры (EMR_01, первая суммаризация)")
    print("=" * 70)
    sample = df[df["emr_id"] == "EMR_01"].iloc[0]
    src_struct = segment_emr(sample["source_text"])
    sum_struct = segment_summary(sample["summary_text"])

    print("\n  -- ЭМК: категории и объём текста --")
    for cat in list(CATEGORY_LABELS_RU) :
        secs = src_struct.by_category.get(cat, [])
        if secs:
            total_len = sum(len(s.text) for s in secs)
            print(f"    {CATEGORY_LABELS_RU[cat]:<32} {len(secs):2d} фрагм., {total_len:6d} симв.")

    print("\n  -- Суммаризация: порядок и категории разделов --")
    for s in sum_struct.sections:
        print(f"    [{CATEGORY_LABELS_RU[s.category]:<28}] «{s.title[:60]}» ({len(s.text)} симв.)")
    print(f"\n  D1 (соответствие схеме): {schema_compliance(sum_struct)}")

    print()
    print("=" * 70)
    print("Синтетический stress-датасет: другой формат записи (нарративный)")
    print("=" * 70)
    import glob
    xlsx_paths = glob.glob("../materials/*.xlsx")
    if not xlsx_paths:
        print("  (файл не найден — пропуск)")
        return
    syn = pd.read_excel(xlsx_paths[0], sheet_name="Лист1", dtype=str).fillna("")
    text_cols = [c for c in syn.columns if c != "Unnamed: 0"]
    print(f"  Колонки-варианты: {text_cols}")
    print(f"  ВАЖНО: тексты этого набора НЕ телеграфные (без построчных")
    print(f"  заголовков 'Жалобы\\nАнамнез заболевания\\n...'), а НАРРАТИВНЫЕ —")
    print(f"  метки разделов встроены в поток текста ('Сопутствующие")
    print(f"  заболевания – ...', 'Вредные привычки: ...'). Включён резервный")
    print(f"  сканер встроенных меток (см. _segment_by_inline_labels).")
    print()
    for col in text_cols[:5]:
        txt = syn[col].iloc[0]
        if len(txt) < 200:
            continue
        struct = segment_emr(txt)
        cov = struct.coverage_summary()
        unrec = (cov.get(OTHER, 0) + cov["__preamble__"]) / max(len(txt), 1)
        cats = ", ".join(CATEGORY_LABELS_RU[c] for c in struct.categories_present())
        print(f"    {col}: {len(txt):4d} симв | найдено разделов {len(struct.sections)}"
              f" | нераспознано {unrec*100:4.1f}% | категории: {cats or '—'}")
    print()
    print("  Нераспознанная часть — преимущественно вводный нарративный абзац")
    print("  («Больная И. 56 лет госпитализирована… На момент поступления")
    print("  жалоб нет… беспокоят приступы…») БЕЗ явной метки раздела.")
    print("  Извлечь из него B1/B2 формальными правилами нельзя в принципе —")
    print("  это и есть зона ответственности семантического слоя (Блок 2:")
    print("  LLM-извлечение сущностей поверх структуры, которую даёт Блок 1).")


if __name__ == "__main__":
    _self_check()
