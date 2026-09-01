"""
objective_layer.py — объективный слой: извлечение проверяемых фактов из
текста и фактологическая сверка пары документов (ЭМК ↔ суммаризация,
либо база ↔ кандидат при регрессионном тестировании на синтетике).

Почему «объективный» — в отличие от LLM-as-Judge (Блок 4), здесь нет
свободной интерпретации: числа, единицы измерения, полярность утверждений
и именованные сущности извлекаются ДЕТЕРМИНИРОВАННЫМИ правилами (там, где
это возможно) и сравниваются как структурированные факты, а не как
«мнение модели о тексте». Это даёт:
  1) «заземление» для промптов судей в Блоке 4 (судья получает не сырой
     текст, а конкретный список фактов-кандидатов для верификации);
  2) первый количественный, воспроизводимый сигнал качества — независимый
     от того, какая LLM выступает в роли судьи и насколько она «в духе»
     сегодня переобучилась на JSON-формат.

Состав слоя (по дизайну исследования — гибрид rule-based + LLM-based):
  • RULE-BASED, высокая точность без LLM:
      - числовые факты с единицами измерения (года, длительности, лабораторные
        показатели, размеры, дозировки) — именно то, что ЦЕЛЕНАПРАВЛЕННО
        искажает синтетический stress-датасет («56 лет»→«46 лет»,
        «40 минут»→«40 часов»);
      - полярность клинических утверждений (купирован/НЕ купирован,
        отягощён/НЕ отягощён, включают/исключают, выявлено/не выявлено) —
        под тип искажений «инверсия отрицания».
  • LLM-BASED (опционально, через JudgePanel — Блок 0): семантические
    сущности, для которых простых правил недостаточно — диагнозы,
    препараты, процедуры, отделения/локализации.

Главная функция — `compare_texts(text_a, text_b, panel=None)`, работающая
СИММЕТРИЧНО: сравнение «ЭМК ↔ суммаризация» (поиск галлюцинаций/пропусков
в кандидате относительно источника) и «база ↔ перефразированный вариант»
(валидация на синтетике, Блок 6) — это один и тот же по сути вопрос
«какие проверяемые факты разошлись между двумя текстами», только с разной
интерпретацией результата.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

# ══════════════════════════════════════════════════════════════════
# 1. ЧИСЛОВЫЕ ФАКТЫ С ЕДИНИЦАМИ ИЗМЕРЕНИЯ
# ══════════════════════════════════════════════════════════════════
# Стратегия: число + единица — самый надёжный, проверяемый правилами факт.
# Группируем единицы по смысловой категории — это и определяет, с чем
# вообще имеет смысл сравнивать данное число (года не сравнивают с мм рт.ст.).

_NUM = r"\d{1,4}(?:[.,]\d{1,3})?"

_UNIT_GROUPS: list[tuple[str, str]] = [
    # (категория, regex-группа единиц измерения)
    ("age",          r"лет(?:ний|ня[яму]|ним)?|год(?:а|ов)?(?!\w)"),
    ("duration",     r"час(?:ов|а)?|минут[ауы]?|сут(?:ок|ки|ках)?|"
                     r"дн(?:я|ей|ь|ях)|недел[яиьюе]|мес[яц](?:ев|а|ы)?"),
    ("vital_sign",   r"мм\s?рт\.?\s?ст\.?|уд(?:ар(?:а|ов)?)?\s?/\s?мин|"
                     r"в\s+минуту|/\s?мин(?:ут)?(?!\w)|°\s?[CС]"),
    ("lab_value",    r"г/л|мг/л|ммоль/л|мкмоль/л|мкг/л|нг/мл|пг/мл|ед/л|"
                     r"мм/ч|×\s?10[⁰¹²³⁴⁵⁶⁷⁸⁹\^\d]*\s?/\s?[лmлmл]|%"),
    ("size",         r"мм(?!\s?рт)|см|мл"),
    ("dose",         r"мг(?:/(?:кг|сут|мл))?|мкг|г(?!/л)|ед\.?(?!/л)"),
]

_UNIT_RE = re.compile(
    r"(?P<value>" + _NUM + r")\s?(?P<unit>" +
    "|".join(f"(?:{u})" for _, u in _UNIT_GROUPS) + r")",
    re.IGNORECASE | re.UNICODE,
)


def _classify_unit(unit_text: str) -> str:
    norm = unit_text.lower().replace(" ", "")
    for category, pattern in _UNIT_GROUPS:
        if re.fullmatch(rf"(?:{pattern})".replace(" ", ""), norm, re.IGNORECASE):
            return category
        if re.match(rf"(?:{pattern})".replace(r"\s?", ""), norm, re.IGNORECASE):
            return category
    return "other"


@dataclass(frozen=True)
class NumericFact:
    """Число + единица измерения + подпись (что именно измерено) + контекст."""
    value: float
    unit_raw: str
    category: str          # age | duration | vital_sign | lab_value | size | dose | other
    label: str             # ближайшее «название показателя» слева (гемоглобин, возраст, …)
    context: str           # ~40 символов слева — вторичный сигнал при сопоставлении
    raw: str               # как фраза выглядела в тексте целиком


def _to_float(num_str: str) -> float:
    return float(num_str.replace(",", "."))


# Границы между соседними показателями в «телеграфном» перечислении
# («гемоглобин 115 г/л, эритроциты 3.43×10¹²/л, …», «ЧСС: 94 /мин. Пульс: …»):
# подпись показателя — это слово(а) ПОСЛЕ ближайшей такой границы слева.
_LABEL_BOUNDARY_RE = re.compile(r"[,;:.\n()]")


def _extract_label(text: str, match_start: int, window: int = 60) -> str:
    """
    Возвращает «подпись» числового факта — ближайшее название показателя
    слева от числа, отделённое от соседних показателей знаком препинания.
    Это и есть ключ сопоставления: «гемоглобин 115» нельзя путать с
    «эритроциты 115», даже если оба упомянуты в одном предложении про ОАК.
    """
    start = max(0, match_start - window)
    segment = text[start:match_start]
    boundaries = list(_LABEL_BOUNDARY_RE.finditer(segment))
    if boundaries:
        segment = segment[boundaries[-1].end():]
    # длина >= 2 — иначе теряются значимые лабораторные сокращения
    # («КФК МБ», «АД», «ЧСС»), которые как раз и отличают один показатель
    # от другого («КФК общ.» vs «КФК МБ»); короткие предлоги/союзы убираем
    # отдельным списком в _GENERIC_LABEL_WORDS, а не длиной токена.
    words = re.findall(r"[а-яёa-z]{2,}", segment.lower())
    return " ".join(words[-2:])


def extract_numeric_facts(text: str, context_window: int = 40) -> list[NumericFact]:
    """
    Извлекает все пары «число + единица измерения» вместе с подписью
    показателя и левым контекстом — нужны при сравнении, чтобы сопоставлять
    «возраст пациента: 56 лет» именно с «56 лет» из другого текста, а не с
    произвольным соседним числом той же категории единиц (как в лабораторной
    панели, где подряд идут гемоглобин/эритроциты/лейкоциты — все «X г/л
    или ×10⁹/л» — и без учёта подписи их легко перепутать).
    """
    facts = []
    for m in _UNIT_RE.finditer(text):
        start = max(0, m.start() - context_window)
        context = re.sub(r"\s+", " ", text[start:m.start()]).strip().lower()
        facts.append(NumericFact(
            value=_to_float(m.group("value")),
            unit_raw=m.group("unit"),
            category=_classify_unit(m.group("unit")),
            label=_extract_label(text, m.start()),
            context=context,
            raw=m.group(0).strip(),
        ))
    return facts


# ══════════════════════════════════════════════════════════════════
# 2. ПОЛЯРНОСТЬ КЛИНИЧЕСКИХ УТВЕРЖДЕНИЙ (детектор инверсии отрицания)
# ══════════════════════════════════════════════════════════════════
# Берём куст однокоренных форм клинически значимого предиката (без учёта
# отрицания) и ищем его вхождения; для каждого вхождения определяем
# полярность по наличию отрицательной частицы НЕПОСРЕДСТВЕННО рядом
# (не более 2 слов слева — «не был полностью купирован», «болей не отмечает»).
#
# Отдельно — антонимичные пары, где инверсия выражена не отрицательной
# частицей, а заменой корня («включают» → «исключают», «подтверждён» →
# «исключён») — они перечислены явными парами синонимов полярности.

_NEGATION_MARKERS = r"(?:не|нет|ни|отсутств\w*|без)"

_POLARITY_PREDICATES: list[str] = [
    r"купирова\w*",
    r"отягощ[её]н\w*",
    r"выявлен\w*",
    r"обнаружен\w*",
    r"подтвержд[её]н\w*",
    r"наблюда(?:ется|ются|лась|лось)",
    r"отмеча(?:ет|ется|ются|ла|лось)",
    r"определя(?:ется|ются|лось)",
    r"диагностирован\w*",
    r"эффект\w*",
    # статус госпитализации — клинически значимая полярность: «не госпитализирован»,
    # «не поступал» переворачивает смысл. Общий предикат (не подгонка под тест).
    r"госпитализир\w*",
    r"поступил\w*",
]

_PREDICATE_RE = re.compile(
    r"(?P<neg>(?:" + _NEGATION_MARKERS + r")\s+(?:\w+\s+){0,2})?"
    r"(?P<predicate>" + "|".join(_POLARITY_PREDICATES) + r")",
    re.IGNORECASE,
)

# Антонимичные корневые пары: (положительная форма-маркер, отрицательная).
# При встрече ЛЮБОЙ из форм фиксируем нормализованный предикат и полюс —
# сравнение увидит несовпадение полюсов как и в случае с «не +предикат».
_ANTONYM_PAIRS: list[tuple[str, str, str]] = [
    # (нормализованное_имя_предиката, regex_положительный_полюс, regex_отрицательный_полюс)
    #
    # АНТИ-ОВЕРФИТ: ниже — только КЛИНИЧЕСКИ-ОБЩИЕ инверсии состояния, которые
    # обязан знать любой медицинский NLP (госпитализация↔выписка, тест +/−,
    # купирование↔провокация, нарастание↔регресс, подтверждение↔исключение).
    # Это НЕ подгонка под конкретные слова синтетического теста: открытые
    # семантические подмены (диагноз↔диагноз, отделение↔отделение, редкие
    # антонимы вроде «потеря↔прибавка») сознательно отданы LLM-слою
    # (compare_entity_sets), а не захардкожены сюда — иначе recall на наборе
    # становится фикцией. См. synthetic.py (режим A: «rule-based + LLM»).
    ("включение",  r"включ\w*",   r"исключ\w*"),
    ("норма",      r"в\s?норме|без\s?патологи\w*|не\s?изменен\w*", r"патологи\w*|изменен\w*(?!\s?не)"),
    ("госпитализация", r"госпитализир\w*|поступил\w*",  r"выписан\w*|выписал\w*"),
    ("результат_теста", r"отрицательн\w*",              r"положительн\w*"),
    ("купирование",    r"купир\w*",                     r"спровоцир\w*|усугуб\w*"),
    ("динамика",       r"нараст\w*|усилива\w*|усилен\w*|прогрессир\w*", r"регресс\w*|ослаблен\w*|ослабева\w*|стиха\w*"),
    ("подтверждение",  r"подтвержд\w*|демонстрир\w*",   r"опроверг\w*|отрица\w*"),
]

_ANTONYM_RE = re.compile(
    "|".join(
        rf"(?P<pos_{i}>{pos})|(?P<neg_{i}>{neg})"
        for i, (_, pos, neg) in enumerate(_ANTONYM_PAIRS)
    ),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PolarityFact:
    """Клинический предикат + его полярность (есть/нет) + контекст."""
    predicate: str         # нормализованное имя предиката (корень или название антонимичной пары)
    polarity: bool         # True = утверждение в положительной форме, False = отрицание
    context: str           # окружение (для сопоставления при сравнении)
    raw: str


def extract_polarity_facts(text: str, context_window: int = 50) -> list[PolarityFact]:
    """
    Извлекает клинические утверждения вместе с их полярностью.

    Два механизма обнаружения инверсии:
      1. Отрицательная частица рядом с предикатом ("не купирован", "болей
         не отмечает") — самый частый и явный случай.
      2. Антонимичные корневые пары ("включают" ↔ "исключают", "в норме" ↔
         "патология") — инверсия БЕЗ отрицательной частицы, которую
         формальный поиск "не …" не поймает.
    """
    facts: list[PolarityFact] = []

    for m in _PREDICATE_RE.finditer(text):
        predicate_raw = m.group("predicate")
        predicate_norm = re.match(r"\w+", predicate_raw.lower()).group(0)[:6]
        polarity = m.group("neg") is None
        start = max(0, m.start() - context_window)
        context = re.sub(r"\s+", " ", text[start:m.start()]).strip().lower()
        facts.append(PolarityFact(
            predicate=predicate_norm, polarity=polarity,
            context=context, raw=m.group(0).strip(),
        ))

    for m in _ANTONYM_RE.finditer(text):
        for i, (name, _pos, _neg) in enumerate(_ANTONYM_PAIRS):
            if m.group(f"pos_{i}"):
                polarity, raw = True, m.group(f"pos_{i}")
            elif m.group(f"neg_{i}"):
                polarity, raw = False, m.group(f"neg_{i}")
            else:
                continue
            start = max(0, m.start() - context_window)
            context = re.sub(r"\s+", " ", text[start:m.start()]).strip().lower()
            facts.append(PolarityFact(predicate=f"antonym:{name}", polarity=polarity,
                                       context=context, raw=raw))
            break

    return facts


# ══════════════════════════════════════════════════════════════════
# 3. СЕМАНТИЧЕСКИЕ СУЩНОСТИ (опционально, через LLM-панель из Блока 0)
# ══════════════════════════════════════════════════════════════════

_ENTITY_EXTRACTION_PROMPT = """\
Ты — помощник для извлечения структурированных медицинских данных из текста.
Извлеки из текста ниже все упоминания следующих категорий сущностей.
Каждую сущность приведи к краткой нормальной форме (именительный падеж,
без вводных слов). Не придумывай ничего, чего нет в тексте.

Категории:
  diagnoses    — диагнозы и клинические состояния
  medications  — лекарственные препараты
  procedures   — процедуры, операции, исследования (КТ, УЗИ, операции и т.п.)
  departments  — отделения, специализации врачей, локализации в организме
  symptoms     — жалобы и симптомы (боль, слабость, одышка, кашель, тошнота,
                 головокружение, потливость и т.п.)
  anamnesis    — значимые факты анамнеза жизни: наследственность, вредные
                 привычки (курение, алкоголь), аллергии, перенесённые травмы

Текст:
\"\"\"
{text}
\"\"\"

Ответь СТРОГО в формате JSON:
{{"diagnoses": ["..."], "medications": ["..."], "procedures": ["..."],
  "departments": ["..."], "symptoms": ["..."], "anamnesis": ["..."]}}
Если категория не встречается — пустой список.
"""

ENTITY_CATEGORIES = ("diagnoses", "medications", "procedures", "departments",
                     "symptoms", "anamnesis")

# Внутрипроцессный кэш извлечения сущностей: идентичный текст+роль -> тот же
# результат. Безопасно (детерминирует повторные вызовы на одинаковом тексте) и
# критично для валидации на синтетике, где ОДИН эталон сравнивается с десятками
# искажённых строк — без кэша эталон переизвлекался бы сотни раз. В проде тексты
# уникальны, кэш почти не срабатывает. Очистка — clear_entity_cache().
_ENTITY_CACHE: dict[tuple[str, str], dict[str, list[str]]] = {}


def clear_entity_cache() -> None:
    _ENTITY_CACHE.clear()


def extract_semantic_entities(panel, text: str, role: str = "judge_1",
                              desc: str = "извлечение сущностей") -> dict[str, list[str]]:
    """
    LLM-извлечение сущностей через абстрактную роль из JudgePanel (Блок 0).
    panel=None — функция не вызывается (обёртки выше должны проверять сами).

    Намеренно тонкий слой над `panel.ask_json`: вся специфика — в промпте
    и схеме, retry/JSON-восстановление остаются в llm_client. Результат
    кэшируется по (роль, текст) на время процесса (см. _ENTITY_CACHE).
    """
    cache_key = (role, text[:6000])
    cached = _ENTITY_CACHE.get(cache_key)
    if cached is not None:
        return {cat: list(v) for cat, v in cached.items()}

    def _validate(parsed: dict, _raw: str) -> None:
        if not any(parsed.get(cat) for cat in ENTITY_CATEGORIES):
            # Пустой результат подозрителен для содержательного мед. текста —
            # просим модель попробовать ещё раз, а не молча возвращаем пустоту.
            if len(text.strip()) > 200:
                raise ValueError("модель не нашла ни одной сущности — похоже на сбой генерации")

    result = panel.ask_json(
        role,
        _ENTITY_EXTRACTION_PROMPT.format(text=text[:6000]),
        desc=desc, validate_fn=_validate, max_attempts=2,
        # think=False — критично для «мыслящих» моделей семейства qwen3:
        # диагностировано прямым вызовом Ollama (тот же промпт), что при их
        # поведении по умолчанию (think включён даже с format="json") скрытые
        # рассуждения <think>...</think> съедают весь бюджет num_predict ДО
        # самого JSON-ответа — итог обрывается без закрывающей скобки или
        # оказывается пустым (см. подробности в llm_client.generate).
        # Здесь — чисто экстрактивная задача без нужды в рассуждениях:
        # явное отключение чинит обрыв И ускоряет вызов (тот же промпт с
        # think=False уложился в 592 из 1024 токенов, done_reason="stop").
        think=False,
    )
    out = {cat: [str(x).strip() for x in result.get(cat, []) if str(x).strip()]
           for cat in ENTITY_CATEGORIES}
    _ENTITY_CACHE[cache_key] = {cat: list(v) for cat, v in out.items()}
    return out


def _normalize_entity(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower()).strip(".,;:")


# Порог посимвольной близости (SequenceMatcher.ratio) для признания двух
# сущностей одной и той же. Подобран по реальной природе шума синтетического
# набора: ДОЛЖЕН поглощать опечатки и словоформы
#   «госпитализирована» vs «госпиталлизирована» ≈ 0.97
#   «креатинфосфокиназа» vs «креатинфосфокиназы» ≈ 0.97
# и НЕ ДОЛЖЕН склеивать реальные клинические подмены (иначе подмена_сущности
# станет невидимой):
#   «кардиореанимация» vs «нейрореанимация» ≈ 0.73
#   «слабость» vs «головокружение» ≈ 0.17
# 0.85 лежит в широком зазоре между этими двумя режимами.
_ENTITY_FUZZY_THRESHOLD = 0.85


def _entity_tokens(name: str) -> set[str]:
    """Значимые токены сущности (>=3 символов) — для сопоставления по
    подмножеству слов: «инфаркт» ⊂ «инфаркт миокарда», иной порядок слов."""
    return {t for t in re.findall(r"[а-яёa-z0-9]+", name.lower()) if len(t) >= 3}


def _entity_match(a: str, b: str) -> bool:
    """
    Две (уже нормализованные) сущности считаются одной и той же, если:
      • строки совпадают;
      • одна содержится в другой (аббревиатура и её расшифровка:
        «кфк» ⊂ «кфк (креатинфосфокиназа)»);
      • значимые токены одной — подмножество токенов другой (расширение
        названия / иной порядок слов: «инфаркт» vs «инфаркт миокарда»);
      • посимвольная близость ≥ порога (опечатки, морфологические варианты).

    Это заменяет прежнее «точное равенство ИЛИ подстрока (len>3)», которое
    штрафовало любую опечатку/словоформу как расхождение сущностей и было
    главным источником ложных срабатываний LLM-слоя (benign FPR). Порог
    `_ENTITY_FUZZY_THRESHOLD` намеренно оставляет реальные подмены различимыми.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) > 3 and (a in b or b in a):
        return True
    ta, tb = _entity_tokens(a), _entity_tokens(b)
    if ta and tb and (ta <= tb or tb <= ta):
        return True
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= _ENTITY_FUZZY_THRESHOLD


def _stem_overlap(a: str, b: str) -> bool:
    """Делят ли два слова общий стем достаточной длины — устойчивая к падежным
    окончаниям замена точного совпадения. Порог адаптивен: для коротких слов
    допускается лишь короткое расхождение в конце («боль»/«боли» — стем «бол»),
    для длинных требуется длинный общий стем («одышка»/«одышку» — «одышк»),
    что отсекает случайные совпадения начала разных слов («боль»/«белок»)."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n >= max(3, min(len(a), len(b)) - 2)


# Обрамляющие слова, которые LLM-экстрактор добавляет к сущности, но которых
# нет в исходном тексте: лабораторный показатель «гемоглобин 115 г/л» извлекается
# как «определение гемоглобина», «КФК общая 126 ед/л» — как «определение КФК
# общая». Идентифицирует сущность ЯДРО («гемоглобин», «кфк»), а не рамка — по нему
# и проверяем присутствие в тексте, иначе ложно теряем реально присутствующий факт.
_ENTITY_FRAMING_WORDS = frozenset({
    "определение", "тест", "анализ", "исследование", "проба", "уровень",
    "содержание", "оценка", "измерение", "показатель", "показатели",
    "общий", "общая", "общее", "на", "и", "в", "с", "по",
})


def _entity_present_in_text(entity_norm: str, text_norm: str) -> bool:
    """
    Содержательно присутствует ли сущность в СЫРОМ тексте (а не в списке,
    извлечённом LLM): каждый её ЗНАЧИМЫЙ (не-обрамляющий) токен имеет в тексте
    слово с общим стемом (см. _stem_overlap). Стем-сопоставление, а не точная
    подстрока — LLM нормализует сущность к именительному падежу («одышка»), а в
    тексте косвенный («одышку»). Обрамляющие слова (_ENTITY_FRAMING_WORDS)
    игнорируются — экстрактор добавляет «определение/тест/общий», которых в тексте
    нет. Все значимые токены обязаны присутствовать: реально отсутствующая
    сущность (удалённая или выдуманная экстрактором) ядра в тексте не найдёт.
    """
    if not text_norm:
        return False
    text_tokens = re.findall(r"[а-яёa-z0-9]+", text_norm)
    toks = re.findall(r"[а-яёa-z0-9]+", entity_norm)
    significant = [t for t in toks if len(t) >= 3 and t not in _ENTITY_FRAMING_WORDS]
    check = significant or [t for t in toks if len(t) >= 3] or toks
    if not check:
        return False
    return all(any(_stem_overlap(t, tt) for tt in text_tokens) for t in check)


def compare_entity_sets(entities_a: dict[str, list[str]],
                        entities_b: dict[str, list[str]],
                        *, text_a: str = "", text_b: str = "") -> dict[str, dict]:
    """
    Precision/Recall/F1 по каждой категории сущностей: насколько B покрывает A
    (recall — «не упустили ли», precision — «не придумали ли лишнего»).

    Сопоставление — через `_entity_match` (точное / подстрока / подмножество
    токенов / посимвольная близость), УСТОЙЧИВОЕ к опечаткам, словоформам,
    расшифровкам аббревиатур и перестановке слов. ОДНО правило применяется и
    для подсчёта matched, и для вывода missing_in_b/extra_in_b (раньше это были
    два слегка разных инлайновых правила — латентный источник рассогласования).

    ФИЛЬТР АРТЕФАКТОВ ИЗВЛЕЧЕНИЯ (если переданы сырые text_a/text_b): LLM
    извлекает сущности недетерминированно — один и тот же препарат может быть
    извлечён из эталона, но пропущен при извлечении из кандидата, давая ФАНТОМНЫЙ
    «пропуск» там, где сущность реально есть в тексте кандидата. Поэтому перед
    выводом missing_in_b проверяем, действительно ли сущность ОТСУТСТВУЕТ в сыром
    тексте B (а не только в списке B); симметрично extra_in_b (кандидат на
    галлюцинацию) проверяется против сырого текста A. Это убирает доминирующий
    остаточный шум (пунктуация/перестановка/пояснение «теряли» морфин из-за
    сэмплинга), НЕ затрагивая настоящие пропуск_контента (удалённый текст реально
    отсутствует) и галлюцинации (выдуманный факт реально не в источнике).
    """
    report = {}
    text_a_norm = re.sub(r"\s+", " ", text_a.lower())
    text_b_norm = re.sub(r"\s+", " ", text_b.lower())
    for cat in ENTITY_CATEGORIES:
        raw_a, raw_b = entities_a.get(cat, []), entities_b.get(cat, [])
        # SELF-GROUNDING: доверяем извлечённой сущности, ТОЛЬКО если она реально
        # присутствует в СВОЁМ тексте. LLM-экстрактор регулярно «додумывает»
        # типичные для контекста симптомы/показатели, которых в тексте нет
        # (напр. «тошнота», «одышка» к кардиологическому случаю) — без этого
        # фильтра такая выдумка появляется/исчезает между прогонами и даёт
        # фантомные missing/extra. При отсутствии текста (text_*="" — вызов без
        # сырого текста) фильтр не применяется. См. cross-text проверку ниже —
        # она ловит обратный случай (сущность В тексте, но не извлечена).
        if text_a_norm:
            raw_a = [x for x in raw_a if _entity_present_in_text(_normalize_entity(x), text_a_norm)]
        if text_b_norm:
            raw_b = [x for x in raw_b if _entity_present_in_text(_normalize_entity(x), text_b_norm)]
        norm_a = [_normalize_entity(x) for x in raw_a]
        norm_b = [_normalize_entity(x) for x in raw_b]

        matched_a: set[int] = set()
        matched_b: set[int] = set()
        for i, a in enumerate(norm_a):
            for j, b in enumerate(norm_b):
                if j in matched_b:
                    continue
                if _entity_match(a, b):
                    matched_a.add(i)
                    matched_b.add(j)
                    break

        # Сущность A не нашла пары в списке B, но реально присутствует в ТЕКСТЕ B
        # -> это промах извлечения, а не пропуск контента: считаем покрытой.
        covered_a = set(matched_a)
        for i in range(len(norm_a)):
            if i not in covered_a and _entity_present_in_text(norm_a[i], text_b_norm):
                covered_a.add(i)
        # Сущность B без пары, но присутствует в ТЕКСТЕ A -> не галлюцинация.
        real_extra_b = [j for j in range(len(norm_b))
                        if j not in matched_b and not _entity_present_in_text(norm_b[j], text_a_norm)]

        missing_in_b = [raw_a[i] for i in range(len(norm_a)) if i not in covered_a]
        extra_in_b = [raw_b[j] for j in real_extra_b]

        recall = (len(norm_a) - len(missing_in_b)) / len(norm_a) if norm_a else 1.0
        precision = (len(norm_b) - len(extra_in_b)) / len(norm_b) if norm_b else (1.0 if not norm_a else 0.0)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        report[cat] = {
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "in_a": len(norm_a), "in_b": len(norm_b), "matched": len(matched_b),
            "missing_in_b": missing_in_b,
            "extra_in_b": extra_in_b,
        }
    return report


# ══════════════════════════════════════════════════════════════════
# 4. СВЕРКА ЧИСЛОВЫХ И ПОЛЯРНЫХ ФАКТОВ МЕЖДУ ДВУМЯ ТЕКСТАМИ
# ══════════════════════════════════════════════════════════════════

def _context_overlap(ctx_a: str, ctx_b: str) -> float:
    """Доля общих слов в контекстах — грубая мера «это один и тот же факт»."""
    words_a = set(re.findall(r"[а-яёa-z]{3,}", ctx_a))
    words_b = set(re.findall(r"[а-яёa-z]{3,}", ctx_b))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


@dataclass
class NumericMismatch:
    fact_a: NumericFact
    fact_b: NumericFact
    similarity: float


# Частые модификаторы, которые сами по себе НЕ идентифицируют показатель —
# «общий белок» vs «общий холестерин», «КФК общ.» vs «КФК МБ» делят слово
# «общий», но это разные показатели. Совпадение только по таким словам
# не должно считаться сопоставлением подписей.
_GENERIC_LABEL_WORDS = {
    "общий", "общая", "общее", "общ", "анализ", "крови", "при", "поступлении",
    "пациент", "пациента", "больной", "больного", "около", "более", "менее",
    "по", "на", "от", "из", "со", "до", "за", "об", "во", "не", "его", "её",
    "ее", "из-за", "без", "для", "под", "над",
}


def _label_match(label_a: str, label_b: str) -> bool:
    """
    Подписи показателей совпадают, если МНОЖЕСТВА их значимых слов
    (без частых модификаторов из _GENERIC_LABEL_WORDS) равны целиком.

    Почему именно равенство множеств, а не пересечение: составные названия
    лабораторных показателей различаются ИМЕННО уточняющим словом —
    «КФК общ.» vs «КФК МБ», «общий белок» vs «общий холестерин» — общая
    часть названия не делает их одним и тем же показателем. Пересечение
    по одному слову («КФК», «общий») даёт ложные пары; точное совпадение
    оставшегося набора слов — нет (ценой того, что переформулировка подписи
    «возраст: 56» -> «пациенту, 56 лет» может остаться несопоставленной —
    это осознанный выбор в пользу точности, а не полноты: ложное обвинение
    в «числовом искажении» там, где его нет, дороже пропуска одного факта).
    """
    if not label_a or not label_b:
        return False
    significant_a = set(label_a.split()) - _GENERIC_LABEL_WORDS
    significant_b = set(label_b.split()) - _GENERIC_LABEL_WORDS
    if not significant_a or not significant_b:
        return False
    return significant_a == significant_b


_PERSON_AGE_CUE_RE = re.compile(
    r"паци\w*|больн\w*|мужчин\w*|женщин\w*|возраст\w*|\bг\.?\s?р\.?\b",
    re.IGNORECASE,
)


def _is_person_age_context(context: str) -> bool:
    """Контекст указывает, что число рядом — именно возраст человека
    (а не длительность в годах вроде «стаж курения 45 пачка-лет»)."""
    return bool(_PERSON_AGE_CUE_RE.search(context))


@dataclass
class UnitMismatch:
    """Совпадающее число, но другая единица измерения той же категории —
    «40 минут» -> «40 часов» (искажение единицы при сохранении значения)."""
    fact_a: NumericFact
    fact_b: NumericFact


def _norm_unit(unit_raw: str) -> str:
    return re.sub(r"\s+", "", unit_raw.lower())


def compare_numeric_facts(facts_a: list[NumericFact], facts_b: list[NumericFact]) -> dict:
    """
    Ищет пары «один и тот же показатель, иное значение» между двумя наборами —
    это и есть «числовое искажение» («56 лет» → «46 лет», «40 минут» → «40 часов»).

    Применяет сопоставление в порядке убывания надёжности:

      1) БЕЗАЛЬТЕРНАТИВНАЯ пара по категории: если в обоих текстах ровно
         один факт данной категории (типично для «возраст» — упоминается
         один раз), они заведомо об одном и том же — сопоставляем напрямую,
         не полагаясь на подпись (которой у одиночного «56 лет» в формате
         «Больная И., 56 лет» попросту нет — только имя слева).

      2) По СОВПАДЕНИЮ ПОДПИСИ показателя (`label`) — ближайшее слово-название
         слева от числа. Это основной канал для лабораторных панелей
         («гемоглобин 115 г/л, эритроциты 3.43×10¹²/л, …»): сравнение по
         широкому контексту путает соседние показатели одной размерности
         (оба упомянуты в «общем анализе крови»), а подпись — нет.
         Контекст остаётся вторичным сигналом — тай-брейк при повторных
         упоминаниях одного показателя (несколько визитов/анализов).

    Несопоставленные факты не считаются ни совпадением, ни расхождением —
    мы не можем достоверно утверждать, что речь об одном измерении
    (точность важнее полноты: ложное «искажение» дороже пропуска одного факта).

    ОТДЕЛЬНО (вне пар «значение изменилось») — `unit_mismatches`: то же число,
    та же категория, но другая единица («40» осталась «40», но «минут» стало
    «часов»). Совпадение конкретного числового значения в одной размерностной
    категории — само по себе сильный, маловероятно случайный сигнал, поэтому
    эта проверка не требует совпадения подписи.
    """
    mismatches: list[NumericMismatch] = []
    unit_mismatches: list[UnitMismatch] = []
    matched_pairs: list[tuple[NumericFact, NumericFact]] = []
    used_b: set[int] = set()

    by_category_a: dict[str, list[NumericFact]] = {}
    by_category_b: dict[str, list[NumericFact]] = {}
    for f in facts_a:
        by_category_a.setdefault(f.category, []).append(f)
    for f in facts_b:
        by_category_b.setdefault(f.category, []).append(f)

    unambiguous_pairs: dict[int, int] = {}   # index in facts_a -> index in facts_b
    for category, list_a in by_category_a.items():
        list_b = by_category_b.get(category, [])
        if len(list_a) == 1 and len(list_b) == 1:
            fa0, fb0 = list_a[0], list_b[0]
            # «age» — категория с реальной двусмысленностью русского языка:
            # «лет/года» означают и возраст человека, и просто длительность
            # в годах («курение более 45 пачка-лет», «болеет 2 года»). Без
            # подписи единственный факт категории «age» в каждом тексте может
            # оказаться РАЗНЫМИ вещами — требуем явного указания на возраст
            # человека («пациент…», «больная…», «мужчина, 65 лет», «возраст…»),
            # иначе безальтернативное сопоставление слишком рискованно.
            if category == "age" and not (_is_person_age_context(fa0.context)
                                           and _is_person_age_context(fb0.context)):
                continue
            unambiguous_pairs[facts_a.index(fa0)] = facts_b.index(fb0)

    for i, fa in enumerate(facts_a):
        best_j, best_sim = None, -1.0

        if i in unambiguous_pairs:
            best_j = unambiguous_pairs[i]
            best_sim = 1.0
        else:
            for j, fb in enumerate(facts_b):
                if j in used_b or fb.category != fa.category:
                    continue
                if not _label_match(fa.label, fb.label):
                    continue
                sim = _context_overlap(fa.context, fb.context)
                if sim > best_sim:
                    best_sim, best_j = sim, j

        if best_j is not None and best_j not in used_b:
            fb = facts_b[best_j]
            used_b.add(best_j)
            if abs(fa.value - fb.value) > 1e-9:
                mismatches.append(NumericMismatch(fact_a=fa, fact_b=fb, similarity=best_sim))
            else:
                matched_pairs.append((fa, fb))

    # Дополнительный, независимый проход — поиск «то же число, другая единица».
    # Не помечаем найденные факты как «использованные»: одна и та же пара
    # могла уже быть учтена выше (или не быть — единицы ведь разные, и
    # _label_match ничего об этом не знает); здесь важно не пропустить сигнал.
    seen_pairs = set()
    for fa in facts_a:
        for fb in facts_b:
            if (fb.category != fa.category
                    or abs(fa.value - fb.value) > 1e-9
                    or _norm_unit(fa.unit_raw) == _norm_unit(fb.unit_raw)):
                continue
            key = (fa.raw, fb.raw)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            unit_mismatches.append(UnitMismatch(fact_a=fa, fact_b=fb))

    # Подсчёт «покрытия» по категориям — для шлюза (Блок 3) важно не только
    # «совпадает ли значение», но и сам факт «было ли вообще упомянуто»:
    # совпадение, расхождение значения ИЛИ расхождение единицы — в любом из
    # этих случаев источник факта в B был найден, то есть факт «покрыт».
    covered_ids = ({id(fa) for fa, _ in matched_pairs}
                   | {id(mm.fact_a) for mm in mismatches}
                   | {id(um.fact_a) for um in unit_mismatches})
    by_category: dict[str, dict] = {}
    for f in facts_a:
        bucket = by_category.setdefault(f.category, {"total": 0, "covered": 0})
        bucket["total"] += 1
        if id(f) in covered_ids:
            bucket["covered"] += 1
    for bucket in by_category.values():
        bucket["coverage"] = round(bucket["covered"] / bucket["total"], 3)

    return {
        "matched": len(matched_pairs),
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "unit_mismatches": unit_mismatches,
        "unit_mismatch_count": len(unit_mismatches),
        "total_in_a": len(facts_a),
        "by_category": by_category,
    }


@dataclass
class PolarityFlip:
    fact_a: PolarityFact
    fact_b: PolarityFact
    similarity: float


def compare_polarity_facts(facts_a: list[PolarityFact], facts_b: list[PolarityFact],
                           context_threshold: float = 0.2) -> dict:
    """
    Находит пары «один и тот же клинический предикат, разная полярность» —
    инверсия отрицания («купирован» → «не купирован», «включают» → «исключают»).
    """
    flips: list[PolarityFlip] = []
    matched_pairs: list[tuple[PolarityFact, PolarityFact]] = []
    used_b: set[int] = set()

    for fa in facts_a:
        best_j, best_sim = None, 0.0
        for j, fb in enumerate(facts_b):
            if j in used_b or fb.predicate != fa.predicate:
                continue
            sim = _context_overlap(fa.context, fb.context)
            if sim > best_sim:
                best_sim, best_j = sim, j

        if best_j is not None and best_sim >= context_threshold:
            fb = facts_b[best_j]
            used_b.add(best_j)
            if fa.polarity != fb.polarity:
                flips.append(PolarityFlip(fact_a=fa, fact_b=fb, similarity=best_sim))
            else:
                matched_pairs.append((fa, fb))

    # Покрытие по предикатам — для шлюза (Блок 3): «упомянут ли вообще
    # критический клинический предикат» (аллергоанамнез, диагностические
    # находки), независимо от того, совпала ли его полярность —
    # расхождение полярности само по себе уже сигнал для объективного слоя,
    # а не повод считать факт «отсутствующим».
    covered_ids = {id(fa) for fa, _ in matched_pairs} | {id(fl.fact_a) for fl in flips}
    by_predicate: dict[str, dict] = {}
    for f in facts_a:
        bucket = by_predicate.setdefault(f.predicate, {"total": 0, "covered": 0})
        bucket["total"] += 1
        if id(f) in covered_ids:
            bucket["covered"] += 1
    for bucket in by_predicate.values():
        bucket["coverage"] = round(bucket["covered"] / bucket["total"], 3)

    return {
        "matched": len(matched_pairs),
        "flips": flips,
        "flip_count": len(flips),
        "total_in_a": len(facts_a),
        "by_predicate": by_predicate,
    }


# ══════════════════════════════════════════════════════════════════
# 4b. ЛОЖНАЯ ПРИЧИННОСТЬ (введение причинной связи, которой нет в источнике)
# ══════════════════════════════════════════════════════════════════
# Тип искажения «ложная_причинность»: нейтральное перечисление фактов в
# источнике («…инфаркт миокарда И курение в анамнезе») превращается в
# причинно-следственное утверждение («…инфаркт миокарда ВСЛЕДСТВИЕ курения»).
# Ни числа, ни полярность, ни набор сущностей при этом не меняются — поэтому
# остальные детекторы слепы к нему. Сигнал прост и устойчив: появление в
# кандидате причинного союза/предлога, которого не было в референсе.
#
# Набор маркеров намеренно УЗКИЙ — только однозначно причинные связки. «На
# фоне», «при», «после» исключены сознательно: они описывают временной/фоновый
# контекст, а не причинность, и часто встречаются в нейтральных перефразах —
# их включение давало бы ложные срабатывания на benign-строках.
# «спровоцирован» сознательно НЕ включён: это полярная инверсия (пара
# купирование↔провокация уже в _ANTONYM_PAIRS) — не вводит новой причинной
# СВЯЗКИ между фактами, а переворачивает исход одного предиката. Дублировать
# его здесь значило бы засчитывать один и тот же случай двум сигналам.
_CAUSAL_CONNECTIVES_RE = re.compile(
    r"вследствие|из-?за|по\s+причине|в\s+результате|благодаря|"
    r"обусловлен\w*|привед\w*\s+к|стал\w*\s+причиной",
    re.IGNORECASE,
)


def compare_causality(text_a: str, text_b: str) -> dict:
    """
    Сравнивает плотность причинных связок в референсе и кандидате. `introduced`
    > 0 означает, что кандидат утверждает причинно-следственную связь, которой
    в референсе не было — кандидат на тип «ложная_причинность».
    """
    a_hits = _CAUSAL_CONNECTIVES_RE.findall(text_a)
    b_hits = _CAUSAL_CONNECTIVES_RE.findall(text_b)
    return {
        "in_a": len(a_hits),
        "in_b": len(b_hits),
        "introduced": max(0, len(b_hits) - len(a_hits)),
        "introduced_markers": [re.sub(r"\s+", " ", m).strip()
                               for m in b_hits][: max(0, len(b_hits) - len(a_hits))],
    }


# ══════════════════════════════════════════════════════════════════
# 4c. ЛОЖНАЯ ИНТЕРПРЕТАЦИЯ ЧИСЛОВОГО ЗНАЧЕНИЯ
# ══════════════════════════════════════════════════════════════════
# Тип «ложная_интерпретация»: к голому числу приписывается оценочное слово
# («повышена», «снижено», «в норме»), ПРОТИВОРЕЧАЩЕЕ референсному диапазону —
# «снижено (150/90 мм рт. ст.)» при систолическом 150 (это гипертензия, не
# гипотензия). От BENIGN «интерпретация_норма» отличается ровно медицинской
# ВЕРНОСТЬЮ: корректная оценка («гемоглобин 115 г/л снижен» — он реально низкий)
# согласуется с диапазоном и НЕ флагается. Поэтому сигнал — не «вставлена
# оценка», а «оценка ПРОТИВОРЕЧИТ значению»: так critical ложь отделяется от
# benign корректной интерпретации, неотличимых по поверхности.
#
# Диапазоны намеренно КОНСЕРВАТИВНЫ и заданы лишь для ОДНОЗНАЧНЫХ по единице
# величин (АД, ЧСС, гемоглобин, креатинин, явный избыток кардиоферментов).
# Неоднозначные (ед/л 126 — КФК общая или МБ? ммоль/л — глюкоза/холестерин/
# калий?) ПРОПУСКАЮТСЯ и остаются на LLM-судьях. Цель — высокая ТОЧНОСТЬ
# сигнала (не раздуть benign FPR), а не полнота; полноту добирает режим B.

_INTERP_POLARITY: list[tuple[str, str]] = [
    ("low",    r"ниже\s+нормы|сниж\w*|пониж\w*|низк\w*|уменьш\w*"),
    ("high",   r"выше\s+нормы|повыш\w*|увелич\w*|высок\w*"),
    ("normal", r"в\s+норме|нормальн\w*"),
]
_INTERP_RE = re.compile(
    "|".join(f"(?P<p{i}>{pat})" for i, (_pol, pat) in enumerate(_INTERP_POLARITY)),
    re.IGNORECASE,
)
# Сразу после оценочного слова — значение (опц. «сис/диа») и опц. единица.
_VALUE_AFTER_RE = re.compile(
    r"^[\s\-—:;(]{0,6}(\d{1,4}(?:[.,]\d{1,3})?)(?:\s*/\s*(\d{1,3}))?"
    r"\s*(мм\s?рт|ед/л|г/л|ммоль/л|мкмоль/л|в\s?минут\w*|уд[\w/]*|пг/мл)?",
    re.IGNORECASE,
)


def _reference_polarity(primary: float, secondary: Optional[float], unit_norm: str) -> Optional[str]:
    """Истинная полярность значения относительно референсной нормы — или None,
    если по единице величина неоднозначна (тогда сигнал не выдаём)."""
    u = unit_norm
    if "ммрт" in u:                              # АД — только пара сис/диа однозначна
        if secondary is None:
            return None                          # одиночное «N мм рт ст» неоднозначно
            #   (диастолическое? пульсовое давление? — пропускаем)
        if primary >= 140 or secondary >= 90:    # систолическое — основной ориентир
            return "high"
        return "low" if primary < 90 else "normal"
    if "минут" in u or u.startswith("уд"):       # ЧСС
        return "low" if primary < 60 else ("high" if primary > 100 else "normal")
    if u == "г/л" and 100 <= primary <= 200:     # гемоглобин (по диапазону значений)
        return "low" if primary < 120 else ("high" if primary > 175 else "normal")
    if "мкмоль" in u:                            # креатинин
        return "low" if primary < 55 else ("high" if primary > 115 else "normal")
    if u == "ед/л":                              # кардиоферменты — лишь явный избыток
        return "high" if primary > 400 else None
    return None


def compare_interpretations(text_a: str, text_b: str) -> dict:
    """
    Ищет в кандидате оценочные слова при числах, ПРОТИВОРЕЧАЩИЕ референсному
    диапазону (ложная интерпретация). Корректные оценки (benign
    «интерпретация_норма») согласованы с диапазоном и не попадают сюда.
    text_a не используется (контрадикция неверна независимо от референса; для
    эталона-источника корректных интерпретаций нет) — параметр для единообразия.
    """
    contradictions: list[dict] = []
    for m in _INTERP_RE.finditer(text_b):
        claimed = next(_INTERP_POLARITY[i][0]
                       for i in range(len(_INTERP_POLARITY)) if m.group(f"p{i}") is not None)
        vm = _VALUE_AFTER_RE.match(text_b[m.end(): m.end() + 24])
        if not vm:
            continue
        primary = float(vm.group(1).replace(",", "."))
        secondary = float(vm.group(2)) if vm.group(2) else None
        unit_norm = re.sub(r"\s+", "", (vm.group(3) or "").lower())
        true_pol = _reference_polarity(primary, secondary, unit_norm)
        if true_pol is None or claimed == true_pol:
            continue
        contradictions.append({
            "claimed": claimed, "true": true_pol,
            "fragment": re.sub(r"\s+", " ", (m.group(0) + " " + vm.group(0)).strip())[:48],
        })
    return {"contradictions": contradictions, "count": len(contradictions)}


# ══════════════════════════════════════════════════════════════════
# 5. ОБЪЕДИНЁННЫЙ ОТЧЁТ
# ══════════════════════════════════════════════════════════════════

@dataclass
class ObjectiveComparisonReport:
    numeric:   dict
    polarity:  dict
    entities:  Optional[dict] = None
    causality: Optional[dict] = None
    interpretation: Optional[dict] = None

    def hard_findings(self) -> list[str]:
        """Человекочитаемый список «жёстких» расхождений — то, что обязано
        попасть в обоснование судьи и потенциально включить стоп-правило E1."""
        out = []
        for mm in self.numeric["mismatches"]:
            out.append(f"число «{mm.fact_a.raw}» -> «{mm.fact_b.raw}» "
                       f"(контекст: …{mm.fact_a.context[-30:]})")
        for um in self.numeric["unit_mismatches"]:
            out.append(f"единица измерения «{um.fact_a.raw}» -> «{um.fact_b.raw}» "
                       f"(значение совпадает, единица — нет; контекст: …{um.fact_a.context[-30:]})")
        for fl in self.polarity["flips"]:
            out.append(f"полярность «{fl.fact_a.raw}» -> «{fl.fact_b.raw}» "
                       f"(контекст: …{fl.fact_a.context[-30:]})")
        if self.causality and self.causality.get("introduced"):
            markers = ", ".join(self.causality.get("introduced_markers") or []) or "причинная связка"
            out.append(f"причинность: в кандидате введена связь, отсутствующая "
                       f"в референсе ({markers})")
        if self.interpretation and self.interpretation.get("count"):
            for c in self.interpretation["contradictions"]:
                out.append(f"ложная интерпретация: «{c['fragment']}» — оценка "
                           f"«{c['claimed']}» противоречит значению (норма: {c['true']})")
        if self.entities:
            for cat, rep in self.entities.items():
                for missing in rep["missing_in_b"]:
                    out.append(f"{cat}: «{missing}» не найдено в сравниваемом тексте")
                # ВНИМАНИЕ: «лишние» сущности (extra_in_b) НЕ выносятся сюда как
                # жёсткая находка. При сверке СВОДКИ с СЫРЫМ ИСТОЧНИКОМ добротная
                # сводка законно НАЗЫВАЕТ диагнозы-синтез («стенокардия
                # напряжения», «артериальная гипертензия»), которых нет дословно в
                # источнике, — объективный слой не отличит корректный клинический
                # синтез от настоящей галлюцинации. Это суждение требует
                # клинических знаний и оставлено судьям (они видят extra_in_b
                # отдельно в заземлении блока A — _ground_block_a). Иначе эталон
                # ложно получает «галлюцинации».
        return out

    def to_summary(self) -> dict:
        """
        Компактное, JSON-сериализуемое резюме объективного слоя — Блок 7
        («объективные метрики рядом с LLM-оценками»): не полный отчёт
        (там dataclass'ы NumericFact/PolarityFlip/… не сериализуются
        напрямую), а числовые сводки + читаемые строки расхождений —
        ровно то, что нужно audit-логу (JSONL) и Excel-отчёту, чтобы
        показать рядом с оценкой судей «что объективный слой увидел сам».
        """
        numeric = self.numeric or {}
        polarity = self.polarity or {}

        def _fmt_pair(item) -> str:
            return f"«{item.fact_a.raw}» → «{item.fact_b.raw}»"

        out = {
            "numeric": {
                "total_in_a": numeric.get("total_in_a", 0),
                "matched": numeric.get("matched", 0),
                "mismatch_count": numeric.get("mismatch_count", 0),
                "unit_mismatch_count": numeric.get("unit_mismatch_count", 0),
                "mismatches": [_fmt_pair(m) for m in numeric.get("mismatches", [])],
                "unit_mismatches": [_fmt_pair(m) for m in numeric.get("unit_mismatches", [])],
            },
            "polarity": {
                "total_in_a": polarity.get("total_in_a", 0),
                "matched": polarity.get("matched", 0),
                "flip_count": polarity.get("flip_count", 0),
                "flips": [_fmt_pair(f) for f in polarity.get("flips", [])],
            },
            "causality": dict(self.causality) if self.causality else None,
            "interpretation": dict(self.interpretation) if self.interpretation else None,
            "entities": None,
        }
        if self.entities:
            out["entities"] = {
                cat: {"precision": rep["precision"], "recall": rep["recall"], "f1": rep["f1"],
                      "in_a": rep["in_a"], "in_b": rep["in_b"], "matched": rep["matched"],
                      "missing_in_b": list(rep["missing_in_b"]), "extra_in_b": list(rep["extra_in_b"])}
                for cat, rep in self.entities.items()
            }
        return out


def empty_report() -> ObjectiveComparisonReport:
    """Пустой (нейтральный) отчёт — для режима «только LLM-судьи» (grounded=False),
    когда объективный слой сознательно НЕ запускается. `hard_findings()` на нём
    возвращает [], `to_summary()` — нулевые счётчики. Позволяет собрать
    JudgeContext без вызова детерминированных проверок."""
    return ObjectiveComparisonReport(
        numeric={"total_in_a": 0, "matched": 0, "mismatch_count": 0,
                 "unit_mismatch_count": 0, "mismatches": [], "unit_mismatches": []},
        polarity={"total_in_a": 0, "matched": 0, "flip_count": 0, "flips": []},
        entities=None, causality=None, interpretation=None,
    )


def compare_texts(text_a: str, text_b: str, *,
                  panel=None, entity_role: str = "judge_1") -> ObjectiveComparisonReport:
    """
    Главная точка входа: полная фактологическая сверка двух текстов.

    text_a — «эталон» (исходная ЭМК при оценке суммаризации; базовый текст
             при регрессии на синтетике);
    text_b — «проверяемый» (суммаризация / перефразированный вариант).

    panel — JudgePanel (Блок 0) для LLM-извлечения сущностей; если None —
            семантический слой пропускается, отчёт строится только на
            детерминированных правилах (числа + полярность). Это намеренно:
            objective_layer должен давать содержательный результат и БЕЗ LLM.
    """
    facts_num_a = extract_numeric_facts(text_a)
    facts_num_b = extract_numeric_facts(text_b)
    numeric = compare_numeric_facts(facts_num_a, facts_num_b)

    facts_pol_a = extract_polarity_facts(text_a)
    facts_pol_b = extract_polarity_facts(text_b)
    polarity = compare_polarity_facts(facts_pol_a, facts_pol_b)

    causality = compare_causality(text_a, text_b)
    interpretation = compare_interpretations(text_a, text_b)

    entities = None
    if panel is not None:
        ent_a = extract_semantic_entities(panel, text_a, role=entity_role, desc="сущности (A)")
        ent_b = extract_semantic_entities(panel, text_b, role=entity_role, desc="сущности (B)")
        entities = compare_entity_sets(ent_a, ent_b, text_a=text_a, text_b=text_b)

    return ObjectiveComparisonReport(numeric=numeric, polarity=polarity,
                                     entities=entities, causality=causality,
                                     interpretation=interpretation)


# ══════════════════════════════════════════════════════════════════
# САМОПРОВЕРКА
# ══════════════════════════════════════════════════════════════════

def _self_check():
    import glob
    import pandas as pd

    print("=" * 70)
    print("Нечёткое сопоставление сущностей и детектор причинности (юнит-проверки)")
    print("=" * 70)
    # Опечатки/словоформы/расшифровки — ДОЛЖНЫ считаться той же сущностью:
    assert _entity_match("госпитализирована", "госпиталлизирована")
    assert _entity_match("креатинфосфокиназа", "креатинфосфокиназы")
    assert _entity_match("кфк (креатинфосфокиназа)", "кфк")
    assert _entity_match("инфаркт", "инфаркт миокарда")
    # Реальные клинические подмены — ДОЛЖНЫ остаться различимыми:
    assert not _entity_match("кардиореанимация", "нейрореанимация")
    assert not _entity_match("слабость", "головокружение")
    # Введение ложной причинности ловится, нейтральное перечисление — нет:
    assert compare_causality("инфаркт миокарда и курение",
                             "инфаркт миокарда вследствие курения")["introduced"] == 1
    assert compare_causality("инфаркт миокарда и курение",
                             "инфаркт миокарда, курение в анамнезе")["introduced"] == 0
    # Фильтр артефактов извлечения: «пропущенная» сущность, реально
    # присутствующая в тексте кандидата, НЕ считается пропуском; реально
    # удалённая — считается:
    _art = compare_entity_sets({"medications": ["морфин"]}, {"medications": []},
                               text_a="назначен морфин", text_b="назначен морфином внутривенно")
    assert _art["medications"]["missing_in_b"] == [], "фантомный пропуск не отфильтрован"
    _real = compare_entity_sets({"medications": ["морфин"]}, {"medications": []},
                                text_a="назначен морфин", text_b="терапия скорректирована")
    assert _real["medications"]["missing_in_b"] == ["морфин"], "реальный пропуск потерян"
    # Обрамляющие слова экстрактора («определение») не мешают заземлению ядра:
    assert _entity_present_in_text("определение гемоглобина", "гемоглобин 115 г/л")
    assert not _entity_present_in_text("тошнота", "приступ загрудинных болей")
    # SELF-GROUNDING: выдуманный экстрактором симптом (нет ни в одном тексте)
    # не порождает ни missing, ни extra; реально присутствующий — порождает:
    _hall = compare_entity_sets({"symptoms": ["тошнота"]}, {"symptoms": []},
                                text_a="боль за грудиной", text_b="боль за грудиной")
    assert _hall["symptoms"]["missing_in_b"] == [], "галлюцинация извлечения не отфильтрована"
    _grounded = compare_entity_sets({"symptoms": ["тошнота"]}, {"symptoms": []},
                                    text_a="тошнота и боль", text_b="боль за грудиной")
    assert _grounded["symptoms"]["missing_in_b"] == ["тошнота"], "реальный пропуск симптома потерян"
    # Ложная интерпретация: оценка ПРОТИВОРЕЧИТ диапазону -> флаг; согласованная
    # (benign интерпретация_норма) и неоднозначная по единице -> НЕ флаг:
    assert compare_interpretations("", "АД снижено (150/90 мм рт. ст.)")["count"] == 1
    assert compare_interpretations("", "гемоглобин повышен (115 г/л)")["count"] == 1
    assert compare_interpretations("", "ЧСС снижена (86 в минуту)")["count"] == 1
    assert compare_interpretations("", "гемоглобин снижен (115 г/л)")["count"] == 0   # верно: низкий
    assert compare_interpretations("", "АД повышено (174/98 мм рт. ст.)")["count"] == 0  # верно: высокое
    assert compare_interpretations("", "КФК в норме (126 ед/л)")["count"] == 0        # ед/л неоднозначно
    assert compare_interpretations("", "глюкоза в норме (5,6 ммоль/л)")["count"] == 0  # ммоль/л неоднозначно
    print("  OK: опечатки склеиваются, подмены различимы, причинность ловится, "
          "артефакты/галлюцинации извлечения отфильтрованы, ложная интерпретация "
          "отделена от корректной по референсным диапазонам")

    print()
    print("=" * 70)
    print("Извлечение фактов на реальной ЭМК (EMR_01)")
    print("=" * 70)
    df = pd.read_excel("data/summaries.xlsx")
    sample = df[df["emr_id"] == "EMR_01"].iloc[0]
    src, summ = sample["source_text"], sample["summary_text"]

    nf = extract_numeric_facts(src)
    pf = extract_polarity_facts(src)
    print(f"  Источник: {len(nf)} числовых фактов, {len(pf)} полярных утверждений")
    print("  Примеры числовых фактов:")
    for f in nf[:8]:
        print(f"    {f.raw!r:<20} категория={f.category:<11} контекст=…{f.context[-28:]!r}")
    print("  Примеры полярных утверждений:")
    for f in pf[:8]:
        sign = "(+)" if f.polarity else "(-)"
        print(f"    {sign} {f.predicate:<10} «{f.raw}»  контекст=…{f.context[-28:]!r}")

    report = compare_texts(src, summ)
    print(f"\n  Сверка ЭМК vs суммаризация (модель {sample['model_id']}):")
    print(f"    числовых фактов сверено: {report.numeric['matched'] + report.numeric['mismatch_count']}"
          f" | расхождений: {report.numeric['mismatch_count']}")
    print(f"    полярных утверждений сверено: {report.polarity['matched'] + report.polarity['flip_count']}"
          f" | инверсий: {report.polarity['flip_count']}")
    for finding in report.hard_findings()[:5]:
        print(f"    !! {finding}")

    print()
    print("=" * 70)
    print("Валидация на синтетическом stress-наборе (база vs кандидат)")
    print("=" * 70)
    print("  (полная авто-классификация типов искажений — задача Блока 6;")
    print("   здесь — точечная проверка, что объективный слой ЛОВИТ контрольные")
    print("   случаи инверсии отрицания / числового искажения / подмены сущности,")
    print("   которые мы вручную нашли в датасете при разведочном анализе)")
    print()

    paths = [p for p in glob.glob("../materials/*.xlsx") if "~$" not in p]
    if not paths:
        print("  (синтетический датасет не найден — пропуск)")
        return
    syn = pd.read_excel(paths[0], sheet_name="Лист1", dtype=str).fillna("")
    base = syn["Ж1"].iloc[0]

    # Вручную отобранные по результатам разведочного diff-анализа строки
    # с известным (на глаз) типом искажения — для точечной, читаемой проверки.
    known_cases = {
        6:  "инверсия отрицания (купирован -> НЕ купирован)",
        8:  "подмена сущности (кардиореанимация -> нейрореанимация)",
        9:  "числовое искажение (40 минут -> 40 часов) + подмена сущности",
        13: "числовое искажение возраста (56 лет -> 46 лет)",
    }
    for row_idx, expected in known_cases.items():
        candidate = syn["Ж1"].iloc[row_idx]
        rep = compare_texts(base, candidate)
        findings = rep.hard_findings()
        caught = "ПОЙМАНО" if findings else "не поймано"
        print(f"  строка {row_idx:>2} | ожидание: {expected}")
        print(f"            -> {caught}; находок: {len(findings)}")
        for f in findings[:3]:
            print(f"               - {f}")
        print()

    print("  Полный прогон по всем 38 кандидатам (база vs строки 1..38) —")
    print("  доля строк с находками = верхняя граница false-positive rate")
    print("  (часть находок здесь — настоящие искажения, не ложные срабатывания;")
    print("   точная разметка типов искажений по всем строкам — задача Блока 6):")
    total, with_findings = 0, 0
    for row_idx in range(1, len(syn)):
        candidate = syn["Ж1"].iloc[row_idx]
        if not candidate.strip():
            continue
        total += 1
        rep = compare_texts(base, candidate)
        findings = rep.hard_findings()
        if findings:
            with_findings += 1
            marker = " ".join(f"[{f.split(chr(34))[0].strip()}]" for f in findings)
            print(f"    строка {row_idx:>2}: {len(findings)} находок — {marker}")
    print(f"\n  Итого: находки в {with_findings}/{total} строках "
          f"({100 * with_findings / total:.0f}%)")


if __name__ == "__main__":
    _self_check()
