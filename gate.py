"""
gate.py — pre-evaluation gate: бинарный фильтр перед LLM-as-Judge.

Зачем нужен отдельный этап ДО привлечения тяжёлых LLM-судей: прогон
суммаризации, которая вообще не упоминает критические факты источника
(диагнозы, ключевые лабораторные показатели, аллергоанамнез, перенесённые
операции/процедуры), через 3-раундовую LLM-оценку — это трата GPU-времени
на документ, по которому и так понятно, что он неполон. Шлюз отсекает
такие случаи раньше, не вызывая ни одной модели-судьи.

Отличие шлюза от объективного слоя (Блок 2) — по СМЫСЛУ проверки:
  • объективный слой ищет ИСКАЖЕНИЯ — факт упомянут, но неверно
    («56 лет» -> «46 лет», «купирован» -> «не купирован»);
  • шлюз ищет ПРОПУСКИ — факт критической категории просто не встретился
    в кандидате ни в каком виде (ни верном, ни искажённом).

Поэтому шлюз не дублирует извлечение фактов, а строится НАД отчётом
объективного слоя: тот уже даёт по каждой категории/предикату разбивку
«всего в источнике / нашлась пара в кандидате» (`by_category`,
`by_predicate`) — шлюзу остаётся интерпретировать «не нашлась пара»
как «потенциальный пропуск критической информации».

Решение детерминированное и прозрачное (никаких LLM на этом этапе —
ускоряется процесс gate'ования, обязательного по дизайну исследования)
и опционально усиливается LLM-сущностями, когда модели доступны.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import preprocessor
from objective_layer import (
    NumericFact, PolarityFact, ENTITY_CATEGORIES,
    extract_numeric_facts, extract_polarity_facts,
    compare_entity_sets, extract_semantic_entities,
)

# ══════════════════════════════════════════════════════════════════
# КРИТЕРИИ «КРИТИЧНОСТИ» — какие категории фактов считаем обязательными
# к упоминанию в суммаризации (по духу плана: «диагнозы, ключевые лаб.
# показатели, аллергии, операции»)
# ══════════════════════════════════════════════════════════════════

# Числовые категории объективного слоя, пропуск которых — потеря
# клинически значимой информации (лабораторные показатели, дозировки,
# возраст пациента — стандартный обязательный атрибут любой ЭМК).
CRITICAL_NUMERIC_CATEGORIES = ("lab_value", "dose", "age")

# Предикаты-стебли из объективного слоя, покрывающие критические
# клинические утверждения: аллергоанамнез («отягощён»/«не отягощён»),
# диагностические находки («выявлено», «обнаружено», «диагностирован»,
# «подтверждён») — пропуск любого из них означает, что суммаризация не
# сообщает, было ли что-то найдено/исключено у пациента.
CRITICAL_PREDICATES = ("отягощ", "выявле", "обнаруж", "диагно", "подтве")


# D1: суммаризация обязана содержать как минимум столько из 5 разделов
# схемы (B1-B5), чтобы можно было считать её структурно полноценной.
MIN_SCHEMA_SECTIONS = 3

# Если используется LLM-извлечение сущностей — порог recall (покрытие
# сущностей источника в кандидате) по диагнозам/процедурам/препаратам.
MIN_ENTITY_RECALL = 0.5

# Категории, полное отсутствие которых (recall==0) — клинически критический
# провал покрытия, а НЕ закономерная компрессия. Только их нулевое покрытие
# блокирует общую (scope=None) сжимающую суммаризацию. Диагнозы и препараты —
# то, без чего сводка по сути бесполезна/опасна; описательные категории
# (departments/symptoms) добротная сводка может закономерно не покрыть.
CRITICAL_ENTITY_CATEGORIES = ("diagnoses", "medications")


@dataclass
class GateReason:
    """Одна конкретная причина непрохождения шлюза — для аудита и анализа."""
    code: str           # машиночитаемый код причины (для статистики/фильтров)
    message: str        # человекочитаемое описание (для отчёта/лога)
    severity: str       # "reject" | "rework" — насколько критична причина


@dataclass
class GateDecision:
    """
    Итог работы шлюза.

    status:
      "pass"   — суммаризация проходит дальше, к LLM-as-Judge
      "rework" — формально проходима, но есть значимые пропуски —
                 имеет смысл доработать перед повторной оценкой
      "reject" — фундаментальные пропуски критической информации —
                 LLM-оценка не запускается вовсе (экономия GPU)

    "rework" vs "reject" — это всё ещё ОДИН и тот же бинарный исход
    («не прошла» -> дальше не идёт), просто с разной тяжестью причины:
    решение «прошла/не прошла» принимается по `passed`, а это разделение
    нужно только для приоритизации при последующем ручном разборе причин.
    """
    status: str
    reasons: list[GateReason] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def reason_codes(self) -> list[str]:
        return [r.code for r in self.reasons]


def _category_counts(facts: list[NumericFact]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in facts:
        counts[f.category] = counts.get(f.category, 0) + 1
    return counts


def _predicate_counts(facts: list[PolarityFact]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in facts:
        counts[f.predicate] = counts.get(f.predicate, 0) + 1
    return counts


# Почему ШЛЮЗ использует ПРОСТОЙ ПОДСЧЁТ числа фактов по категориям, а не
# попарное сопоставление из объективного слоя (Блок 2): то сопоставление
# нарочно консервативно — оно требует совпадения подписи показателя именно
# затем, чтобы не путать «гемоглобин» с «эритроциты» при поиске ИСКАЖЕНИЙ.
# При сильном перефразе подписи расходятся, и доля «сопоставленных» падает
# даже там, где факты на месте.
#
# Почему порог — НАЛИЧИЕ (>0), а не ДОЛЯ от количества в источнике: суммаризация
# по определению сжимает текст — ЭМК упоминает дозу препарата 25 раз (на каждое
# назначение), нормальная сводная суммаризация упомянет её 1-2 раза. Падение
# «доли» в таком случае — ожидаемое и желательное сжатие, а не потеря факта.
# Первый прогон с порогом по доле (15%) давал 100% reject на реальном корпусе —
# не потому что факты вырезаны, а потому что хорошие суммаризации СЖИМАЮТ.
# Шлюз по плану — «бинарный фильтр покрытия критических сущностей»: вопрос не
# «сколько раз», а «упомянута ли категория вообще хоть раз» — присутствует
# целиком или вырезана целиком.
def _numeric_coverage_reasons(facts_a: list[NumericFact],
                              facts_b: list[NumericFact]) -> list[GateReason]:
    counts_a, counts_b = _category_counts(facts_a), _category_counts(facts_b)
    reasons = []
    for category in CRITICAL_NUMERIC_CATEGORIES:
        total = counts_a.get(category, 0)
        if total == 0:
            continue   # в источнике вообще нет фактов этой категории — нечего проверять
        present = counts_b.get(category, 0)
        if present == 0:
            reasons.append(GateReason(
                code=f"numeric_coverage_low:{category}",
                message=(f"в источнике есть числовые факты категории «{category}» "
                         f"({total} шт.), в кандидате — ни одного: похоже, "
                         f"блок вырезан целиком"),
                severity="reject",
            ))
    return reasons


def _predicate_coverage_reasons(facts_a: list[PolarityFact],
                                facts_b: list[PolarityFact]) -> list[GateReason]:
    counts_a, counts_b = _predicate_counts(facts_a), _predicate_counts(facts_b)
    reasons = []
    for predicate in CRITICAL_PREDICATES:
        total = counts_a.get(predicate, 0)
        if total == 0:
            continue
        present = counts_b.get(predicate, 0)
        if present == 0:
            reasons.append(GateReason(
                code=f"predicate_coverage_low:{predicate}",
                message=(f"в источнике есть упоминания «{predicate}…» ({total} шт.), "
                         f"в кандидате — ни одного: похоже, утверждение "
                         f"вообще не отражено"),
                severity="reject",
            ))
    return reasons


def _schema_reason(summary_text: str) -> Optional[GateReason]:
    structure = preprocessor.segment_summary(summary_text)
    compliance = preprocessor.schema_compliance(structure)
    present = len(compliance["present_categories"])
    if present < MIN_SCHEMA_SECTIONS:
        missing_ru = ", ".join(preprocessor.CATEGORY_LABELS_RU[c]
                               for c in compliance["missing_categories"])
        return GateReason(
            code="schema_incomplete",
            message=(f"структура суммаризации неполна: присутствует только "
                     f"{present}/5 обязательных разделов схемы "
                     f"(отсутствуют: {missing_ru})"),
            severity="reject" if present == 0 else "rework",
        )
    return None


def _entity_recall_reasons(entity_report: dict, *, scope: Optional[str]) -> list[GateReason]:
    """
    Покрытие сущностей источника в кандидате.

    scope=None (общая, СЖИМАЮЩАЯ суммаризация полного исходника): низкое покрытие —
    ЗАКОНОМЕРНАЯ компрессия (сводка на ~500 симв. не повторяет каждую сущность
    исходника на ~2000 симв.); полноту по сути оценивают судьи (блок B). Поэтому
    здесь блокируем ТОЛЬКО полное отсутствие клинически КРИТИЧЕСКОЙ категории
    (диагнозы/препараты, recall==0) — реальный провал, а не сжатие. Иначе любой
    добротный конспект (включая эталон) ложно отклонялся бы из-за описательных
    категорий вроде departments/symptoms, которые сводка закономерно опускает.

    scope задан (целевая выжимка под адресата): сущности источника уже сужены
    фильтром релевантности до значимых для задачи — их непокрытие значимо, поэтому
    действует обычный порог MIN_ENTITY_RECALL.
    """
    reasons = []
    for category, rep in entity_report.items():
        if rep["in_a"] == 0:
            continue
        recall = rep["recall"]
        if scope is None:
            if category in CRITICAL_ENTITY_CATEGORIES and recall == 0:
                reasons.append(GateReason(
                    code=f"entity_recall_zero:{category}",
                    message=(f"в источнике есть сущности категории «{category}», в "
                             f"кандидате — НИ ОДНОЙ ({', '.join(rep['missing_in_b'][:5]) or '—'}): "
                             f"критическая информация полностью отсутствует"),
                    severity="reject"))
        elif recall < MIN_ENTITY_RECALL:
            reasons.append(GateReason(
                code=f"entity_recall_low:{category}",
                message=(f"низкая полнота сущностей категории «{category}»: "
                         f"recall={recall:.0%} (порог {MIN_ENTITY_RECALL:.0%}); "
                         f"пропущено: {', '.join(rep['missing_in_b'][:5]) or '—'}"),
                severity="reject" if recall == 0 else "rework"))
    return reasons


# ══════════════════════════════════════════════════════════════════
# АУДИТОРИЯ/ЗАДАЧА СУММАРИЗАЦИИ — параметр прогона, а не догадка по тексту
# ══════════════════════════════════════════════════════════════════
#
# Реальные суммаризации в корпусе — НЕ сжатые пересказы всей карты, а целевые
# выжимки под конкретную задачу (по словам заказчика — весь нынешний корпус из
# 26 ЭМК собран для оценки суммаризации именно «для врача-рентгенолога»; ЭМК на
# ~19К символов -> выжимка на ~1.3К). Такая суммаризация ЗАКОННО опускает
# разделы, не относящиеся к задаче адресата (например, офтальмологию при
# направлении на КТ органов брюшной полости) — проверка «упомянуто ли вообще»
# по сырым числам/предикатам в этом случае систематически давала ложный reject
# (первый прогон на корпусе — 156/156 reject, хотя суммаризации по сути
# корректны: см. разбор EMR_10/модель 2).
#
# ВАЖНО: пытались определять адресата по заголовку суммаризации (регэксп на
# «Данные для рентгенолога» и т.п.) — не сработало: разные модели формулируют
# заголовок по-разному («Резюме для врача-рентгенолога», «Жалобы, ставшие
# причиной выполнения КТ…», «Медицинское резюме» вообще без адресата) — проверка
# по факту находила совпадение лишь в 11 из 156 пар. Сама задача суммаризации
# («для кого и зачем») определяется ПОСТАНОВКОЙ ЭКСПЕРИМЕНТА (промптом, который
# дали оцениваемой модели), а не тем, что модель решила написать в заголовке —
# поэтому это ВХОДНОЙ ПАРАМЕТР прогона, а не то, что нужно угадывать по тексту.
#
# Синтетический набор — наоборот: пары «база/перефраз» одного объёма, без
# заявленной задачи-адресата — там «полное покрытие» и есть правильный
# ориентир (уже проверено: 10/10 pass на лёгких перефразах, устойчивый reject
# на обрезанных/вырожденных текстах).
#
# Поэтому `evaluate_gate` принимает `scope` явным параметром:
#   scope задан   -> «критичность» сужается до того, что важно ИМЕННО этой
#                    аудитории (LLM-фильтр поверх сущностей Блока 2, см.
#                    filter_relevant_entities); сырая сверка чисел/предикатов
#                    по ПОЛНОМУ объёму источника не применяется — она по
#                    конструкции не различает «вырезали раздел» и «раздел
#                    не нужен этой аудитории»;
#   scope=None    -> исходный «полный» критерий (сырые числа/предикаты) —
#                    обе стороны должны покрывать один и тот же материал
#                    (used для синтетики и для любых будущих наборов без
#                    выраженной задачи-адресата).
#
# Текущий охват — только "radiologist": по словам заказчика, весь нынешний
# корпус собран именно под эту специальность. Масштабирование на другие
# специальности — добавление новых ключей в _RELEVANCE_FILTER_PROMPTS и
# передача соответствующего `scope` при вызове шлюза для нужного набора,
# без изменения механизма.

_RELEVANCE_FILTER_PROMPTS = {
    "radiologist": """\
Ты помогаешь определить, какие сведения из карты пациента ДЕЙСТВИТЕЛЬНО нужны
врачу-рентгенологу перед/при выполнении и трактовке лучевого исследования
(КТ, МРТ, рентген, УЗИ).

Ниже — пронумерованные списки сущностей, извлечённых из карты пациента.
Отметь номера тех пунктов в КАЖДОЙ категории, которые рентгенологу важно
знать для:
  • понимания причины направления и того, что искать на снимках;
  • оценки противопоказаний/безопасности (аллергия на контраст,
    металлические импланты, беременность и т.п.);
  • корректной интерпретации находок с учётом текущих диагнозов и ранее
    выполненных визуализаций того же органа/области.

НЕ отмечай сущности, относящиеся к другим профилям и не влияющие на задачу
рентгенолога (например, офтальмологические находки при направлении на КТ
органов брюшной полости).

{lists}

Ответь СТРОГО в формате JSON со списками номеров (целые числа) по КАЖДОЙ
категории, например: {{"diagnoses": [0, 2], "medications": [], "procedures": [1],
"departments": [], "symptoms": [0], "anamnesis": [1]}}
Если в категории нет релевантных пунктов — пустой список.
""",
}


def filter_relevant_entities(panel, entities: dict[str, list[str]], scope: str,
                             role: str = "judge_1",
                             desc: str = "шлюз: фильтр релевантности") -> dict[str, list[str]]:
    """
    Сужает извлечённые сущности источника (Блок 2) до тех, что важны именно
    для заявленного адресата `scope`. Возвращает подмножество ИСХОДНЫХ строк
    (модель выбирает номера, а не переписывает текст — так результат остаётся
    проверяемым подмножеством того, что уже извлёк объективный слой, и не
    вносит собственных перефразов/искажений).

    Поддерживаются только адресаты из _RELEVANCE_FILTER_PROMPTS; для прочих
    (включая scope=None) фильтрация не имеет смысла — возвращается вход как есть.
    """
    prompt_template = _RELEVANCE_FILTER_PROMPTS.get(scope)
    if prompt_template is None or not any(entities.get(cat) for cat in ENTITY_CATEGORIES):
        return entities

    lists = "\n".join(
        f"{cat}:\n" + "\n".join(f"  {i}: {item}" for i, item in enumerate(entities.get(cat) or []))
        for cat in ENTITY_CATEGORIES
    )

    result = panel.ask_json(
        role, prompt_template.format(lists=lists),
        desc=desc, max_attempts=2,
        # think=False — это чисто экстрактивная/фильтрующая задача (отобрать
        # релевантные пункты из уже извлечённых списков), рассуждения тут не
        # нужны и не помогают. На qwen3:8b (Ollama 0.24) скрытые <think>-токены
        # по умолчанию включены даже при format="json" и съедают весь бюджет
        # num_predict, из-за чего JSON обрезается/приходит пустым (см. тот же
        # фикс и комментарий в objective_layer.extract_semantic_entities).
        think=False,
    )

    filtered: dict[str, list[str]] = {}
    for cat in ENTITY_CATEGORIES:
        original = entities.get(cat) or []
        idxs = result.get(cat) or []
        filtered[cat] = [original[i] for i in idxs
                         if isinstance(i, int) and 0 <= i < len(original)]
    return filtered


def evaluate_gate(source_text: str, summary_text: str, *,
                  scope: Optional[str] = None,
                  panel=None, entity_role: str = "judge_1") -> GateDecision:
    """
    Прогоняет суммаризацию через pre-evaluation gate.

    source_text  — исходная ЭМК (эталон критических фактов)
    summary_text — проверяемая суммаризация

    scope — заявленная ЗАДАЧА/АУДИТОРИЯ суммаризации, известная из постановки
            прогона (а не угадываемая по тексту, см. комментарий выше про
            «АУДИТОРИЯ/ЗАДАЧА — параметр прогона»). Текущий поддерживаемый
            ключ — "radiologist" (нынешний корпус из 26 ЭМК). None — задача
            не сужена: ожидается, что кандидат покрывает источник целиком
            (синтетические пары база/перефраз и подобные наборы).

    panel — JudgePanel (Блок 0); нужна для двух вещей:
            1) recall семантических сущностей (диагнозы/препараты/процедуры/
               отделения) — всегда, если задана;
            2) при заданном `scope` — сужение состава «критичных» сущностей
               источника до релевантных этой задаче (filter_relevant_entities),
               иначе recall считался бы против ВСЕЙ карты, что для целевой
               выжимки даёт системный ложный reject.
            Без panel шлюз `scope` не учитывает (фильтровать нечем) и
            работает на детерминированных правилах — числовые факты,
            полярность, структура (D1, информационно).

    Решение:
      • есть причины severity="reject"  -> status="reject"
      • нет «reject», но есть «rework»  -> status="rework"
      • причин нет                      -> status="pass"
    """
    facts_num_a = extract_numeric_facts(source_text)
    facts_num_b = extract_numeric_facts(summary_text)

    facts_pol_a = extract_polarity_facts(source_text)
    facts_pol_b = extract_polarity_facts(summary_text)

    reasons: list[GateReason] = []
    if scope is None:
        # Задача не сужена — обе стороны должны охватывать один и тот же
        # материал (синтетические пары база/перефраз и т.п.): полное сырое
        # сопоставление чисел/предикатов уместно.
        reasons += _numeric_coverage_reasons(facts_num_a, facts_num_b)
        reasons += _predicate_coverage_reasons(facts_pol_a, facts_pol_b)
    # При заданном scope сырая сверка по ПОЛНОМУ объёму источника не
    # применяется — целевая выжимка законно опускает нерелевантные разделы;
    # «критичность» там проверяется через сущности, см. ниже.

    # Структурная полнота (D1) — НЕ критерий шлюза: план относит структурную
    # метрику к Блоку 2 (диагностика «что не так»), а шлюз — к проверке
    # ФАКТИЧЕСКОГО покрытия. К тому же формат документов может законно
    # отличаться от строгой схемы B1-B5 (см. синтетический набор — это
    # повествовательные перефразы, не структурированные суммаризации, и
    # их «несоответствие схеме» не означает потери фактов). Несём как
    # информационный сигнал в отчёт — пригодится для аудита (Блок 7) и
    # ручного разбора причин, но решение «пропустить/не пропустить» не меняет.
    schema_reason = _schema_reason(summary_text)

    entity_report = None
    if panel is not None:
        ent_a = extract_semantic_entities(panel, source_text, role=entity_role,
                                          desc="шлюз: сущности источника")
        ent_b = extract_semantic_entities(panel, summary_text, role=entity_role,
                                          desc="шлюз: сущности кандидата")
        ent_a_for_recall = (filter_relevant_entities(panel, ent_a, scope, role=entity_role,
                                                      desc="шлюз: фильтр релевантности")
                            if scope is not None else ent_a)
        # Сырые тексты -> фильтр артефактов извлечения LLM в compare_entity_sets:
        # «пропущенная» сущность, реально присутствующая в тексте суммаризации,
        # не считается пропуском (это промах извлечения, а не потеря контента).
        entity_report = compare_entity_sets(ent_a_for_recall, ent_b,
                                            text_a=source_text, text_b=summary_text)
        reasons += _entity_recall_reasons(entity_report, scope=scope)

    if any(r.severity == "reject" for r in reasons):
        status = "reject"
    elif reasons:
        status = "rework"
    else:
        status = "pass"

    coverage = {
        "declared_scope": scope,
        "numeric_counts_source": _category_counts(facts_num_a),
        "numeric_counts_candidate": _category_counts(facts_num_b),
        "predicate_counts_source": _predicate_counts(facts_pol_a),
        "predicate_counts_candidate": _predicate_counts(facts_pol_b),
        "schema_note": schema_reason.message if schema_reason else None,
        "entities": entity_report,
    }
    return GateDecision(status=status, reasons=reasons, coverage=coverage)


# ══════════════════════════════════════════════════════════════════
# САМОПРОВЕРКА
# ══════════════════════════════════════════════════════════════════

def _self_check():
    import glob
    import pandas as pd

    print("=" * 70)
    print("Шлюз на реальных парах ЭМК / суммаризация (scope='radiologist',")
    print("без LLM-панели — проверяем только rule-based часть: числа/предикаты")
    print("по полному объёму источника ОТКЛЮЧЕНЫ, т.к. задача сужена; остаётся")
    print("структура (информационно). Цель прогона — убедиться, что сырые")
    print("rule-based проверки больше НЕ дают ложный 100% reject на целевых")
    print("выжимках (как было до учёта scope, см. комментарий к evaluate_gate)")
    print("=" * 70)
    df = pd.read_excel("data/summaries.xlsx")
    counts = {"pass": 0, "rework": 0, "reject": 0}
    reason_counter: dict[str, int] = {}
    examples: dict[str, tuple] = {}

    for _, row in df.iterrows():
        decision = evaluate_gate(row["source_text"], row["summary_text"],
                                 scope="radiologist")
        counts[decision.status] += 1
        for code in decision.reason_codes():
            reason_counter[code] = reason_counter.get(code, 0) + 1
        if decision.status != "pass" and decision.status not in examples:
            examples[decision.status] = (row["emr_id"], row["model_id"], decision)

    total = len(df)
    print(f"  Всего пар: {total}")
    for status in ("pass", "rework", "reject"):
        print(f"    {status:<8}: {counts[status]:>4}  ({100*counts[status]/total:.1f}%)")
    if reason_counter:
        print("\n  Частота причин (по всем парам, может быть несколько на пару):")
        for code, n in sorted(reason_counter.items(), key=lambda kv: -kv[1]):
            print(f"    {code:<35} {n:>4}")
        print("\n  Примеры непрошедших (по одной на статус):")
        for status, (emr_id, model_id, decision) in examples.items():
            print(f"\n  [{status}] {emr_id} / модель {model_id}:")
            for r in decision.reasons:
                print(f"      - ({r.severity}) {r.message}")
    else:
        print("  Причин для rework/reject не нашлось — rule-based часть шлюза")
        print("  больше не штрафует целевые выжимки за компрессию (как и должно")
        print("  быть: единственный гейтинг для них — через сущности, ниже).")

    print()
    print("=" * 70)
    print("Шлюз на реальных парах с LLM-панелью (scope-aware entity recall) —")
    print("выборка из нескольких пар: проверяем, что фильтр релевантности +")
    print("сверка сущностей действительно ловит/пропускает осмысленно, а не")
    print("просто отключает критерий целиком")
    print("=" * 70)
    try:
        from llm_client import JudgePanel
        panel = JudgePanel()
        sample_ids = [(emr_id, mid) for emr_id, mid in
                      [("EMR_10", 2), ("EMR_10", 38), ("EMR_11", 2)]
                      if not df[(df["emr_id"] == emr_id) & (df["model_id"] == mid)].empty]
        for emr_id, model_id in sample_ids:
            row = df[(df["emr_id"] == emr_id) & (df["model_id"] == model_id)].iloc[0]
            decision = evaluate_gate(row["source_text"], row["summary_text"],
                                     scope="radiologist", panel=panel)
            print(f"\n  {emr_id} / модель {model_id}: {decision.status}  {decision.reason_codes()}")
            if decision.coverage.get("entities"):
                for cat, rep in decision.coverage["entities"].items():
                    if rep["in_a"] or rep["in_b"]:
                        print(f"      сущности[{cat}]: recall={rep['recall']:.0%} "
                              f"(релевантно в источнике={rep['in_a']}, в кандидате={rep['in_b']}) "
                              f"пропущено={rep['missing_in_b'][:3]}")
    except Exception as e:
        print(f"  (пропуск — LLM-панель недоступна: {e})")

    print()
    print("=" * 70)
    print("Шлюз на синтетическом stress-наборе — должен ловить «вырезанные»")
    print("критические факты и пропускать нормальные перефразы")
    print("=" * 70)
    paths = [p for p in glob.glob("../materials/*.xlsx") if "~$" not in p]
    if not paths:
        print("  (синтетический датасет не найден — пропуск)")
        return
    syn = pd.read_excel(paths[0], sheet_name="Лист1", dtype=str).fillna("")
    base = syn["Ж1"].iloc[0]

    print("\n  (а) Контроль на лёгких перефразах базы (строки 1..10) — ожидаем pass:")
    light_pass = 0
    for i in range(1, 11):
        d = evaluate_gate(base, syn["Ж1"].iloc[i])
        light_pass += d.passed
        print(f"    строка {i:>2}: {d.status:<7} {d.reason_codes()}")
    print(f"    -> pass: {light_pass}/10")

    print("\n  (б) Контроль на «вырезанных» критических блоках —")
    print("      эмулируем намеренное удаление целых разделов из базового текста:")
    # Синтетический набор — повествовательный текст, не телеграфный ЭМК,
    # поэтому «вырезание раздела» эмулируем грубо — по абзацам/предложениям,
    # отбрасывая хвост текста (типичная деградация суммаризации — обрыв).
    sentences = [s.strip() for s in base.replace("\n", " ").split(".") if s.strip()]
    for cut_ratio, label in ((0.3, "обрезано ~70% текста"), (0.6, "обрезано ~40% текста")):
        truncated = ". ".join(sentences[:max(1, int(len(sentences) * cut_ratio))]) + "."
        d = evaluate_gate(base, truncated)
        print(f"    {label:<22}: {d.status:<7} {d.reason_codes()}")

    print("\n  (в) Контроль на пустой/вырожденной суммаризации — ожидаем reject:")
    d = evaluate_gate(base, "Пациентка обратилась с жалобами.")
    print(f"    почти пустой текст     : {d.status:<7} {d.reason_codes()}")


if __name__ == "__main__":
    _self_check()
