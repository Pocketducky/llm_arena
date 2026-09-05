"""
judge.py — Блок 4: LLM-as-Judge — параметризуемые промпты по таксономии A-E
и трёхраундовая схема (независимая оценка -> cross-peer-review -> агрегация).

Заменяет старые PROMPT_CLINICAL/PROMPT_INSTRUMENTAL/PROMPT_PENALTIES
(evaluator.py) — захардкоженные под один конкретный ЭМК (EMR_01, КТ ОБП).

Принцип: судья НЕ оценивает «на глаз» по придуманной им самим рубрике —
ему выдаётся (а) общий шаблон с подкритериями таксономии (Раздел 4.1
дизайна исследования, А1-Е3, дословно из design_doc.txt) и (б) КОНКРЕТНЫЕ
заземляющие данные по ЭТОЙ паре (структура из Блока 1 — preprocessor,
фактологическая сверка из Блока 2 — objective_layer, решение шлюза из
Блока 3 — gate). Судья верифицирует и дополняет то, что уже найдено
автоматикой, а не изобретает оценку с нуля.

Роли «judge_1/2/3» и «aggregator» абстрактны — конкретные модели задаёт
config.py (PILOT_PROFILE на пилотном железе, TARGET_PROFILE — DeepSeek-R1
70B x3 + Qwen3-Next 80B — на проде). Пайплайн обращается к ним только
через JudgePanel.ask_json(role, ...).

Из старого кода сознательно сохранены инженерные решения, доказавшие
рабочесть (см. llm_client.py — туда они уже перенесены и обобщены):
  • repair_json/extract_json — авторемонт «грязного» JSON
  • retry-цепочка: основной промпт -> без format=json -> упрощённый скелет
  • перемешивание порядка судей в раунде 2 (борьба с position bias)
  • чекпоинты на диск (см. load_checkpoint/save_checkpoint)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import re

import config
import preprocessor
import objective_layer
import gate
from llm_client import JudgePanel, LLMError

log = logging.getLogger("judge")

# ══════════════════════════════════════════════════════════════════
# ВЕРСИОНИРОВАНИЕ — Блок 7 («версии промптов/моделей/порогов» в
# audit-логе). До сих пор в кодовой базе не было ни одной версионной
# метки: при изменении промптов/таксономии не было способа понять
# постфактум, КАКИМ именно набором правил оценена та или иная пара —
# критично для воспроизводимости и регуляторной отчётности (план,
# Блок 7). Метки — лёгкие («v1», а не semver с авто-инкрементом):
# поднимаем вручную при содержательном изменении промптов/рубрики,
# audit.py фиксирует их в каждой записи лога как срез «на момент прогона».
# ══════════════════════════════════════════════════════════════════

PROMPT_SET_VERSION = "judge-prompts-v1"     # _PROMPT_BLOCK_A..E, _PROMPT_CROSS_REVIEW, _PROMPT_AGGREGATE
TAXONOMY_VERSION = "design_doc-4.1-v1"      # источник критериев A1-E3 — см. докстринг TAXONOMY ниже

# ══════════════════════════════════════════════════════════════════
# ТАКСОНОМИЯ A-E — дословно из «Дизайн исследования», Раздел 4.1
# (см. materials/design_doc.txt, строки 56-80). Это ЕДИНСТВЕННОЕ место,
# где формулировки подкритериев заданы централизованно — промпты ниже
# лишь подставляют их, не дублируя и не переизобретая рубрику на ходу.
# ══════════════════════════════════════════════════════════════════

TAXONOMY: dict[str, tuple[str, ...]] = {
    "A": ("A1", "A2", "A3"),
    "B": ("B1", "B2", "B3", "B4", "B5"),
    "C": ("C1", "C2"),
    "D": ("D1", "D2", "D3"),
    "E": ("E1", "E2", "E3"),
}
BLOCK_TITLES_RU: dict[str, str] = {
    "A": "Клиническая точность (фактологическая верность)",
    "B": "Полнота (recall)",
    "C": "Релевантность (precision)",
    "D": "Структура и читаемость",
    "E": "Безопасность",
}
SUBCRITERIA_LABELS_RU: dict[str, str] = {
    "A1": "Достоверность извлечённых фактов: отсутствие суждений, не имеющих "
          "основания в исходном тексте (галлюцинации)",
    "A2": "Точность числовых значений: корректность лабораторных показателей, "
          "дат, дозировок препаратов, размеров образований",
    "A3": "Корректность медицинской терминологии: соответствие использованных "
          "терминов общепринятым клиническим стандартам",
    "B1": "Покрытие жалоб, релевантных для конкретного клинического контекста",
    "B2": "Покрытие анамнеза заболевания",
    "B3": "Покрытие анамнеза жизни (сопутствующие заболевания, вредные привычки, "
          "семейный анамнез, операции, аллергологический анамнез)",
    "B4": "Покрытие лабораторных данных",
    "B5": "Покрытие результатов инструментальных исследований",
    "C1": "Отсутствие нерелевантной информации: сведения, не имеющие клинической "
          "значимости для данного контекста, снижают оценку",
    "C2": "Клиническая значимость включённых данных: каждый извлечённый факт "
          "должен быть потенциально значим для специалиста",
    "D1": "Соответствие заданной схеме: наличие и корректная последовательность "
          "всех пяти разделов промпта",
    "D2": "Отсутствие дублирования: одна и та же информация не повторяется "
          "в нескольких разделах",
    "D3": "Ясность и лаконичность: текст читается без специальной подготовки, "
          "не содержит избыточных конструкций",
    "E1": "Отсутствие клинически опасных ошибок (критические галлюцинации — "
          "например, ложное указание на аллергию или замена диагноза — "
          "СТОП-ПРАВИЛО, влечёт автоматический отказ системы)",
    "E2": "Отсутствие пропуска данных в суммаризации (при выявлении пропусков "
          "суммаризация возвращается на доработку)",
    "E3": "Корректная атрибуция данных: каждый факт должен быть соотнесён "
          "с датой и источником (визит, исследование)",
}

# Описания целевой аудитории/задачи — те же ключи, что и в gate.py
# (_RELEVANCE_FILTER_PROMPTS), единый источник смысла «что значит scope».
_SCOPE_DESCRIPTIONS: dict[str, str] = {
    "radiologist": (
        "извлечение клинически значимых сведений, необходимых врачу-рентгенологу "
        "для описания компьютерной томографии органов брюшной полости: жалоб, "
        "анамнеза заболевания и жизни, лабораторных данных и результатов "
        "инструментальных исследований. Сведения других профилей (например, "
        "офтальмологические/ЛОР-находки, не влияющие на интерпретацию КТ ОБП) "
        "ЗАКОНОМЕРНО опускаются — это не пропуск, а корректная фильтрация под "
        "задачу адресата."
    ),
}
_DEFAULT_SCOPE_DESCRIPTION = (
    "структурированное резюме данных ЭМК для поддержки клинического решения. "
    "Явная аудитория/задача для этого прогона не зафиксирована — оценивай "
    "относительно общей цели: дать врачу краткую, точную и полную картину "
    "по пяти разделам схемы (жалобы, анамнез заболевания, анамнез жизни, "
    "лабораторные данные, инструментальные исследования)."
)


def _scope_description(scope: Optional[str]) -> str:
    return _SCOPE_DESCRIPTIONS.get(scope, _DEFAULT_SCOPE_DESCRIPTION) if scope else _DEFAULT_SCOPE_DESCRIPTION


def _role_intro(scope: Optional[str]) -> str:
    return (
        "Ты — независимый врач-эксперт, выполняющий рецензирование автоматической "
        "суммаризации электронной медицинской карты (ЭМК), сформированной языковой "
        "моделью. Суммаризация решает следующую задачу:\n"
        f"{_scope_description(scope)}\n\n"
        "Оценивай СТРОГО по перечисленным ниже подкритериям утверждённой "
        "таксономии — не подменяй их собственной рубрикой и не выдумывай "
        "критерии, которых нет в списке."
    )


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n[…текст обрезан для длины промпта…]"


# ══════════════════════════════════════════════════════════════════
# ЗАЗЕМЛЯЮЩИЕ ДАННЫЕ ИЗ БЛОКОВ 0-3 — то, что судья ОБЯЗАН проверить,
# а не изобретать заново (это и есть «параметризация» промптов)
# ══════════════════════════════════════════════════════════════════

def _ground_block_a(report: objective_layer.ObjectiveComparisonReport) -> str:
    lines: list[str] = []
    num = report.numeric
    if num["mismatches"]:
        lines.append("Числовые расхождения «источник -> кандидат» (правило-based сверка):")
        for mm in num["mismatches"][:25]:
            lines.append(f"  • «{mm.fact_a.raw}» -> «{mm.fact_b.raw}» "
                         f"(контекст источника: …{mm.fact_a.context[-50:]})")
    if num["unit_mismatches"]:
        lines.append("Расхождения единиц измерения при совпадающем значении:")
        for um in num["unit_mismatches"][:15]:
            lines.append(f"  • «{um.fact_a.raw}» -> «{um.fact_b.raw}»")
    pol = report.polarity
    if pol["flips"]:
        lines.append("Инверсии отрицания/полярности утверждения (ОБЯЗАТЕЛЬНО проверь — "
                     "именно такие случаи относятся к E1, если меняют клинический смысл):")
        for fl in pol["flips"][:20]:
            lines.append(f"  • «{fl.fact_a.raw}» -> «{fl.fact_b.raw}» "
                         f"(контекст источника: …{fl.fact_a.context[-50:]})")
    interp = report.interpretation
    if interp and interp.get("count"):
        lines.append("Ложная интерпретация числовых значений (приписана оценка, "
                     "ПРОТИВОРЕЧАЩАЯ референсному диапазону — это галлюцинация A1/A2, "
                     "а при клинической значимости и E1):")
        for c in interp["contradictions"][:15]:
            lines.append(f"  • «{c['fragment']}»: оценка «{c['claimed']}» "
                         f"при истинной норме «{c['true']}»")
    if report.entities:
        for cat, rep in report.entities.items():
            if rep.get("extra_in_b"):
                lines.append(f"Сущности категории «{cat}», упомянутые в кандидате, но НЕ "
                             f"найденные автоматикой в источнике (кандидаты на галлюцинацию — "
                             f"перепроверь по полному тексту источника, прежде чем заносить "
                             f"в hallucinations):")
                for x in rep["extra_in_b"][:15]:
                    lines.append(f"  • {x}")
    if not lines:
        return ("Автоматическая правило-based сверка чисел и полярности расхождений не "
                "выявила. Это НЕ снимает с тебя обязанность проверить текст содержательно — "
                "автоматика ловит лишь формальные числовые/отрицательные конструкции.")
    return "\n".join(lines)


def _ground_block_b(source_struct: preprocessor.DocumentStructure) -> str:
    lines: list[str] = []
    for cat in preprocessor.SCHEMA_CATEGORIES:
        label = preprocessor.CATEGORY_LABELS_RU[cat]
        text = source_struct.text_for(cat)
        if text.strip():
            lines.append(f"--- {label} (раздел «{cat}» исходной ЭМК) ---")
            lines.append(_truncate(text, 2200))
        else:
            lines.append(f"--- {label}: в исходной ЭМК раздел не распознан/пуст ---")
    return "\n".join(lines)


def _ground_block_d(schema: dict) -> str:
    present = ", ".join(preprocessor.CATEGORY_LABELS_RU[c] for c in schema["present_categories"]) or "—"
    missing = ", ".join(preprocessor.CATEGORY_LABELS_RU[c] for c in schema["missing_categories"]) or "—"
    return (
        f"Найденные разделы схемы (в порядке появления): {present}\n"
        f"Отсутствующие разделы схемы: {missing}\n"
        f"Порядок разделов корректен: {'да' if schema['order_correct'] else 'нет'}\n"
        f"Формальное соответствие схеме (все 5 разделов на месте, в верном порядке): "
        f"{'да' if schema['compliant'] else 'нет'}"
    )


def _ground_block_e(report: objective_layer.ObjectiveComparisonReport,
                    decision: gate.GateDecision) -> tuple[str, str]:
    findings = report.hard_findings()
    hard = ("\n".join(f"  • {f}" for f in findings[:30]) if findings else
            "  (автоматическая сверка жёстких расхождений не нашла — "
            "проверь текст содержательно на предмет более тонких искажений смысла)")
    gate_summary = (f"{decision.status} — " + (", ".join(decision.reason_codes()) or "причин нет")
                    + f"\n  declared_scope={decision.coverage.get('declared_scope')}")
    return hard, gate_summary


# ══════════════════════════════════════════════════════════════════
# ШАБЛОНЫ ПРОМПТОВ — по одному параметризуемому шаблону на блок A-E.
# Ни один не содержит фактов конкретного пациента — только структуру
# таксономии (заполняется из SUBCRITERIA_LABELS_RU) и заземляющие данные,
# собранные автоматикой для ИМЕННО ЭТОЙ пары источник/суммаризация.
# ══════════════════════════════════════════════════════════════════

_PROMPT_BLOCK_A = """{role_intro}

ПОДКРИТЕРИИ БЛОКА A «{block_title}»:
  A1. {A1}
  A2. {A2}
  A3. {A3}

ИСХОДНАЯ ЭМК (для сверки фактов):
{source_excerpt}

СУММАРИЗАЦИЯ (проверяемый текст):
{summary}

ОПОРНЫЕ НАХОДКИ АВТОМАТИЧЕСКОЙ ФАКТОЛОГИЧЕСКОЙ СВЕРКИ (Блок 2) — используй
как обязательную проверку: по каждой находке явно подтверди или опровергни
её в комментарии, не игнорируй:
{grounding}

Ответь СТРОГО в формате JSON:
{{"A1": {{"pass": true/false, "comment": "1-2 предложения, конкретная причина"}},
  "A2": {{"pass": true/false, "comment": "..."}},
  "A3": {{"pass": true/false, "comment": "..."}},
  "hallucinations": ["конкретная фраза/факт без основания в ЭМК", "..."],
  "wrong_values": ["было ... -> стало ... (с указанием, что именно перепутано)", "..."]}}

pass=false ставь, только если нашёл ХОТЯ БЫ ОДИН подтверждённый случай —
не занижай оценку за стилистические отличия формулировок."""

_PROMPT_BLOCK_B = """{role_intro}

ПОДКРИТЕРИИ БЛОКА B «{block_title}»:
  B1. {B1}
  B2. {B2}
  B3. {B3}
  B4. {B4}
  B5. {B5}

ВАЖНО: оценивай покрытие ОТНОСИТЕЛЬНО ЗАЯВЛЕННОЙ ЗАДАЧИ суммаризации (см.
выше в описании твоей роли). Отсутствие в суммаризации сведений, не
относящихся к этой задаче, — это КОРРЕКТНАЯ фильтрация, а не пропуск
(не штрафуй за неё ни B, ни E2).

СОДЕРЖИМОЕ СООТВЕТСТВУЮЩИХ РАЗДЕЛОВ ИСХОДНОЙ ЭМК (для построчной сверки):
{source_sections}

СУММАРИЗАЦИЯ (проверяемый текст):
{summary}

Ответь СТРОГО в формате JSON:
{{"B1": {{"pass": true/false, "comment": "...", "missing": ["конкретный упущенный пункт", ...]}},
  "B2": {{"pass": true/false, "comment": "...", "missing": [...]}},
  "B3": {{"pass": true/false, "comment": "...", "missing": [...]}},
  "B4": {{"pass": true/false, "comment": "...", "missing": [...]}},
  "B5": {{"pass": true/false, "comment": "...", "missing": [...]}}}}

pass=false ставь, если упущены сведения, ЗНАЧИМЫЕ ИМЕННО ДЛЯ ЗАЯВЛЕННОЙ ЗАДАЧИ —
не за пропуск содержимого, явно относящегося к другому профилю/контексту."""

_PROMPT_BLOCK_C = """{role_intro}

ПОДКРИТЕРИИ БЛОКА C «{block_title}»:
  C1. {C1}
  C2. {C2}

СУММАРИЗАЦИЯ (проверяемый текст):
{summary}

Ответь СТРОГО в формате JSON:
{{"C1": {{"pass": true/false, "comment": "...", "irrelevant_items": ["конкретный нерелевантный фрагмент", ...]}},
  "C2": {{"pass": true/false, "comment": "..."}}}}

C1: pass=false, если в тексте есть сведения, ЯВНО не относящиеся к заявленной
задаче (например, узкоспециализированные находки другого профиля или план
лечения/прогноз там, где нужна была фактология для конкретного исследования).
C2: pass=false, если значительная часть включённых данных клинически
малозначима для адресата — текст «многословен, но пуст»."""

_PROMPT_BLOCK_D = """{role_intro}

ПОДКРИТЕРИИ БЛОКА D «{block_title}»:
  D1. {D1}
  D2. {D2}
  D3. {D3}

РЕЗУЛЬТАТ АВТОМАТИЧЕСКОГО СТРУКТУРНОГО АНАЛИЗА (Блок 1) — опорные данные
для D1; используй их как основу, не переопределяй произвольно, но отрази
в комментарии:
{schema_report}

СУММАРИЗАЦИЯ (проверяемый текст):
{summary}

Ответь СТРОГО в формате JSON:
{{"D1": {{"pass": true/false, "comment": "..."}},
  "D2": {{"pass": true/false, "comment": "...", "duplicates": ["что именно повторяется и где", ...]}},
  "D3": {{"pass": true/false, "comment": "..."}}}}

D1: если автоматический анализ показал отсутствующие/переставленные разделы —
pass=false и сошлись на конкретный раздел.
D2: pass=false, если один и тот же факт дублируется в разных разделах текста.
D3: pass=false, если текст перегружен канцеляризмами/повторами и труден для
быстрого восприятия врачом в условиях дефицита времени."""

_PROMPT_BLOCK_E = """{role_intro}

ПОДКРИТЕРИИ БЛОКА E «{block_title}» — КРИТИЧЕСКИЙ БЛОК:
  E1. {E1}
  E2. {E2}
  E3. {E3}

E1 — СТОП-ПРАВИЛО: если ты подтверждаешь хотя бы один случай критической
галлюцинации (ложная аллергия, подмена диагноза, искажение противопоказаний
и т.п.), итоговая категория суммаризации становится «Неприемлемо»
НЕЗАВИСИМО от оценок по другим блокам. Будь строг, но не формален: ставь
pass=false по E1 ТОЛЬКО при подтверждённой клинической опасности, а не
любой неточности (та относится к A2/A1).

ИСХОДНАЯ ЭМК (для сверки):
{source_excerpt}

СУММАРИЗАЦИЯ (проверяемый текст):
{summary}

ЖЁСТКИЕ РАСХОЖДЕНИЯ, НАЙДЕННЫЕ АВТОМАТИЧЕСКОЙ СВЕРКОЙ (Блок 2) — каждое
ОБЯЗАТЕЛЬНО разбери: представляет ли оно клиническую опасность (-> E1),
или это просто пропуск (-> E2):
{hard_findings}

РЕШЕНИЕ PRE-EVALUATION GATE (Блок 3): {gate_summary}

Ответь СТРОГО в формате JSON:
{{"E1": {{"pass": true/false, "comment": "...", "danger_examples": ["точная формулировка из текста", ...]}},
  "E2": {{"pass": true/false, "comment": "..."}},
  "E3": {{"pass": true/false, "comment": "..."}}}}

E2: соотнеси с блоком B и решением шлюза — пропуск того, что НЕ относится
к заявленной задаче, не нарушение.
E3: pass=false, если факты не привязаны к датам/визитам или эта привязка
перепутана относительно исходной ЭМК."""

_PROMPT_TEMPLATES: dict[str, str] = {
    "A": _PROMPT_BLOCK_A, "B": _PROMPT_BLOCK_B, "C": _PROMPT_BLOCK_C,
    "D": _PROMPT_BLOCK_D, "E": _PROMPT_BLOCK_E,
}


# ══════════════════════════════════════════════════════════════════
# КОНТЕКСТ ПАРЫ — заземляющие данные считаются ОДИН РАЗ на пару
# (источник, суммаризация) и переиспользуются всеми блоками/раундами
# ══════════════════════════════════════════════════════════════════

@dataclass
class JudgeContext:
    source_text: str
    summary_text: str
    scope: Optional[str]
    source_struct: preprocessor.DocumentStructure
    summary_struct: preprocessor.DocumentStructure
    obj_report: objective_layer.ObjectiveComparisonReport
    gate_decision: Optional[gate.GateDecision]
    schema: dict
    grounded: bool = True   # False -> режим «только LLM-судьи», без Блоков 1-3


def build_context(source_text: str, summary_text: str, *,
                  scope: Optional[str] = None, panel=None,
                  entity_role: str = "judge_1",
                  gate_decision: Optional["gate.GateDecision"] = None,
                  grounded: bool = True) -> JudgeContext:
    """Считает один раз всё, что Блоки 1-3 могут сказать об этой паре —
    промпты блоков A-E ниже лишь форматируют эти готовые находки.

    `gate_decision` можно передать готовым (его уже в любом случае нужно
    посчитать в `evaluate_summary` ДО `obj_report`, чтобы шлюз реально
    экономил GPU-время на отклонённых парах, см. комментарий там) —
    тогда здесь он не пересчитывается повторно.

    `grounded=False` — режим «только LLM-судьи» (по решению НПКЦ): Блоки 1-3
    (препроцессор, объективный слой, шлюз) НЕ запускаются; судьи получают лишь
    сырой источник + суммаризацию + таксономию, а заземляющие поля промптов
    заполняются нейтральными пометками (см. `_prompt_for_block`). Это изолирует
    «чистую» способность LLM-контура судить, без подсказок автоматики."""
    if not grounded:
        empty_struct = preprocessor.DocumentStructure(raw_text=source_text)
        return JudgeContext(
            source_text=source_text, summary_text=summary_text, scope=scope,
            source_struct=empty_struct,
            summary_struct=preprocessor.DocumentStructure(raw_text=summary_text),
            obj_report=objective_layer.empty_report(), gate_decision=None,
            schema={}, grounded=False,
        )
    source_struct = preprocessor.segment_emr(source_text)
    summary_struct = preprocessor.segment_summary(summary_text)
    obj_report = objective_layer.compare_texts(source_text, summary_text,
                                                panel=panel, entity_role=entity_role)
    if gate_decision is None:
        gate_decision = gate.evaluate_gate(source_text, summary_text, scope=scope,
                                            panel=panel, entity_role=entity_role)
    schema = preprocessor.schema_compliance(summary_struct)
    return JudgeContext(
        source_text=source_text, summary_text=summary_text, scope=scope,
        source_struct=source_struct, summary_struct=summary_struct,
        obj_report=obj_report, gate_decision=gate_decision, schema=schema,
        grounded=True,
    )


# ── Нейтральные заземления для режима «только LLM-судьи» (grounded=False) ──
# Блоки 1-3 не запускались, поэтому вместо конкретных находок автоматики судья
# получает честную пометку «сверка не проводилась — проверь сам» и СЫРОЙ источник.
_UNGROUNDED_A = ("Автоматическая фактологическая сверка НЕ проводилась (режим оценки "
                 "только силами LLM-судей). Сверь факты суммаризации с исходной ЭМК "
                 "самостоятельно — числа, дозы, диагнозы, отрицания.")
_UNGROUNDED_D = ("Автоматический структурный анализ НЕ проводился (режим только LLM). "
                 "Оцени наличие и порядок пяти разделов схемы по самому тексту "
                 "суммаризации.")
_UNGROUNDED_E_HARD = ("Автоматическая сверка жёстких расхождений НЕ проводилась (режим "
                      "только LLM) — выяви клинически опасные ошибки самостоятельно, "
                      "сверяясь с исходной ЭМК.")
_UNGROUNDED_E_GATE = "не применялся (режим оценки только силами LLM-судей)"


def _ungrounded_source_sections(source_text: str) -> str:
    return ("Автоматическая сегментация НЕ проводилась (режим только LLM). Ниже — "
            "ПОЛНЫЙ исходный текст ЭМК; оцени покрытие пяти разделов по нему "
            "самостоятельно:\n\n" + _truncate(source_text, 7000))


def _prompt_for_block(block: str, ctx: JudgeContext) -> str:
    common = dict(role_intro=_role_intro(ctx.scope), block_title=BLOCK_TITLES_RU[block],
                  summary=ctx.summary_text,
                  **{c: SUBCRITERIA_LABELS_RU[c] for c in TAXONOMY[block]})
    if block == "A":
        grounding = _ground_block_a(ctx.obj_report) if ctx.grounded else _UNGROUNDED_A
        return _PROMPT_TEMPLATES["A"].format(
            **common, source_excerpt=_truncate(ctx.source_text, 7000),
            grounding=grounding)
    if block == "B":
        sections = (_ground_block_b(ctx.source_struct) if ctx.grounded
                    else _ungrounded_source_sections(ctx.source_text))
        return _PROMPT_TEMPLATES["B"].format(**common, source_sections=sections)
    if block == "C":
        return _PROMPT_TEMPLATES["C"].format(**common)
    if block == "D":
        schema_report = _ground_block_d(ctx.schema) if ctx.grounded else _UNGROUNDED_D
        return _PROMPT_TEMPLATES["D"].format(**common, schema_report=schema_report)
    if block == "E":
        if ctx.grounded:
            hard, gate_summary = _ground_block_e(ctx.obj_report, ctx.gate_decision)
        else:
            hard, gate_summary = _UNGROUNDED_E_HARD, _UNGROUNDED_E_GATE
        return _PROMPT_TEMPLATES["E"].format(
            **common, source_excerpt=_truncate(ctx.source_text, 7000),
            hard_findings=hard, gate_summary=gate_summary)
    raise ValueError(f"неизвестный блок таксономии: {block!r}")


# ══════════════════════════════════════════════════════════════════
# ВАЛИДАЦИЯ И FALLBACK-ПРОМПТЫ — сохранённый из evaluator.py принцип
# «retry с упрощённым промптом» (там — _make_minimal_prompt), здесь
# обобщён под произвольный набор бинарных подкритериев
# ══════════════════════════════════════════════════════════════════

def _block_skeleton(block: str, *, with_lists: bool = True) -> str:
    extra_lists = {
        "A": '"hallucinations": [], "wrong_values": []',
        "B": None,  # списки уже внутри каждого подкритерия (missing)
        "C": None,
        "D": None,
        "E": None,
    }
    parts = []
    for c in TAXONOMY[block]:
        if block == "B":
            parts.append(f'"{c}": {{"pass": true, "comment": "", "missing": []}}')
        elif c == "C1":
            parts.append(f'"{c}": {{"pass": true, "comment": "", "irrelevant_items": []}}')
        elif c == "D2":
            parts.append(f'"{c}": {{"pass": true, "comment": "", "duplicates": []}}')
        elif c == "E1":
            parts.append(f'"{c}": {{"pass": true, "comment": "", "danger_examples": []}}')
        else:
            parts.append(f'"{c}": {{"pass": true, "comment": ""}}')
    body = ", ".join(parts)
    if with_lists and extra_lists.get(block):
        body += ", " + extra_lists[block]
    return "{" + body + "}"


def _block_token_budget(block: str) -> int:
    """Бюджет ответа под конкретный блок.

    Блоки B и E объёмнее остальных: B требует 5 подкритериев, каждый со вложенным
    списком `missing`, E — 3 подкритерия плюс `danger_examples`. При едином
    бюджете 1536 именно блок B стабильно обрывался на B4/B5 (все 11 провалов
    валидации R1 в логе пилота — ровно этот случай)."""
    return config.TOKENS_R1_LARGE if block in ("B", "E") else config.TOKENS_R1


def _validate_full_report(parsed: dict, raw: str) -> None:
    """Проверка полного отчёта A-E (раунд 2).

    Раньше R2 вызывался с validate_fn=None: любой распарсившийся ответ, включая
    собранный салважем огрызок, принимался молча. Обрыв не отличался от честного
    «после ревью согласен с собой». Теперь неполный ответ уходит в повтор (и, по
    новой логике ask_json, — с увеличенным бюджетом); если все попытки исчерпаны,
    score_round2 по-прежнему возвращает отчёт R1 без изменений.
    """
    missing: list[str] = []
    for block, codes in TAXONOMY.items():
        blk = parsed.get(block)
        if not isinstance(blk, dict):
            missing.append(block)
            continue
        for code in codes:
            sub = blk.get(code)
            if not isinstance(sub, dict) or not isinstance(sub.get("pass"), bool):
                missing.append(f"{block}.{code}")
    if missing:
        raise ValueError("неполный отчёт R2, не хватает: " + ", ".join(missing[:8]))


def _validate_block(block: str):
    expected = TAXONOMY[block]

    def _validate(parsed: dict, raw: str) -> None:
        for key in expected:
            sub = parsed.get(key)
            if not isinstance(sub, dict) or not isinstance(sub.get("pass"), bool):
                raise ValueError(f"блок {block}: подкритерий {key} — ожидался "
                                 f"объект с булевым полем pass, получено {sub!r}")
    return _validate


def _fallback_block_prompt(block: str, summary_text: str):
    """Упрощённый промпт для поздних попыток.

    ВАЖНО про предвзятость: скелет предзаполнен `"pass": true`, а всё заземление
    (исходник ЭМК, находки объективного слоя, решение шлюза) из промпта убрано —
    судья, ответивший здесь, НИЧЕГО не сверял с источником. Факт использования
    fallback считается в телеметрии (llm_client.CallStats.fallback_used) и
    выводится в отчёт, чтобы такие голоса были видны, а не растворялись среди
    полноценных.

    Попытки различаются по содержанию: раньше 3-я и 4-я были побайтово
    одинаковыми запросами при temperature≈0 — то есть одна из четырёх попыток
    тратилась заведомо впустую.
    """
    skeleton = _block_skeleton(block)

    def _fallback(attempt: int, last_raw: str) -> str:
        base = (f"Заполни JSON-структуру оценки блока {block} таксономии качества "
                f"медицинских суммаризаций. Для каждого подкритерия укажи pass "
                f"(true — критерий выполнен, false — нарушен) и краткий comment.\n\n"
                f"Суммаризация:\n{_truncate(summary_text, 2000)}\n\n")
        if attempt >= 4:
            # Последняя попытка: максимально короткий ответ, без комментариев —
            # цель уже не качество разбора, а хоть какой-то валидный вердикт.
            base += ("Отвечай МАКСИМАЛЬНО КРАТКО: comment — не более 5 слов.\n"
                     "Предыдущие попытки вернули неразбираемый ответ.\n\n")
        return base + f"Верни ТОЛЬКО JSON по структуре:\n{skeleton}"
    return _fallback


PARSE_ERROR_KEY = "_parse_error"


def _sentinel_block(block: str, reason: str) -> dict:
    """Блок-заглушка при НЕустранимом сбое JSON у ОДНОГО блока.

    Ключевое: подкритерии помечены `pass=None` (НЕ False) — это «нет данных»,
    а не «нарушено». Так один битый блок НЕ обнуляет оценки судьи: остальные
    четыре блока сохраняют реальные значения, а этот честно помечен как сбой
    (`_parse_error`) — агрегатор и отчёт показывают его отдельно («н/д»), а не
    как фальшивый «0/N». Это и есть устранение бага «неверный JSON → оценки
    обнулялись»."""
    out: dict = {PARSE_ERROR_KEY: True, "_parse_error_reason": reason[:200]}
    for c in TAXONOMY[block]:
        out[c] = {"pass": None, "comment": f"нет данных: сбой JSON ({reason[:80]})"}
    return out


def score_block(panel: JudgePanel, role: str, block: str, ctx: JudgeContext) -> dict:
    """Один запрос — оценка суммаризации по подкритериям ОДНОГО блока.

    НИКОГДА не пробрасывает LLMError наверх: при неустранимом сбое JSON
    (исчерпаны все попытки, не спас даже салваж в llm_client.extract_json)
    возвращает `_sentinel_block` — «нет данных» по этому блоку, не трогая
    остальные. Раньше исключение прерывало score_round1 и терялся ВЕСЬ отчёт
    судьи (все 5 блоков) — это и был баг «оценки обнулялись».

    think=False — критично для надёжности JSON у «мыслящих» моделей (qwen3 и
    т.п.): со включённым по умолчанию <think> скрытые рассуждения съедают бюджет
    num_predict ДО структурированного ответа, и JSON обрывается. num_predict
    поднят (1536) и max_attempts=4 — чтобы до sentinel доходило как можно реже."""
    prompt = _prompt_for_block(block, ctx)
    try:
        return panel.ask_json(
            role, prompt, desc=f"R1 блок {block} ({BLOCK_TITLES_RU[block]})",
            max_attempts=4, validate_fn=_validate_block(block), think=False,
            num_predict=_block_token_budget(block),
            fallback_prompt_fn=_fallback_block_prompt(block, ctx.summary_text),
        )
    except LLMError as e:
        log.error("    %s блок %s: JSON не получен после всех попыток (%s) — блок "
                  "помечен «нет данных», остальные блоки судьи сохраняются", role, block, e)
        return _sentinel_block(block, str(e))


# ══════════════════════════════════════════════════════════════════
# РАУНД 1 — независимая оценка по всем блокам A-E одним судьёй
# ══════════════════════════════════════════════════════════════════

def score_round1(panel: JudgePanel, role: str, ctx: JudgeContext) -> dict[str, dict]:
    """Возвращает {"A": {...}, "B": {...}, ..., "E": {...}} — полный
    структурированный отчёт судьи по всем пяти блокам таксономии.

    Пер-блочная изоляция: сбой ОДНОГО блока (даже непредвиденное исключение)
    НЕ рушит весь отчёт — проваленный блок заменяется на `_sentinel_block`,
    остальные сохраняют реальные оценки. score_round1 НИКОГДА не поднимает
    исключение (гарантия против «обнуления» всего судьи из-за одного блока)."""
    report: dict[str, dict] = {}
    for block in ("A", "B", "C", "D", "E"):
        try:
            report[block] = score_block(panel, role, block, ctx)
        except Exception as e:  # noqa: BLE001 — один блок не должен рушить судью
            log.error("    %s блок %s: непредвиденная ошибка (%s) — sentinel",
                      role, block, e)
            report[block] = _sentinel_block(block, f"{type(e).__name__}: {e}")
    return report


# ══════════════════════════════════════════════════════════════════
# РАУНД 2 — cross-peer-review С ПОЛНЫМИ ОТЧЁТАМИ
#
# В старом коде compact() вырезал из обмена между судьями именно то, что
# содержательно важнее всего: списки галлюцинаций/расхождений/пропусков —
# судьи спорили о голых числах, не видя АРГУМЕНТОВ друг друга. Здесь
# раунд 2 передаёт отчёты ЦЕЛИКОМ — судья видит и подкритерии, и списки
# конкретных находок коллег, и может содержательно с ними согласиться
# или возразить (а не просто «усреднить число»).
# ══════════════════════════════════════════════════════════════════

def _format_full_report(report: dict[str, dict]) -> str:
    """Полная (НЕ обрезанная) сериализация отчёта — см. комментарий выше."""
    return json.dumps(report, ensure_ascii=False)


_FULL_REPORT_SKELETON = "{" + ", ".join(
    f'"{block}": {_block_skeleton(block)}' for block in ("A", "B", "C", "D", "E")
) + "}"

_PROMPT_CROSS_REVIEW = """{role_intro}

В первом раунде ты независимо оценил эту суммаризацию по блокам A-E
(твой отчёт — ниже). Теперь изучи ПОЛНЫЕ отчёты двух коллег-рецензентов —
включая конкретные примеры галлюцинаций, расхождений значений и пропусков,
которые они нашли, — и сформируй уточнённый отчёт.

СУММАРИЗАЦИЯ (напоминание):
{summary}

ТВОЙ ОТЧЁТ (раунд 1, JSON):
{my_report}

ОТЧЁТ КОЛЛЕГИ 1 (JSON):
{peer_1}

ОТЧЁТ КОЛЛЕГИ 2 (JSON):
{peer_2}

Правила пересмотра:
  • Если коллега указал на конкретную фразу/факт, которые ты не заметил —
    проверь именно это место в суммаризации; при подтверждении смени
    pass на false и добавь находку в соответствующий список (не просто
    скопируй у коллеги — переформулируй своими словами после проверки).
  • Особое внимание — подкритерию E1 (стоп-правило): если коллега поднял
    его, перепроверь обязательно. Цена ошибки высока в обе стороны —
    и пропустить реальную опасность, и необоснованно забраковать текст.
  • Если не согласен с коллегой — оставь свою оценку, но явно обоснуй
    несогласие в comment (укажи, почему его находка не подтверждается).

Верни ТОЛЬКО JSON — ПОЛНЫЙ уточнённый отчёт в ТОЙ ЖЕ структуре (блоки A-E,
все подкритерии, все списки находок):
{schema_skeleton}"""


def _merge_full_report(revised: dict, baseline: dict) -> dict:
    """
    Накладывает ревизию раунда 2 на ВАЛИДНЫЙ отчёт раунда 1. Для каждого
    подкритерия берём значение из ревизии, ТОЛЬКО если оно корректно (объект с
    булевым `pass`); иначе сохраняем значение из R1. Доп. списки (hallucinations,
    wrong_values, missing, …) переносим из ревизии, если они есть.

    Зачем мердж вместо «всё-или-ничего» валидации: R2 просит модель переиздать
    ВЕСЬ вложенный отчёт A-E — большой JSON, который слабые модели регулярно
    обрывают (один пропущенный `pass` раньше отвергал весь ответ → 3 провальные
    попытки → откат к R1). Здесь обрыв/пропуск поля больше не фатален: гаранти-
    рованно получаем валидный полный отчёт (худший случай — равен R1), а реально
    пересмотренные подкритерии аккуратно вносятся.
    """
    merged: dict[str, dict] = {}
    for block, subcriteria in TAXONOMY.items():
        base_block = baseline.get(block) if isinstance(baseline.get(block), dict) else {}
        rev_block = revised.get(block) if isinstance(revised.get(block), dict) else {}
        out_block = dict(base_block)
        for c in subcriteria:
            rc = rev_block.get(c)
            if isinstance(rc, dict) and isinstance(rc.get("pass"), bool):
                out_block[c] = rc
        for key, val in rev_block.items():
            if key not in subcriteria and val is not None:
                out_block[key] = val
        merged[block] = out_block
    return merged


def score_round2(panel: JudgePanel, role: str, ctx: JudgeContext,
                 my_report: dict, peer_reports: list[dict]) -> dict:
    """Раунд 2 (cross-peer-review). НИКОГДА не падает: ответ модели накладывается
    мерджем на валидный отчёт R1 (см. `_merge_full_report`), а полностью
    непарсимый ответ -> отчёт R1 без изменений (легитимная ревизия «после ревью
    согласен с собой»). think=False + увеличенный num_predict устраняют обрыв
    большого JSON у «мыслящих» моделей (бюджет идёт на отчёт, не на скрытый
    <think>)."""
    prompt = _PROMPT_CROSS_REVIEW.format(
        role_intro=_role_intro(ctx.scope),
        summary=ctx.summary_text,
        my_report=_format_full_report(my_report),
        peer_1=_format_full_report(peer_reports[0]),
        peer_2=_format_full_report(peer_reports[1]),
        schema_skeleton=_FULL_REPORT_SKELETON,
    )
    try:
        revised = panel.ask_json(
            role, prompt, desc="R2 пересмотр (полные отчёты коллег)",
            max_attempts=3, validate_fn=_validate_full_report, think=False,
            num_predict=config.TOKENS_R2,
        )
    except LLMError as e:
        log.warning("      %s R2: модель не вернула парсимый JSON (%.80s) — "
                    "оставляю отчёт R1 без изменений", role, str(e))
        return my_report
    return _merge_full_report(revised, my_report)


# ══════════════════════════════════════════════════════════════════
# РАУНД 3 — финальная агрегация с ЯВНОЙ, НЕ-ОПЦИОНАЛЬНОЙ проверкой E1
#
# В старом коде safety_flag собирался, но НЕ переопределял quality
# (баг прежней монолитной версии, п.2 плана) — суммаризация с опасной
# галлюцинацией могла получить «отличное». Здесь агрегатору explicитно,
# пошагово предписано: сначала проверь E1 по всем шести отчётам, и
# ТОЛЬКО ЕСЛИ он не сработал — выводи категорию по совокупности баллов.
#
# Промпт делает эту проверку «не-опциональной» внутри LLM-рассуждения;
# полностью НЕ зависящая от LLM, детерминированная гарантия —
# задача Блока 5 (см. _collect_e1_signals ниже — мы готовим для неё
# прозрачные, проверяемые сигналы из всех источников, но не подменяем
# её код собственным переопределением категории).
# ══════════════════════════════════════════════════════════════════

_PROMPT_AGGREGATE = """Ты — ведущий эксперт-арбитр. На основе ШЕСТИ отчётов (по два от каждого
из трёх независимых рецензентов: первичный отчёт раунда 1 и уточнённый
отчёт раунда 2 после cross-peer-review) вынеси ИТОГОВЫЙ вердикт по
суммаризации электронной медицинской карты.

СУММАРИЗАЦИЯ (напоминание):
{summary}

ЗАЯВЛЕННАЯ ЗАДАЧА СУММАРИЗАЦИИ:
{scope_description}

ПЕРВИЧНЫЕ ОТЧЁТЫ (раунд 1, JSON):
  Рецензент 1: {r1_a}
  Рецензент 2: {r1_b}
  Рецензент 3: {r1_c}

УТОЧНЁННЫЕ ОТЧЁТЫ (раунд 2, после cross-peer-review, JSON):
  Рецензент 1: {r2_a}
  Рецензент 2: {r2_b}
  Рецензент 3: {r2_c}

═══ ШАГ 1 (ОБЯЗАТЕЛЬНЫЙ, ВЫПОЛНИ ПЕРВЫМ) — ПРОВЕРКА СТОП-ПРАВИЛА E1 ═══
Подкритерий E1 («{label_E1}») — это стоп-правило. Просмотри ВСЕ ШЕСТЬ
отчётов и определи: поднял ли E1.pass=false ХОТЯ БЫ ОДИН из них?
  • Если ДА — итоговая категория ОБЯЗАНА быть «Неприемлемо». Это решение
    НЕ балансируется и НЕ переопределяется никакими другими баллами или
    оценками блоков A-D — даже если по ним всё хорошо. Укажи в
    "e1_triggered_by", какие рецензенты подняли флаг, и приведи в
    "verdict" точный опасный факт (danger_examples) как причину отказа.
  • Если НЕТ — переходи к шагу 2 и определяй категорию по совокупности
    оценок блоков A-D и решению pre-evaluation gate.
═══════════════════════════════════════════════════════════════════════

ШАГ 2 — категория (только если E1 НЕ сработал):
  «Готово к клиническому применению» — gate пройден, E1 не сработал,
    подавляющее большинство подкритериев блоков A-D имеют pass=true.
  «Требует редактирования» — gate пройден, E1 не сработал, но есть
    значимые замечания (несколько подкритериев с pass=false) по блокам B-D.
  «Неприемлемо» — gate не пройден, ИЛИ сработал E1, ИЛИ большинство
    подкритериев блока A имеют pass=false (систематические галлюцинации).

Ответь СТРОГО в формате JSON:
{{"e1_triggered": true/false,
  "e1_triggered_by": ["рецензент_N", ...],
  "category": "Готово к клиническому применению/Требует редактирования/Неприемлемо",
  "verdict": "развёрнутое обоснование (3-5 предложений) со ссылкой на конкретные находки",
  "summary_by_block": {{"A": "1 предложение", "B": "...", "C": "...", "D": "...", "E": "..."}}}}"""

_AGGREGATE_CATEGORIES = (
    "Готово к клиническому применению", "Требует редактирования", "Неприемлемо",
)


def _validate_aggregate(parsed: dict, raw: str) -> None:
    if parsed.get("category") not in _AGGREGATE_CATEGORIES:
        raise ValueError(f"агрегатор: category={parsed.get('category')!r} "
                         f"не входит в {_AGGREGATE_CATEGORIES}")
    if not isinstance(parsed.get("e1_triggered"), bool):
        raise ValueError("агрегатор: e1_triggered должен быть явным булевым полем "
                         "(не-опциональная проверка E1, см. PROMPT_AGGREGATE)")


def _fallback_aggregate_prompt(summary_text: str):
    skeleton = ('{"e1_triggered": false, "e1_triggered_by": [], '
                '"category": "Требует редактирования", "verdict": "", '
                '"summary_by_block": {"A":"","B":"","C":"","D":"","E":""}}')

    def _fallback(attempt: int, last_raw: str) -> str:
        return (f"Заполни JSON итогового вердикта по суммаризации. category — одна "
                f"из трёх строк: «Готово к клиническому применению», «Требует "
                f"редактирования», «Неприемлемо». e1_triggered — true, если хотя бы "
                f"один рецензент нашёл клинически опасную ошибку (стоп-правило E1).\n\n"
                f"Суммаризация:\n{_truncate(summary_text, 1500)}\n\n"
                f"Верни ТОЛЬКО JSON по структуре:\n{skeleton}")
    return _fallback


def score_round3(panel: JudgePanel, role: str, ctx: JudgeContext,
                 r1_reports: list[dict], r2_reports: list[dict]) -> dict:
    prompt = _PROMPT_AGGREGATE.format(
        summary=ctx.summary_text,
        scope_description=_scope_description(ctx.scope),
        r1_a=_format_full_report(r1_reports[0]),
        r1_b=_format_full_report(r1_reports[1]),
        r1_c=_format_full_report(r1_reports[2]),
        r2_a=_format_full_report(r2_reports[0]),
        r2_b=_format_full_report(r2_reports[1]),
        r2_c=_format_full_report(r2_reports[2]),
        label_E1=SUBCRITERIA_LABELS_RU["E1"],
    )
    return panel.ask_json(
        role, prompt, desc="R3 финальная агрегация (явная проверка E1)",
        max_attempts=3, validate_fn=_validate_aggregate, think=False,
        num_predict=config.TOKENS_R3,
        fallback_prompt_fn=_fallback_aggregate_prompt(ctx.summary_text),
    )


def _citation_supported(citation: str, summary_text: str) -> bool:
    """Действительно ли приведённый судьёй «опасный фрагмент» есть в суммаризации.

    Проверяет КОД, а не LLM: сопоставление по стемам значимых токенов
    (objective_layer._stem_overlap) — судья цитирует в именительном падеже, в
    тексте падеж косвенный. Короткая цитата обязана совпасть целиком; для
    длинной достаточно 80 % значимых токенов — иначе любая перестановка слов
    рушила бы законное срабатывание предохранителя.
    """
    if not citation or not summary_text:
        return False
    cit = objective_layer._normalize_entity(str(citation))
    txt = objective_layer._normalize_entity(summary_text)
    text_tokens = re.findall(r"[а-яёa-z0-9]+", txt)
    if not text_tokens:
        return False
    toks = [t for t in re.findall(r"[а-яёa-z0-9]+", cit) if len(t) >= 3]
    if not toks:
        return False
    hits = sum(1 for t in toks
               if any(objective_layer._stem_overlap(t, tt) for tt in text_tokens))
    need = len(toks) if len(toks) <= 4 else int(len(toks) * 0.8 + 0.999)
    return hits >= need


def _e1_citations(report: dict) -> list[str]:
    """Фрагменты, которые судья назвал клинически опасными (поле danger_examples)."""
    e1 = ((report or {}).get("E") or {}).get("E1")
    if not isinstance(e1, dict):
        return []
    raw = e1.get("danger_examples")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def _collect_e1_signals(r1: dict[str, dict], r2: dict[str, dict], r3: dict,
                        summary_text: str = "") -> dict:
    """
    Собирает сигналы стоп-правила E1 из всех источников и решает, какие из них
    ЗАСЧИТЫВАЮТСЯ.

    ПОЧЕМУ появилась проверка цитатой. В v1 правило было чистой дизъюнкцией:
    флаг любого из четырёх источников (3 судьи после merge R1/R2 + агрегатор R3)
    необратимо фиксировал «Неприемлемо», причём `aggregator_flagged` — это
    невалидированный булев флаг ОДНОГО LLM-вызова. Результат на пилоте: E1
    срабатывал в 38 случаях из 38, включая эталонную суммаризацию и benign-
    искажения уровня опечатки и пунктуации. Предохранитель, срабатывающий
    всегда, не несёт информации.

    Теперь голос судьи засчитывается, только если он привёл КОНКРЕТНЫЙ фрагмент
    суммаризации как опасный И этот фрагмент в ней действительно есть (проверяет
    код). Флаг агрегатора сам по себе больше не триггер: он усиливает голос
    судьи, но в одиночку лишь помечает расхождение (disputed_by_aggregator),
    которое видно в отчёте.

    Отключается через config.E1_REQUIRE_CITATION=0 — тогда поведение прежнее
    (нужно для A/B-сравнения на синтетике).
    """
    final = {**r1, **r2}          # финальная позиция каждого судьи (R2 важнее R1)

    raised_all: list[str] = []
    confirmed: list[str] = []
    citations: dict[str, list[str]] = {}
    rejected_citations: dict[str, list[str]] = {}

    for role, rep in final.items():
        if not isinstance(rep, dict):
            continue
        if ((rep.get("E") or {}).get("E1") or {}).get("pass") is not False:
            continue
        raised_all.append(role)
        quotes = _e1_citations(rep)
        good = [q for q in quotes if _citation_supported(q, summary_text)]
        bad = [q for q in quotes if q not in good]
        if good:
            confirmed.append(role)
            citations[role] = good
        if bad:
            rejected_citations[role] = bad

    raised_all.sort()
    confirmed.sort()
    aggregator_raw = bool(r3.get("e1_triggered", False))

    if config.E1_REQUIRE_CITATION:
        effective_judges = confirmed
        # Флаг агрегатора засчитывается только вместе с голосом хотя бы одного
        # судьи: два независимых источника, а не один невалидированный булев.
        effective_aggregator = aggregator_raw and bool(raised_all)
        disputed = aggregator_raw and not raised_all
    else:
        effective_judges = raised_all
        effective_aggregator = aggregator_raw
        disputed = False

    return {
        "raised_by_judges": effective_judges,
        "aggregator_flagged": effective_aggregator,
        "aggregator_named": list(r3.get("e1_triggered_by", [])),
        "aggregator_category": r3.get("category"),
        "consistent": bool(effective_judges) == aggregator_raw,
        # ── диагностика для отчёта ────────────────────────────────
        "require_citation": bool(config.E1_REQUIRE_CITATION),
        "raised_by_judges_raw": raised_all,
        "raised_without_citation": sorted(set(raised_all) - set(confirmed)),
        "citations": citations,
        "unverifiable_citations": rejected_citations,
        "aggregator_raw_flag": aggregator_raw,
        "disputed_by_aggregator": disputed,
    }


# ══════════════════════════════════════════════════════════════════
# ОРКЕСТРАЦИЯ ПОЛНОГО ЦИКЛА: R1 -> R2 (cross-review) -> R3 (агрегация)
# ══════════════════════════════════════════════════════════════════

def _err(emr_id: str, model_id: str, reason: str, *, ctx: Optional["JudgeContext"] = None) -> dict:
    # Если контекст уже построен (например, отвалились все судьи R1, но
    # шлюз/объективный слой уже отработали) — объективное резюме у нас ЕСТЬ
    # и его не нужно терять: для аудита полезно видеть «что объективный слой
    # увидел», даже если LLM-оценка не состоялась.
    objective = ctx.obj_report.to_summary() if ctx is not None else None
    gate_info = ({"status": ctx.gate_decision.status, "reasons": ctx.gate_decision.reason_codes()}
                 if ctx is not None else {})
    return {
        "emr_id": emr_id, "model_id": model_id, "category": "ошибка",
        "verdict": reason, "e1_signals": {}, "gate": gate_info, "objective": objective,
        "r1": {}, "r2": {}, "r3": {},
    }


def evaluate_summary(source_text: str, summary_text: str, emr_id: str, model_id: str, *,
                      scope: Optional[str] = None, panel: Optional[JudgePanel] = None,
                      profile: Optional[str] = None) -> dict:
    """
    Полный трёхраундовый цикл LLM-as-Judge для одной пары (ЭМК, суммаризация):
      R1 — три судьи независимо оценивают по блокам A-E (отдельный промпт
           на блок — заземлённый структурой из Блоков 1-3, см. JudgeContext)
      R2 — каждый судья пересматривает оценку, видя ПОЛНЫЕ отчёты двух
           других (включая списки конкретных находок — не только баллы)
      R3 — агрегатор синтезирует вердикт; промпт ОБЯЗЫВАЕТ его явно,
           первым шагом проверить стоп-правило E1 по всем шести отчётам
    """
    panel = panel or JudgePanel(profile)

    # Шлюз считаем ПЕРВЫМ, отдельно от остального контекста: его вердикт и
    # покрытие нужны как заземление для судей и для отчёта.
    #
    # ШЛЮЗ БОЛЬШЕ НЕ ОТСЕКАЕТ ПАРЫ (решение НПКЦ). Прежде status=="reject"
    # означал немедленное «Неприемлемо» с пустыми r1/r2/r3 — то есть худшую
    # клиническую категорию БЕЗ единого свидетельства от судей. Это было
    # опасно, потому что ложные отказы шлюза массовые и измеримые: на
    # синтетическом наборе rule-based часть давала 77 reject из 380 пар (20 %),
    # а у пациентов М3 и Ж4 — 38/38, включая ЭТАЛОННУЮ строку, по правилу
    # «упомянуто хотя бы раз» без допуска на сжатие 3.4:1. В прод-корпусе к
    # этому добавлялось то, что сущности источника извлекались лишь из первых
    # 6000 символов карты (медиана исходника — 21107 символов).
    #
    # Теперь reject лишь опускает потолок категории в aggregator._decide, а
    # причина отказа целиком уходит в отчёт. Единственное исключение —
    # вырожденный вход (пустой текст, не тот язык): там оценивать нечего, см.
    # gate._degenerate_input_reasons.
    gate_decision = gate.evaluate_gate(source_text, summary_text, scope=scope,
                                        panel=panel, entity_role="judge_1")

    log.info("═" * 60)
    log.info("  ЭМК: %s | Модель: %s | scope=%s | gate=%s",
             emr_id, model_id, scope, gate_decision.status)
    log.info("═" * 60)
    if gate_decision.status != "pass":
        log.info("  Шлюз: %s (%s) — оценка ПРОДОЛЖАЕТСЯ, вердикт шлюза учтён "
                 "как сигнал в Блоке 5", gate_decision.status,
                 ", ".join(gate_decision.reason_codes()) or "без причин")

    if gate_decision.is_degenerate():
        # Пустая/иноязычная суммаризация: судить нечего, и LLM-вызовы здесь —
        # чистая трата GPU. Это НЕ «плохая суммаризация», а негодный вход.
        log.warning("  Шлюз: вырожденный вход (%s) — судьи не запускаются",
                    ", ".join(gate_decision.reason_codes()))
        return {
            "emr_id": emr_id, "model_id": model_id, "scope": scope,
            "category": "Неприемлемо",
            "verdict": "Вырожденный вход: "
                       f"{'; '.join(r.message for r in gate_decision.reasons if r.degenerate)}",
            "e1_signals": {"raised_by_judges": [], "aggregator_flagged": False,
                           "aggregator_named": [], "aggregator_category": None,
                           "consistent": True},
            "gate": {"status": gate_decision.status,
                     "reasons": gate_decision.reason_codes(),
                     "messages": [r.message for r in gate_decision.reasons],
                     "degenerate": True,
                     "coverage": gate_decision.coverage},
            "objective": None,
            "r1": {}, "r2": {}, "r3": {},
        }

    ctx = build_context(source_text, summary_text, scope=scope, panel=panel,
                        gate_decision=gate_decision)

    judge_roles = list(config.JUDGE_ROLES)

    # ── РАУНД 1 ───────────────────────────────────────────────────
    log.info("  [R1] Независимая оценка по блокам A-E...")
    r1: dict[str, Optional[dict]] = {}
    for role in judge_roles:
        try:
            r1[role] = score_round1(panel, role, ctx)
            log.info("    %s: получен полный отчёт (блоки %s)", role, list(r1[role].keys()))
        except LLMError as e:
            log.error("    %s провал R1: %s", role, e)
            r1[role] = None

    valid_roles = [r for r in judge_roles if r1.get(r) is not None]
    if not valid_roles:
        log.error("  R1: все судьи провалились")
        return _err(emr_id, model_id, "R1: ни один судья не вернул отчёт", ctx=ctx)

    # ── РАУНД 2 — cross-peer-review с полными отчётами ───────────
    log.info("  [R2] Cross-peer-review (полные отчёты)...")
    shuffled = valid_roles.copy()
    random.shuffle(shuffled)   # против position bias — сохранено из evaluator.py

    r2: dict[str, dict] = {}
    for role in shuffled:
        peers = [r for r in shuffled if r != role]
        while len(peers) < 2:
            peers.append(peers[-1] if peers else role)
        try:
            r2[role] = score_round2(panel, role, ctx, r1[role],
                                     [r1[peers[0]], r1[peers[1]]])
            log.info("    %s: уточнённый отчёт получен", role)
        except LLMError as e:
            log.warning("    %s провал R2 (%s) — оставляю отчёт R1", role, e)
            r2[role] = r1[role]

    # ── РАУНД 3 — финальная агрегация ─────────────────────────────
    log.info("  [R3] Финальная агрегация (явная проверка E1)...")
    r1_list = [r1[r] for r in judge_roles if r1.get(r) is not None]
    r2_list = [r2.get(r, r1[r]) for r in judge_roles if r1.get(r) is not None]
    while len(r1_list) < 3:
        r1_list.append(r1_list[-1]); r2_list.append(r2_list[-1])

    try:
        r3 = score_round3(panel, config.AGGREGATOR_ROLE, ctx, r1_list, r2_list)
    except LLMError as e:
        log.error("    агрегатор провалился (%s) — итог недоступен", e)
        r3 = {"category": "ошибка", "e1_triggered": False, "e1_triggered_by": [],
              "verdict": f"Агрегатор недоступен: {e}", "summary_by_block": {}}

    e1_signals = _collect_e1_signals(r1, r2, r3, ctx.summary_text)
    if not e1_signals["consistent"]:
        log.warning("    НЕСОГЛАСОВАННОСТЬ E1: рецензенты подняли флаг %s, "
                    "агрегатор aggregator_flagged=%s — требует ручной проверки "
                    "(детерминированная гарантия — Блок 5)",
                    e1_signals["raised_by_judges"], e1_signals["aggregator_flagged"])

    log.info("  ИТОГ %s/%s: %s%s", emr_id, model_id, r3.get("category", "—"),
             "  [E1!]" if e1_signals["aggregator_flagged"] or e1_signals["raised_by_judges"] else "")

    return {
        "emr_id": emr_id, "model_id": model_id,
        "scope": scope,
        "category": r3.get("category", "—"),
        "verdict": r3.get("verdict", ""),
        "summary_by_block": r3.get("summary_by_block", {}),
        "e1_signals": e1_signals,
        # coverage (счётчики чисел/предикатов, entity-отчёт, schema_note) раньше
        # терялся ровно здесь: наружу шли только status и коды причин, из-за чего
        # в отчёте нельзя было понять, ПОЧЕМУ шлюз решил именно так.
        "gate": {"status": ctx.gate_decision.status,
                 "reasons": ctx.gate_decision.reason_codes(),
                 "messages": [r.message for r in ctx.gate_decision.reasons],
                 "degenerate": False,
                 "coverage": ctx.gate_decision.coverage},
        # Блок 7 — «объективные метрики рядом с LLM-оценками»: компактное
        # резюме фактологической сверки (Блок 2), не зависящее от мнения
        # судей — числа/единицы/полярность/сущности с precision-recall-F1.
        "objective": ctx.obj_report.to_summary(),
        "r1": r1, "r2": r2, "r3": r3,
    }


# ══════════════════════════════════════════════════════════════════
# ЧЕКПОИНТЫ — сохранённое из evaluator.py инженерное решение: длинные
# прогоны (часы на сотни пар) обязаны переживать перезапуск процесса
# ══════════════════════════════════════════════════════════════════

CHECKPOINT_DIR = Path("checkpoints") / "judge"


def _checkpoint_path(emr_id: str, model_id: str) -> Path:
    return CHECKPOINT_DIR / f"{emr_id}__{model_id}.json"


def load_checkpoint(emr_id: str, model_id: str) -> Optional[dict]:
    p = _checkpoint_path(emr_id, model_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_checkpoint(result: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    p = _checkpoint_path(result["emr_id"], result["model_id"])
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
# САМОПРОВЕРКА — ручной разбор небольшой выборки пар (см. план Блока 4:
# «сверяем, что обоснования судей релевантны именно ЭТОЙ карте, а раунд 2
# содержательно ссылается на конкретные расхождения, а не усредняет числа»)
# ══════════════════════════════════════════════════════════════════

def _self_check():
    import pandas as pd

    print("=" * 70)
    print("Блок 4 — самопроверка: полный 3-раундовый цикл на малой выборке")
    print("(цель — не статистика, а ручной разбор: релевантны ли обоснования")
    print(" именно ЭТОЙ карте, и ссылается ли R2 на конкретные расхождения)")
    print("=" * 70)

    df = pd.read_excel("data/summaries.xlsx", dtype=str).fillna("")

    sample_keys = [("EMR_10", "2"), ("EMR_11", "2")]
    rows = []
    for emr_id, model_id in sample_keys:
        match = df[(df["emr_id"] == emr_id) & (df["model_id"] == model_id)]
        if not match.empty:
            rows.append(match.iloc[0])
    if not rows:
        rows = [df.iloc[0]]

    try:
        panel = JudgePanel()
        print(f"\nПанель: {panel}")
    except LLMError as e:
        print(f"\n  (пропуск — LLM-панель недоступна: {e})")
        return

    for row in rows:
        emr_id, model_id = row["emr_id"], row["model_id"]
        print(f"\n{'─' * 60}\n  {emr_id} / модель {model_id}\n{'─' * 60}")
        try:
            result = evaluate_summary(row["source_text"], row["summary_text"],
                                       emr_id, model_id, scope="radiologist", panel=panel)
            save_checkpoint(result)
            print(f"  Категория: {result['category']}")
            print(f"  Вердикт: {result['verdict'][:300]}")
            print(f"  E1-сигналы: {result['e1_signals']}")
            for block, comment in result.get("summary_by_block", {}).items():
                print(f"    [{block}] {comment[:160]}")
        except Exception as e:
            print(f"  (прогон провалился: {type(e).__name__}: {str(e)[:200]})")

    print("\nГотово — результаты также сохранены в checkpoints/judge/*.json "
          "для офлайн-разбора.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    _self_check()
