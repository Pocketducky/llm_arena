from app.core.prompts import PROMPT_EXTRACT_FACTS, PROMPT_COVERAGE, PROMPT_ERRORS
from app.core.utils import _calc_coverage_score
from app.giga_core.giga_response import ask_gigachat


async def score_gigachat_r1(source: str, summary: str) -> dict:
    """
    Оценка суммаризации ЭМК по 2 разделам и 12 критериям, состоящая из 3 запросов

    coverage:
      "complaints":      {{"covered": [], "missing": [], "iodine_allergy_noted": false}},
      "disease_history": {{"covered": [], "missing": []}},
      "comorbidities":   {{"covered": [], "missing": []}},
      "habits":          {{"covered": [], "missing": []}},
      "labs":            {{"covered": [], "missing": []}},
      "imaging":         {{"covered": [], "missing": []}}

    errors:
      "iodine_missing": false,
      "wrong_focus": false,
      "hallucinations": [],
      "wrong_values": [],
      "irrelevant": [],
      "penalties": 0,
      "safety_flag": false,
      "safety_reason": ""

    :param source: ЭМК
    :param summary: Суммаризация ЭМК
    """

    facts_prompt = PROMPT_EXTRACT_FACTS.format(
        source = source
    )
    facts = await ask_gigachat(facts_prompt, "Факты ЭМК")

    coverage_prompt = PROMPT_COVERAGE.format(
        emr_facts = facts,
        summary = summary
    )
    coverage = await ask_gigachat(coverage_prompt, "Соответствие ЭМК/суммаризации", temperature=0.0)

    # Для поиска ошибок берём только первые 6000 символов ЭМК,
    # чтобы экономить токены — ошибки обычно в клинической части
    source_short = source[:6000]
    if len(source) > 6000:
        source_short += "\n[... текст обрезан ...]"

    errors_prompt = PROMPT_ERRORS.format(
        source_short=source_short,
        summary=summary,
    )
    errors = await ask_gigachat(errors_prompt, "Штрафы", temperature=0.0)

    # Считаем баллы в коде
    MAXES = {"complaints": 15, "disease_history": 15, "comorbidities": 20,
             "habits": 5, "labs": 20, "imaging": 25}
    scores = {cat: _calc_coverage_score(coverage.get(cat, {}), mx)
              for cat, mx in MAXES.items()}

    pen = float(errors.get("penalties", 0))

    # iodine: только если аллергия реально есть в ЭМК
    iodine_in_source = bool(facts.get("iodine_allergy_in_source", False))
    iodine_noted = bool(
        coverage.get("complaints", {}).get("iodine_allergy_noted", False) or
        coverage.get("disease_history", {}).get("iodine_allergy_noted", False)
    )
    iodine_missing = (iodine_in_source and not iodine_noted
                      and bool(errors.get("iodine_missing", False)))

    positive = sum(scores.values())
    final_score = max(0.0, min(100.0, round(positive - abs(pen), 1)))

    return {
        **scores,
        "penalties": pen,
        "final_score": final_score,
        "iodine_flag": iodine_missing,
        "safety_flag": bool(errors.get("safety_flag", False)),
        "safety_reason": str(errors.get("safety_reason", "")),
        "hallucinations": list(errors.get("hallucinations", [])),
        "wrong_values": list(errors.get("wrong_values", [])),
        "irrelevant": list(errors.get("irrelevant", [])),
        "iodine_missing": bool(errors.get("iodine_missing", False)),
        "wrong_focus": bool(errors.get("wrong_focus", False)),
        "coverage_detail": {cat: {
            "covered": coverage.get(cat, {}).get("covered", []),
            "missing": coverage.get(cat, {}).get("missing", []),
        } for cat in MAXES},
    }


