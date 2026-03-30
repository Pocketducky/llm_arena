import json

from app.core.prompts import PROMPT_R3
from app.core.utils import _score_to_quality
from app.giga_core.giga_response import ask_gigachat


async def score_gigachat_r3(r2_results: list[dict]) -> dict:
    """
    Финальная оценка суммаризации
    :param r2_results: Результаты перекресной оценки на втором этапе
    :return: Оценка в формате: {"complaints":0,"disease_history":0,"comorbidities":0,"habits":0,"labs":0,"imaging":0,"penalties":0,"iodine_flag":false,"safety_flag":false,"hallucinations":[],"quality":"отличное/хорошее/удовлетворительное/неудовлетворительное/опасное"}

    """
    if len(r2_results) < 3:
        final_score = sum(r.get("final_score", 0) for r in r2_results) / len(r2_results)
        quality = _score_to_quality(final_score)
        return {
            "final_score": final_score,
            "quality": quality,
            "verdict": "Автоматическое среднее (недостаточно оценок для R3)",
            "safety_flag": any(r.get("safety_flag", False) for r in r2_results),
            "iodine_flag": any(r.get("iodine_flag", False) for r in r2_results),
            "hallucinations": [h for r in r2_results for h in r.get("hallucinations", [])],
        }

    prompt_r3 = PROMPT_R3.format(
        report_a=json.dumps({k: v for k, v in r2_results[0].items()
                           if k not in ("missing_clinical","safety_reason",
                                        "wrong_values","error")},
                          ensure_ascii=False),
        report_b=json.dumps({k: v for k, v in r2_results[1].items()
                           if k not in ("missing_clinical","safety_reason",
                                        "wrong_values","error")},
                          ensure_ascii=False),
        report_c=json.dumps({k: v for k, v in r2_results[2].items()
                           if k not in ("missing_clinical","safety_reason",
                                        "wrong_values","error")},
                          ensure_ascii=False),
    )
    result = await ask_gigachat(prompt_r3, "R3-арбитраж", temperature=0.0)

    criteria = {
        "complaints": float(result.get("complaints", 0)),
        "disease_history": float(result.get("disease_history", 0)),
        "comorbidities": float(result.get("comorbidities", 0)),
        "habits": float(result.get("habits", 0)),
        "labs": float(result.get("labs", 0)),
        "imaging": float(result.get("imaging", 0)),
        "penalties": float(result.get("penalties", 0)),
    }

    comp = float(criteria.get("complaints", 0))
    dh = float(criteria.get("disease_history", 0))
    co = float(criteria.get("comorbidities", 0))
    hab = float(criteria.get("habits", 0))
    lab = float(criteria.get("labs", 0))
    img = float(criteria.get("imaging", 0))
    pen = float(criteria.get("penalties", 0))

    positive = comp + dh + co + hab + lab + img
    final_score = max(0.0, min(100.0, round(positive - pen if pen > 0 else positive + pen, 1)))

    quality = str(result.get("quality", "—"))
    verdict_text = str(result.get("verdict", ""))

    safety_flag = (bool(result.get("safety_flag", False)))
    iodine_flag = (bool(result.get("iodine_flag", False)))

    if quality in ("—", ""):
        quality = _score_to_quality(final_score)

    return {
        "final_score": final_score,
        "quality": quality,
        "safety_flag": safety_flag,
        "iodine_flag": iodine_flag,
        "verdict": verdict_text,
        "criteria": criteria,
    }

