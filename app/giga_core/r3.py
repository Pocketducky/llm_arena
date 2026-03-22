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

    prompt = PROMPT_R3.format(
        report_a=json.dumps({k: v for k, v in r2_results[0].items()
                             if k not in ("hallucinations", "quality")}, ensure_ascii=False),
        report_b=json.dumps({k: v for k, v in r2_results[1].items()
                             if k not in ("hallucinations", "quality")}, ensure_ascii=False),
        report_c=json.dumps({k: v for k, v in r2_results[2].items()
                             if k not in ("hallucinations", "quality")}, ensure_ascii=False),
    )
    result = await ask_gigachat(prompt, "R3-арбитраж")

    final_score = float(result.get("final_score", 0))
    quality = str(result.get("quality", _score_to_quality(final_score)))
    verdict = str(result.get("verdict", ""))
    safety_flag = bool(result.get("safety_flag", False) or any(r.get("safety_flag") for r in r2_results))
    iodine_flag = bool(result.get("iodine_flag", False) or any(r.get("iodine_flag") for r in r2_results))
    hallucinations = list(set(h for r in r2_results for h in r.get("hallucinations", [])))

    return {
        "final_score": final_score,
        "quality": quality,
        "verdict": verdict,
        "safety_flag": safety_flag,
        "iodine_flag": iodine_flag,
        "hallucinations": hallucinations,
    }

