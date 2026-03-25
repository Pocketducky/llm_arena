from app.core.prompts import PROMPT_CLINICAL, FEWSHOT_GOOD_CLINICAL, FEWSHOT_BAD_SUMMARY, FEWSHOT_BAD_CLINICAL, \
    FEWSHOT_GOOD_SUMMARY, PROMPT_INSTRUMENTAL, PROMPT_PENALTIES, FEWSHOT_GOOD_PENALTIES, FEWSHOT_BAD_PENALTIES
from app.core.utils import split_source
from app.giga_core.giga_response import ask_gigachat


async def score_gigachat_r1(source: str, summary: str) -> dict:
    """
    Оценка суммаризации ЭМК по 2 разделам и 12 критериям, состоящая из 3 запросов
    :param source: ЭМК
    :param summary: Суммаризация ЭМК
    :return: Словарь с результатами оценки:
        { summary : Обобщенныый результат, только самая важная информация по оценкам
        clinical_part, instrumental_part, penalties : Подробный отчет о результатах каждого блока оценки }
    """
    src_clinical, src_labs = split_source(source)

    clinical_prompt = PROMPT_CLINICAL.format(
        source=src_clinical, summary=summary,
        fewshot_good_summary=FEWSHOT_GOOD_SUMMARY,
        fewshot_good_clinical=FEWSHOT_GOOD_CLINICAL,
        fewshot_bad_summary=FEWSHOT_BAD_SUMMARY,
        fewshot_bad_clinical=FEWSHOT_BAD_CLINICAL,
    )
    clinical = await ask_gigachat(clinical_prompt, "клиника")

    instrumental_prompt = PROMPT_INSTRUMENTAL.format(
        source=src_labs, summary=summary
    )
    instrumental = await ask_gigachat(instrumental_prompt, "лаб+инстр")

    penalties_prompt = PROMPT_PENALTIES.format(
        source=source, summary=summary,
        fewshot_good_penalties=FEWSHOT_GOOD_PENALTIES,
        fewshot_bad_penalties=FEWSHOT_BAD_PENALTIES,
    )
    penalties = await ask_gigachat(penalties_prompt, "штрафы")

    # Временная обработка
    try:
        comp = float(clinical.get("complaints", {}).get("score", 0))
        dh = float(clinical.get("disease_history", {}).get("score", 0))
        co = float(clinical.get("comorbidities", {}).get("score", 0))
        hab = float(clinical.get("habits", {}).get("score", 0))
        lab = float(instrumental.get("labs", {}).get("score", 0))
        img = float(instrumental.get("imaging", {}).get("score", 0))
        pen = float(penalties.get("penalties", 0))

        positive = comp + dh + co + hab + lab + img
        final_score = max(0.0, min(100.0, round(positive + pen, 1)))

        return {
            "summary": {
                "complaints": comp,
                "disease_history": dh,
                "comorbidities": co,
                "habits": hab,
                "labs": lab,
                "imaging": img,
                "penalties": pen,
                "final_score": final_score,
                "iodine_flag": bool(penalties.get("iodine_missing", False)),
                "safety_flag": bool(penalties.get("safety_flag", False)),
                "safety_reason": str(penalties.get("safety_reason", "")),
                "hallucinations": [str(h) for h in penalties.get("hallucinations", [])],
                "wrong_values": [str(w) for w in penalties.get("wrong_values", [])],
            },
            "full": {
                "clinical_part": clinical,
                "instrumental_part": instrumental,
                "penalties": penalties,
            }
        }

    except Exception as e:
        print(f"❌Ошибка тут: {e}")


