import json

from app.core.prompts import PROMPT_R2
from app.giga_core.giga_response import ask_gigachat


async def score_gigachat_r2(my_r1: dict, peer1_r1: dict, peer2_r1: dict, summary: str) -> dict:
    """
    Перекрестная оценка суммаризаций на первом этапе

    Промпт:

    Ты — старший врач-рентгенолог. Ты уже оценил суммаризацию в первом раунде.
    Теперь изучи оценки двух коллег и скорректируй свою позицию если нужно.

    СУММАРИЗАЦИЯ (напоминание):
    {summary}

    ТВОЯ ОЦЕНКА R1:
    {my_report}

    ОЦЕНКА КОЛЛЕГИ 1:
    {peer_1}

    ОЦЕНКА КОЛЛЕГИ 2:
    {peer_2}

    Верни ТОЛЬКО JSON — финальные скорректированные баллы (без пояснений):
    {{"complaints":0,"disease_history":0,"comorbidities":0,"habits":0,"labs":0,"imaging":0,"penalties":0,"iodine_flag":false,"safety_flag":false,"hallucinations":[],"quality":"отличное/хорошее/удовлетворительное/неудовлетворительное/опасное"}}

    Максимумы: complaints 15, disease_history 15, comorbidities 20, habits 5, labs 20, imaging 25.
    Если коллеги нашли то что ты пропустил — учти. Если не согласен — держи свою позицию.
    Аллергия на йод: если хоть один коллега поднял iodine_flag — проверь и ты.

    :param my_r1: Собственная оценка суммаризации
    :param peer1_r1: Оценка суммаризации второй модели
    :param peer2_r1: Оценка суммаризации третьей модели
    :param summary: Суммаризация
    :return: Оценка в формате: {"complaints":0,"disease_history":0,"comorbidities":0,"habits":0,"labs":0,"imaging":0,"penalties":0,"iodine_flag":false,"safety_flag":false,"hallucinations":[],"quality":"отличное/хорошее/удовлетворительное/неудовлетворительное/опасное"}
    """
    def compact(r: dict) -> str:
        return json.dumps({k: v for k, v in r.items()
                           if k not in ("safety_reason", "wrong_values", "hallucinations")},
                          ensure_ascii=False)

    prompt = PROMPT_R2.format(
        summary=summary,
        my_report=compact(my_r1),
        peer_1=compact(peer1_r1),
        peer_2=compact(peer2_r1),
    )
    result = await ask_gigachat(prompt, "R2-пересмотр")

    comp = float(result.get("complaints", 0))
    dh = float(result.get("disease_history", 0))
    co = float(result.get("comorbidities", 0))
    hab = float(result.get("habits", 0))
    lab = float(result.get("labs", 0))
    img = float(result.get("imaging", 0))
    pen = float(result.get("penalties", 0))

    positive = comp + dh + co + hab + lab + img
    final_score = max(0.0, min(100.0, round(positive + pen, 1)))

    return {
        "complaints": comp,
        "disease_history": dh,
        "comorbidities": co,
        "habits": hab,
        "labs": lab,
        "imaging": img,
        "penalties": pen,
        "final_score": final_score,
        "iodine_flag": bool(result.get("iodine_flag", False)),
        "safety_flag": bool(result.get("safety_flag", False)),
        "hallucinations": list(result.get("hallucinations", [])),
        "quality": str(result.get("quality", "—")),
    }
