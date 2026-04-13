import json

from app.core.prompts import PROMPT_R2
from app.giga_core.giga_response import ask_gigachat


def _compact_for_r2(r: dict) -> str:
    """
    Формирует отчёт для R2, включающий не только баллы, но и
    диагностические данные: что покрыто/пропущено и какие ошибки найдены.
    Это позволяет модели R2 понять ПРИЧИНУ снижения баллов коллегами.
    """
    # Базовые баллы и флаги
    compact = {
        "complaints": r.get("complaints", 0),
        "disease_history": r.get("disease_history", 0),
        "comorbidities": r.get("comorbidities", 0),
        "habits": r.get("habits", 0),
        "labs": r.get("labs", 0),
        "imaging": r.get("imaging", 0),
        "penalties": r.get("penalties", 0),
        "final_score": r.get("final_score", 0),
        "iodine_flag": r.get("iodine_flag", False),
        "safety_flag": r.get("safety_flag", False),
        "quality": r.get("quality", "—"),
    }

    # Покр. детализация — сокращаем до только missing (пропущенные факты)
    coverage_detail = r.get("coverage_detail", {})
    if coverage_detail:
        missing_only = {}
        for cat, detail in coverage_detail.items():
            missing = detail.get("missing", [])
            if missing:
                missing_only[cat] = missing
        if missing_only:
            compact["missing_facts"] = missing_only

    # Ошибки — только если есть
    hallucinations = r.get("hallucinations", [])
    if hallucinations:
        compact["hallucinations"] = hallucinations

    wrong_values = r.get("wrong_values", [])
    if wrong_values:
        compact["wrong_values"] = wrong_values

    irrelevant = r.get("irrelevant", [])
    if irrelevant:
        compact["irrelevant"] = irrelevant

    return json.dumps(compact, ensure_ascii=False, indent=2)


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

    prompt_r2 = PROMPT_R2.format(
        summary=summary,
        my_report=_compact_for_r2(my_r1),
        peer_1=_compact_for_r2(peer1_r1),
        peer_2=_compact_for_r2(peer2_r1),
    )

    result = await ask_gigachat(prompt_r2, "R2-пересмотр")

    comp = float(result.get("complaints", 0))
    dh = float(result.get("disease_history", 0))
    co = float(result.get("comorbidities", 0))
    hab = float(result.get("habits", 0))
    lab = float(result.get("labs", 0))
    img = float(result.get("imaging", 0))
    pen = float(result.get("penalties", 0))

    positive = comp + dh + co + hab + lab + img
    final_score = max(0.0, min(100.0, round(positive - abs(pen), 1)))

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
        "r2_reason": str(result.get("r2_reason", "")),
    }
