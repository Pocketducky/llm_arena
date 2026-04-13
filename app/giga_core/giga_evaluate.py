import asyncio

from app.giga_core.r1 import score_gigachat_r1
from app.giga_core.r2 import score_gigachat_r2
from app.giga_core.r3 import score_gigachat_r3


async def evaluate_with_gigachat(source: str, summary: str) -> dict:
    """
    Полный цикл оценки суммаризации:

    Три независимые оценки на первом этапе (параллельно) →
    Перекрестная оценка на втором этапе (параллельно) →
    Финальная оценка на третьем этапе

    Возвращает словарь с ключами:
        final: dict - финальный результат (баллы, качество, флаги и т.д.)
        r1_results: list[dict] - три результата первого этапа
        r2_results: list[dict] - три результата второго этапа
    """
    # R1: три независимые оценки — запускаем параллельно
    r1_tasks = [score_gigachat_r1(source, summary) for _ in range(3)]
    r1_results: list[dict] = await asyncio.gather(*r1_tasks)

    # R2: каждая из трёх оценок пересматривается с учётом двух других — параллельно
    r2_tasks = []
    for i in range(3):
        peer1 = r1_results[(i + 1) % 3]
        peer2 = r1_results[(i + 2) % 3]
        r2_tasks.append(score_gigachat_r2(r1_results[i], peer1, peer2, summary))
    r2_results: list[dict] = await asyncio.gather(*r2_tasks)

    # R3: финальный арбитраж (один запрос)
    final: dict = await score_gigachat_r3(r2_results)
    final["all_hallucinations"] = list({
        x for d in r1_results + r2_results for x in d.get("hallucinations", [])
    })

    return {
        "final": final,
        "r1_results": r1_results,
        "r2_results": r2_results,
    }
