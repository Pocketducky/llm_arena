from app.giga_core.r1 import score_gigachat_r1
from app.giga_core.r2 import score_gigachat_r2
from app.giga_core.r3 import score_gigachat_r3


async def evaluate_with_gigachat(source: str, summary: str) -> dict:
    """
    Полный цикл оценки суммаризации:

    Три независимые оценки на первом этапе -> Перекрестная оценка на втором этапе -> Финальная оценка на третьем этапе
    :param source: ЭМК
    :param summary: Суммаризация ЭМК
    :return: Финальный результат по 12 метрикам
    """
    # R1: три независимые оценки
    r1_results = []
    for _ in range(3):
        r1 = await score_gigachat_r1(source, summary)
        r1_results.append(r1)

    # R2: каждая из трёх оценок пересматривается с учётом двух других
    r2_results = []
    for i in range(3):
        peer1 = r1_results[(i+1)%3]
        peer2 = r1_results[(i+2)%3]
        r2 = await score_gigachat_r2(r1_results[i], peer1, peer2, summary)
        r2_results.append(r2)

    # R3: финальный арбитраж
    final = await score_gigachat_r3(r2_results)
    return final
