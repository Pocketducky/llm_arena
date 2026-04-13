from app.core.prompts import SUMMARIZE_PROMPT
from app.giga_core.giga_response import ask_gigachat


async def summarize_with_gigachat(source_text: str) -> str:
    """
    Суммаризация ЭМК (простая однозапросная)

    :param source_text: текст ЭМК
    :return: суммаризация ЭМК
    :raises ValueError: если GigaChat не вернул поле summary
    """
    prompt = SUMMARIZE_PROMPT.format(source_text=source_text)

    res = await ask_gigachat(prompt, "Суммаризация")
    summary = res.get("summary")

    if not summary or not isinstance(summary, str):
        raise ValueError(
            f"GigaChat не вернул суммаризацию. Ответ: {res}"
        )

    return summary