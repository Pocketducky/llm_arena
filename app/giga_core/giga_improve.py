from app.core.prompts import IMPROVED_SUMMARIZATION
from app.giga_core.giga_response import ask_gigachat


async def improve_summarization_gigachat(source: str, summary: str, score_1: dict, score_2: dict, score_3: dict) -> str:

    improvement_prompt = IMPROVED_SUMMARIZATION.format(
        source_text=source, summary=summary,
        peer_1=score_1, peer_2=score_2, peer_3=score_3,
    )

    response = await ask_gigachat(improvement_prompt, "Улучшение суммаризации")
    improved_summary = response.get("improved_summary")

    return improved_summary