from app.core.utils import extract_json, repair_json
from app.services.giga import gigachat_completion
import json
from app.services.giga_logs import logger

async def ask_gigachat(prompt: str, desc: str = "", temperature: float = 0.1) -> dict | str:
    """
    Полный запрос в гигачат с форматированием ответа
    :param prompt:
    :param desc: Метка запроса (R1, R2, R3)
    :return:
    """
    raw = await gigachat_completion(prompt, temperature=temperature)
    try:
        result = extract_json(raw)
        logger.info(
            "Уровень запроса: %s\nОтвет модели: %s",   # Можно добавить: \nПромпт: %s; prompt
            desc, result
        )
    except Exception:
        try:
            repaired = repair_json(raw)
            result = json.loads(repaired)
        except Exception:
            raise ValueError(f"❌GigaChat не вернул JSON для {desc}: {raw}")
    return result