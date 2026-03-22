from app.core.utils import extract_json, repair_json
from app.services.giga import gigachat_completion
import json

async def ask_gigachat(prompt: str, desc: str = "") -> dict | str:
    """
    Полный запрос в гигачат с форматированием ответа
    :param prompt:
    :param desc: Метка запроса (R1, R2, R3)
    :return:
    """
    raw = await gigachat_completion(prompt)
    try:
        result = extract_json(raw)
    except Exception:
        try:
            repaired = repair_json(raw)
            result = json.loads(repaired)
        except Exception:
            print(f"GigaChat не вернул JSON для {desc}: {raw[:200]}")
            return raw
    return result
