from app.core.utils import extract_json
from app.services.giga import gigachat_completion
from app.services.giga_logs import logger
import asyncio

MAX_RETRIES = 2
RETRY_DELAY = 3.0


async def ask_gigachat(prompt: str, desc: str = "", temperature: float = 0.1) -> dict:
    """
    Запрос в GigaChat с автоматическим retry при transient-ошибках.

    :param prompt: текст промпта
    :param desc: метка запроса для логов (R1, R2, R3, Суммаризация и т.д.)
    :param temperature: температура генерации
    :return: распарсенный dict из JSON-ответа
    :raises ValueError: если после всех попыток не удалось получить валидный JSON
    :raises RuntimeError: если API недоступен после всех retry
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            raw = await gigachat_completion(prompt, temperature=temperature)
            result = extract_json(raw)
            logger.info("Уровень запроса: %s\nОтвет модели: %s", desc, result)
            return result

        except ValueError as e:
            # JSON parsing error — retry with repaired JSON
            last_error = e
            logger.warning(f"  [{desc}] Попытка {attempt}: ошибка парсинга JSON — {e}")
            if attempt <= MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)

        except Exception as e:
            # Network, timeout, token expiry, etc.
            last_error = e
            logger.warning(f"  [{desc}] Попытка {attempt}: {type(e).__name__} — {e}")
            if attempt <= MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY * attempt)

    raise RuntimeError(
        f"❌ GigaChat не ответил после {MAX_RETRIES + 1} попыток для [{desc}]: {last_error}"
    )