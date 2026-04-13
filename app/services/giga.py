import asyncio
import httpx
from app.services.giga_logs import logger
from app.services.giga_token import get_giga_token

# Семафор ограничивает одновременные запросы к GigaChat до 1 (бесплатная версия API).
# Все запросы сериализуются и выполняются строго по очереди.
_gigachat_semaphore = asyncio.Semaphore(1)

# Сколько секунд запрос может ждать своей очереди в semaphore.
# При 1-потоковом GigaChat и 3 R1-вызовах + 3 R2-вызовах + 1 R3 = 7 запросов на оценку.
# Каждый вызов ~10-30 сек → полная оценка ~1-3 мин.
# 5 мин — достаточно чтобы не умереть при нормальной нагрузке.
QUEUE_TIMEOUT = 300

# Таймаут самого HTTP-запроса к GigaChat.
# LLM-ответы могут занимать 30-90 сек, особенно для длинных промптов.
HTTP_TIMEOUT = 120


async def gigachat_completion(prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
    """
    Функция получения ответа от GigaChat.

    Все вызовы сериализуются через asyncio.Semaphore(1) — запросы встают в очередь.
    Токен запрашивается внутри semaphore-блока, чтобы гарантированно не протухнуть
    пока запрос ждал в очереди.

    :raises TimeoutError: если запрос не получил semaphore за QUEUE_TIMEOUT секунд
    :raises httpx.HTTPStatusError: если GigaChat вернул ошибку
    """
    # ── Ожидание semaphore с таймаутом ──────────────────────────────
    try:
        await asyncio.wait_for(_gigachat_semaphore.acquire(), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Запрос не смог получить доступ к GigaChat за {QUEUE_TIMEOUT} сек. "
            f"Возможно, очередь слишком длинная или сервис недоступен."
        )

    try:
        # Токен запрашиваем ВНУТРИ semaphore — гарантия что он свежий.
        token = get_giga_token()

        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        payload = {
            "model": "GigaChat-2",
            "messages": [
                {"role": "system", "content": (
                    "Все твои ответы должны быть строго в формате JSON. "
                    "Не добавляй никаких комментариев, объяснений или дополнительных символов вне фигурных скобок."
                )},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        async with httpx.AsyncClient(verify=False, timeout=HTTP_TIMEOUT) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            logs = {
                "created_at": data.get("created"),
                "model": data.get("model"),
                "tokens": data.get("usage"),
            }
            logger.info(
                "Запрос к GigaChat:\n"
                "  Модель: %(model)s\n"
                "  Токены: %(tokens)s",
                {
                    "model": logs["model"],
                    "tokens": logs["tokens"]
                }
            )

            return data["choices"][0]["message"]["content"]

    finally:
        _gigachat_semaphore.release()
