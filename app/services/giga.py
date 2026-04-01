import asyncio
import httpx
from app.services.giga_logs import logger
from app.services.giga_token import get_giga_token

# Семафор ограничивает одновременные запросы до 1
_gigachat_semaphore = asyncio.Semaphore(1)

async def gigachat_completion(prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
    """
    Функция получения ответа от ГЧ. Подключено обновление и сохранение Access Token в кэш

    Model:

    "GigaChat-2" — Простая, много токенов
    "GigaChat-2-Pro" — Хорошенькая (80 к токенов)
    "GigaChat-2-Max" — Самая лучшая (80 к токенов)
    """
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    token = get_giga_token()
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    payload = {
        "model": "GigaChat-2",
        "messages": [
            {"role": "system", "content": "Все твои ответы должны быть строго в формате JSON. Не добавляй никаких комментариев, объяснений или дополнительных символов вне фигурных скобок."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    async with _gigachat_semaphore:
        async with httpx.AsyncClient(verify=False, timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload,)
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
                # "  Промпт: %(prompt)s\n"
                # "  Ответ: %(response)s\n"
                "  Токены: %(tokens)s",
                {
                    "model": logs["model"],
                    # "prompt": prompt,
                    # "response": data["choices"][0]["message"]["content"],
                    "tokens": logs["tokens"]
                }
            )

            return data["choices"][0]["message"]["content"]