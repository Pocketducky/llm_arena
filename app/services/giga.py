import httpx
from app.services.giga_token import get_giga_token


async def gigachat_completion(prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
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
                     {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    async with httpx.AsyncClient(verify=False, timeout=60) as client:
        response = await client.post(url, headers=headers, json=payload,)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]