import json
import os
import re
import logging


# Файловый лог — DEBUG (все сырые ответы моделей)
# Консоль — INFO (только прогресс)
_file_handler   = logging.FileHandler("evaluation.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(_file_handler)
log.addHandler(_console_handler)


def repair_json(text: str) -> str:
    """
    Исправляет все типичные ошибки JSON которые делают LLM:
    1. Trailing comma:    {"a":1,}        -> {"a":1}
    2. Одинарные кавычки: {'a':'b'}       -> {"a":"b"}
    3. Python literals:   True/False/None -> true/false/null
    4. Пропущенные кавычки у ключей: {a:1} -> {"a":1}
    5. Незакрытые скобки в конце
    6. Пропущенное двоеточие: {"a" "b"}  -> {"a":"b"}  (редко)
    7. Лишние запятые между ключами
    """
    # Python boolean/None
    text = re.sub(r'\bTrue\b',  "true",  text)
    text = re.sub(r'\bFalse\b', "false", text)
    text = re.sub(r'\bNone\b',  "null",  text)

    # Одинарные кавычки -> двойные
    # Осторожно: не трогаем апострофы внутри строк
    text = re.sub(r"(?<!\\)'([^']*)'", r'"\1"', text)

    # Trailing comma перед } или ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Ключи без кавычек: {score:5} -> {"score":5}
    text = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', text)

    # Двойные запятые: {"a":1,,} -> {"a":1}
    text = re.sub(r",{2,}", ",", text)

    # Пропущенное двоеточие между строкой и значением: "key" "value" -> "key": "value"
    text = re.sub(r'("\s*)(\s+")', r'\1:\2', text)

    # Закрываем незакрытые скобки в конце
    opens  = text.count("{") - text.count("}")
    opens2 = text.count("[") - text.count("]")
    if opens > 0:
        text = text.rstrip() + "}" * opens
    if opens2 > 0:
        text = text.rstrip() + "]" * opens2

    return text


def extract_json(raw: str) -> dict:
    """
    Извлекает JSON из ответа модели.
    Обрабатывает: markdown, текст вокруг JSON, think-блоки.
    При ошибке пробует авторемонт.
    """
    text = raw.strip()

    # Убираем think-блок (на случай если модель его генерирует)
    if "<think>" in text:
        if "</think>" in text:
            text = text[text.find("</think>") + len("</think>"):].strip()
        else:
            # Незакрытый think — убираем тег и берём остаток
            text = text[text.find("<think>") + len("<think>"):]
            text = re.sub(r"^.*?(?=\{)", "", text, flags=re.DOTALL).strip()

    # Убираем markdown-обёртку
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Берём от первой { до последней }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Фигурные скобки не найдены. Ответ: {raw[:300]}")

    json_str = text[start:end]

    # Попытка 1: чистый JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Попытка 2: с авторемонтом
    try:
        return json.loads(repair_json(json_str))
    except json.JSONDecodeError:
        pass

    # Попытка 3: найти первый полный объект — иногда модель пишет два JSON подряд
    depth, i, best_end = 0, start, -1
    for idx in range(start, min(end, len(text))):
        if text[idx] == "{": depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                best_end = idx + 1
                break
    if best_end > start:
        try:
            return json.loads(repair_json(text[start:best_end]))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"JSON не удалось распарсить даже после ремонта. "
                     f"Фрагмент: {json_str[:300]}")


def _warn_if_all_zeros(model_key: str, result: dict, raw: str):
    """
    Если все числовые поля = 0 — модель скопировала шаблон.
    Бросает ValueError чтобы запустить retry с другим подходом.
    """
    nums = [v for v in result.values() if isinstance(v, (int, float))]
    if nums and all(v == 0 for v in nums) and len(nums) > 1:
        log.warning(f"      ⚠ {model_key} вернул ВСЕ нули — запускаю retry")
        log.warning(f"      Ответ: {raw[:200]}")
        raise ValueError(f"all_zeros: {model_key} вернул все нули")


def _make_minimal_prompt(desc: str, failed_raw: str) -> str:
    """
    Минималистичный промпт для retry3.
    Когда модель не справляется с длинным промптом — даём только
    структуру JSON и просим её заполнить по предыдущему ответу.
    """
    if "клиника" in desc:
        skeleton = (
            '{"complaints":{"score":?,"max":15,"missing":[]},'
            '"disease_history":{"score":?,"max":15,"missing":[],"iodine_allergy_noted":false},'
            '"comorbidities":{"score":?,"max":20,"missing":[]},'
            '"habits":{"score":?,"max":5,"missing":[]}}'
        )
    elif "лаб" in desc or "инстр" in desc:
        skeleton = (
            '{"labs":{"score":?,"max":20,"missing":[]},'
            '"imaging":{"score":?,"max":25,"missing":[]}}'
        )
    elif "штраф" in desc:
        skeleton = (
            '{"iodine_missing":false,"kidney_cyst_missing":false,'
            '"wrong_focus":false,"hallucinations":[],"wrong_values":[],'
            '"irrelevant":[],"penalties":0,"safety_flag":false,"safety_reason":""}'
        )
    elif "R2" in desc or "пересмотр" in desc:
        skeleton = (
            '{"complaints":0,"disease_history":0,"comorbidities":0,'
            '"habits":0,"labs":0,"imaging":0,"penalties":0,'
            '"iodine_flag":false,"safety_flag":false,"hallucinations":[],'
            '"quality":"удовлетворительное"}'
        )
    else:
        skeleton = (
            '{"complaints":0,"disease_history":0,"comorbidities":0,'
            '"habits":0,"labs":0,"imaging":0,"penalties":0,'
            '"final_score":0,"iodine_flag":false,"safety_flag":false,'
            '"hallucinations":[],"quality":"удовлетворительное","verdict":""}'
        )

    # Заменяем ? на реальные числа из предыдущего ответа если можно
    return (
        f"Заполни JSON-структуру числами от 0 до максимума. "
        f"Замени все ? на реальные числа оценки. "
        f"Верни ТОЛЬКО JSON:\n{skeleton}"
    )


def split_source(source: str) -> tuple[str, str]:
    """
    Делит исходный текст ЭМК на две части:
      part1 — осмотры врачей (жалобы, анамнез, сопутствующие)
      part2 — лабораторные и инструментальные данные

    Это сокращает размер каждого промпта вдвое (~10k вместо 20k символов)
    и помогает 7B моделям сфокусироваться на нужном разделе.
    """
    # Ищем разделители между секциями ЭМК
    lab_markers  = ["Результаты лабораторных", "Лабораторные исследования",
                    "Биохимическое исследование", "Клинический анализ крови",
                    "Компьютерная томография", "Ультразвуковое исследование",
                    "Выписной эпикриз"]
    split_pos = len(source)  # по умолчанию — весь текст в part1
    for marker in lab_markers:
        pos = source.find(marker)
        if 0 < pos < split_pos:
            split_pos = pos

    part1 = source[:split_pos].strip()   # клиника
    part2 = source[split_pos:].strip()   # лаб + инструментальные

    # Если разделить не получилось — дублируем весь текст
    if not part2:
        part2 = source

    log.debug(f"    split_source: part1={len(part1)} симв, part2={len(part2)} симв")
    return part1, part2


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.txt':
        return file_bytes.decode('utf-8')
    elif ext == '.docx':
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return '\n'.join([p.text for p in doc.paragraphs])
    elif ext == '.pdf':
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    else:
        raise ValueError('Unsupported file format')


def _score_to_quality(score: float) -> str:
    """Определяет качество суммаризации по итоговому баллу."""
    if score >= 80:  return "отличное"
    if score >= 65:  return "хорошее"
    if score >= 45:  return "удовлетворительное"
    if score >= 25:  return "неудовлетворительное"
    return "опасное"