import json
import os
import re
import logging


# Файловый лог — DEBUG (все сырые ответы моделей)
# Консоль — INFO (только прогресс)
_file_handler = logging.FileHandler("evaluation.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

log = logging.getLogger("arena_llm")
log.setLevel(logging.DEBUG)
log.addHandler(_file_handler)
log.addHandler(_console_handler)


def repair_json(text: str) -> str:
    """Исправляет типичные ошибки JSON от LLM."""
    text = re.sub(r'\bTrue\b',  "true",  text)
    text = re.sub(r'\bFalse\b', "false", text)
    text = re.sub(r'\bNone\b',  "null",  text)
    text = re.sub(r"(?<!\\)'([^']*)'", r'"\1"', text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', text)
    text = re.sub(r",{2,}", ",", text)
    opens  = text.count("{") - text.count("}")
    opens2 = text.count("[") - text.count("]")
    if opens  > 0: text = text.rstrip() + "}" * opens
    if opens2 > 0: text = text.rstrip() + "]" * opens2
    return text


def extract_json(raw: str) -> dict:
    """Извлекает JSON из ответа модели."""
    text = raw.strip()
    if "<think>" in text:
        if "</think>" in text:
            text = text[text.find("</think>") + len("</think>"):].strip()
        else:
            text = text[text.find("<think>") + len("<think>"):]
            text = re.sub(r"^.*?(?=\{)", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Фигурные скобки не найдены. Ответ: {raw[:300]}")
    json_str = text[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(repair_json(json_str))
    except json.JSONDecodeError:
        pass
    depth, best_end = 0, -1
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
    raise ValueError(f"JSON не распарсить. Фрагмент: {json_str[:300]}")


def _warn_if_all_zeros(model_key: str, result: dict, raw: str):
    """
    Предупреждает если все числовые поля = 0.
    ВАЖНО: исключаем bool — False == 0 в Python, но это валидное значение!
    Проверяем только реальные числа (int/float, не bool).
    """
    # type(v) проверяет точный тип, не учитывая наследование bool от int
    nums = [v for v in result.values() if type(v) in (int, float)]
    if nums and all(v == 0 for v in nums) and len(nums) > 1:
        log.warning(f"      ⚠ {model_key} вернул ВСЕ нули — запускаю retry")
        log.warning(f"      Ответ: {raw[:200]}")
        raise ValueError(f"all_zeros: {model_key} вернул все нули")


def _make_minimal_prompt(desc: str, failed_raw: str) -> str:
    """Минималистичный промпт для retry3."""
    if "покрытие" in desc:
        skeleton = (
            '{"complaints":{"covered":[],"missing":[],"score":0,"max":15},'
            '"disease_history":{"covered":[],"missing":[],"score":0,"max":15,"iodine_allergy_noted":false},'
            '"comorbidities":{"covered":[],"missing":[],"score":0,"max":20},'
            '"habits":{"covered":[],"missing":[],"score":0,"max":5},'
            '"labs":{"covered":[],"missing":[],"score":0,"max":20},'
            '"imaging":{"covered":[],"missing":[],"score":0,"max":25}}'
        )
    elif "ошибки" in desc:
        skeleton = (
            '{"iodine_missing":false,"wrong_focus":false,'
            '"hallucinations":[],"wrong_values":[],"irrelevant":[],'
            '"penalties":0,"safety_flag":false,"safety_reason":""}'
        )
    elif "R2" in desc or "пересмотр" in desc:
        skeleton = (
            '{"complaints":0,"disease_history":0,"comorbidities":0,'
            '"habits":0,"labs":0,"imaging":0,"penalties":0,'
            '"iodine_flag":false,"safety_flag":false,'
            '"hallucinations":[],"quality":"удовлетворительное"}'
        )
    else:
        skeleton = (
            '{"complaints":0,"disease_history":0,"comorbidities":0,'
            '"habits":0,"labs":0,"imaging":0,"penalties":0,'
            '"final_score":0,"iodine_flag":false,"safety_flag":false,'
            '"hallucinations":[],"quality":"удовлетворительное","verdict":""}'
        )
    return (
        "Верни ТОЛЬКО JSON. Замени 0 на реальные числа оценки.\n"
        f"{skeleton}"
    )


def split_source(source: str) -> tuple[str, str]:
    """Делит ЭМК на клиническую и лабораторную части."""
    lab_markers = ["Результаты лабораторных", "Лабораторные исследования",
                   "Биохимическое исследование", "Клинический анализ крови",
                   "Компьютерная томография", "Ультразвуковое исследование",
                   "Выписной эпикриз"]
    split_pos = len(source)
    for marker in lab_markers:
        pos = source.find(marker)
        if 0 < pos < split_pos:
            split_pos = pos
    part1 = source[:split_pos].strip()
    part2 = source[split_pos:].strip()
    if not part2:
        part2 = source
    log.debug(f"    split_source: part1={len(part1)}, part2={len(part2)}")
    return part1, part2


def _score_to_quality(score: float) -> str:
    """Определяет качество по итоговому баллу."""
    if score >= 80: return "отличное"
    if score >= 65: return "хорошее"
    if score >= 45: return "удовлетворительное"
    if score >= 25: return "неудовлетворительное"
    return "опасное"


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    import io
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
        return ''.join(page.extract_text() for page in reader.pages)
    else:
        raise ValueError('Unsupported file format')


def _calc_coverage_score(cat_data: dict, max_score: float) -> float:
    """Считает балл покрытия по covered/missing. Код считает — не модель."""
    covered = cat_data.get("covered", [])
    missing = cat_data.get("missing", [])
    total   = len(covered) + len(missing)
    if total == 0:
        return max_score   # нет фактов → нечего проверять → максимум
    return round(len(covered) / total * max_score, 1)