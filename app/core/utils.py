import json
import os
import re


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
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback: detect encoding via chardet
            import chardet
            detected = chardet.detect(file_bytes)
            encoding = detected.get('encoding') or 'cp1251'
            return file_bytes.decode(encoding, errors='replace')
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