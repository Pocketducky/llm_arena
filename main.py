# main.py
########################################
# Запуск:                              #
#  ''' uvicorn main:app --reload '''   #
########################################

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uuid
import asyncio
from typing import Dict

from app.giga_core.giga_evaluate import evaluate_with_gigachat
from app.giga_core.giga_summarize import summarize_with_gigachat
from app.core.utils import extract_text_from_file

# Инициализация
app = FastAPI(title="Оценка суммаризаций ЭМК")

# Подключаем статику
app.mount("/static", StaticFiles(directory="static"), name="static")

# Хранилище задач
tasks: Dict[str, dict] = {}

# Эндпоинты
@app.post("/upload_for_evaluation")
async def upload_for_evaluation(file: UploadFile = File(...)):
    """
    Загрузка документа -> Суммаризация -> Оценка
    :param file: загруженный файл в формате txt, docx, pdf
    :return: суммаризация ЭМК, запрос на оценку
    """
    content = await file.read()
    try:
        source_text = extract_text_from_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Ошибка извлечения текста: {e}")

    summary = await summarize_with_gigachat(source_text)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "running",
        "source": source_text,
        "summary": summary,
        "result": None
    }
    asyncio.create_task(run_evaluation_task(task_id, source_text, summary))
    return {"task_id": task_id, "summary": summary}


@app.post("/evaluate_text")
async def evaluate_text(source: str = Form(...), summary: str = Form(...)):
    """
    Оценка суммаризации ЭМК

    :param source: ЭМК
    :param summary: Суммаризация ЭМК
    :return: Метрики оценки
    """
    try:
        result = await evaluate_with_gigachat(source, summary)
        return result
    except Exception as e:
        raise HTTPException(500, f"Ошибка оценки: {str(e)}")


@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    """
    Суммаризация ЭМК
    :param file: загруженный файл в формате txt, docx, pdf
    :return: суммаризация ЭМК
    """
    content = await file.read()
    try:
        source_text = extract_text_from_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Ошибка извлечения текста: {e}")
    summary = await summarize_with_gigachat(source_text)
    return {"summary": summary}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Получение статуса по запросу на оценку
    :param task_id: номер запроса на оценку
    :return:
    """
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    return tasks[task_id]


async def run_evaluation_task(task_id: str, source: str, summary: str):
    """
    Запуск оценки суммаризвции ЭМК c отслеживанием прогресса
    :param task_id: номер запроса на оценку
    :param source: ЭМК
    :param summary: Суммаризация ЭМК
    :return:
    """
    try:
        result = await evaluate_with_gigachat(source, summary)
        tasks[task_id]["result"] = result
        tasks[task_id]["status"] = "completed"
    except Exception as e:
        tasks[task_id]["result"] = {"error": str(e)}
        tasks[task_id]["status"] = "error"


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()