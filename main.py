# main.py
########################################
# Запуск:
#  Разработка: uvicorn main:app --reload
#  Прод: 1) uvicorn main:app --host 0.0.0.0 --port 8000
#        2) В другом терминале: nport http 8000 --subdomain {название} ИЛИ ssh -R 80:localhost:8000 serveo.net
#
# Если по локальной сети: ifconfig
# Затем найти в выводе: en0 и ы нем inet (вида 10.246.24.106)
# Затем: uvicorn main:app --host 0.0.0.0 --port 8000
# И заходим: http://<IP_вашего_компьютера>:8000
########################################
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.params import Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uuid
import asyncio
from typing import Dict

from app.giga_core.giga_evaluate import evaluate_with_gigachat
from app.giga_core.giga_improve import improve_summarization_gigachat
from app.giga_core.giga_summarize import summarize_with_gigachat
from app.core.utils import extract_text_from_file

# Инициализация
app = FastAPI(title="Оценка суммаризаций ЭМК")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для теста можно "*", но в проде лучше ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def evaluate_text(payload: dict = Body(...)):
    source = payload.get("source")
    summary = payload.get("summary")
    if not source or not summary:
        raise HTTPException(400, "source and summary required")
    try:
        result = await evaluate_with_gigachat(source, summary)
        return result
    except Exception as e:
        raise HTTPException(500, f"Ошибка оценки: {str(e)}")


@app.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    """
    Извлекает текст из загруженного файла (txt, docx, pdf) и возвращает его.
    """
    content = await file.read()
    try:
        text = extract_text_from_file(content, file.filename)
        return {"text": text}
    except Exception as e:
        raise HTTPException(400, f"Ошибка извлечения текста: {str(e)}")


@app.post("/improve_summarization")
async def improve_summarization(payload: dict = Body(...)):
    source = payload.get("source")
    summary = payload.get("summary")
    r1_results_full = payload.get("r1_results_full")
    if not source or not summary or r1_results_full is None:
        raise HTTPException(400, "Missing fields")
    try:
        # r1_results_full может быть списком или строкой – приводим к списку
        r1_list = r1_results_full if isinstance(r1_results_full, list) else json.loads(r1_results_full)
        if len(r1_list) != 3:
            raise ValueError("Ожидается ровно три оценки R1")
        improved = await improve_summarization_gigachat(source, summary, *r1_list)
        return {"improved_summary": improved}
    except Exception as e:
        raise HTTPException(500, f"Ошибка улучшения: {str(e)}")


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


@app.post("/summarize_text")
async def summarize_text(text: str = Body(..., embed=True)):
    if not text:
        raise HTTPException(400, "No text provided")
    summary = await summarize_with_gigachat(text)
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