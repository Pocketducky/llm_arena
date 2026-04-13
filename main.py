# main.py
########################################
# Запуск:
#  Разработка: uvicorn main:app --reload
#  Прод: uvicorn main:app --host 0.0.0.0 --port 8000
########################################
import json
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
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
from app.core.schemas import (
    EvaluateRequest,
    ImproveRequest,
    SummarizeTextRequest,
    SummarizeResponse,
    UploadResponse,
    ExtractTextResponse,
    ImproveResponse,
    StatusResponse,
    EvaluateResponse,
)

# ── Константы ────────────────────────────────────────────────────

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
TASK_TTL = 3600  # 1 час — после этого задача удаляется из памяти
TASK_CLEANUP_INTERVAL = 300  # проверять каждые 5 минут

# ── Инициализация ────────────────────────────────────────────────

app = FastAPI(
    title="Оценка суммаризаций ЭМК",
    description="API для суммаризации и оценки медицинских ЭМК через GigaChat",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # credentials=True с origins=["*"] невалиден по CORS-spec
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Хранилище задач ──────────────────────────────────────────────

tasks: Dict[str, dict] = {}


def _cleanup_expired_tasks():
    """Удаляет завершённые/ошибочные задачи старше TASK_TTL."""
    now = time.time()
    expired = [
        tid for tid, t in tasks.items()
        if t["status"] in ("completed", "error") and (now - t.get("created_at", 0)) > TASK_TTL
    ]
    for tid in expired:
        del tasks[tid]
    if expired:
        pass  # можно добавить логирование при необходимости


async def periodic_cleanup():
    while True:
        await asyncio.sleep(TASK_CLEANUP_INTERVAL)
        _cleanup_expired_tasks()


@app.on_event("startup")
async def startup():
    asyncio.create_task(periodic_cleanup())


# ── Эндпоинты ────────────────────────────────────────────────────

@app.post("/upload_for_evaluation", response_model=UploadResponse)
async def upload_for_evaluation(file: UploadFile = File(...)):
    """Загрузка документа → суммаризация → фоновая оценка."""
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"Файл слишком большой (макс. {MAX_FILE_SIZE // 1024 // 1024} MB)")

    try:
        source_text = extract_text_from_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Ошибка извлечения текста: {e}")

    try:
        summary = await summarize_with_gigachat(source_text)
    except Exception as e:
        raise HTTPException(500, f"Ошибка суммаризации: {e}")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "running",
        "source": source_text,
        "summary": summary,
        "result": None,
        "created_at": time.time(),
    }
    asyncio.create_task(run_evaluation_task(task_id, source_text, summary))
    return UploadResponse(task_id=task_id, summary=summary)


@app.post("/evaluate_text", response_model=EvaluateResponse)
async def evaluate_text(payload: EvaluateRequest):
    """Синхронная оценка суммаризации."""
    try:
        result = await evaluate_with_gigachat(payload.source, payload.summary)
        return result
    except Exception as e:
        raise HTTPException(500, f"Ошибка оценки: {e}")


@app.post("/extract_text", response_model=ExtractTextResponse)
async def extract_text(file: UploadFile = File(...)):
    """Извлекает текст из загруженного файла (txt, docx, pdf)."""
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"Файл слишком большой (макс. {MAX_FILE_SIZE // 1024 // 1024} MB)")
    try:
        text = extract_text_from_file(content, file.filename)
        return ExtractTextResponse(text=text)
    except Exception as e:
        raise HTTPException(400, f"Ошибка извлечения текста: {e}")


@app.post("/improve_summarization", response_model=ImproveResponse)
async def improve_summarization(payload: ImproveRequest):
    """Улучшение суммаризации на основе оценок R1."""
    try:
        r1_list = (
            payload.r1_results_full
            if isinstance(payload.r1_results_full, list)
            else json.loads(payload.r1_results_full)
        )
        if len(r1_list) != 3:
            raise HTTPException(400, "Ожидается ровно три оценки R1")
        improved = await improve_summarization_gigachat(
            payload.source, payload.summary, *r1_list
        )
        return ImproveResponse(improved_summary=improved)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка улучшения: {e}")


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(file: UploadFile = File(...)):
    """Суммаризация ЭМК из файла."""
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"Файл слишком большой (макс. {MAX_FILE_SIZE // 1024 // 1024} MB)")
    try:
        source_text = extract_text_from_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Ошибка извлечения текста: {e}")
    try:
        summary = await summarize_with_gigachat(source_text)
        return SummarizeResponse(summary=summary)
    except Exception as e:
        raise HTTPException(500, f"Ошибка суммаризации: {e}")


@app.post("/summarize_text", response_model=SummarizeResponse)
async def summarize_text(payload: SummarizeTextRequest):
    """Суммаризация переданного текста."""
    try:
        summary = await summarize_with_gigachat(payload.text)
        return SummarizeResponse(summary=summary)
    except Exception as e:
        raise HTTPException(500, f"Ошибка суммаризации: {e}")


@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """Получение статуса по запросу на оценку."""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    return StatusResponse(**tasks[task_id])


async def run_evaluation_task(task_id: str, source: str, summary: str):
    """Фоновый запуск оценки суммаризации ЭМК."""
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
