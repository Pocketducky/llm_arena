"""Pydantic-схемы для запросов и ответов API."""

from pydantic import BaseModel, Field
from typing import Optional


# ── Запросы ──────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Текст ЭМК")
    summary: str = Field(..., min_length=1, description="Суммаризация ЭМК")


class ImproveRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Текст ЭМК")
    summary: str = Field(..., min_length=1, description="Текущая суммаризация")
    r1_results_full: list[dict] | str = Field(
        ..., description="Три оценки R1 (список dict или JSON-строка)"
    )


class SummarizeTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст для суммаризации")


# ── Ответы ───────────────────────────────────────────────────────

class SummarizeResponse(BaseModel):
    summary: str


class UploadResponse(BaseModel):
    task_id: str
    summary: str


class ExtractTextResponse(BaseModel):
    text: str


class ImproveResponse(BaseModel):
    improved_summary: str


class ScoreDetail(BaseModel):
    complaints: float = 0
    disease_history: float = 0
    comorbidities: float = 0
    habits: float = 0
    labs: float = 0
    imaging: float = 0
    penalties: float = 0


class FinalResult(BaseModel):
    final_score: float
    quality: str
    safety_flag: bool = False
    iodine_flag: bool = False
    verdict: str = ""
    criteria: Optional[ScoreDetail] = None
    all_hallucinations: list[str] = Field(default_factory=list)


class CoverageDetail(BaseModel):
    covered: list[str | dict] = Field(default_factory=list)
    missing: list[str | dict] = Field(default_factory=list)


class R1Result(BaseModel):
    complaints: float
    disease_history: float
    comorbidities: float
    habits: float
    labs: float
    imaging: float
    penalties: float
    final_score: float
    iodine_flag: bool = False
    safety_flag: bool = False
    safety_reason: str = ""
    hallucinations: list[str] = Field(default_factory=list)
    wrong_values: list[str] = Field(default_factory=list)
    irrelevant: list[str] = Field(default_factory=list)
    iodine_missing: bool = False
    wrong_focus: bool = False
    coverage_detail: dict[str, CoverageDetail] = Field(default_factory=dict)


class R2Result(BaseModel):
    complaints: float
    disease_history: float
    comorbidities: float
    habits: float
    labs: float
    imaging: float
    penalties: float
    final_score: float
    iodine_flag: bool = False
    safety_flag: bool = False
    hallucinations: list[str] = Field(default_factory=list)
    quality: str = "—"
    r2_reason: str = ""


class EvaluateResponse(BaseModel):
    final: FinalResult
    r1_results: list[R1Result]
    r2_results: list[R2Result]


class StatusResponse(BaseModel):
    status: str  # "running" | "completed" | "error"
    source: Optional[str] = None
    summary: Optional[str] = None
    result: Optional[dict] = None


class ErrorResponse(BaseModel):
    detail: str
