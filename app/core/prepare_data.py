"""
Скрипт подготовки данных: создаёт summaries.xlsx из твоих файлов.
Запускай один раз перед основным скриптом.

Структура входных данных (пример на основе text1_*.txt):
  - text1_source.txt       → исходный текст ЭМК №1
  - text1_model1.txt ... test_summary.txt → суммаризации
"""

import os
import re
from pathlib import Path

import pandas as pd
import chardet


# ─────────────────────────────────────────────
# Настройки — измени под свою папку с данными
# ─────────────────────────────────────────────

DATA_DIR    = Path("data/raw_data")   # папка с исходными .txt файлами
OUTPUT_PATH = Path("data/result_data/summaries.xlsx")

# Соответствие ID модели из ТЗ → файловому суффиксу
# ТЗ: ID 2, 11, 23, 89, 38, 19  → файлы model1..model6
MODEL_ID_MAP = {
    "model1": "2",
    "model2": "11",
    "model3": "23",
    "model4": "89",
    "model5": "38",
    "model6": "19",
}


def read_file(path: Path) -> str:
    raw = path.read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    return raw.decode(encoding, errors="ignore").strip()


def prepare_from_txt_files():
    """
    Собирает данные из .txt файлов в формате:
      text{N}_source.txt       → исходная ЭМК
      text{N}_model{M}.txt     → суммаризация

    Создаёт summaries.xlsx с колонками:
      emr_id | model_id | source_text | summary_text
    """
    rows = []

    # Ищем все source-файлы
    source_files = sorted(DATA_DIR.glob("*_source.txt"))

    if not source_files:
        print(f"⚠ Не найдено *_source.txt файлов в {DATA_DIR}")
        print("  Убедись что папка raw_data/ содержит твои файлы")
        return

    print(f"Найдено {len(source_files)} исходных ЭМК")

    for src_path in source_files:
        # Определяем номер ЭМК из имени файла (text1_, text2_, ...)
        match = re.match(r"text(\d+)_source\.txt", src_path.name)
        if not match:
            print(f"  Пропускаю: {src_path.name} (не подходит формат)")
            continue

        emr_num    = match.group(1)
        emr_id     = f"EMR_{emr_num.zfill(2)}"
        source_txt = read_file(src_path)

        print(f"  ЭМК {emr_id}: источник {len(source_txt)} символов")

        # Ищем суммаризации для этой ЭМК
        for file_suffix, model_id in MODEL_ID_MAP.items():
            summary_path = DATA_DIR / f"text{emr_num}_{file_suffix}.txt"
            if summary_path.exists():
                summary_txt = read_file(summary_path)
                rows.append({
                    "emr_id":       emr_id,
                    "model_id":     model_id,
                    "source_text":  source_txt,
                    "summary_text": summary_txt,
                })
                print(f"    + Модель {model_id}: {len(summary_txt)} символов")
            else:
                print(f"    ✗ Файл не найден: {summary_path.name}")

    if not rows:
        print("❌ Данные не собраны. Проверь структуру файлов.")
        return

    df = pd.DataFrame(rows, columns=["emr_id", "model_id",
                                      "source_text", "summary_text"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False)

    print(f"\n✅ Сохранено {len(df)} записей → {OUTPUT_PATH}")
    print(f"   ЭМК: {df['emr_id'].nunique()} | "
          f"Модели: {df['model_id'].nunique()}")


def prepare_from_excel(source_xlsx: str):
    """
    Альтернатива: если у тебя уже есть исходный Excel от организаторов,
    переформатируй его под нужную структуру.

    Организаторы предоставляют map.xlsx — адаптируй логику под его структуру.
    """
    df_raw = pd.read_excel(source_xlsx, dtype=str).fillna("")
    print(f"Загружен Excel: {df_raw.shape}, колонки: {list(df_raw.columns)}")
    # TODO: адаптировать под реальную структуру map.xlsx
    print("Адаптируй функцию под структуру своего Excel-файла от организаторов")


if __name__ == "__main__":
    print("=" * 50)
    print("Подготовка данных для оценки")
    print("=" * 50)

    # Вариант 1: из .txt файлов (как в примере из ТЗ)
    prepare_from_txt_files()

    # Вариант 2: раскомментируй если данные в Excel
    # prepare_from_excel("result_data/map.xlsx")
