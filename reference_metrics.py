"""
reference_metrics.py — reference-based метрики качества суммаризации: ROUGE-L и
BERTScore. ПРИМЕНЯЮТСЯ ТОЛЬКО ТАМ, ГДЕ ЕСТЬ ЭТАЛОННЫЙ ТЕКСТ СУММАРИЗАЦИИ.

Обе метрики сравнивают кандидат-суммаризацию с ЭТАЛОНОМ (reference). На текущем
этапе эталон есть только у синтетического набора (строка 1 каждого пациента —
эталонная суммаризация); поэтому метрики вызываются исключительно из synthetic.py.
В продуктивном пути (исходник ЭМК -> суммаризация, scope="radiologist") эталонной
суммаризации НЕТ, и этот модуль там не используется (см. дизайн, Раздел 4.2:
«ROUGE-L — лексическое перекрытие с эталонным текстом; BERTScore — семантическое
сходство с эталоном»).

Состав:
  • ROUGE-L — ручная реализация через наибольшую общую подпоследовательность (LCS)
    токенов. БЕЗ внешних зависимостей (как и прочие rule-based метрики проекта).
  • BERTScore — через пакет `bert-score` (тянет torch+transformers). Импортируется
    ЛЕНИВО: ядро пайплайна остаётся лёгким, тяжёлая зависимость нужна только при
    фактическом запросе BERTScore (см. requirements-metrics.txt).
"""

from __future__ import annotations

import re

# Токенизация — тот же принцип, что в objective_layer (слова из букв/цифр, lower).
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Модель по умолчанию для BERTScore на русском: многоязычный BERT (покрывает
# русский, скачивается из общедоступного хранилища). Заменяемо параметром.
DEFAULT_BERTSCORE_MODEL = "bert-base-multilingual-cased"


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Длина наибольшей общей подпоследовательности двух списков токенов.
    DP за O(len(a)·len(b)) с памятью O(min(len(a),len(b)))."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else (prev[j] if prev[j] >= cur[j - 1] else cur[j - 1])
        prev = cur
    return prev[len(b)]


def rouge_l(reference: str, candidate: str, *, beta: float = 1.0) -> dict:
    """
    ROUGE-L между эталоном и кандидатом на уровне токенов.

    precision = LCS / |кандидат|, recall = LCS / |эталон|,
    F = (1+β²)·P·R / (R + β²·P)  (β=1 → обычная F1).
    Возвращает {"precision","recall","f1"} в [0,1]. Пустой текст → нули.
    """
    ref, cand = _tokens(reference), _tokens(candidate)
    if not ref or not cand:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(ref, cand)
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    b2 = beta * beta
    denom = recall + b2 * precision
    f1 = ((1 + b2) * precision * recall / denom) if denom > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def bertscore_available() -> bool:
    """Установлен ли пакет bert-score (без его импорта тяжёлых зависимостей до нужды)."""
    import importlib.util
    return importlib.util.find_spec("bert_score") is not None


def bertscore(references: list[str], candidates: list[str], *,
              model_type: str = DEFAULT_BERTSCORE_MODEL, lang: str = "ru",
              batch_size: int = 32) -> list[dict]:
    """
    BERTScore для списков (эталон, кандидат) — семантическое сходство с эталоном.

    Пакет `bert-score` импортируется ЛЕНИВО; при его отсутствии — понятная ошибка
    с инструкцией установки. Внутренний вызов bert_score.score(cands, refs, ...)
    (гипотеза — кандидат, эталон — reference). Возвращает список
    {"precision","recall","f1"} по парам в исходном порядке.
    """
    if len(references) != len(candidates):
        raise ValueError("references и candidates должны быть одной длины")
    try:
        from bert_score import score as _bert_score_fn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "BERTScore требует пакет bert-score (тянет torch+transformers). "
            "Установите: pip install -r requirements-metrics.txt"
        ) from exc
    P, R, F = _bert_score_fn(candidates, references, model_type=model_type, lang=lang,
                             batch_size=batch_size, rescale_with_baseline=False, verbose=False)
    return [{"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)}
            for p, r, f in zip(P.tolist(), R.tolist(), F.tolist())]


def _self_check() -> None:
    print("--- ROUGE-L ---")
    same = "пациентка 56 лет госпитализирована с инфарктом миокарда"
    assert rouge_l(same, same)["f1"] == 1.0, "идентичные тексты должны давать F1=1"
    partial = rouge_l("пациентка 56 лет, инфаркт миокарда",
                      "пациентка 46 лет, инфаркт миокарда")
    assert 0.0 < partial["f1"] < 1.0, partial
    drop = rouge_l("гемоглобин 115 эритроциты 3.4 лейкоциты 10",
                   "гемоглобин 115")
    assert drop["recall"] < drop["precision"], drop  # кандидат короче эталона
    assert rouge_l("", "что-то")["f1"] == 0.0
    print(f"  идентичные -> F1=1.0 ✓; частичное перекрытие -> {partial}; "
          f"усечение -> {drop}")

    print("--- BERTScore ---")
    if bertscore_available():
        res = bertscore(["пациентка 56 лет, инфаркт миокарда"],
                        ["пациентка 56 лет, инфаркт миокарда"])
        print(f"  идентичные -> {res[0]} (F1 близок к 1)")
    else:
        print("  пакет bert-score не установлен — пропуск "
              "(pip install -r requirements-metrics.txt)")
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")


if __name__ == "__main__":
    _self_check()
