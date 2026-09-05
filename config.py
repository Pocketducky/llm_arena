"""
config.py — конфигурация бэкенда LLM, ролей судей и связи «роль → модель».

ИДЕЯ: код пайплайна (preprocessing, objective layer, judge, aggregation) никогда
не обращается к именам моделей или адресам серверов напрямую. Он работает с
АБСТРАКТНЫМИ РОЛЯМИ ("judge_1", "judge_2", "judge_3", "aggregator"), а конкретное
соответствие роль → модель и параметры подключения задаются здесь.

ВСЁ настраивается через ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ (см. .env.example) — оператору не
нужно править Python. Значения по умолчанию рассчитаны на запуск в НПКЦ через
vLLM. Для локальной разработки достаточно выставить EMR_LLM_BACKEND=ollama.

Поддерживаются два бэкенда:
  • vllm   — продакшн НПКЦ: OpenAI-совместимый API (qwen3.5-122b-fp8, qwen3.6-27b-fp8);
  • ollama — локальная разработка на лёгких моделях.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА .env (без внешних зависимостей)
# ══════════════════════════════════════════════════════════════════

def _load_dotenv() -> None:
    """Мини-загрузчик файла `.env` рядом с config.py: строки вида KEY=VALUE.
    Уже установленные переменные окружения имеют приоритет (setdefault), поэтому
    `.env` — это удобные значения по умолчанию, которые всегда можно переопределить
    через `export`. Комментарии (#) и пустые строки игнорируются."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()


# ══════════════════════════════════════════════════════════════════
# ЧТЕНИЕ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ══════════════════════════════════════════════════════════════════

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


# ══════════════════════════════════════════════════════════════════
# ВЫБОР БЭКЕНДА
# ══════════════════════════════════════════════════════════════════
# "vllm" — сервер(ы) vLLM в НПКЦ (OpenAI-совместимый API); "ollama" — локально.
LLM_BACKEND = _env("EMR_LLM_BACKEND", "vllm").strip().lower()


# ══════════════════════════════════════════════════════════════════
# ПОДКЛЮЧЕНИЕ К vLLM (продакшн НПКЦ)
# ══════════════════════════════════════════════════════════════════
# vLLM поднимает OpenAI-совместимый API. Как правило, ОДИН сервер обслуживает
# ОДНУ модель, поэтому крупная и средняя модели обычно живут на РАЗНЫХ портах.
# Базовый адрес — общий по умолчанию; при разных портах задайте отдельные
# endpoint'ы под каждую модель (EMR_VLLM_ENDPOINT_LARGE/_MEDIUM).

VLLM_BASE_URL = _env("EMR_VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = _env("EMR_VLLM_API_KEY", "EMPTY")  # vLLM по умолчанию не требует ключа

# Управление режимом «мышления» Qwen3 через chat_template_kwargs.enable_thinking.
# По умолчанию ВЫКЛЮЧЕНО: скрытые рассуждения съедают бюджет max_tokens и обрывают
# JSON (диагностировано на пилоте). Включайте только осознанно.
VLLM_ENABLE_THINKING = _env_bool("EMR_VLLM_ENABLE_THINKING", False)


# ══════════════════════════════════════════════════════════════════
# ПОДКЛЮЧЕНИЕ К OLLAMA (локальная разработка)
# ══════════════════════════════════════════════════════════════════

OLLAMA_HOST = _env("EMR_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
NUM_CTX = _env_int("EMR_OLLAMA_NUM_CTX", 16384)  # Ollama по умолчанию режет контекст до 2048


# ══════════════════════════════════════════════════════════════════
# ОБЩИЕ ПАРАМЕТРЫ ГЕНЕРАЦИИ И ПОВТОРОВ
# ══════════════════════════════════════════════════════════════════

TIMEOUT_SECONDS = _env_int("EMR_TIMEOUT", 600)       # таймаут одного запроса (крупные модели медленнее)
RETRY_SLEEP_SECONDS = _env_int("EMR_RETRY_SLEEP", 10)
REQUEST_PAUSE_SECONDS = _env_int("EMR_REQUEST_PAUSE", 2)  # пауза между запросами (щадим сервер)

# Температура 0.0 по умолчанию: вердикт должен быть воспроизводим. Вместе с
# EMR_SEED (ниже) два прогона на одних данных обязаны дать одинаковые категории.
DEFAULT_TEMPERATURE = _env_float("EMR_TEMPERATURE", 0.0)

# Seed генерации и перемешиваний (порядок судей в R2/R3 — см. judge.py). Раньше
# random.shuffle шёл от глобального ГСЧ без seed, из-за чего прогон был
# невоспроизводим, что обесценивало версионирование промптов/таксономии.
SEED = _env_int("EMR_SEED", 42)


# ══════════════════════════════════════════════════════════════════
# БЮДЖЕТЫ ГЕНЕРАЦИИ ПО РАУНДАМ
# ══════════════════════════════════════════════════════════════════
# ДИАГНОСТИКА (пилот НПКЦ): единый бюджет 1024-1536 токенов применялся и к
# ответу из двух булевых полей (блок C), и к финальной агрегации R3 по шести
# полным отчётам. 1536 токенов ≈ 3000-3800 символов русского JSON — полный
# отчёт A-E туда не помещается, ответ обрывался, и срабатывал салваж
# («JSON собран салважем верхнеуровневых полей; ответ был оборван»). Все
# зафиксированные в логе провалы валидации R1 — блок B на подкритериях B4/B5,
# то есть обрыв ровно на хвосте самого объёмного блока.
#
# Бюджет теперь задаётся ПО РАУНДУ и соразмерен ожидаемому ответу. Контекст
# vLLM в НПКЦ (>=131072) позволяет держать значения с запасом.
DEFAULT_NUM_PREDICT = _env_int("EMR_MAX_TOKENS", 4096)
# Потолок автоподъёма бюджета при обрыве ответа (llm_client.ask_json удваивает
# бюджет на следующей попытке, а не повторяет тот же обречённый запрос).
MAX_TOKENS_CEILING = _env_int("EMR_MAX_TOKENS_CEILING", 16384)

TOKENS_R1 = _env_int("EMR_MAX_TOKENS_R1", 3072)              # блоки A, C, D
TOKENS_R1_LARGE = _env_int("EMR_MAX_TOKENS_R1_LARGE", 4096)  # блоки B и E — вложенные списки
TOKENS_R2 = _env_int("EMR_MAX_TOKENS_R2", 8192)              # переиздание всего отчёта A-E
TOKENS_R3 = _env_int("EMR_MAX_TOKENS_R3", 6144)              # вердикт + сводка по 5 блокам
TOKENS_ENTITIES = _env_int("EMR_MAX_TOKENS_ENTITIES", 4096)  # извлечение сущностей / фильтр релевантности


# ══════════════════════════════════════════════════════════════════
# ПОРОГИ РЕШЕНИЙ И ПРОПУСКНАЯ СПОСОБНОСТЬ
# ══════════════════════════════════════════════════════════════════
# E1 засчитывается, только если судья привёл КОНКРЕТНЫЙ фрагмент суммаризации
# как опасный И этот фрагмент действительно в ней есть (проверяет код, не LLM).
# Отключение (=0) возвращает прежнее поведение «любой поднятый флаг = E1» —
# нужно для A/B-сравнения на синтетике.
E1_REQUIRE_CITATION = _env_bool("EMR_E1_REQUIRE_CITATION", True)

# Сколько блоков таксономии могут остаться БЕЗ ДАННЫХ (сбой JSON), прежде чем
# оценка признаётся неполной. Раньше «нет данных» приравнивалось к «провален»,
# и три оборванных ответа давали клиническое «Неприемлемо».
MAX_NODATA_BLOCKS = _env_int("EMR_MAX_NODATA_BLOCKS", 1)

# Число пар, обрабатываемых параллельно (внутри пары порядок R1->R2->R3 сохраняется).
CONCURRENCY = _env_int("EMR_CONCURRENCY", 4)

# Извлечение сущностей: раньше текст жёстко резался на первых 6000 символах,
# из-за чего на прод-корпусе (медиана исходника 21107 симв.) терялось ~72 %
# медкарты, а entity_recall шлюза считался по первой трети. Длинные тексты
# режутся на перекрывающиеся куски и объединяются, а не обрезаются.
ENTITY_MAX_CHARS = _env_int("EMR_ENTITY_MAX_CHARS", 40000)
ENTITY_CHUNK_CHARS = _env_int("EMR_ENTITY_CHUNK_CHARS", 12000)
ENTITY_CHUNK_OVERLAP = _env_int("EMR_ENTITY_CHUNK_OVERLAP", 500)


# ══════════════════════════════════════════════════════════════════
# РОЛИ
# ══════════════════════════════════════════════════════════════════
# judge_1..judge_3 — независимые судьи-эксперты (Round 1), затем участники
#                    cross-peer-review (Round 2);
# aggregator       — финальный арбитр (Round 3): синтезирует вердикт и обязан
#                    явно проверить stop-rule E1.

JUDGE_ROLES = ("judge_1", "judge_2", "judge_3")
AGGREGATOR_ROLE = "aggregator"
ALL_ROLES = (*JUDGE_ROLES, AGGREGATOR_ROLE)


# ══════════════════════════════════════════════════════════════════
# ИМЕНА МОДЕЛЕЙ НПКЦ
# ══════════════════════════════════════════════════════════════════
# ВАЖНО: строки должны совпадать с тем, под каким именем модель обслуживается
# в vLLM (флаг --served-model-name; проверяется через GET /v1/models,
# см. check_environment.py). При расхождении имён vLLM вернёт 404.

NPKC_MODEL_LARGE = _env("EMR_MODEL_LARGE", "qwen3.5-122b-fp8")
NPKC_MODEL_MEDIUM = _env("EMR_MODEL_MEDIUM", "qwen3.6-27b-fp8")

# Карта «модель → endpoint». Заполняется, только если модели на разных адресах
# (типичный случай: одна модель на один сервер vLLM). Иначе используется
# VLLM_BASE_URL для всех. См. vllm_endpoint_for().
VLLM_MODEL_ENDPOINTS: dict[str, str] = {}
if os.getenv("EMR_VLLM_ENDPOINT_LARGE"):
    VLLM_MODEL_ENDPOINTS[NPKC_MODEL_LARGE] = os.environ["EMR_VLLM_ENDPOINT_LARGE"]
if os.getenv("EMR_VLLM_ENDPOINT_MEDIUM"):
    VLLM_MODEL_ENDPOINTS[NPKC_MODEL_MEDIUM] = os.environ["EMR_VLLM_ENDPOINT_MEDIUM"]


def vllm_endpoint_for(model_name: str) -> str:
    """Адрес vLLM для конкретной модели: индивидуальный endpoint, если задан,
    иначе общий VLLM_BASE_URL."""
    return VLLM_MODEL_ENDPOINTS.get(model_name, VLLM_BASE_URL)


def vllm_endpoints() -> list[str]:
    """Уникальные адреса vLLM, задействованные активным набором моделей —
    для проверки связи в check_environment.py."""
    seen, out = set(), []
    for url in (*VLLM_MODEL_ENDPOINTS.values(), VLLM_BASE_URL):
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


# ══════════════════════════════════════════════════════════════════
# ПРОФИЛИ — соответствие «роль → модель»
# ══════════════════════════════════════════════════════════════════

# --- НПКЦ (продакшн, vLLM) ---
# Два сильных судьи + один поменьше: даёт архитектурное разнообразие (декорреляция
# ошибок судей) и экономит ресурс; сильный арбитр — на безопасность (стоп-правило E1).
NPKC_PROFILE: dict[str, str] = {
    "judge_1":    NPKC_MODEL_LARGE,
    "judge_2":    NPKC_MODEL_LARGE,
    "judge_3":    NPKC_MODEL_MEDIUM,
    "aggregator": NPKC_MODEL_LARGE,
}

# --- Целевая конфигурация из «Дизайна исследования» (раздел 4.3) ---
TARGET_PROFILE: dict[str, str] = {
    "judge_1":    "deepseek-r1:70b",
    "judge_2":    "deepseek-r1:70b",
    "judge_3":    "deepseek-r1:70b",
    "aggregator": "qwen3-next:80b",
}

# --- Пилотные профили для локальной разработки через Ollama ---
PILOT_PROFILE: dict[str, str] = {
    "judge_1": "qwen3:8b", "judge_2": "qwen3:8b",
    "judge_3": "qwen3:8b", "aggregator": "qwen3:8b",
}
PILOT_PROFILE_DIVERSE: dict[str, str] = {
    "judge_1": "qwen3:8b", "judge_2": "llama3.1:8b",
    "judge_3": "mistral:7b", "aggregator": "qwen3:8b",
}
# qwen2.5 — не «мыслящая» модель: весь бюджет генерации идёт на ответ, а не на
# скрытый <think>, поэтому крупные промпты R2/R3 укладываются в таймаут.
PILOT_FAST_PROFILE: dict[str, str] = {
    "judge_1": "qwen2.5:7b", "judge_2": "qwen2.5:7b",
    "judge_3": "qwen2.5:7b", "aggregator": "qwen2.5:7b",
}
PILOT_FAST_14B_PROFILE: dict[str, str] = {
    "judge_1": "qwen2.5:14b", "judge_2": "qwen2.5:14b",
    "judge_3": "qwen2.5:14b", "aggregator": "qwen2.5:14b",
}

PROFILES: dict[str, dict[str, str]] = {
    "npkc":           NPKC_PROFILE,
    "target":         TARGET_PROFILE,
    "pilot":          PILOT_PROFILE,
    "pilot_diverse":  PILOT_PROFILE_DIVERSE,
    "pilot_fast":     PILOT_FAST_PROFILE,
    "pilot_fast_14b": PILOT_FAST_14B_PROFILE,
}

# Профиль по умолчанию (переопределяется EMR_PROFILE или флагом --profile).
ACTIVE_PROFILE = _env("EMR_PROFILE", "npkc")


# ══════════════════════════════════════════════════════════════════
# СУЖЕНИЕ ЗАДАЧИ ОЦЕНКИ (scope) — ЧТО ИМЕННО СУММАРИЗИРУЮТ МОДЕЛИ
# ══════════════════════════════════════════════════════════════════
# Датасет data/summaries.xlsx — целевые выжимки под адресата (врач-рентгенолог,
# подготовка к описанию КТ ОБП), а не полные сводки. При scope="radiologist"
# судьи и gate не штрафуют законные пропуски сведений других профилей.
# Подробности — в gate._RELEVANCE_FILTER_PROMPTS и judge._SCOPE_DESCRIPTIONS.
DATASET_SCOPE: str | None = _env("EMR_DATASET_SCOPE", "radiologist") or None


# ══════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Profile:
    """Снимок профиля: имя + словарь роль → модель."""
    name: str
    roles: dict[str, str]

    def model_for(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError:
            raise KeyError(
                f"В профиле '{self.name}' не задана модель для роли '{role}'. "
                f"Доступные роли профиля: {sorted(self.roles)}"
            )

    def unique_models(self) -> list[str]:
        """Уникальные имена моделей (для проверки доступности / прогрева)."""
        seen, out = set(), []
        for name in self.roles.values():
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out


def get_profile(profile_name: str | None = None) -> Profile:
    """
    Возвращает снимок профиля по имени (по умолчанию — ACTIVE_PROFILE).

        profile = get_profile()            # активный профиль
        profile = get_profile("npkc")      # явно выбранный
        model_name = profile.model_for("judge_1")
    """
    name = profile_name or ACTIVE_PROFILE
    if name not in PROFILES:
        raise KeyError(
            f"Неизвестный профиль '{name}'. Доступные профили: {sorted(PROFILES)}"
        )
    return Profile(name=name, roles=dict(PROFILES[name]))
