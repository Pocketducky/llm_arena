"""
config.py — конфигурация ролей LLM-судей и связь «роль → модель Ollama».

ИДЕЯ: код пайплайна оценки (preprocessing, objective layer, judge, aggregation)
никогда не обращается к именам моделей напрямую. Он работает с АБСТРАКТНЫМИ
РОЛЯМИ ("judge_1", "judge_2", "judge_3", "aggregator"), а конкретное соответствие
роль → модель задаётся здесь, в виде именованных ПРОФИЛЕЙ.

Чтобы переключиться на другой набор моделей — смените ACTIVE_PROFILE
(или передайте имя профиля явно при запуске) и/или отредактируйте словарь
профиля. Изменения в коде пайплайна не требуются.

Зафиксированная целевая конфигурация (см. «Дизайн исследования», раздел 4.3):
    3 × DeepSeek-R1 70B (независимые судьи) + Qwen3-Next-80B (финальный агрегатор)
Она прописана как профиль "target" и будет использоваться, когда появится
GPU-инфраструктура. До этого момента работаем на профиле "pilot" — лёгких
моделях, развёрнутых локально через Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass

# ══════════════════════════════════════════════════════════════════
# ПОДКЛЮЧЕНИЕ К OLLAMA
# ══════════════════════════════════════════════════════════════════

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"

NUM_CTX = 16384          # без явного указания Ollama использует 2048 — обрезает ЭМК
TIMEOUT_SECONDS = 300    # таймаут одного запроса к модели
RETRY_SLEEP_SECONDS = 10
REQUEST_PAUSE_SECONDS = 2   # пауза между последовательными запросами (щадим GPU/CPU)

DEFAULT_TEMPERATURE = 0.1
DEFAULT_NUM_PREDICT = 1024

# ══════════════════════════════════════════════════════════════════
# РОЛИ
# ══════════════════════════════════════════════════════════════════
# judge_1..judge_3 — независимые судьи-эксперты (Round 1), затем
#                    участники cross-peer-review (Round 2)
# aggregator       — финальный арбитр (Round 3), синтезирует вердикт
#                    и обязан явно проверить stop-rule E1

JUDGE_ROLES = ("judge_1", "judge_2", "judge_3")
AGGREGATOR_ROLE = "aggregator"
ALL_ROLES = (*JUDGE_ROLES, AGGREGATOR_ROLE)


# ══════════════════════════════════════════════════════════════════
# ПРОФИЛИ — конкретные модели Ollama для каждой роли
# ══════════════════════════════════════════════════════════════════

# --- Целевая (production) конфигурация согласно дизайну исследования ---
# Разворачивается на GPU-инфраструктуре (2×H100 80GB, последовательный инференс,
# INT4-квантизация). Имена моделей соответствуют тегам, под которыми они,
# предположительно, будут опубликованы в Ollama registry / собраны из GGUF;
# при необходимости поправьте под реальные локальные теги (`ollama list`).
TARGET_PROFILE: dict[str, str] = {
    "judge_1":   "deepseek-r1:70b",
    "judge_2":   "deepseek-r1:70b",
    "judge_3":   "deepseek-r1:70b",
    "aggregator": "qwen3-next:80b",
}

# --- Пилотная конфигурация — то, что реально стоит локально прямо сейчас ---
# `ollama list` на момент написания показывает только qwen3:8b.
# Используем её для всех ролей. Это не идеальный ансамбль (нет разнообразия
# точек зрения между судьями), но пайплайн остаётся полностью рабочим
# и тестируемым на лёгком железе уже сегодня — диверсификацию добавим,
# когда будут подтянуты другие модели (см. PILOT_PROFILE_DIVERSE ниже).
PILOT_PROFILE: dict[str, str] = {
    "judge_1":   "qwen3:8b",
    "judge_2":   "qwen3:8b",
    "judge_3":   "qwen3:8b",
    "aggregator": "qwen3:8b",
}

# --- Расширенный пилотный профиль — разные модели на разные роли ---
# Активируйте после того, как подтянете дополнительные лёгкие модели:
#   ollama pull llama3.1:8b
#   ollama pull mistral:7b
# Разнообразие моделей-судей снижает корреляцию их ошибок (одна и та же
# модель, оценивающая сама себя трижды, скорее согласится сама с собой,
# чем независимые архитектуры — это ослабляет смысл cross-peer-review).
PILOT_PROFILE_DIVERSE: dict[str, str] = {
    "judge_1":   "qwen3:8b",
    "judge_2":   "llama3.1:8b",
    "judge_3":   "mistral:7b",
    "aggregator": "qwen3:8b",
}

PROFILES: dict[str, dict[str, str]] = {
    "pilot":         PILOT_PROFILE,
    "pilot_diverse": PILOT_PROFILE_DIVERSE,
    "target":        TARGET_PROFILE,
}

# Профиль, используемый по умолчанию, если не указан явно
ACTIVE_PROFILE = "pilot"


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
    Возвращает снимок профиля по имени.
    Если имя не задано — берётся ACTIVE_PROFILE.

    Использование в коде пайплайна:
        profile = get_profile()                  # активный профиль
        profile = get_profile("target")          # явно выбранный
        model_name = profile.model_for("judge_1")
    """
    name = profile_name or ACTIVE_PROFILE
    if name not in PROFILES:
        raise KeyError(
            f"Неизвестный профиль '{name}'. Доступные профили: {sorted(PROFILES)}"
        )
    return Profile(name=name, roles=dict(PROFILES[name]))
