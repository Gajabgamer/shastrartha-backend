"""Settings from environment variables only (CLAUDE.md rule 3)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.split("#")[0].strip().lower() in {"1", "true", "yes", "on"}


def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).split("#")[0].strip()


def _list(name: str) -> list[str]:
    return [v.strip() for v in _str(name).split(",") if v.strip()]


def tts_voice(persona_id: str) -> str:
    """Voice model id for a persona, e.g. TTS_VOICE_MARCUS_AURELIUS."""
    return _str(f"TTS_VOICE_{persona_id.upper()}")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    llm_provider: str
    llm_model: str
    anthropic_api_key: str
    openai_api_key: str
    gemini_api_keys: list[str]
    llm_fallback_models: list[str]
    llm_max_tokens: int
    fish_api_key: str
    fish_tts_model: str
    fish_tts_latency: str
    tts_enabled: bool
    nli_model: str
    nli_enabled: bool
    nli_threshold: float
    max_verify_retries: int
    frontend_url: str
    demo_mode: bool
    token_delay_ms: int
    rounds_ranked: int
    rounds_demo: int
    pvp_round_seconds: int
    sabha_public_url: str
    sabha_final_seconds: int
    supabase_url: str
    supabase_service_key: str
    dense_enabled: bool
    chroma_dir: Path
    chroma_collection: str

    @property
    def llm_available(self) -> bool:
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key and self.llm_model)
        if self.llm_provider == "openai":
            return bool(self.openai_api_key and self.llm_model)
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_keys and self.llm_model)
        return False


@lru_cache
def get_settings() -> Settings:
    return Settings(
        data_dir=Path(_str("DATA_DIR", str(REPO_ROOT / "data"))),
        llm_provider=_str("LLM_PROVIDER", "anthropic").lower(),
        llm_model=_str("LLM_MODEL"),
        anthropic_api_key=_str("ANTHROPIC_API_KEY"),
        openai_api_key=_str("OPENAI_API_KEY"),
        gemini_api_keys=_list("GEMINI_API_KEYS") or _list("GEMINI_API_KEY"),
        llm_fallback_models=_list("LLM_FALLBACK_MODELS"),
        llm_max_tokens=int(_str("LLM_MAX_TOKENS", "700")),
        fish_api_key=_str("FISH_API_KEY"),
        fish_tts_model=_str("FISH_TTS_MODEL", "s2.1-pro"),
        fish_tts_latency=_str("FISH_TTS_LATENCY", "balanced"),  # normal | balanced | low
        tts_enabled=_str("TTS_PROVIDER").lower() == "fish" and bool(_str("FISH_API_KEY")),
        nli_model=_str("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
        nli_enabled=_bool("NLI_ENABLED", True),
        nli_threshold=float(_str("NLI_ENTAILMENT_THRESHOLD", "0.7")),
        max_verify_retries=int(_str("MAX_VERIFY_RETRIES", "2")),
        frontend_url=_str("FRONTEND_URL", "http://localhost:3000"),
        demo_mode=_bool("DEMO_MODE", True),
        token_delay_ms=int(_str("TOKEN_DELAY_MS", "35")),
        rounds_ranked=int(_str("ROUNDS_RANKED", "3")),
        rounds_demo=int(_str("ROUNDS_DEMO", "2")),
        pvp_round_seconds=int(_str("PVP_ROUND_SECONDS", "180")),
        # Base URL phones open for Live Sabha voting; default = this laptop's LAN address + the frontend port.
        sabha_public_url=_str("SABHA_PUBLIC_URL"),
        sabha_final_seconds=int(_str("SABHA_FINAL_SECONDS", "60")),
        supabase_url=_str("SUPABASE_URL"),
        supabase_service_key=_str("SUPABASE_SECRET_KEY") or _str("SUPABASE_SERVICE_ROLE_KEY"),
        dense_enabled=_bool("DENSE_RETRIEVAL", True),
        chroma_dir=Path(_str("CHROMA_DIR", str(REPO_ROOT / "data" / "chroma"))),
        chroma_collection=_str("CHROMA_COLLECTION", "shastrartha_corpus"),
    )
