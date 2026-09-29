from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project structure:
# rag_part/
# └── backend/
#     ├── .env
#     └── app/
#         └── config.py

BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    """
    Application configuration.

    Environment variables are loaded from backend/.env
    and can also be supplied directly by the deployment environment.
    """

    gemini_api_key: str

    gemini_embedding_model: str = "gemini-embedding-001"
    
    cohere_api_key: str

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached settings instance.
    """

    return Settings()


settings = get_settings()