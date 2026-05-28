import os
import json
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from cryptography.fernet import Fernet

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    SECRET_KEY: str = "949fdfefde5d1182181d331448b11116c278bc0f82df973950ef777b73273e97"
    ENCRYPTION_KEY: str = ""
    DATABASE_URL: str = "sqlite:///./medrag.db"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    UPLOAD_DIR: str = "./uploads"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("ENCRYPTION_KEY", mode="before")
    @classmethod
    def validate_or_generate_encryption_key(cls, value):
        if not value:
            # Generate a key if it doesn't exist
            return Fernet.generate_key().decode()
        # Verify if it is a valid Fernet key
        try:
            Fernet(value.encode())
            return value
        except Exception:
            # Re-generate if invalid
            return Fernet.generate_key().decode()

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
