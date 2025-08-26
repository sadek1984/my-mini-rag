from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY: str
    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    GENERATION_BACKEND: str = "COHERE"
    EMBEDDING_BACKEND: str = "COHERE"  # huggingface, openai, cohere

    OPENAI_API_KEY: str = None
    OPENAI_API_URL: str = None
    COHERE_API_KEY: str = None

    # إعدادات HuggingFace 🆕
    # HF_MODEL_NAME: str = "intfloat/multilingual-e5-small"  # النموذج الافتراضي
    # HF_CACHE_DIR: str = "./models"  # مجلد حفظ النماذج

    GENERATION_MODEL_ID_LITERAL: List[str] = None
    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None
    INPUT_DAFAULT_MAX_CHARACTERS: int = None
    GENERATION_DAFAULT_MAX_TOKENS: int = None
    GENERATION_DAFAULT_TEMPERATURE: float = None
    GEMINI_API_KEY: str = None

    VECTOR_DB_BACKEND_LITERAL: List[str] = None
    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str 
    VECTOR_DB_DISTANCE_METHOD: str = None
    VECTOR_DB_PGVEC_INDEX_THRESHOLD: int = 100

    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"

    POSTGRES_USERNAME: str 
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_MAIN_DATABASE: str

    sample_code: str
    sample_name: str
    pesticide_name: str
    limits: str
    device_reading: str
    result: str
    source_file: str 
    row_index: int

    MAIN_FOLDER_PATH: str
    MONTHS_IN_ORDER: List[str]


    model_config = SettingsConfigDict(env_file=".env")

def get_settings():
    return Settings()
