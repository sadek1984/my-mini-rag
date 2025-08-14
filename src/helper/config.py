from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY: str
    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int
     
    MONGODB_URL: str
    MONGODB_DATABASE: str

    GENERATION_BACKEND: str = "COHERE"
    EMBEDDING_BACKEND: str = "COHERE"  # huggingface, openai, cohere

    OPENAI_API_KEY: str = None
    OPENAI_API_URL: str = None
    COHERE_API_KEY: str = None

    # إعدادات HuggingFace 🆕
    # HF_MODEL_NAME: str = "intfloat/multilingual-e5-small"  # النموذج الافتراضي
    # HF_CACHE_DIR: str = "./models"  # مجلد حفظ النماذج

    GENERATION_MODEL_ID_LITERAL: list[str] = None
    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None
    INPUT_DAFAULT_MAX_CHARACTERS: int = None
    GENERATION_DAFAULT_MAX_TOKENS: int = None
    GENERATION_DAFAULT_TEMPERATURE: float = None

    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str 
    VECTOR_DB_DISTANCE_METHOD: str = None

    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"


    model_config = SettingsConfigDict(env_file=".env")

def get_settings():
    return Settings()
