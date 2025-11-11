from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    USE_WHISPER_API: bool = False
    USE_LLAMA_API: bool = False
    USE_EXTERNAL_APIS: bool = False

    WHISPER_API_BASE: str | None = None
    WHISPER_API_KEY: str | None = None
    WHISPER_MODEL: str = "medium"

    LLAMA_API_BASE: str | None = None
    LLAMA_API_KEY: str | None = None
    LLAMA_MODEL: str = "llama2"


    HOST: str = "127.0.0.1"
    PORT: int = 8000


    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()