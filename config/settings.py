from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    claude_api_key: str
    vector_db_type: str = "chroma"
    vector_db_path: str = "./data/vectors"
    log_level: str = "INFO"
    environment: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
