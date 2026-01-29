from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()  

class Settings(BaseSettings):
    DEEPSEEK_API_KEY : str
    MODEL_NAME : str
    MODEL_BASE_URL: str
    GEMINI_API_KEY : str
    GEMINI_MODEL_NAME : str
    GROK_API_KEY : str
    GROK_MODEL : str


class Config:
        env_file = ".env"

env = Settings()
