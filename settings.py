from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    dagshub_user_token: str
    mlflow_tracking_uri: str
    mlflow_tracking_username: str
    mlflow_tracking_password: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
