import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from root of recall-rag-service
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

PORT: int = int(os.getenv("PORT", "8000"))
MONGODB_URI: str = os.getenv("MONGODB_URI", "")
JWT_SECRET: str = os.getenv("JWT_SECRET", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
UPSTASH_REDIS_REST_URL: str = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN: str = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
