import os

class Settings:
    # KVEN2 HARD-CODED CONFIG
    # No environment variables used. Fully isolated.

    OWUI_API_KEY = "owui-admin-secret"
    LLM_BACKEND_URL = "http://192.168.143.193:8080/v1"
    SMALL_MODEL_URL = "http://192.168.143.191:8080/v1"
    MAIN_MODEL = "Qwen_Qwen3.6-35B-A3B-Q8_0.gguf"
    PROXY_PORT = 10000
    LOG_LEVEL = "info"
    MEMORY_DIR = "/opt/kven2/data/kven2"
    DB_PATH = os.path.join(MEMORY_DIR, "memory.db")
    INDEX_PATH = os.path.join(MEMORY_DIR, "hnsw_index.bin")

settings = Settings()
