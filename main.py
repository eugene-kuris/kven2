import sys
import os
import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config import settings

# --- НАСТРОЙКА ЛОГИРОВАНИЯ (Этап A: Instrumentation) ---
# Создаем директорию для логов, если её нет
LOG_DIR = "/agent/data/kven2"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "kven2.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Вывод в консоль
        logging.FileHandler(LOG_FILE, encoding='utf-8')  # Запись в файл
    ]
)
logger = logging.getLogger(__name__)

# Инициализация компонентов
import sqlite
import embedder
import hnsw
import decay

app = FastAPI(title="Kven II Proxy")

# Добавляем CORS для удобства отладки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Kven II Starting up...")
    
    # 1. Инициализация БД
    await sqlite.init_db()
    logger.info("[MAIN] Database initialized.")

    # 2. Инициализация Embedder и HNSW
    try:
        embedder.init_embedder()
        logger.info("[MAIN] Embedder initialized.")
        hnsw.init_hnsw()
        logger.info("[MAIN] HNSW index initialized.")
    except Exception as e:
        logger.error(f"[MAIN] Failed to initialize RAG components: {e}")
        sys.exit(1)

    # 3. Первичная гигиена памяти
    logger.info("[MAIN] Running initial memory hygiene check...")
    await decay.run_decay()
    logger.info("[MAIN] Startup complete.")

@app.get("/models")
async def models():
    # Заглушка для совместимости
    return {"data": [{"id": settings.MAIN_MODEL}]}

@app.get("/slots")
async def slots():
    return []

# Подключение роутеров
from routes import router
app.include_router(router, prefix="/v1")

if __name__ == "__main__":
    logger.info(f"Starting server on port {settings.PROXY_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=settings.PROXY_PORT)