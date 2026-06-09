import logging
import sys
import httpx
import atexit
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Импорт модулей проекта
import sqlite
import decay
import routes
from config import settings
import embedder
import hnsw

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Initializing Kven2 Gateway...")
    
    # Инициализация БД
    await sqlite.init_db()
    
    # Инициализация Embedder и HNSW
    try:
        embedder.init_embedder()
        logger.info("[EMBEDDER] Embedder initialized.")
        hnsw.init_hnsw()
        logger.info("[HNSW] HNSW index initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")

    logger.info("🧹 Running initial memory hygiene check...")
    await decay.run_decay()
    
    # Регистрируем сохранение HNSW на случайной остановке
    def save_hnsw_on_exit():
        hnsw.save_hnsw()
    atexit.register(save_hnsw_on_exit)
    
    models_to_check = [
        ("Small Model (30B)", settings.SMALL_MODEL_URL),
        ("Large Model (35B)", settings.LLM_BACKEND_URL)
    ]
    
    for name, url in models_to_check:
        base_url = url.rstrip('/')
        try:
            logger.info(f"🔍 Checking {name} at {base_url}/models ...")
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.get(f"{base_url}/models", timeout=5.0)
                logger.info(f"✅ {name} is online and healthy.")
        except Exception as e:
            logger.critical(f"❌ {name} FAILED connection check! Shutting down.")
            logger.critical(f"Error details: {e}")
            raise RuntimeError(f"Critical Backend Failure: {name} unreachable. Aborting startup.") from e
    
    yield
    # При graceful shutdown save_hnsw_on_exit вызовется через atexit

app = FastAPI(lifespan=lifespan)

# Включаем роутер напрямую
app.include_router(routes.router, prefix="/v1") 

@app.get("/models")
async def root_models():
    return await routes.router.routes[0].endpoint()

@app.get("/slots")
async def root_slots():
    return [{"id": "default", "name": "Kven Gateway", "object": "slot"}]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PROXY_PORT)
