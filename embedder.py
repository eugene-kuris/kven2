# /opt/kven2/agent/storage/embedder.py
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
_embed_model = None
_model_lock = threading.Lock()

_EMBEDDING_CPU_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed_cpu")

def init_embedder() -> SentenceTransformer:
    global _embed_model
    with _model_lock:
        if _embed_model is None:
            logger.info(f"[RAG] Loading embedding model: {EMBEDDING_MODEL_NAME}...")
            try:
                _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info("[RAG] Embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"[RAG] Failed to initialize: {e}")
                raise
    return _embed_model

def get_embedding_sync(text, normalize_embeddings: bool = True) -> list:
    model = init_embedder()
    try:
        return model.encode(text, normalize_embeddings=normalize_embeddings).tolist()
    except Exception as e:
        logger.error(f"[RAG] Embedding generation failed: {e}")
        raise

async def get_embedding(text: str, normalize_embeddings: bool = True) -> list:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _EMBEDDING_CPU_EXECUTOR,
        get_embedding_sync,
        text,
        normalize_embeddings
    )
