import logging
import json
import asyncio
from sqlite import insert_memory, get_recent_episodic
from small_model import call_small_model
from embedder import get_embedding
import hnsw

logger = logging.getLogger("Kven.Memory")

CONSOLIDATION_PROMPT = """
SYSTEM: You are a memory consolidator.
TASK: You are given a list of raw conversation logs (Episodic Memories).
ACTION: Identify patterns, rules, code snippets, and decisions made during these interactions. Convert them into high-level, reusable semantic facts.
OUTPUT: A JSON list of strings. No markdown, no explanation.
CONTEXT:
{episodic_logs}
"""

async def consolidate():
    logger.info("[CONSOLIDATION] Starting consolidation process...")
    
    episodic_logs = await get_recent_episodic(limit=10)
    if not episodic_logs:
        logger.info("[CONSOLIDATION] No episodic logs to consolidate.")
        return

    logs_text = "\n".join([f"- {log}" for log in episodic_logs])
    prompt = CONSOLIDATION_PROMPT.format(episodic_logs=logs_text)
    
    raw_response = await call_small_model(prompt, grammar=None, max_tokens=1024)
    
    cleaned = raw_response.strip().strip("```json").strip("```").strip()
    try:
        items = json.loads(cleaned)
        if not isinstance(items, list):
            items = [items]
            
        saved = 0
        for item in items:
            content = str(item)[:2500]
            new_id = await insert_memory(
                content=content,
                kind="semantic",
                importance=0.8,
                tags="[]",
                decay_rate=0.98,
                table_name="semantic_memory",
                epistemic_type="Inference",
                source="model_inference"
            )
            saved += 1
            
            # HNSW Integration
            if new_id:
                try:
                    vector = await get_embedding(content)
                    await asyncio.to_thread(hnsw.add_to_hnsw, [new_id], [vector])
                except Exception as e:
                    logger.error(f"[HNSW] Failed to add semantic memory {new_id}: {e}")
                    
        logger.info(f"[CONSOLIDATION] Consolidated {saved} new semantic memories.")
    except Exception as e:
        logger.error(f"[CONSOLIDATION] Error: {e}")
