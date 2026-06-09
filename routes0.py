#routes.py
import json
import httpx
import logging
import asyncio
from fastapi import Request, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse, Response

# Исправлены импорты на прямые названия файлов
from sqlite import (
    load_active_state, 
    save_active_state, 
    save_history_snapshot,
    get_semantic_context,
    get_project_context
)
# ИСПРАВЛЕНО: Импорт обновлен на новое имя файла kven2_profile
from kven2_profile import load_agent_profile
import kven2_time as sys_time 
from write_path import process_episodic, strip_reasoning

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

BASE_BACKEND = settings.LLM_BACKEND_URL

@router.get("/models")
async def list_models():
    try:
        logger.info("[ROUTE] >>> Fetching models list from backend")
        async with httpx.AsyncClient(timeout=1000.0) as client:
            r = await client.get(f"{BASE_BACKEND}/models")
            r.raise_for_status()
            logger.info("[ROUTE] ✅ Models list retrieved successfully")
            return Response(content=r.content, media_type="application/json")
    except Exception as e:
        logger.error(f"[MODELS] Error: {e}", exc_info=True)
        return JSONResponse(content={"object": "list", "data": []}, status_code=200)

@router.get("/slots")
async def list_slots():
    return [{"id": "default", "name": "Kven Gateway", "object": "slot"}]

@router.post("/chat/completions")
async def handle_chat(request: Request):
    try:
        logger.info("[ROUTE] >>> Incoming request to /chat/completions")
        body = await request.json()
        messages = body.get("messages", [])
        model_name = body.get("model") or settings.MAIN_MODEL
        msg_count = len(messages)
        logger.info(f"[ROUTE] Model: {model_name}, Messages count: {msg_count}")
        
        # 1. Версионирование
        logger.info("[ROUTE] Loading active state from DB...")
        active_state = await load_active_state()
        logger.info(f"[ROUTE] Active state keys: {list(active_state.keys())}")
        
        await save_history_snapshot(active_state)
        logger.info("[ROUTE] ✅ History snapshot saved")
        
        # 2. Сборка промпта
        profile = load_agent_profile()
        current_time = await sys_time.get_external_time()
        logger.info(f"[ROUTE] Profile loaded. Time fetched: {current_time}")
        
        sys_block = ""
        if profile:
            sys_block += f"name: {profile.get('agent_name', 'Kven II')}\n"
            sys_block += "role: You are my friend.\n\n"
            sys_block += f"You are {profile.get('agent_name', 'Kven II')}.\n"
            sys_block += f"Agent Role: {profile.get('agent_role', '')}\n"
            sys_block += f"Project History: {profile.get('project_history', '')}\n"
            sys_block += f"Owner: {profile.get('owner', '')}\n"
            sys_block += f"Mission: {profile.get('mission', '')}\n\n"
            
        sys_block += f"Current server datetime: {current_time}\n"
        sys_block += "You have access to the current server time above.\n"
        sys_block += "Use it when answering questions about date or time.\n"
        sys_block += "Do not say that you lack realtime access.\n\n"
        
        # PHASE 2: Семантическая память
        semantic_memories = await get_semantic_context(limit=5)
        if semantic_memories:
            sys_block += f"\n--- SEMANTIC MEMORY (Learned Knowledge) ---\n{semantic_memories}\n"
            logger.debug(f"[ROUTE] Semantic context added. Length: {len(semantic_memories)} chars")

        # PHASE 3: Проект
        current_project_id = active_state.get('current_project_id', 1)
        project_goal = await get_project_context(current_project_id)
        if project_goal:
            sys_block += f"\n{project_goal}\n"
            logger.debug(f"[ROUTE] Project context added: {project_goal}")

        sys_block += "ACTIVE STATE:\n"
        sys_block += f"- Active Problem: {active_state.get('active_problem', 'None')}\n"
        sys_block += f"- Salience: {active_state.get('salience', 0.0)}\n"
        sys_block += f"- Confidence: {active_state.get('confidence', 0.0)}\n"
        
        if len(sys_block) > 3200:
            sys_block = sys_block[:2900] + "\n[TRUNCATED: SYSTEM BLOCK LIMIT]"
            logger.warning(f"[ROUTE] ⚠️ System block truncated. Original length > 3200 chars")
        
        enriched_messages = [{"role": "system", "content": sys_block}] + messages
        logger.info(f"[ROUTE] System block length: {len(sys_block)} chars. Total payload messages: {len(enriched_messages)}")
        
        payload = {
            "model": model_name,
            "messages": enriched_messages,
            "stream": body.get("stream", False),
            "temperature": body.get("temperature", 0.7)
        }
        
        chat_url = f"{BASE_BACKEND}/chat/completions"
        logger.info(f"[ROUTE] Forwarding to backend: {chat_url}")
        
        backend_chunks = []
        async with httpx.AsyncClient(timeout=1200.0) as client:
            async with client.stream("POST", chat_url, json=payload) as response:
                logger.info(f"[ROUTE] Backend response status: {response.status_code}")
                if response.headers.get("content-type", "").startswith("text/event-stream"):
                    logger.info("[ROUTE] ✅ SSE stream detected. Parsing chunks...")
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            backend_chunks.append(line)
                    logger.info(f"[ROUTE] ✅ Stream completed. Total chunks: {len(backend_chunks)}")
                else:
                    logger.info("[ROUTE] ⚠️ Non-SSE response detected. Falling back to JSON read...")
                    data = await response.aread()
                    resp_json = json.loads(data)
                    content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        backend_chunks.append(f"data: {json.dumps({'content': content})}")
                        logger.info(f"[ROUTE] ✅ JSON fallback content extracted. Length: {len(content)}")
        
        async def generate():
            for chunk in backend_chunks:
                yield chunk + "\n"
        
        # FIX: Robust parsing for both flat and OpenAI-style (delta) chunks
        assistant_reply = ""
        for line in backend_chunks:
            if line.startswith("data: "):
                raw = line[6:].strip()
                if raw == "[DONE]": continue
                try:
                    chunk_obj = json.loads(raw)
                    
                    # 1. Try standard OpenAI structure: choices -> delta -> content
                    choices = chunk_obj.get("choices", [])
                    content = None
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                    
                    # 2. Fallback: flat structure {"content": ...}
                    if not content:
                        content = chunk_obj.get("content")

                    if content:
                        assistant_reply += content
                            
                except json.JSONDecodeError:
                    logger.debug(f"[ROUTE] Non-JSON chunk ignored: {raw[:50]}...")
        
        # ВНЕДРЕНИЕ: Очистка ответа от мыслительных блоков перед сохранением
        assistant_reply = strip_reasoning(assistant_reply)
        
        logger.info(f"[ROUTE] Final assistant_reply length: {len(assistant_reply)} chars. First 100: {assistant_reply[:100]}")
        
        if assistant_reply and len(assistant_reply.strip()) > 10:
            logger.info("[ROUTE] ✅ Triggering background task [WRITE_PATH]")
            asyncio.create_task(process_episodic(messages, assistant_reply, active_state))
        else:
            logger.warning("[ROUTE] ⚠️ Assistant reply too short/empty. Skipping [WRITE_PATH] to prevent garbage accumulation.")
        
        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"[ROUTE] Gateway Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
