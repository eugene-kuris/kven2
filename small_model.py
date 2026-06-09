# /opt/kven2/agent/models/small_model.py
import asyncio
import logging
import httpx
import json

logger = logging.getLogger(__name__)

import sys
sys.path.append('/opt/kven2')
from config import settings

SM_URL = f"{settings.SMALL_MODEL_URL}/chat/completions"
SM_MODEL = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_S.gguf"

MEMORY_GRAMMAR = """
root ::= array
array ::= "[" ws (chunk ("," ws chunk)*)? ws "]"
chunk ::= "{" ws
    "\"save\"" ws ":" ws bool ws ","
    "\"kind\"" ws ":" ws kind ws ","
    "\"epistemic_type\"" ws ":" ws epistemic_type ws ","
    "\"source\"" ws ":" ws source ws ","
    "\"summary\"" ws ":" ws string ws ","
    "\"facts\"" ws ":" ws fact_list ws
"}"
bool ::= "true" | "false"
kind ::= "\"procedural\"" | "\"semantic\"" | "\"failure\"" | "\"episodic\""
epistemic_type ::= "\"Observation\"" | "\"Hypothesis\"" | "\"Verified Invariant\"" | "\"Temporary State\""
source ::= "\"direct_user\"" | "\"tool_verification\"" | "\"model_inference\"" | "\"external_rule\""
fact_list ::= "[" ws (string ("," ws string)*)? ws "]"
string ::= "\"" ([^"\\\\] | "\\\\" [ntrb"/\\\\] | "\\\\" u{4})* "\""
ws ::= [ \\t\\n\\r]*
"""

async def call_small_model(prompt: str, grammar: str = None, max_tokens: int = 512) -> str:
    payload = {
        "model": SM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    if grammar:
        payload["grammar"] = grammar
    else:
        payload["response_format"] = {"type": "json_object"}

    # ✅ PHASE 4 FIX: Увеличен таймаут до 15 минут (900 секунд) для медленных CPU offload сценариев
    async with httpx.AsyncClient(timeout=900.0) as client:
        try:
            r = await client.post(SM_URL, json=payload)
            r.raise_for_status()
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip()
        except Exception as e:
            logger.error(f"[SM] Call failed: {e}")
            return ""

async def extract_memory(dialogue: str, active_state: dict = None) -> list:
    context = ""
    if active_state and active_state.get("active_problem"):
        context = f"\nCURRENT PROBLEM: {active_state['active_problem']}"

    prompt = f"""
SYSTEM: You are a memory assistant.
TASK: Extract knowledge from the dialogue.
OUTPUT: Strict JSON list.
{context}

Dialogue:
{dialogue}

Output:
[
"""
    result = await call_small_model(prompt, grammar=MEMORY_GRAMMAR, max_tokens=1024)
    return result # Simplified for now, full parser in routes
