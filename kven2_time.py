# /opt/kven2/agent/utils/time.py
import httpx
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TIME_SOURCE = "http://127.0.0.1:8954/time"

async def get_external_time() -> str:
    """
    Запрашивает точное время у доверенного источника.
    Если источник недоступен, возвращает время системы.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(TIME_SOURCE)
            return resp.text.strip()
    except Exception:
        # Fallback: локальное время
        return datetime.now().strftime("%a %b %d %H:%M:%S %Y")
