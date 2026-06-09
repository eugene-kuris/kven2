#kven2_profile.py
import json
import os
import logging

logger = logging.getLogger(__name__)

# Абсолютный путь к файлу профиля (исправлено для стабильности)
PROFILE_PATH = '/opt/kven2/agent_profile.json'

def load_agent_profile() -> dict:
    """
    Загружает профиль агента.
    Синхронная функция, безопасна для вызова в async-контексте (FastAPI).
    """
    try:
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"[PROFILE] Invalid JSON in {PROFILE_PATH}: {e}")
    except Exception as e:
        logger.error(f"[PROFILE] Failed to load profile: {e}")
    
    return {}
