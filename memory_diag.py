import sys
import os
import sqlite3
import asyncio
import importlib
import traceback
from config import settings

# Пути из конфигурации [1]
DB_PATH = settings.DB_PATH
MEMORY_DIR = settings.MEMORY_DIR

def run_sql_diagnostics():
    print("📊 === SQL ДИАГНОСТИКА ===")
    if not os.path.exists(DB_PATH):
        print(f"❌ БД не найдена по пути: {DB_PATH}")
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM episodic_memory")
    print(f"✅ episodic_memory: {cur.fetchone()[0]} записей")
    cur.execute("SELECT COUNT(*) FROM semantic_memory")
    print(f"✅ semantic_memory: {cur.fetchone()[0]} записей")
    conn.close()

async def run_vector_diagnostics():
    print("🔍 ПРОВЕРКА ВЕКТОРНОГО ПОИСКА...")
    EMBEDDER_PATH = "embedder"
    HNSW_PATH = "hnsw"

    try:
        print(f"📥 Импортирую эмбеддер: {EMBEDDER_PATH}")
        embedder_mod = importlib.import_module(EMBEDDER_PATH)
        
        print(f"📥 Импортирую HNSW: {HNSW_PATH}")
        # 🔧 ИСПРАВЛЕНИЕ: Импортируем и сохраняем в переменную ДО использования
        hnsw_mod = importlib.import_module(HNSW_PATH)

        # 🔧 ИСПРАВЛЕНИЕ: Принудительная инициализация индекса из файла
        print("🔄 Инициализация HNSW...")
        hnsw_mod.init_hnsw()

        if hnsw_mod.hnsw_index is not None:
            count = hnsw_mod.hnsw_index.get_current_count()
            print(f"✅ Индекс загружен. Записей в индексе: {count}")
        else:
            print("❌ hnsw_index всё ещё None. Проверь INDEX_PATH в config.py [1].")
            return

        get_nearest_neighbors = hnsw_mod.get_nearest_neighbors

        query_text = "memory system"
        print(f"⏳ Вычисляю эмбеддинг для: '{query_text}'")
        vector = await embedder_mod.get_embedding(query_text)

        print(f"✅ Эмбеддинг получен. Вызываю HNSW поиск (k=5)...")
        results = get_nearest_neighbors(vector, k=5)
        
        if results:
            print("🎯 Найденные соседние записи:")
            for label, distance in results:
                print(f"   ID: {label} | Dist: {distance:.4f}")
        else:
            print("⚠️ HNSW не вернул соседей. Возможно, индекс пуст или требует инициализации.")

    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Проверь, что пути в конфигурации совпадают с проектом.")
    except Exception as e:
        print(f"❌ Ошибка векторного поиска: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Запуск диагностики памяти Kven II...")
    print(f"📁 Путь к БД: {DB_PATH}")
    run_sql_diagnostics()
    asyncio.run(run_vector_diagnostics())
    print("\n✅ Диагностика завершена.")