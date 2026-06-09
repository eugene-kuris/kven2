import sys
sys.path.append('/opt/kven2')
from sqlite import get_connection
from hnsw import get_nearest_neighbors

def run_phase0():
    conn = get_connection()
    try:
        # 1. Счётчики таблиц
        sem_count = conn.execute("SELECT COUNT(*) FROM semantic_memory WHERE deleted = 0").fetchone()[0]
        epi_count = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        print(f"[PHASE 0] semantic_memory: {sem_count}")
        print(f"[PHASE 0] episodic_memory: {epi_count}")
        
        if sem_count == 0:
            print("❌ FAIL: Нет семантической памяти. Проверь consolidation.py и WRITE_PATH.")
            return False

        # 2. Тест HNSW-поиска
        test_vector = [0.0] * 768  # nomic-embed-text-v1 выдает 768-мерные векторы
        neighbors = get_nearest_neighbors(test_vector, k=5)
        print(f"[PHASE 0] HNSW neighbors returned: {len(neighbors)}")
        
        if len(neighbors) == 0:
            print("❌ FAIL: HNSW индекс пуст или не инициализирован. Проверь hnsw.py и индексацию.")
            return False
            
        print("✅ PASS: Baseline verified. Pipeline is operational.")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    run_phase0()