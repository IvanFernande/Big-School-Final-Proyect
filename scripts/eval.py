from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import Embedder
from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.generator import SimpleGenerator

# Preguntas de prueba ampliadas. ASCII para evitar encoding.
TEST_SET = [
    # KPIs CSV
    {"question": "Cual es el SLA cumplido en Q4 y cuantos tickets abiertos hubo?", "expected": ["93.1", "1500"]},
    {"question": "Cual es el CSAT promedio en Q3?", "expected": ["4.4"]},
    {"question": "Cual es el SLA cumplido en Q2 y cuantos tickets cerrados hubo?", "expected": ["92.7", "1290"]},
    {"question": "Que trimestre tuvo el SLA mas bajo?", "expected": ["q3", "89.9"]},
    {"question": "Ordena los trimestres por tickets abiertos de mayor a menor", "expected": ["q4", "q3", "q2", "q1"]},
    # Config JSON
    {"question": "Cual es el servicio principal configurado?", "expected": ["core_api"]},
    {"question": "Cual es la version del servicio principal?", "expected": ["2.3.1"]},
    {"question": "Cual es el SLA objetivo definido?", "expected": ["95", "sla", "objetivo"]},
    {"question": "Cuales son los umbrales de alerting cpu, memoria y latencia?", "expected": ["85", "80", "250"]},
    {"question": "Quien es la responsable de soporte y su email?", "expected": ["laura", "gomez", "@technova.com"]},
    # FAQ TXT
    {"question": "Como restablecer la contrasena?", "expected": ["restablecer", "portal", "ticket"]},
    {"question": "Que hacer si no puedo acceder a la vpn?", "expected": ["vpn", "ticket"]},
    {"question": "Que datos debo incluir al reportar una incidencia?", "expected": ["capturas", "descripcion", "pasos"]},
    {"question": "Cuales son los tiempos normales de respuesta?", "expected": ["15", "2 horas"]},
    # Manual/README
    {"question": "Resume el flujo de escalado del sistema.", "expected": ["escalado", "sla"]},
    {"question": "Que modulos componen el sistema y como interactuan?", "expected": ["tickets", "monitorizacion", "diagnostico", "kpi"]},
    {"question": "Que pasos de troubleshooting antes de escalar?", "expected": ["verificar", "credenciales", "logs"]},
    # On-call
    {"question": "Cual es el horario de guardia entre semana y fin de semana?", "expected": ["18:00", "08:00", "24h"]},
    {"question": "Que tiempo maximo tiene el tecnico oncall para responder una alerta?", "expected": ["10 minutos", "10"]},
    {"question": "Que hacer ante una alerta de servicio degradado?", "expected": ["latencia", "reiniciar", "estado"]},
    {"question": "Que comandos se usan en oncall?", "expected": ["tn-monitor", "tn-restart", "tn-alert"]},
    # Politicas operativas
    {"question": "Que canales oficiales deben usarse para comunicacion interna?", "expected": ["tickets", "correo"]},
    {"question": "Que requiere la politica de gestion de cambios antes de produccion?", "expected": ["aprobado", "evaluado"]},
    {"question": "Como se priorizan incidencias segun impacto y urgencia?", "expected": ["prioridad", "impacto", "urgencia"]},
    # Inventarios
    {"question": "Cual es el estado del firewall principal y del ups en inventario hardware?", "expected": ["firewall", "operativo", "ups", "mantenimiento"]},
    {"question": "Que servicio tiene criticidad alta y owner backend_team?", "expected": ["core_api", "alta", "backend_team"]},
    # Historico incidentes
    {"question": "Cual fue la causa raiz del incidente INC-2024-001?", "expected": ["pool de conexiones"]},
    {"question": "Que acciones correctivas se tomaron en INC-2024-009?", "expected": ["rollback", "automatico"]},
    {"question": "Que incidente tuvo perdida de nodo y cuanto duro la resolucion?", "expected": ["srv-db-002", "4h 22m"]},
]

SLEEP_BETWEEN_REQUESTS = 5  # segundos entre preguntas para evitar 429
MAX_RETRIES = 3
RETRY_SLEEP = 60  # segundos para reintentos ante 429/quota


def score_answer(answer: str, expected_substrings) -> float:
    ans_low = (answer or "").lower()
    if not ans_low:
        return 0.0
    hits = sum(1 for sub in expected_substrings if sub.lower() in ans_low)
    return hits / len(expected_substrings) if expected_substrings else 0.0


def main():
    index_dir = Path("index")
    try:
        store = VectorStore.load(index_dir)
    except FileNotFoundError:
        print(f"No se encontro un indice en {index_dir}. Ejecuta primero scripts/build_index.py.")
        return

    embedder = Embedder()
    retriever = Retriever(embedder, store, k=16)
    try:
        generator = SimpleGenerator()
    except Exception as e:
        print(f"No se pudo inicializar el generador (Gemini). Configura GEMINI_API_KEY o config.json. Error: {e}")
        return

    results = []
    for i, test in enumerate(TEST_SET):
        q = test["question"]
        expected = test["expected"]
        retrieved = retriever.retrieve(q)
        answer = None
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                answer = generator.answer(q, retrieved, max_contexts=8)
                break
            except Exception as e:
                last_error = e
                msg = str(e)
                if "ResourceExhausted" in msg or "429" in msg:
                    wait = RETRY_SLEEP * (attempt + 1)
                    print(f"[{q}] Cuota/429, reintentando en {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    break
        if answer is None:
            answer = f"[ERROR LLM] {last_error}"
            score = 0.0
        else:
            score = score_answer(answer, expected)
        results.append((q, score, expected, answer))
        if i < len(TEST_SET) - 1:
            print(f"Esperando {SLEEP_BETWEEN_REQUESTS}s antes de la siguiente pregunta...")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("Evaluacion RAG")
    print("-" * 80)
    for q, score, expected, answer in results:
        print(f"P: {q}")
        print(f"  Esperado: {expected}")
        print(f"  Respuesta: {answer}")
        print(f"  Score: {score:.2f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
