import os
from openai import OpenAI
from dotenv import load_dotenv
from collections import OrderedDict

# Configuración
CSV_FILE = "knowledge_base.csv"
DB_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"



load_dotenv()
# Cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Inicializamos cachés ---
embedding_cache = {}        # Cache de embeddings para queries
qa_cache = OrderedDict()    # Cache de respuestas finales
QA_CACHE_SIZE = 10
