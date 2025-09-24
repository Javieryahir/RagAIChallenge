import os
from collections import OrderedDict
from openai import OpenAI
from dotenv import load_dotenv

# Configuración
CSV_FILE = "knowledge_base.csv"
DB_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"
QA_CACHE_SIZE = 10


load_dotenv()
# Cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cachés
embedding_cache = {}
qa_cache = OrderedDict()

