import numpy as np
import hashlib
import re
import unicodedata
from sentence_transformers import SentenceTransformer

# --- Inicializamos modelo de embeddings ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Parámetros de cache semántico ---
SIMILARITY_THRESHOLD = 0.78  

# --- Helpers ---
def normalize(text: str) -> str:
    """Lower, remove accents and punctuation, collapse spaces."""
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def embed_text(text: str) -> np.ndarray:
    """Return 1D numpy embedding (float32)."""
    emb = embedding_model.encode(text)
    return np.array(emb, dtype=np.float32)

def find_best_in_cache(embedding: np.ndarray, embedding_cache: dict):
    """
    Busca la entrada más similar en embedding_cache.
    embedding_cache expected structure:
        emb_key -> {"embedding": [..], "answer": "...", "query": "..."}
    Devuelve (best_key, best_sim, best_answer) o (None, 0.0, None).
    """
    candidates = []
    for k, v in embedding_cache.items():
        if isinstance(v, dict) and "embedding" in v:
            try:
                e = np.array(v["embedding"], dtype=np.float32)
                candidates.append((k, e, v.get("answer"), v.get("query")))
            except Exception:
                continue

    if not candidates:
        return None, 0.0, None

    mat = np.stack([c[1] for c in candidates])  # shape (N, D)
    # cosine similarity
    emb_norm = np.linalg.norm(embedding) + 1e-12
    mat_norms = np.linalg.norm(mat, axis=1) + 1e-12
    sims = (mat @ embedding) / (mat_norms * emb_norm)

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_key, _, best_answer, _ = candidates[best_idx]

    if best_sim >= SIMILARITY_THRESHOLD:
        return best_key, best_sim, best_answer
    return None, best_sim, None

def make_emb_key(embedding: np.ndarray, precision: int = 1):
    """Genera un key compacto a partir de embedding redondeado (solo para guardar)."""
    rounded = np.round(embedding, decimals=precision)
    return hashlib.md5(rounded.tobytes()).hexdigest()
