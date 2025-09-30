from config import client, embedding_cache, qa_cache, QA_CACHE_SIZE
from retrieval.chroma_store import retrieve_chroma
from retrieval.bm25_store import retrieve_bm25
from .query_rewriter import rewrite_query
from retrieval.cache_utils import normalize, embed_text, find_best_in_cache, make_emb_key

# --- Función principal ---
def rag_answer(query: str, top_k: int = 5) -> str:
    # Reescribir y normalizar la query para embeddings
    rewritten_query = rewrite_query(query)
    normalized = normalize(rewritten_query)
    print(f"Rewritten query: {rewritten_query}")
    print(f"Normalized for embedding: {normalized}")

    # 1) calcular embedding de la query
    embedding = embed_text(normalized)

    # 2) buscar la mejor coincidencia semántica en el cache
    best_key, best_sim, best_answer = find_best_in_cache(embedding, embedding_cache)
    if best_key is not None:
        print(f"⚡ Respuesta desde cache semántico (sim={best_sim:.3f})")
        return best_answer

    # 3) Recuperación híbrida (Chroma + BM25)
    retrieved_chroma = retrieve_chroma(rewritten_query, top_k=top_k)
    retrieved_bm25 = retrieve_bm25(rewritten_query, top_k=top_k)

    # Deduplicar resultados
    combined_docs = retrieved_bm25 + retrieved_chroma
    seen = set()
    unique_docs = []
    for doc in combined_docs:
        doc_text = doc.get("document", "")
        if doc_text and doc_text not in seen:
            unique_docs.append(doc)
            seen.add(doc_text)

    context = "\n".join(
        [f"{i+1}. {doc['document']} (Context: {doc.get('context')})"
         for i, doc in enumerate(unique_docs[: top_k * 2])]
    )

    prompt = f"""
You are a helpful assistant.

Here is the context retrieved from the knowledge base (BM25 + Chroma, deduplicated):
{context}

User's original question: {query}
Rewritten query used for retrieval: {rewritten_query}

Answer:
"""

    # Llamada al LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()

    # Guardar en cache semántico
    emb_key = make_emb_key(embedding, precision=1)
    embedding_cache[emb_key] = {
        "embedding": embedding.tolist(),
        "answer": answer,
        "query": rewritten_query
    }

    # Mantener QA cache (LRU simple)
    try:
        if len(qa_cache) >= QA_CACHE_SIZE:
            qa_cache.popitem(last=False)
    except Exception:
        pass
    try:
        qa_cache[query] = answer
    except Exception:
        pass

    return answer
