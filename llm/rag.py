
from config import client, embedding_cache, qa_cache, QA_CACHE_SIZE
from retrieval.chroma_store import retrieve_chroma
from retrieval.bm25_store import retrieve_bm25
from .query_rewriter import rewrite_query



def rag_answer(query: str, top_k: int = 1) -> str:
        # --- Reescribir la query para mejor recuperación ---
    rewritten_query = rewrite_query(query)
    print(f"Rewritten query: {rewritten_query}")

    # --- Revisar si la pregunta ya está en caché ---
    if rewritten_query in qa_cache:
        print("⚡ Respuesta desde caché de QA")
        return qa_cache[rewritten_query]



    # --- Recuperación híbrida ---
    retrieved_chroma = retrieve_chroma(rewritten_query, top_k=top_k)
    retrieved_bm25 = retrieve_bm25(rewritten_query, top_k=top_k)

    # --- Reducir contexto duplicado ---
    combined_docs = retrieved_bm25 + retrieved_chroma
    seen = set()
    unique_docs = []
    for doc in combined_docs:
        # deduplicamos por texto de la respuesta
        ans = doc.get("answer", "")
        if ans and ans not in seen:
            unique_docs.append(doc)
            seen.add(ans)

    # Construir el contexto (limitamos a top_k * 2 fragmentos para no pasarnos)
    context = "\n".join([f"{i+1}. {doc['answer']}" for i, doc in enumerate(unique_docs[: top_k * 2])])

    # --- Debug: imprimir resultados de recuperación ---
    print("=== BM25 Results ===")
    for i, doc in enumerate(retrieved_bm25):
        print(f"{i+1}. Question: {doc.get('question')}")
        print(f"   Answer: {doc.get('answer')}\n")

    print("=== Chroma Embeddings Results ===")
    for i, doc in enumerate(retrieved_chroma):
        print(f"{i+1}. Question: {doc.get('question')}")
        print(f"   Answer: {doc.get('answer')}\n")

    # --- Imprimir Embedding Cache (resumen) ---
    print("=== Embedding Cache ===")
    for i, (txt, emb) in enumerate(embedding_cache.items()):
        display_txt = txt if len(txt) <= 60 else txt[:60] + "..."
        print(f"{i+1}. Text: {display_txt}")
        try:
            print(f"   Embedding length: {len(emb)}")
        except Exception:
            print("   Embedding: (no length available)")

    # --- Prompt al LLM ---
    prompt = f"""
You are a helpful assistant.

Here is the context retrieved from the knowledge base (BM25 + Chroma, deduplicated):
{context}

User's original question: {query}
Rewritten query used for retrieval: {rewritten_query}

Answer:
"""
    # Llamada al LLM (igual que en tu ejemplo original)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    # Extraer texto de la respuesta (según estructura del cliente que usas)
    answer = response.choices[0].message.content.strip()

    # --- Guardar en caché QA (con tamaño máximo) ---
    if len(qa_cache) >= QA_CACHE_SIZE:
        # pop el item más antiguo
        qa_cache.popitem(last=False)
    qa_cache[query] = answer

    return answer
