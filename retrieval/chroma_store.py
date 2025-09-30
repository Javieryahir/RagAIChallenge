import pandas as pd
import chromadb
from config import CSV_FILE, DB_DIR, COLLECTION_NAME
from retrieval.embeddings import get_embedding


def build_chroma_collection():
    # Cargar CSV con columnas: id, document, context
    df = pd.read_csv(CSV_FILE)
    chroma_client = chromadb.PersistentClient(path=DB_DIR)

    # Borrar colección anterior si existe
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except:
        pass

    # Crear colección nueva
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    # Calcular embeddings a partir de "document"
    embeddings = [get_embedding(doc) for doc in df["document"].values]

    # Insertar en Chroma
    collection.add(
        embeddings=embeddings,
        documents=df["document"].tolist(),  # Texto base del documento
        metadatas=[
            {"id": str(row["id"]), "context": row["context"]}
            for _, row in df.iterrows()
        ],
        ids=[str(i) for i in range(len(df))]
    )
    return collection


def retrieve_chroma(query, top_k=1):
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Calcular embedding del query
    q_emb = get_embedding(query)

    # Recuperar documentos más cercanos
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)

    retrieved_docs = []
    for i in range(len(results["ids"][0])):
        retrieved_docs.append({
            "id": results["metadatas"][0][i]["id"],
            "document": results["documents"][0][i],
            "context": results["metadatas"][0][i]["context"]
        })

    return retrieved_docs

def ensure_chroma_collection():
    """
    Verifica si la colección de Chroma existe.
    Si no existe, la crea desde el CSV.
    """
    # Cliente Chroma
    chroma_client = chromadb.PersistentClient(path=DB_DIR)

    try:
        # Intentamos obtener la colección
        chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"✅ Chroma collection '{COLLECTION_NAME}' exists.")
    except chromadb.errors.NotFoundError:
        # Si no existe, la creamos
        print(f"⚡ Chroma collection '{COLLECTION_NAME}' not found. Building it now...")
        build_chroma_collection()
        print(f"✅ Collection '{COLLECTION_NAME}' created successfully.")