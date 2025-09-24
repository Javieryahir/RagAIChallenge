import pandas as pd
import chromadb
from config import CSV_FILE, DB_DIR, COLLECTION_NAME
from retrieval.embeddings import get_embedding

def build_chroma_collection():
    df = pd.read_csv(CSV_FILE)
    chroma_client = chromadb.PersistentClient(path=DB_DIR)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    embeddings = [get_embedding(q) for q in df["question"].values]

    collection.add(
        embeddings=embeddings,
        documents=df["answer"].tolist(),
        metadatas=[{"question": q} for q in df["question"].tolist()],
        ids=[str(i) for i in range(len(df))]
    )
    return collection

def retrieve_chroma(query, top_k=1):
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    q_emb = get_embedding(query)
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)

    return [
        {"question": results["metadatas"][0][i]["question"],
         "answer": results["documents"][0][i]}
        for i in range(len(results["ids"][0]))
    ]
