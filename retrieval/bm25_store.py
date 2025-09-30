import pandas as pd
from rank_bm25 import BM25Okapi
from config import CSV_FILE


df_bm25 = pd.read_csv(CSV_FILE)


bm25 = BM25Okapi([doc.split() for doc in df_bm25["document"].tolist()])

def retrieve_bm25(query, top_k=1):
    tokenized_query = query.split()
    top_docs = bm25.get_top_n(tokenized_query, df_bm25["document"].tolist(), n=top_k)

    retrieved_docs = []
    for doc in top_docs:
        idx = df_bm25[df_bm25["document"] == doc].index[0]
        retrieved_docs.append({
            "id": df_bm25.loc[idx, "id"],
            "document": df_bm25.loc[idx, "document"],
            "context": df_bm25.loc[idx, "context"]
        })
    return retrieved_docs

