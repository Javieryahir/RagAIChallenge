import pandas as pd
from rank_bm25 import BM25Okapi
from config import CSV_FILE

df_bm25 = pd.read_csv(CSV_FILE)
bm25 = BM25Okapi([q.split() for q in df_bm25["question"].tolist()])

def retrieve_bm25(query, top_k=1):
    tokenized_query = query.split()
    doc_scores = bm25.get_top_n(tokenized_query, df_bm25["answer"].tolist(), n=top_k)
    retrieved_docs = []
    for doc in doc_scores:
        idx = df_bm25[df_bm25["answer"] == doc].index[0]
        retrieved_docs.append({
            "question": df_bm25.loc[idx, "question"],
            "answer": df_bm25.loc[idx, "answer"]
        })
    return retrieved_docs
