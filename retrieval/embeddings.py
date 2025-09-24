from config import client, embedding_cache

def get_embedding(text, model="text-embedding-3-small"):
    if text in embedding_cache:
        return embedding_cache[text]
    emb = client.embeddings.create(input=[text], model=model).data[0].embedding
    embedding_cache[text] = emb
    return emb
