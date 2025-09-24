from config import client

def rewrite_query(query):
    prompt = f"""
You are an assistant that reformulates user questions for better retrieval.
If the query is not in English, rewrite it in English.

Original question: {query}

Rewritten query:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
