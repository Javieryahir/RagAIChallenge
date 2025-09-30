from langgraph.graph import StateGraph, END
from typing import TypedDict
from llm.rag import rag_answer
from openai import OpenAI
import datetime
from retrieval.chroma_store import ensure_chroma_collection

client = OpenAI()

class State(TypedDict):
    query: str
    intent: str
    answer: str

# Nodo que usa el LLM para detectar la intención
def detect_intent_llm(state: State):
    prompt = f"""
    You are a classifier. Given a user query, respond ONLY with either 'date' or 'rag'.
    User query: {state['query']}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5
    )

    intent = response.choices[0].message.content.strip().lower()
    state["intent"] = intent
    return state

def answer_with_rag(state: State):
    state["answer"] = rag_answer(state["query"])
    return state

def answer_with_date(state: State):
    today = datetime.date.today()
    state["answer"] = f"Hoy es {today.strftime('%d/%m/%Y')}."
    return state

workflow = StateGraph(State)

workflow.add_node("detect_intent", detect_intent_llm)
workflow.add_node("rag", answer_with_rag)
workflow.add_node("date", answer_with_date)

workflow.set_entry_point("detect_intent")

workflow.add_conditional_edges(
    "detect_intent",
    lambda state: state["intent"],
    {
        "rag": "rag",
        "date": "date",
    }
)

workflow.add_edge("rag", END)
workflow.add_edge("date", END)

# Compilar
app = workflow.compile()

# Verificar o crear colección Chroma
ensure_chroma_collection()


if __name__ == "__main__":
    while True:
        q = input("Ask me something (or 'quit'): ")
        if q.lower() == "quit":
            break
        result = app.invoke({"query": q})
        print("Answer:", result["answer"])



