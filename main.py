# main.py
from langgraph.graph import StateGraph, END
from typing import TypedDict
from llm.rag import rag_answer
from openai import OpenAI
import datetime

client = OpenAI()

class State(TypedDict):
    query: str
    intent: str
    answer: str


def detect_intent(state: State):
    """Clasifica si la intención es 'rag' o 'date'"""
    query = state["query"]

    # Ejemplo MUY básico (puedes cambiarlo por un LLM)
    if "fecha" in query.lower() or "día" in query.lower():
        state["intent"] = "date"
    else:
        state["intent"] = "rag"
    return state

def answer_with_rag(state: State):
    state["answer"] = rag_answer(state["query"])
    return state

def answer_with_date(state: State):
    today = datetime.date.today()
    state["answer"] = f"Hoy es {today.strftime('%d/%m/%Y')}."
    return state

# ---- Construcción del grafo ----
workflow = StateGraph(State)

workflow.add_node("detect_intent", detect_intent)
workflow.add_node("rag", answer_with_rag)
workflow.add_node("date", answer_with_date)

workflow.set_entry_point("detect_intent")

# Condiciones de ramificación
workflow.add_conditional_edges(
    "detect_intent",
    lambda state: state["intent"],  # clave: "rag" o "date"
    {
        "rag": "rag",
        "date": "date",
    }
)

workflow.add_edge("rag", END)
workflow.add_edge("date", END)

# ---- Compilar ----
app = workflow.compile()

# ---- Loop interactivo ----
if __name__ == "__main__":
    while True:
        q = input("Ask me something (or 'quit'): ")
        if q.lower() == "quit":
            break
        result = app.invoke({"query": q})
        print("Answer:", result["answer"])



