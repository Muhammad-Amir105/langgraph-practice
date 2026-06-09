import os
from typing import TypedDict

from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def set_retriever(new_retriever):

    global retriever

    retriever = new_retriever

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks):

    if os.path.exists("./chroma_db"):

        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding_model
        )

        vectorstore.add_documents(chunks)

    else:

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db"
        )

    return vectorstore


# -------------------------
# LLM
# -------------------------

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# -------------------------
# Embeddings
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Vector DB
# -------------------------

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

set_retriever(retriever)

# -------------------------
# State
# -------------------------

class GraphState(TypedDict):
    question: str
    context: str
    route: str
    answer: str
    retries: int


# -------------------------
# Retrieve Node
# -------------------------

def retrieve(state):

    global retriever

    if retriever is None:

        return {
            "context": ""
        }

    docs = retriever.invoke(
        state["question"]
    )

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    return {
        "context": context
    }


# -------------------------
# Grade Context Node
# -------------------------

def grade_context(state):

    question = state["question"]
    context = state["context"]

    prompt = f"""
    You are a relevance checker.

    Question:
    {question}

    Context:
    {context}

    Determine whether the context contains information useful for answering the question.

    Reply with ONLY one word:

    yes
    or
    no
    """

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    if "yes" in decision:
        route = "rag"
    else:
        route = "llm"

    return {
        "route": route
    }


# -------------------------
# Router
# -------------------------
def route_question(state):

    if state["route"] == "rag":
        return "rag"

    if state.get("retries", 0) >= 2:
        return "llm"

    return "rewrite"
# def route_question(state):
#     return state["route"]


# -------------------------
# RAG Answer Node
# -------------------------

def rag_answer(state):

    question = state["question"]
    context = state["context"]

    prompt = f"""
    Answer ONLY from the provided context.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# -------------------------
# General LLM Node
# -------------------------

def general_answer(state):

    question = state["question"]

    response = llm.invoke(question)

    return {
        "answer": response.content
    }

# -------------------------
# Rewrite Question Node
# -------------------------

def rewrite_question(state):

    question = state["question"]

    prompt = f"""
Rewrite the following question
to make it clearer and better for document retrieval.

Question:
{question}

Return only the rewritten question.
"""

    response = llm.invoke(prompt)

    new_question = response.content.strip()

    return {
        "question": new_question,
        "retries": state.get("retries", 0) + 1
    }


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve)
builder.add_node("grade_context", grade_context)
builder.add_node("rag_answer", rag_answer)
builder.add_node("general_answer", general_answer)
builder.add_node("rewrite_question", rewrite_question)

builder.set_entry_point("retrieve")

builder.add_edge(
    "retrieve",
    "grade_context"
)

builder.add_conditional_edges(
    "grade_context",
    route_question,
    {
        "rag": "rag_answer",
        "rewrite": "rewrite_question",
        "llm": "general_answer"
    }
)

builder.add_edge(
    "rewrite_question",
    "retrieve"
)

builder.add_edge(
    "rag_answer",
    END
)

builder.add_edge(
    "general_answer",
    END
)

graph = builder.compile()