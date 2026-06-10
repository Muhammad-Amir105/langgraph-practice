import os
from typing import TypedDict
from dotenv import load_dotenv
from pathlib import Path

from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
root = Path(__file__).resolve().parent
load_dotenv(root / ".env")
load_dotenv(root / "uploads" / ".env", override=False)

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
    research: str
    final_answer: str
    review: str
    retries: int


# -------------------------
# resercher node
# -------------------------
def researcher(state):

    question = state["question"]

    prompt = f"""
You are a researcher.

Provide detailed research notes.

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {
        "research": response.content,
        "retries": state.get("retries", 0) + 1
    }


# -------------------------
# summarizer node
# -------------------------
def summarizer(state):

    research = state["research"]

    prompt = f"""
You are a summarizer.

Summarize the following research.

Research:
{research}
"""

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content
    }

# -------------------------
# critic node
# -------------------------


def critic(state):

    research = state["research"]

    prompt = f"""
You are a research reviewer.

Review the research below.

Research:
{research}

If the research is detailed and complete reply only:

good

Otherwise reply only:

bad
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    print("Critic Decision:", decision)

    return {
        "review": decision
    }

def route_after_critic(state):

    review = state["review"]

    if "good" in review:
        return "summarizer"

    if state.get("retries", 0) >= 2:
        return "summarizer"

    return "researcher"
# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(GraphState)

builder.add_node("researcher", researcher)
builder.add_node("summarizer", summarizer)
builder.add_node("critic", critic)

builder.set_entry_point("researcher")

builder.add_edge(
    "researcher",
    "critic"
)

builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "researcher": "researcher",
        "summarizer": "summarizer"
    }
)

builder.add_edge(
    "summarizer",
    END
)

graph = builder.compile()