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
print('abc', os.getenv("GROQ_API_KEY"))

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
    tool: str
    answer: str


# -------------------------
# resercher node
# -------------------------
def agent(state):

    question = state["question"]

    prompt = f"""
    You are a router.

    If the question contains arithmetic or mathematical calculations
    (addition, subtraction, multiplication, division, percentages, powers, etc.),
    reply only:

    calculator

    Otherwise reply only:

    search

    Question:
    {question}
    """

    response = llm.invoke(prompt)
    print("Agent response:", response.content.strip().lower())

    return {
        "tool": response.content.strip().lower()
    }


def route_tool(state):
    return state["tool"]

def retrieve(state):

    docs = retriever.invoke(
        state["question"]
    )
    print("Retrieved Docs:", len(docs))


    context = "\n".join(
        doc.page_content
        for doc in docs
    )
    print("Context:")
    print(context[:500])

    return {
        "context": context
    }

def grade_context(state):

    prompt = f"""
You are a routing system.

Question:
{state["question"]}

Retrieved Context:
{state["context"]}

If the retrieved context contains information
that can answer the question, output exactly:

pdf

Otherwise output exactly:

llm

Return one word only.
"""

    response = llm.invoke(prompt)
    print("Grade Decision:", response.content.strip().lower())

    return {
        "tool": response.content.strip().lower()
    }

def calculator_tool(state):
    print("Invoking calculator tool")

    question = state["question"]
    print("Question:", question)
    prompt = f"""
    Convert this question into a valid Python math expression.

    Question:
    {question}

    Return only the expression.
    """

    expression = llm.invoke(prompt).content.strip()
    print("expression", expression)

    result = eval(expression)

    return {
        "answer": str(result)
    }

def pdf_tool(state):
    print("Invoking PDF tool")

    question = state["question"]

    docs = retriever.invoke(question)

    context = "\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
    Answer using the provided context.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def llm_tool(state):
    print("Invoking LLM tool")

    question = state["question"]

    response = llm.invoke(question)

    return {
        "answer": response.content
    }
# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(GraphState)

builder.add_node("agent", agent)
builder.add_node("calculator_tool", calculator_tool)
builder.add_node("grade_context", grade_context)
builder.add_node("retrieve", retrieve)
builder.add_node("pdf_tool", pdf_tool)
builder.add_node("llm_tool", llm_tool)

builder.set_entry_point("agent")

builder.add_conditional_edges(
    "agent",
    route_tool,
    {
        "calculator": "calculator_tool",
        "search": "retrieve"
    }
)

builder.add_edge(
    "retrieve",
    "grade_context"
)

builder.add_conditional_edges(
    "grade_context",
    route_tool,
    {
        "pdf": "pdf_tool",
        "llm": "llm_tool"
    }
)

builder.add_edge(
    "calculator_tool",
    END
)

builder.add_edge(
    "pdf_tool",
    END
)

builder.add_edge(
    "llm_tool",
    END
)

graph = builder.compile()