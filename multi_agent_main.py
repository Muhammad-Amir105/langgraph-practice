from fastapi import FastAPI
import os
from schemas import QuestionRequest

from fastapi import UploadFile, File

retriever = None

app = FastAPI()

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...)
):

    global retriever

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Import heavy helpers lazily to avoid importing large ML deps at app startup
    from rag import load_pdf, split_documents, create_vectorstore

    documents = load_pdf(file_path)

    chunks = split_documents(documents)

    vectorstore = create_vectorstore(chunks)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return {
        "message": "PDF uploaded successfully"
    }


@app.post("/ask")
async def ask_question(data: QuestionRequest):

    # Import graph lazily to avoid loading heavy dependencies at module import
    from multi_agent_rag import graph

    result = graph.invoke(
        {
            "question": data.question
        }
    )

    return {
        "question": result["question"],
        "answer": result["answer"]
    }