from mcp.server.fastmcp import FastMCP
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

mcp = FastMCP("Company Assistant")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
print(
    "Documents:",
    vectorstore._collection.count()
)   

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


@mcp.tool()
def calculator(expression: str) -> str:
    """
    Calculate math expression.
    """
    return str(eval(expression))

@mcp.resource("pdf://search/{query}")
def search_pdf(query: str):

    docs = retriever.invoke(query)

    return "\n".join(
        doc.page_content
        for doc in docs
    )

@mcp.prompt()
def summarize_pdf():

    return """
    You are a professional document summarizer.

    Summarize the document in bullet points.
    """

if __name__ == "__main__":
    mcp.run()