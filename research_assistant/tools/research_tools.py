from langchain_core.tools import Tool
import requests
import os
from config import OUTPUT_DIR, VECTOR_DIR, EMBEDDINGS_MODEL

# RAG imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def web_search_tool(query: str) -> str:
    """Simulated web search tool (replace with actual API)"""
    return f"Search results for '{query}': [Simulated search results with relevant information about {query}]"

def save_document_tool(content: str, filename: str) -> str:
    """Tool to save research documents"""
    try:
        filepath = os.path.join(OUTPUT_DIR, f"{filename}.txt")
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(content)
        return f"Document saved as {filepath}"
    except Exception as e:
        return f"Error saving document: {e}"

def ingest_pdf_to_vectorstore(pdf_path: str, collection_name: str = "research") -> str:
    """Extract text from PDF -> chunk -> embed -> upsert to Chroma"""
    try:
        if not os.path.exists(pdf_path):
            return f"Error: PDF not found at {pdf_path}"

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=VECTOR_DIR
        )
        vectordb.add_documents(chunks)
        vectordb.persist()
        return f"Ingested {len(chunks)} chunks into {VECTOR_DIR} (collection: {collection_name})"
    except Exception as e:
        return f"Error during ingestion: {e}"

def get_retriever(collection_name: str = "research", k: int = 5):
    """Return a retriever over the persisted Chroma collection"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=VECTOR_DIR
    )
    return vectordb.as_retriever(search_kwargs={"k": k})

# Create LangChain tools
tools = [
    Tool(
        name="web_search",
        description="Search the web for information",
        func=web_search_tool
    ),
    Tool(
        name="save_document",
        description="Save content to a text file",
        func=save_document_tool
    ),
    Tool(
        name="ingest_pdf",
        description="Extract text from a PDF and ingest into vector DB",
        func=ingest_pdf_to_vectorstore
    )
]