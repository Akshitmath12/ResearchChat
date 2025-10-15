import os
from dotenv import load_dotenv

load_dotenv()


LLM_PROVIDER = "ollama"  


OLLAMA_MODEL = "llama2"  
OLLAMA_BASE_URL = "http://localhost:11434"


HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


TEMPERATURE = 0.7


OUTPUT_DIR = "outputs"
MAX_SUBTOPICS = 4

# RAG / Vector settings
VECTOR_DIR = "vectordb"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure vector directory exists
os.makedirs(VECTOR_DIR, exist_ok=True)


os.makedirs(OUTPUT_DIR, exist_ok=True)