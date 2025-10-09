from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

USE_HUGGINGFACE = True
# MODEL = "all-MiniLM-L6-v2"
# --- Hugging Face Model Selection ---
# all-MiniLM-L6-v2 is fast, but BGE-small is the standard for high-performance open-source RAG.
MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings():
    if USE_HUGGINGFACE:
        return HuggingFaceEmbeddings(model_name=MODEL)
    else:
        return OpenAIEmbeddings()
