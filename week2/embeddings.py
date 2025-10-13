# imported libraries
# Imports for OpenAI embedding functionality
from langchain_openai import OpenAIEmbeddings

# Imports for Hugging Face (local/remote) embedding functionality
from langchain_huggingface import HuggingFaceEmbeddings

# CONFIGURATION
USE_HUGGINGFACE = True

# MODEL = "all-MiniLM-L6-v2"
# --- Hugging Face Model Selection ---
# all-MiniLM-L6-v2 is fast, but BGE-small is the standard for high-performance open-source RAG.
# MODEL = "BAAI/bge-small-en-v1.5"


# --- Hugging Face Model Selection ---
# Upgraded to BGE-Large for better overall performance compared to BGE-Small.
# BAAI/bge-small-en-v1.5 is fast, but BGE-large provides higher semantic accuracy.
# MODEL = "BAAI/bge-large-en-v1.5"

# --- Hugging Face Model Selection ---
# Switched to BGE-Base, which offers a better balance of high performance
# and moderate computational resources compared to the larger BGE-Large model.
MODEL = "BAAI/bge-base-en-v1.5"

"""
If you're looking for an alternative that is highly popular and often tops benchmarks,
consider nomic-ai/nomic-embed-text-v1.5 or thenlper/gte-large.
"""


def get_embeddings():
    """
    Initializes and returns the selected embedding function.
    """
    if USE_HUGGINGFACE:
        # Returns the BGE-Base model, which downloads the model locally on first run.
        return HuggingFaceEmbeddings(model_name=MODEL)
    else:
        # Returns the OpenAI embedding model (requires OPENAI_API_KEY environment variable).
        return OpenAIEmbeddings()
