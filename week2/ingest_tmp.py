import os
import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_chroma import Chroma
from embeddings import get_embeddings
from dotenv import load_dotenv

# Import necessary components for parent-child chunking
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

MODEL = "gpt-4.1-nano"
db_name = "vector_db"
knowledge_base_path = "knowledge-base/*"

# --- Configuration for Experiments ---
# Set the strategy you want to use here:
# 1. "DENSE_CHUNKS" (Smaller, overlapping chunks)
# 2. "PARENT_CHILD" (Small chunks for retrieval, large chunks for generation)
# 3. "DEFAULT" (Your original 500/100 configuration)
CHUNKING_STRATEGY = "PARENT_CHILD"
# -----------------------------------


USE_HUGGINGFACE = True

load_dotenv(override=True)


def fetch_documents():
    """Loads all markdown files from the knowledge-base directory."""
    folders = glob.glob("knowledge-base/*")
    documents = []
    print(f"📄 Found {len(folders)} knowledge base folders to process.")
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    print(f"📝 Total documents loaded: {len(documents)}")
    return documents


def create_chunks(documents):
    """
    Creates chunks based on the selected CHUNKING_STRATEGY.
    Returns:
        tuple: (chunks, retriever_type, parent_document_retriever_instance or None)
    """
    strategy = CHUNKING_STRATEGY

    if strategy == "DENSE_CHUNKS":
        print("⚙️  Using Strategy: DENSE_CHUNKS (256/50)")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        return chunks, "Standard", None

    elif strategy == "PARENT_CHILD":
        print("⚙️  Using Strategy: PARENT_CHILD (Retrieval/Context)")

        # 1. Parent Document Splitter (The full context sent to the LLM)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        # 2. Child Document Splitter (The small pieces used for embedding/retrieval)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=20
        )

        # We need to initialize the vectorstore here but won't persist it yet
        embeddings = get_embeddings()
        vectorstore = Chroma(
            persist_directory=db_name,
            embedding_function=embeddings,
        )

        # The Store is used to map child chunk IDs to parent document content
        store = InMemoryStore()

        # Create the ParentDocumentRetriever
        parent_document_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        # Ingest the documents into the retriever (this creates both parent and child chunks internally)
        parent_document_retriever.add_documents(documents)

        # For PARENT_CHILD, we don't return chunks, but the retriever itself,
        # which handles both storage and retrieval logic.
        print("💡 Parent-Child ingestion complete. The retriever is ready.")
        return [], "ParentChild", parent_document_retriever

    else:  # DEFAULT
        print("⚙️  Using Strategy: DEFAULT (500/100)")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        return chunks, "Standard", None


def create_embeddings(chunks, retriever_type, parent_document_retriever):
    """
    Creates or updates the vector database based on the selected chunking strategy.
    """
    embeddings = get_embeddings()

    # Always delete existing database to start fresh for lab experiments
    if os.path.exists(db_name):
        import shutil

        shutil.rmtree(db_name)
        print(f"🗑️  Deleted entire vector database directory at {db_name}")

    if retriever_type == "ParentChild":
        # The data is already stored inside the parent_document_retriever's vectorstore.
        # We just need to persist the vectorstore if it's Chroma (which it is).
        # Note: LangChain's ParentDocumentRetriever handles the storage automatically
        # when add_documents is called in create_chunks.

        # Since the Chroma vectorstore was initialized in create_chunks,
        # we can't easily get the final count here without more effort.
        # For simplicity, we assume the ingestion was successful if we reached this point.
        print("✅ Parent-Child vectorstore created and ready for use.")
        return parent_document_retriever

    else:  # Standard chunking
        # Create new vectorstore from the chunks
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_name,
        )

        collection = vectorstore._collection
        count = collection.count()

        sample_embedding = collection.get(limit=1, include=["embeddings"])[
            "embeddings"
        ][0]
        dimensions = len(sample_embedding)
        print(f"✅ Created new vectorstore with {count:,} documents")
        print(
            f"📊 There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
        )
        return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks, retriever_type, parent_document_retriever = create_chunks(documents)

    if retriever_type == "ParentChild":
        # When using ParentDocumentRetriever, the vectorstore creation is integrated
        # into the create_chunks function (via add_documents). We just return the retriever.
        final_retriever = parent_document_retriever
    else:
        # Standard workflow: create embeddings from the generated chunks
        final_retriever = create_embeddings(chunks, retriever_type, None)

    print("\nIngestion complete.")
    print(
        f"Next step: Use the generated vectorstore or retriever in your RAG pipeline."
    )
