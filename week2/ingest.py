# imported libraries
# # Standard library for interacting with the operating system (e.g., path base names).
import os

# Finds file paths matching a specified pattern (used for iterating through document folders).
import glob

# Needed for extracting employee name from filepath
import re

# Used to load documents from the file system (loads Markdown content from directories).
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# New splitter for structural content
from langchain.text_splitter import MarkdownHeaderTextSplitter

# The ChromaDB vector store wrapper for document storage and retrieval.
from langchain_chroma import Chroma

# Standard library for file system operations (used for deleting directories).
import shutil

# local imports
# # Provides the function to initialize the embedding model (e.g., OpenAI or Hugging Face).
from embeddings import get_embeddings

# Loads environment variables (like API keys) from a .env file.
from dotenv import load_dotenv

MODEL = "gpt-4.1-nano"
db_name = "vector_db"
knowledge_base_path = "knowledge-base/*"

USE_HUGGINGFACE = True
load_dotenv(override=True)


def fetch_documents():
    folders = glob.glob("knowledge-base/*")
    documents = []
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
    return documents


# def create_chunks(documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = text_splitter.split_documents(documents)
#     return chunks


def create_chunks(documents):
    """
    Splits documents based on Markdown headers (##) and prepends contextual
    information (Employee Name + Section Title) to the content of each chunk.
    This replaces the naive RecursiveCharacterTextSplitter.
    """
    # Define the headers to split the document on.
    # We focus on H2 (##) which separates logical sections like Career Progression,
    # Performance, Compensation, and Other HR Notes.
    headers_to_split_on = [("##", "Section_Title")]
    all_chunks = []
    # Initialize the Markdown splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        # We strip the headers from the content since we will use them to tag the chunk.
        strip_headers=True,
    )
    for doc in documents:
        # 1. Extract the Employee Name for contextual tagging
        filename = doc.metadata.get("source", "")
        # Uses regex to extract the name (e.g., 'Nina Patel') from the path
        match = re.search(r"/([^/]+)\.md$", filename)
        employee_name = (
            match.group(1).replace("-", " ") if match else "Unknown Employee"
        )
        # 2. Split the document content based on H2 headers
        chunks_from_doc = markdown_splitter.split_text(doc.page_content)
        # 3. Contextually Tag and Enrich each chunk
        for chunk in chunks_from_doc:
            # The splitter saves the header text (e.g., 'Annual Performance History')
            # into the metadata key 'Section_Title'
            section_title = chunk.metadata.get("Section_Title", "Summary")

            # Check if the chunk is just the preamble before the first H2
            if not chunk.page_content.strip() and section_title == "Summary":
                # Skip chunks that are empty or only contain the high-level HR Record/Name headers
                continue
            # Create the rich, contextually tagged content for better RAG retrieval
            # Example: "Nina Patel's Annual Performance History: [content...]"
            chunk.page_content = (
                f"{employee_name}'s {section_title}:\n" f"{chunk.page_content}"
            )
            # Merge original document metadata (like doc_type) with the new chunk metadata
            chunk.metadata = {**doc.metadata, **chunk.metadata}
            all_chunks.append(chunk)
    # Note: Using the Markdown splitter ensures no overlap, as each chunk is a complete section.
    return all_chunks


# def create_embeddings(chunks):
#     embeddings = get_embeddings()

#     # the deletion attempt will fail if the default collection name changes or is unknown, so make it explicit
#     # COLLECTION_NAME = "my_knowledge_collection"  # Define a name

#     # Always delete existing database to start fresh for lab experiments
#     if os.path.exists(db_name):
#         try:
#             # Try to properly delete the collection first
#             existing_vectorstore = Chroma(
#                 persist_directory=db_name,
#                 embedding_function=embeddings,
#                 # collection_name=COLLECTION_NAME,  # ADD THIS LINE
#             )
#             existing_vectorstore.delete_collection()
#             print(f"🗑️  Deleted existing collection from vector database at {db_name}")

#         except Exception as e:
#             print(f"⚠️  Could not delete collection properly: {e}")
#             # Fallback to directory deletion
#             import shutil

#             shutil.rmtree(db_name)
#             print(f"🗑️  Deleted entire vector database directory at {db_name}")

#     # Create new vectorstore
#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=db_name,
#         # collection_name=COLLECTION_NAME,  # ADD THIS LINE
#     )

#     collection = vectorstore._collection
#     count = collection.count()

#     sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
#     dimensions = len(sample_embedding)
#     print(f"✅ Created new vectorstore with {count:,} documents")
#     print(
#         f"📊 There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
#     )
#     return vectorstore


# EMBEDDING & DELETION
def create_embeddings(chunks):
    embeddings = get_embeddings()

    # Always delete existing database directory to start fresh for lab experiments
    if os.path.exists(db_name):
        try:
            # Use file system deletion (shutil.rmtree) as the primary, most reliable method
            shutil.rmtree(db_name)
            print(f"🗑️  Deleted entire vector database directory at {db_name}")

        except Exception as e:
            # This handles file locks or permission issues
            print(f"⚠️  Could not delete vector database directory at {db_name}: {e}")
            # If deletion fails, return None to halt the ingestion
            return None

    # Create new vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_name,
    )

    # Removed: vectorstore.persist()
    # Rationale: Persistence is handled automatically by recent LangChain/Chroma versions,
    # and calling persist() explicitly throws an AttributeError.

    collection = vectorstore._collection
    count = collection.count()

    # Ensure the collection exists before attempting to retrieve an item
    if count > 0:
        sample_embedding = collection.get(limit=1, include=["embeddings"])[
            "embeddings"
        ][0]
        dimensions = len(sample_embedding)
    else:
        # Handle case where no chunks were created (though checked in __main__)
        dimensions = 0

    print(f"✅ Created new vectorstore with {count:,} documents")
    print(
        f"📊 There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
    )
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")


"""
✅ Robust error handling added: The try/except block ensures that:
        It first tries to properly delete the collection using Chroma's API
        If that fails, it falls back to deleting the entire directory
   This prevents the "readonly database" errors you were seeing
"""


"""
The chunk_size and chunk_overlap parameters are no longer needed in the revised code and
have been removed for a very specific, strategic reason.

The shift from the old RecursiveCharacterTextSplitter to the
MarkdownHeaderTextSplitter changes the entire philosophy of how documents are segmented:

Why chunk_size is gone: Logical Integrity
Old Way (Character-based): The RecursiveCharacterTextSplitter chopped the document into arbitrary blocks of text (e.g., 500 characters), regardless of whether it cut off a sentence, a paragraph, or a section title.

New Way (Structural-based): The MarkdownHeaderTextSplitter groups text based on the logical structure defined by your Markdown headers (##). A chunk now represents a complete, self-contained section (like "Annual Performance History" or "Compensation History").

If we imposed a chunk_size=500 on this, we would defeat the entire purpose, as it might cut off a long performance review and reintroduce fragmentation. Structural splitting prioritizes completeness and context over fixed size.

Why chunk_overlap is gone: No Structural Gaps
Old Way (Needed Overlap): Overlap was essential with the old splitter because it created arbitrary cuts. Overlapping a few sentences ensured that the context didn't get lost when a chunk ended.

New Way (No Need for Overlap): Since each chunk is now a complete, logical section, there are no structural gaps between the chunks. The "Compensation History" chunk naturally begins where the "Annual Performance History" chunk ends, making overlap unnecessary and redundant.

In short, we are no longer relying on brute-force character splitting; we are using the documents' internal structure to create higher-quality, more contextually relevant chunks.

"""
