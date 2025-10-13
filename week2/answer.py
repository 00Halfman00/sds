from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from embeddings import get_embeddings

# --- ADDED IMPORTS FOR EXPONENTIAL BACKOFF ---
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
db_name = "vector_db"

SYSTEM_PROMPT1 = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so. Make sure you reason through each step.

Context:
{context}
"""

# Your current prompt is good, but we can make the Chain-of-Thought requirement more explicit
# and add a specific guardrail against hallucination (a common issue when models are unsure).

SYSTEM_PROMPT = """
You are a knowledgeable, professional, and friendly assistant representing the company Insurellm.
Your primary role is to answer user questions about employee information, performance, and compensation.

### INSTRUCTIONS:
1. **Fact Check:** ONLY use the provided Context below. Do not use external knowledge.
2. **Chain-of-Thought:** Before generating the final answer, internally reason through the following steps:
    a. Identify the core question and the required data fields (e.g., employee name, year, bonus amount).
    b. Scan the Context for exact matches to those fields.
    c. If a fact cannot be found in the Context, explicitly state that the information is unavailable.
3. **Response:** Provide a concise and accurate answer based on your internal reasoning.

Context:
{context}
"""

vectorstore = Chroma(persist_directory=db_name, embedding_function=get_embeddings())
# Current Retriever (No Filter)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
# Proposed Improvement: Filter by a metadata field (e.g., 'Employee_Name')
# This requires knowing the employee name from the question, which is complex.
# The simplest implementation is often to use the 'score threshold'.
# Changed k=10 to k=6 to reduce noise and increase focus on the most relevant chunks.

# New: Use Maximum Marginal Relevance (MMR) for better top-K diversity
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 5,
#     },  # fetch_k: retrieve 50 candidates, then select best 5
# )
llm = ChatOpenAI(temperature=0, model_name=MODEL)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question)


@retry(  # <-- APPLIED DECORATOR HERE
    wait=wait_exponential(multiplier=1, min=10, max=240),
    stop=stop_after_attempt(10),
)
async def answer_question(question: str) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context.
    Protected by @retry to handle RateLimitErrors during LLM calls.
    """
    messages = [("system", SYSTEM_PROMPT), ("user", question)]
    prompt = ChatPromptTemplate.from_messages(messages)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # The actual LLM API call happens inside this ainvoke method
    response = await rag_chain.ainvoke({"input": question})

    return response["answer"], response["context"]


"""
You're absolutely correct! While we protected the LLM-as-a-Judge function in eval.py,
the core RAG generation call inside answer.py is still exposed to rate limits.

To fully guard against the error,
we need to apply the same Exponential Backoff mechanism to the answer_question function in answer.py.
This ensures that if the main RAG call fails due to hitting the tokens-per-minute limit,
it automatically waits and retries.

Here is the updated answer.py file with the necessary imports and the @retry decorator applied to answer_question.
RAG Answer Generation with Tenacity
Oct 10, 2:52 PM

Why This Fix Works
By decorating answer_question, we now have two layers of protection in your answer evaluation workflow:
answer.py (answer_question): Protects the initial RAG generation call,
which is the most token-intensive step.

eval.py (evaluate_answer): Protects the second LLM call,
which is the structured evaluation by the LLM-as-a-Judge.

With both functions using tenacity, your evaluation can now run automatically,
waiting and retrying when the rate limit is momentarily reached.
"""


# from langchain_openai import ChatOpenAI
# from langchain_chroma import Chroma
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document
# from dotenv import load_dotenv
# from embeddings import get_embeddings

# load_dotenv(override=True)

# MODEL = "gpt-4.1-nano"
# db_name = "vector_db"

# SYSTEM_PROMPT = """
# You are a knowledgeable, friendly assistant representing the company Insurellm.
# You are chatting with a user about Insurellm.
# If relevant, use the given context to answer any question.
# If you don't know the answer, say so. Make sure you reason through each step.

# Context:
# {context}
# """

# vectorstore = Chroma(persist_directory=db_name, embedding_function=get_embeddings())
# retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
# llm = ChatOpenAI(temperature=0, model_name=MODEL)


# def fetch_context(question: str) -> list[Document]:
#     """
#     Retrieve relevant context documents for a question.
#     """
#     return retriever.invoke(question)


# async def answer_question(question: str) -> tuple[str, list]:
#     """
#     Answer a question using RAG and return the answer and the retrieved context
#     """
#     messages = [("system", SYSTEM_PROMPT), ("user", question)]
#     prompt = ChatPromptTemplate.from_messages(messages)
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#     response = await rag_chain.ainvoke({"input": question})

#     return response["answer"], response["context"]
