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

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so. Make sure you reason through each step.

Context:
{context}
"""

vectorstore = Chroma(persist_directory=db_name, embedding_function=get_embeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
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
