# ---------- Imports ----------
# loads enviroment variables from a .env file into system environment
from dotenv import load_dotenv

# classes used to define and coordinate AI agents
from agents import Agent, Runner

# quick and simple web interface
import gradio as gr

# stuctured data models; Field adds validation and metadata
from pydantic import BaseModel, Field

# client that interacts with OpenAI APIs
from openai import OpenAI

# provides tools for writing asynchronous code
import asyncio

# provides logs of questiones asked in text file
import logging


# ---------- Setup ----------
# Load environment variables from .env file into system, replacing existing environment variables
load_dotenv(override=True)
# Initialize OpenAI API client (used to send requests to the OpenAI API)
openai_client = OpenAI()
CHAT_MODEL = "gpt-4.1-mini"


# ---------- Helper Functions ----------
# read info from local file
def load_linkedin_profile():
    """load linkedIn profile for Oscar from text file"""
    with open("./backend/linkedIn_profile.txt", "r") as file:
        return file.read()


# clean list of messages to include only the keys: role and content
def clean_messages(messages):
    """Ensure messages have only 'role and 'content' keys."""
    cleaned = []
    for msg in messages:
        clean_msg = {"role": msg.get("role"), "content": msg.get("content")}
        cleaned.append(clean_msg)
    return cleaned


"""  ---------- Data Model ----------  """


# Define a data model for the chat response
class ChatResponse(BaseModel):
    # LLM-generated response text based on Oscar's linkedIn profile
    llm_response: str = Field(
        description="The response from LLM based on linkedIn profile on Oscar."
    )


"""  ----------  Load user's profile  ----------  """
# load profile and normalize whitespace
profile = " ".join(load_linkedin_profile().split())
# include users favorite food
favorite_food = (
    "Sushi with no junk: just fish, rice, ginger, soy sauce and lots of wassabi!!"
)


# ---------- System Prompt ( Instructions for the LLM on how to respond ) ----------
SYSTEM_PROMPT = f"""
You are Oscar Sanchez’s professional digital assistant.

Your purpose is to answer questions about Oscar accurately and concisely
using the information provided in his LinkedIn profile and known context.

Context:
- Oscar’s favorite food is: {favorite_food}.
- The LinkedIn profile text is provided in full (accessible to you as context).
- The user may be a recruiter, employer, collaborator, or colleague.

Guidelines:
1. Only answer questions related to Oscar’s professional life, education, skills,
   career achievements, or professional interests, except for favorite food.
2. If the question is personal or unrelated to work, except for favorite food, politely redirect:
   "I’m Oscar’s professional assistant and can only answer work-related questions."
3. Never fabricate or speculate about Oscar’s life outside of professional context.
4. Respond in a professional, warm, and articulate tone suitable for a business conversation.
5. Keep responses brief (under 120 words) unless a detailed summary is requested.
6. If you don’t have enough information, say:
   "That information isn’t included in Oscar’s profile."

Your responses should sound natural, informative, and reflect Oscar’s expertise.
"""

# ---------- Agent Definition ----------
digital_twin_agent = Agent(
    name="Digital_Twin",
    instructions=SYSTEM_PROMPT,
    model=CHAT_MODEL,
    output_type=ChatResponse,
)

# ---------- Logging Setup ----------
logging.basicConfig(
    filename="chat_logs.txt", level=logging.INFO, format="%(asctime)s %(message)s"
)


# ---------- Chat Function ----------
async def chat(message: str, history: list) -> str:
    """
    message: new user message (string)
    history: list of [user_message, bot_message] pairs maintained by Gradio
    """
    # Build conversation in OpenAI-style message format
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        if user_msg:
            conversation.append({"role": "user", "content": user_msg})
        if bot_msg:
            conversation.append({"role": "assistant", "content": bot_msg})

    # Add the new user message
    conversation.append({"role": "user", "content": message})
    # clean the message list before sending
    safe_history = clean_messages(conversation)

    try:
        response = await Runner.run(digital_twin_agent, safe_history)
        parsed = response.final_output_as(ChatResponse)
        final_response = parsed.llm_response
    except Exception as e:
        logging.error(f"Error during LLM call: {e}")
        final_response = "Sorry, I encountered an issue generating that response."

    logging.info(f"User: {message}\nAssistant: {final_response}\n---")
    return final_response


async def main():
    """launch gradio interface"""
    gr.ChatInterface(
        chat,
        title="Ask Oscar’s Digital Twin",
        description="Ask about Oscar Sanchez’s professional background, skills, and experience.",
        examples=[
            ["What are Oscar’s main programming skills?"],
            ["Can you summarize Oscar’s recent projects?"],
            ["What industries has Oscar worked in?"],
        ],
        theme="gradio/soft",
    ).launch()


# ---------- Launch ----------
if __name__ == "__main__":
    asyncio.run(main())
