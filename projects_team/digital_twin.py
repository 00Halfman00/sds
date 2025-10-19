"""------------------  Imports  ------------------"""

# Load environment variables from a .env file into system environment
from pyexpat.errors import messages
from dotenv import load_dotenv

# Classes used to define and coordinate AI agents
from agents import Agent, Runner

# Quick and simple web interface
import gradio as gr

# Structured, data models; Field adds validation and metadata
from gradio.themes import monochrome
from pydantic import BaseModel, Field

# Client that interacts with OpenAI APIs
from openai import OpenAI

# Provides tools for writting asynchronous code
import asyncio

# Provides logs of queries/inputs into text file
import logging


"""  -------------------  Setup  -----------------  """
# Load environment variables from a .env file into system, replacing existing environment variables
load_dotenv(override=True)
# Initialize OpenAI client (used to send request to the OpenAI API)
openai_client = OpenAI()
# LLM selected for chat
CHAT_MODEL = "gpt-4.1-mini"
# User to be represented by digital twin Agent
PERSONA = "Oscar"


"""  --------------------  Helper functions  ------------------  """


# Load local text file for user
def load_linkedIn_profile(persona: str) -> str:
    """Load persona's linkedIn profile from local file."""
    with open(f"./backend/{persona}_linkedIn_profile.txt", "r") as file:
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
    # LLM generated response text based on Persona's LinkedIn profile
    llm_response: str = Field(
        description="The response from LLM based on Persona's LinkedIn profile."
    )


"""  ----------  Load Persona's profile  ----------  """
# Load Persona's LinkedIn profile
profile_text = " ".join(load_linkedIn_profile(PERSONA)).split()
# Persona's favorite food
favorite_food = (
    "Sushi with no junk: just fish, rice, ginger, soy sauce and lots of wassabi!!"
)

# ---------- System Prompt ( Instructions for the LLM on how to respond ) ----------
SYSTEM_PROMPT = f"""
You are {PERSONA}'s professional digital assistant.

Your purpose is to answer questions about {PERSONA}'s accurately and consicely
using the information provided in the LinkedIn profile and known context.

--- START OF LINKEDIN PROFILE ---
{profile_text}
--- END OF LINKEDIN PROFILE ---

Context:
- {PERSONA}'s favorite food is: {favorite_food}.
- The LinkedIn profile text is provided in full (accessible to you as context).
- The user may be a recruiter, employer, collaborator or colleague.

Guidelines:
1.  Only anser questions related to {PERSONA}'s professional lief, education, skills,
    career achievements or professional interest, except for favorite food.
2.  If the question is personal or unrelated to work, except for favorite food, politely redirect:
    "I’m {PERSONA}'s professional assistant and can only answer work-related questions."
3.  Never fabricate or speculate about {PERSONA}'s life outside of professional context.
4.  Respond in a professional, warm, and articulate tone suitable for a business conversation.
5.  Keep responses brief (in under 120 words) unless a detailed summary is requested.
6.  If you don't have enough information, say:
    "That information isn't included in {PERSONA}'s profile.

Your responses should sound natural, informative, and reflect {PERSONA}'s expertise.
"""

#  ------------  Agent Definition  -----------
digital_twin = Agent(
    name="Digital_Twin",
    instructions=SYSTEM_PROMPT,
    model=CHAT_MODEL,
    output_type=ChatResponse,
)

#  ---------- Logging Setup  ------------
logging.basicConfig(
    filename="chat_logs.txt", level=logging.INFO, format="%(asctime)s %(message)s"
)


# ---------- Chat Function ----------
async def chat(message: str, history: list) -> str:
    """
    message: user message (str)
    history: list of dictionaries in the format [{'role': str, 'content': str}]
             due to gr.ChatInterface(type="messages").
    """
    # 1. Build the conversation payload for the LLM.
    # Start with the essential System Prompt.
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 2. Append the prior conversation history.
    # When type="messages" is used, 'history' is already a list of dictionaries
    # in the correct OpenAI format (excluding the system prompt).
    conversation.extend(history)

    # 3. Append the current user message (passed separately as a string).
    conversation.append({"role": "user", "content": message})

    safe_history = clean_messages(conversation)

    # --- API Call and Response Processing (Try/Except Block) ---
    try:
        # 1. Asynchronously run the agent with the full conversation history.
        response = await Runner.run(digital_twin, safe_history)

        # 2. Parse the unstructured API response, validating it against the Pydantic data model.
        parsed = response.final_output_as(ChatResponse)

        # 3. Extract the final, clean response string.
        final_response = parsed.llm_response
    except Exception as e:
        # CATCHES any error that occurs during the try block.

        # Log the detailed error to the file (chat_logs.txt).
        logging.error(f"Error during LLM call: {e}")

        # Provide a safe, user-friendly fallback message.
        final_response = "Sorry, I encountered an issue generating that response."

    logging.info(f"User:{message}\nAssistant:{final_response}\n ----")
    # Return the final message string to Gradio.
    return final_response


async def main():
    """launch gradio interface"""
    gr.ChatInterface(
        chat,
        title=f"Ask {PERSONA}'s Digital Twin",
        description=f"Ask about {PERSONA}'s professional background, skills and experience.",
        # Re-enabled type="messages" to use the recommended, non-deprecated history format.
        type="messages",
        examples=[
            [f"What are {PERSONA}'s main programming skills?"],
            [f"Can you summarize {PERSONA}'s recent projects?"],
            [f"What industries has {PERSONA} worked in?"],
        ],
        theme="gradio/monochrome",
    ).launch()


#  --------------- Launch Gradio  -------------
if __name__ == "__main__":
    asyncio.run(main())


"""  ------------------  Test functionality of code function by function  -------------------  """
# print(load_linkedIn_profile("oscar_sanchez")) ------ passing
