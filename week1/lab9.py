import os
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import display
import gradio as gr

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4.1-mini"
openai = OpenAI()

system_msg = "You are a helpful assistant for an Airline called FlightAI. "
system_msg += "Give short, witty, snarky answers, no more than 1 sentence."
system_msg += "If relevant, use the get_ticket_price tool to get the price of a ticket to a given city."

""" CHAT LLM """


def chat(message, history):
    messages = (
        [{"role": "system", "content": system_msg}]
        + history
        + [{"role": "user", "content": message}]
    )
    chat_response = openai.chat.completions.create(model=MODEL, messages=messages)
    return chat_response.choices[0].message.content


import base64
from io import BytesIO

""" IMAGE LLM """


def artist_agent(city):
    image_response = openai.images.generate(
        model="dall-e-3",
        prompt=f"An image representing a vacation in {city}, showing tourist sites.",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))
