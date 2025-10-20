import os
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr


#Load Environment Variables
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Selected LLMs and OpenAI Client instance (Http calls to the LLMs)
CHAT_MODEL = "gpt-4.1-mini"
IMAGE_MODEL = "dall-e-3"
openai = OpenAI()


# BASE64 ENCODING
import base64
from io import BytesIO


# IMAGE LLM
def artist_agent(city):
    image_response = openai.images.generate(
        model=IMAGE_MODEL,
        prompt=f"An image representing a vacation in {city}, showing tourist sites.",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


# SYSTEM MESSAGE
system_msg = "You are a helpful assistant for an Airline called FlightAI. "
system_msg += "Give short, witty, snarky answers, no more than 1 sentence."

# TOOLS
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "sydney": "$2999"}


def get_ticket_price(city):
    return f"The price of a ticket to {city.title()} is {ticket_prices[city]}."


""" CHAT LLM WITH IMAGE LLM """


def mixed_chat(history):
    message = history[-1]["content"]
    messages = [{"role": "system", "content": system_msg}] + history

    city = None
    for dest in ticket_prices.keys():
        if dest in message.lower():
            city = dest
            break
    if city:
        reply = get_ticket_price(city)
        image = artist_agent(city)
    else:
        response = openai.chat.completions.create(model=CHAT_MODEL, messages=messages)
        reply = response.choices[0].message.content
        image = None

    history += [{"role": "assistant", "content": reply}]
    return history, image


""" GRADIO UI """
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=400, type="messages")
        image_output = gr.Image(height=400)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")

    def do_entry(message, history):
        history += [{"role": "user", "content": message}]
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        mixed_chat, inputs=chatbot, outputs=[chatbot, image_output]
    )

ui.launch(inbrowser=True)
