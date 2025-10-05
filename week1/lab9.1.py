"""import necessary libraries"""

# environment libraries
import os
from dotenv import load_dotenv

# openai client
from openai import OpenAI

# image processing
from PIL import Image
import base64
from io import BytesIO

# web interface
import gradio as gr

""" load environment variables """
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

""" selected llms and openai client instance (http calls to the llms) """
CHAT_MODEL = "gpt-4.1-mini"
IMAGE_MODEL = "dall-e-3"
openai = OpenAI()

""" create a system prompt """

system_msg = "You are a helpful assistant for an airlined called FlightAI."
system_msg += "Give short, wity, snarky answers that are one sentence long"

""" tools """
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "sydney": "$1700"}


def get_ticket_prices(city):
    return f"A flight to {city} cost {ticket_prices[city]}"


""" function to generate an image of a city """


def artist_agent(city):
    image_response = openai.images.generate(
        model=IMAGE_MODEL,
        prompt=f"An image representing a vacation in a {city}, showing tourist sites in art decko style.",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


""" function to generate chat and call image function  """


def mixed_chat(history):
    message = history[-1]["content"]
    messages = [{"role": "system", "content": system_msg}] + history

    city = None
    for dest in ticket_prices.keys():
        if dest in message.lower():
            city = dest
            break

    if city:
        history.append({"role": "assistant", "content": get_ticket_prices(city)})
        image = artist_agent(city)
    else:
        response = openai.chat.completions.create(model=CHAT_MODEL, messages=messages)
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        image = None

    return history, image


""" function to deploy web interface """

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=400, type="messages")
        image_output = gr.Image(height=400)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI assistant:")

    def do_entry(message, history):
        if history is None:
            history = []
        history += [{"role": "user", "content": message}]
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        mixed_chat, inputs=chatbot, outputs=[chatbot, image_output]
    )

ui.launch(inbrowser=True)
