# =============  imports   =============
# environment
from dotenv import load_dotenv

# HTTP Client Class
from openai import OpenAI

# image processing
from PIL import Image
import base64
from io import BytesIO

# web interface
import gradio as gr

# ==== environment setup ====
load_dotenv(override=True)

# ==== API Client ====
openai_client = OpenAI()

# ===== Model Selection =====
CHAT_MODEL = "gpt-4.1-mini"
IMAGE_MODEL = "dall-e-3"

# ===== System Prompt =====
system_message = (
    "You are a helpful assistant for a sheet metal and welding business called MetalWorksAI. "
    "You provide short, witty, and practical answers in one sentence. "
    "You can give rough estimates for welding jobs based on material type, alloy, size, and work type (indoor or outdoor). "
    "You can also generate images of sheet metal projects or welded structures if requested. "
    "Be professional but friendly, and provide helpful advice about metal types, welding techniques, and project feasibility."
)

# ===== Estimation Data =====
job_estimates = {
    "kitchen hood": "300 per foot",
    "fence panel": "$300 per yard",
    "weld ": "$100 per hour",
    "wall cover": "$250 per sheet",
}


def get_job_estimate(job):
    """
    Purpose:
        Return a rough cost estimate for a known welding or sheet metal job.
    Parameters:
        job (str): The name of the job (e.g., 'kitchen hood', 'fence panel').
    Returns:
        str: A human-readable cost estimate.
    """
    return f"The estimated cost for {job} is {job_estimates[job]}"


def artist_agent(job):
    """
    Purpose:
        Generate an AI image representing the requested metalwork job.
    Parameters:
        job (str): A brief description of the job or item.
    Returns:
        PIL.Image.Image: The generated image for display in Gradio.
    """
    image_response = openai_client.images.generate(
        model=IMAGE_MODEL,
        prompt=f"A image of {job} in art deco style.",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


def multi_agent_chat(history):
    """
    Purpose:
        Handle chat interaction between the user and the AI assistant.
        Detects whether the user is requesting a cost estimate or general advice.
        If a known job is mentioned, returns a price and image; otherwise,
        defers to the language model for a conversational answer.
    Parameters:
        history (list[dict]): List of chat messages exchanged so far.
    Returns:
        tuple: (updated chat history, generated image or None)
    """
    message = history[-1]["content"]
    messages = [{"role": "system", "content": system_message}] + history

    job = None
    for task in job_estimates.keys():
        if task in message.lower():
            job = task
            break

    if job:
        history.append({"role": "assistant", "content": get_job_estimate(job)})
        image = artist_agent(job)
    else:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL, messages=messages
        )
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        image = None

    return history, image


# ===== Web Interface =====
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=400, type="messages")
        image_output = gr.Image(height=400)
    with gr.Row():
        entry = gr.Textbox(label="Chat with AI assistant")

    def do_entry(message, history):
        """
        Purpose:
          Add the user's message to chat history and clear the input box.
        Parameters:
            message (str): Text entered by the user.
            history (list[dict]): Chat history so far.
        Returns:
            tuple: ('', updated_history)
        """
        if history is None:
            history = []
        history.append({"role": "user", "content": message})
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        multi_agent_chat, inputs=chatbot, outputs=[chatbot, image_output]
    )

ui.launch(inbrowser=True)
