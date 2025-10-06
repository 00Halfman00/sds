"""import libraries"""

# env libraries
import os
from dotenv import load_dotenv

# HTTP Client
from openai import OpenAI

""" load environment variables """
load_dotenv(override=True)

""" create two seperate HTTP Client instances using OpenAI client class(one pointing to OpenAI REST API the other to Anthropic REST API) """
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_base_url = "https://api.anthropic.com/v1/"

openai = OpenAI()
anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_base_url)


""" SELECT TWO LLMs, CREATE SYSTEM MESSAGES/INSTRUCTIONS, AND GATHER MESSAGES FOR EACH """

gpt_model = "gpt-4.1-nano"
claude_model = "claude-3-5-haiku-latest"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

claude_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

gpt_messages = ["Hi there."]
claude_messages = ["Hi."]


def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]

    for gpt_msg, claude_msg in zip(gpt_messages, claude_messages):
        messages.append({"role": "assistant", "content": gpt_msg})
        messages.append({"role": "user", "content": claude_msg})

    response = openai.chat.completions.create(model=gpt_model, messages=messages)
    reply = response.choices[0].message.content
    return reply


def call_claude():
    messages = []

    for gpt_msg, claude_msg in zip(gpt_messages, claude_messages):
        messages.append({"role": "user", "content": gpt_msg})
        messages.append({"role": "assistant", "content": claude_msg})

    messages.append({"role": "user", "content": gpt_messages[-1]})
    response = anthropic.chat.completions.create(model=claude_model, messages=messages)
    reply = response.choices[0].message.content
    return reply


print(f"GPT:\n{gpt_messages[0]}\n\n")
print(f"GPT:\n{claude_messages[0]}\n\n")

for i in range(3):
    gpt_next = call_gpt()
    print(f"GPT:\n{gpt_next}\n\n")
    gpt_messages.append(gpt_next)

    claude_next = call_claude()
    print(f"Claude:\n{claude_next}\n\n")
    claude_messages.append(claude_next)
