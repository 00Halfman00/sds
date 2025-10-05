"""import libraries"""

# env library
from dotenv import load_dotenv

# HTTP Client
from openai import OpenAI

""" load environment variables """
load_dotenv(override=True)

""" create instance of  HTTP Client """
openai = OpenAI()

""" create system and user prompts to send to LLM """
messages1 = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the meaning of life?"},
]

""" Send request and get response from the Chat Completions API """
response = openai.chat.completions.create(
    model="gpt-4.1-mini", messages=messages1, max_tokens=50
)

reply = response.choices[0].message.content
print(reply)

""" create second system and user prompts to send to LLM """
messages2 = [
    {
        "role": "system",
        "content": "You are a helpful assistant that talks like a pirate",
    },
    {"role": "user", "content": "What is the meaning of life?"},
]

response = openai.chat.completions.create(
    model="gpt-4.1-mini", messages=messages2, max_tokens=50
)

reply = response.choices[0].message.content
print(reply)


def query_LLM(syst_prompt, user_prompt):
    messages = [
        {"role": "system", "content": syst_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = openai.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, max_tokens=50
    )
    reply = response.choices[0].message.content
    return reply


syst1 = "You are a helpful assistant that talks like a Yoda"
user1 = "What is the meaning of life?"

print(query_LLM(syst1, user1))
