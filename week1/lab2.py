"""import libraries"""

# env library
from dotenv import load_dotenv

# HTTP Client
from openai import OpenAI

# load envrironment variables
load_dotenv(override=True)

# create HTTP Client
openai = OpenAI()

# Send request and get response from the Chat Completions API
response = openai.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content": "What is the capital of England?"}],
    temperature=0.5,
    max_tokens=50,
)

# view reply
reply = response.choices[0].message.content
print(reply)
