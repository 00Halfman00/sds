"""import libraries"""

# HTTP Client
import requests

# env libraries
from dotenv import load_dotenv
import os


""" load environment variables  """
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

""" test api key is not None """
if api_key:
    print(f"Your api_key begins with these two letters {api_key[:2]}")


""" construct raw HTTP request """
url = "https://api.openai.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
data = {
    "model": "gpt-4.1-nano",
    "messages": [{"role": "user", "content": "Tell me a one-sentence fun fact."}],
}

""" make post request """
response = requests.post(url, headers=headers, json=data)

""" view reply """
reply = response.json()["choices"][0]["message"]["content"]

print(reply)
