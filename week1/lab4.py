"""import libraries"""

# env library
import os
from dotenv import load_dotenv

# HTTP Client
from openai import OpenAI

import sys
import time
import json

""" load environment variables """
load_dotenv(override=True)


""" get api keys for LLMs """
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if openai_api_key:
    print(f"The first two characters for open_api_key {openai_api_key[:2]}")

if anthropic_api_key:
    print(f"The first two characters for open_api_key {anthropic_api_key[:2]}")

""" get url for each LLMs """
anthropic_url = "https://api.anthropic.com/v1/"


""" create HTTP Client for each LLM """
openai = OpenAI()
anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)


""" create message to send to LLMS """
message = "What is the meaning of life. Keep it down to a couple of sentences?"
messages = [{"role": "user", "content": message}]

""" how can I test connection to each client? """
models = []
answers = []


def answer(client, model):
    print(f"\n### Response from {model}:\n")
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    reply = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            reply += delta
            # Print new tokens as they arrive, flush immediately
            print(delta, end="", flush=True)

    # Small pause for readability
    time.sleep(0.5)

    # Postprocess (optional)
    words_section = reply.split("</think>")[1] if "</think>" in reply else reply
    word_count = len(words_section.split())

    print(f"\n\n---\nWord count: {word_count}\n")

    # Save results for later comparison
    models.append(model)
    answers.append(reply)

    return reply


answer(openai, "gpt-4.1-mini")
answer(openai, "gpt-5-nano")
answer(anthropic, "claude-sonnet-4-20250514")

together = ""
for index, answer in enumerate(answers):
    together += f"# Response from competitor {index+1}\n\n"
    together += answer + "\n\n"


print(together)

print("\n\n")

judge = f"""You are judging a competition between {len(models)} competitors.
Each model has been given this question:

{message}

Your job is to evaluate each response for clarity and strength of argument and accuracy of word count, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

Here are the responses from each competitor:

{together}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""


judge_messages = [{"role": "user", "content": judge}]
response = openai.chat.completions.create(model="gpt-4.1-mini", messages=judge_messages)
results = response.choices[0].message.content

results_dict = json.loads(results)
ranks = results_dict["results"]
for index, result in enumerate(ranks):
    competitor = models[int(result) - 1]
    print(f"Rank {index+1}: {competitor}")
