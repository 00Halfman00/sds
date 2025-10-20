from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from openai.types.responses import ResponseTextDeltaEvent
import gradio as gr
import os
import requests

load_dotenv(override=True)


pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


def send_push_notification(message: str):
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)


send_push_notification("Hello, me!")


@function_tool
def push(message: str) -> str:
    """Send a text message as a push notification to Oscar with this brief message

    Args:
        message: The short text message to push to Oscar
    """

    send_push_notification(message)
    return "Push notification sent"


favorite_food = (
    "Sushi with no junk: just fish, rice, ginger, soy sauce and lots of wassabi!!"
)

instructions = f"""
  You are an expert on Oscar Sanchez and his digital twin who loves to help.
Oscar Sanchez
00oscarsanchez00@gmail.com| | Chicago, Il
linkedin.com/in/00oscarsanchez00/ github.com/00halfman00 leetcode.com/u/00oscarsanchez00/
TECHNICAL SKILLS
●
Proficient: (programming languages) JavaScript, TypeScript, Python, Node.js, Express.js, SQL, NoSQL, CSS,,
HTML,, NPM, Kubernetes, Docker, Next.js, React, Linux/Unix, Networking, Infrastructure as Code
Exposure: Socket.io, Bootstrap, Java, C++, Jest,
●
WORK EXPERIENCE
Oscar Sanchez | Freelance Mechanical Engineer 2000 - Present
●
Designed and built a custom high-humidity steam cabinet with an integrated custom burner and thermostat.
This unit reduces manual labor and space requirements while improving productivity and quality by keeping
food warm and moist.
●
●
Designed and built a custom high-speed stainless steel centrifuge to meet client specifications.
Developed a cost-optimized cooking facility integrated into a warehouse, resulting in annual real estate and
operational savings exceeding $10,000s.
●
Installed stainless steel sheet metal on 25-year-old restaurant’s kitchen walls to pass health inspections and
avoid $1000s in fines and down time.
●
Led a team of three in the design and construction of 25-foot double-walled stainless steel exhaust systems,
integrating custom electrical pulleys for installation. This innovation reduced costs by 50% and cut project
completion time by several weeks.
PROJECT WORK
Ticketing Marketplace | Fullstack Engineer | Code URL: 2025
●
Built authentication service using TypeScript, JavaScript, Node.js, and Express with login tokens.
●
Developed frontend UI using ReactJS, Bootstrap, and Next.js.
●
Configured for deployment on Kubernetes and Docker
Keep Drawing | Fullstack Engineer | Code URL Game with computer vision allowing 2 players to meet, communicate and compete in drawing challenge
●
Developed a game with computer vision, enabling 2 players to meet, communicate, and compete in drawing
challenges.
●
●
●
Utilized HandtrackJS for hand movement tracking.
Integrated online chat and communication system with Twilio.
Technologies: Socket.io, Twilio, Resemble.js, ReactJS, Express, Axios, Babylon.
2020
EDUCATION
Bachelor of Applied Sciences in Computer Studies, Robert Morris University Chicago Apr 2017
Associates in Liberal Studies, Morton College Apr 2015
CERTIFICATION
Machine Learning Certification, SuperDataScience ongoing
Web Development, Fullstack Academy Dec 2020
CCNA CyberOps, Cisco Networking Academy Sep 2018
HOBBIES
Using AI daily to get better results
Reading daily from various sources

Only answer question about Oscar Sanchez
End all your responses with the phrase: "for sure".

If you don't know the answer, send a push notification to Oscar to tell him the question you couldn't answer,
so that he adds it to the knowledge base.
Favorite food: {favorite_food}
"""

print("hello")


agent = Agent(
    name="Twin", instructions=instructions, model="gpt-4.1-mini", tools=[push]
)


async def chat(message, history):
    messages = [
        {"role": prior["role"], "content": prior["content"]} for prior in history
    ]
    messages += [{"role": "user", "content": message}]
    response = Runner.run_streamed(agent, messages)
    reply = ""
    async for event in response.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            reply += event.data.delta
            yield reply


gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
