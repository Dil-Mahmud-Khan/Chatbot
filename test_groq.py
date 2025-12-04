from groq import Groq
import os
from dotenv import load_dotenv

print("Current directory:", os.getcwd())
print("Files:", os.listdir("."))

# Load .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print("Loaded API KEY:", api_key[:10] + "...")

client = Groq(api_key=api_key)

resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "Hello! What are smart cities?"}
    ]
)

print("\n--- Groq Response ---\n")
print(resp.choices[0].message.content)
