import faiss
import numpy as np
import json
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load FAISS index
index = faiss.read_index("smartcities_index.faiss")

# Load chunks
with open("smartcities_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Initialize embedding model
embedder = TextEmbedding()

def retrieve_relevant_chunks(query, k=1):
    # Generate embedding for the query
    query_vector = None
    for v in embedder.embed(query):
        query_vector = np.array(v).astype("float32").reshape(1, -1)

    # Search FAISS
    distances, indices = index.search(query_vector, k)

    # Retrieve matching chunk text
    return [chunks[i] for i in indices[0]]

def ask_groq(context, question):
    prompt = f"""
You are an AI assistant that answers questions using context from Smart Cities research.

Context:
{context}

Question:
{question}

Answer ONLY using the information in the context.
If the context does not contain the answer, say: "The context does not provide that information."
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def chat():
    print("\nSmart Cities RAG Chatbot")
    print("Ask a question, or type 'exit' to quit.\n")

    while True:
        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        # Retrieve context
        context_chunks = retrieve_relevant_chunks(question, k=1)
        context = "\n\n".join(context_chunks)

        # Get answer from Groq
        answer = ask_groq(context, question)

        print("\nChatbot:", answer, "\n")

if __name__ == "__main__":
    chat()

