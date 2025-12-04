import streamlit as st
import faiss
import numpy as np
import json
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load FAISS index & chunks
index = faiss.read_index("smartcities_index.faiss")

with open("smartcities_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Embedding model
embedder = TextEmbedding()

def retrieve_chunks(query, k=1):
    query_vector = None
    for v in embedder.embed(query):
        query_vector = np.array(v).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

def answer_with_groq(context, question):
    prompt = f"""
You are an AI assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

If the answer is not in the context, say: "The context does not provide that information."
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------

st.title(" Smart Cities RAG Chatbot")
st.write("Welcome to Dil's LLM.")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("Your question:")

if st.button("Ask"):
    if user_input.strip() != "":
        context = "\n\n".join(retrieve_chunks(user_input, k=1))
        answer = answer_with_groq(context, user_input)

        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", answer))


# --------------------------------------------------------------------
# Clean Left-Right ChatGPT Style (No Background)
# --------------------------------------------------------------------

st.markdown("""
<style>
.user-msg {
    text-align: right;
    padding: 5px;
    margin: 5px 0;
    font-weight: bold;
}
.bot-msg {
    text-align: left;
    padding: 5px;
    margin: 5px 0;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.markdown(f"<div class='user-msg'>🧑 <b>You:</b> {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 <b>Bot:</b> {msg}</div>", unsafe_allow_html=True)
