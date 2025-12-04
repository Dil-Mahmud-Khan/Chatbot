import os

# Path to your text file
DATA_PATH = "smart_cities_overview.txt"

def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

text = load_text_file(DATA_PATH)

print("Loaded characters:", len(text))
print(text[:500])  # preview first 500 chars
def clean_text(text):
    # Replace line breaks with spaces
    text = text.replace("\n", " ")

    # Remove multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")

    # Strip spaces at beginning and end
    text = text.strip()

    return text

cleaned_text = clean_text(text)

print("\nCleaned characters:", len(cleaned_text))
print(cleaned_text[:500])  # preview



def chunk_text(text, max_words=250):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

chunks = chunk_text(cleaned_text, max_words=250)

print("\nTotal chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk[:300], "...")

from fastembed import TextEmbedding

embedder = TextEmbedding()

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        for vector in embedder.embed(chunk):
            embeddings.append(vector)  # vector is already a list of floats
    return embeddings

embeddings = embed_chunks(chunks)

print("\nGenerated embeddings:", len(embeddings))
print("Embedding vector length:", len(embeddings[0]))



import faiss
import numpy as np
import json

def build_faiss_index(embeddings):
    vectors = np.array(embeddings).astype("float32")
    
    dimension = vectors.shape[1]  # 384

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index, vectors

index, vector_array = build_faiss_index(embeddings)

print("\nFAISS index built successfully!")
print("Number of vectors indexed:", index.ntotal)


# Save FAISS index
faiss.write_index(index, "smartcities_index.faiss")

# Save chunks to a JSON file
with open("smartcities_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("\nSaved: smartcities_index.faiss and smartcities_chunks.json")


