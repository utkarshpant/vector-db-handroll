import os
import requests
from uuid import uuid4
from dotenv import load_dotenv
from app.utils.openai import client

load_dotenv()

API_URL = "http://localhost:8000/library"


def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def main():
    # 1. Create a new library
    lib_resp = requests.post(
        f"{API_URL}/",
        json={"name": "Demo Library", "metadata": {"purpose": "demo"}}
    )
    lib_resp.raise_for_status()
    lib_id = lib_resp.json()["id"]
    print(f"Created library: {lib_id}")

    # 2. Prepare chunks from paragraphs
    text = """FastAPI is a modern, fast web framework for building APIs with Python.
It is based on standard Python type hints and Pydantic models.
OpenAI provides powerful APIs for natural language processing tasks.
Vector databases enable efficient similarity search for embeddings."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    chunks = []
    for para in paragraphs:
        embedding = get_embedding(para)
        chunks.append({
            "id": str(uuid4()),
            "metadata": {"text": para},
            "embedding": embedding
        })

    # 3. Upsert chunks
    upsert_resp = requests.put(
        f"{API_URL}/{lib_id}/chunks",
        json={"chunks": chunks}
    )
    upsert_resp.raise_for_status()
    print(f"Upserted {len(chunks)} chunks.")

    # 4. Query the database
    query = "What is FastAPI?"
    query_embedding = get_embedding(query)
    search_resp = requests.post(
        f"{API_URL}/{lib_id}/search",
        json={"query": query_embedding},
        params={"k": 2}
    )
    search_resp.raise_for_status()
    results = search_resp.json()
    print("Top results:")
    for res in results:
        print("-", res[0]['metadata'], res[1])

if __name__ == "__main__":
    main()