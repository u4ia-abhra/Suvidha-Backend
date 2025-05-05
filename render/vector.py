# vector.py — Loads FAISS indexes and retrieves domain-specific documents
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load embedding model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieval function
def retrieve_docs(domain, query, top_k=3):
    # Load FAISS index and corresponding docs
    index = faiss.read_index(f"data/{domain}_index.bin")
    with open(f"data/{domain}_documents.pkl", "rb") as f:
        docs = pickle.load(f)

    # Encode query and search
    query_vec = embed_model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)

    # Return the most relevant documents
    return [docs[i] for i in indices[0]]