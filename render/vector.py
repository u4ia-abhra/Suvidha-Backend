# vector.py — Loads FAISS indexes and retrieves domain-specific documents
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

# Load embedding model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieval function
def retrieve_docs(domain, query, top_k=3):
    try:
        index_path = f"data/{domain}_index.bin"
        docs_path = f"data/{domain}_documents.pkl"

        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            print(f"[vector.py] Missing FAISS index or docs for domain: {domain}")
            return None

        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)

        query_vec = embed_model.encode([query]).astype(np.float32)
        distances, indices = index.search(query_vec, top_k)

        retrieved = [docs[i] for i in indices[0] if i < len(docs)]
        if not retrieved:
            print(f"[vector.py] No documents found for query: {query}")
            return None

        return retrieved

    except Exception as e:
        print(f"[vector.py] Retrieval error: {e}")
        return None
