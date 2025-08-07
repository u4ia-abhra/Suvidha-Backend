# vector.py — Loads FAISS indexes and retrieves domain-specific documents
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Optional
import concurrent.futures

from logger import logger

# Load the embedding model once at module level
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

class VectorSearcher:
    """A class to handle vector-based document retrieval for a specific domain."""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.index_path = f"data/{self.domain}_index.bin"
        self.docs_path = f"data/{self.domain}_documents.pkl"
        self.index = None
        self.docs = []
        self._load()

    def _load(self):
        """Loads the FAISS index and documents from disk."""
        logger.info(f"[{self.domain}] Loading FAISS index from {self.index_path}...")
        if not os.path.exists(self.index_path) or not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Missing FAISS index or documents for domain: {self.domain}")
        
        self.index = faiss.read_index(self.index_path)
        logger.info(f"[{self.domain}] FAISS index loaded successfully.")

        logger.info(f"[{self.domain}] Loading documents from {self.docs_path}...")
        with open(self.docs_path, "rb") as f:
            self.docs = pickle.load(f)
        logger.info(f"[{self.domain}] Documents loaded successfully.")

    def retrieve_docs(self, query: str, top_k: int = 10) -> Optional[List[str]]:
        """
        Retrieves the top_k most relevant documents for a given query.
        Uses a timeout to ensure retrieval is fast.
        """
        if self.index is None or not self.docs:
            logger.warning(f"[{self.domain}] Index or documents not loaded.")
            return None
        
        def _search():
            query_vec = embed_model.encode([query]).astype(np.float32)
            distances, indices = self.index.search(query_vec, top_k)
            return [self.docs[i] for i in indices[0] if i < len(self.docs)]

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_search)
                retrieved = future.result(timeout=5.0)  # 5.0s timeout for retrieval
            if not retrieved:
                logger.warning(f"[{self.domain}] No documents found for query: {query}")
                return None
            return retrieved

        except concurrent.futures.TimeoutError:
            logger.error(f"[{self.domain}] Retrieval timed out (>0.5s)")
            return None
        except Exception as e:
            logger.error(f"[{self.domain}] Retrieval error: {e}")
            return None