# build_index.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents for each domain
ecommerce_docs = [
    "ProductID: P001, ProductName: Wireless Mouse, Category: Electronics, Price: 19.99, QuantityInStock: 250, Rating: 4.5, DateAdded: 2024-01-01",
    # ... (add the rest here)
]

medical_docs = [
    "PatientID: P001, Name: John Doe, Age: 34, Gender: Male, Condition: Hypertension, Medication: Amlodipine, AppointmentDate: 2025-02-10, Doctor: Dr. Smith, Contact: 555-1234",
    # ... (add the rest here)
]

banking_docs = [
    "CustomerID: C001, Name: John Doe, Gender: Male, Age: 35, AccountType: Savings, AccountBalance: 15000.50, Branch: New York, TransactionID: T001, TransactionType: Deposit, TransactionAmount: 2000.00, TransactionDate: 2023-12-01",
    # ... (add the rest here)
]

def create_faiss_index(docs, domain):
    vectors = embed_model.encode(docs)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))
    
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, f"data/{domain}_index.bin")
    with open(f"data/{domain}_documents.pkl", "wb") as f:
        pickle.dump(docs, f)

# Build and save
create_faiss_index(ecommerce_docs, "ecommerce")
create_faiss_index(medical_docs, "medical")
create_faiss_index(banking_docs, "banking")

print("Indexes built and saved in 'data/' folder.")
