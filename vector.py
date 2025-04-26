# Converts the given data into vector embeddings (faiss) and retrieves relevant data
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample knowledge base
docs =  [
    "ProductID: P001, ProductName: Wireless Mouse, Category: Electronics, Price: 19.99, QuantityInStock: 250, Rating: 4.5, DateAdded: 2024-01-01",
    "ProductID: P002, ProductName: Laptop Stand, Category: Accessories, Price: 29.99, QuantityInStock: 150, Rating: 4.7, DateAdded: 2024-01-03",
    "ProductID: P003, ProductName: USB-C Cable, Category: Electronics, Price: 9.99, QuantityInStock: 500, Rating: 4.3, DateAdded: 2024-01-05",
    "ProductID: P004, ProductName: Coffee Mug, Category: Home, Price: 12.49, QuantityInStock: 300, Rating: 4.6, DateAdded: 2024-01-08",
    "ProductID: P005, ProductName: Office Chair, Category: Furniture, Price: 129.99, QuantityInStock: 75, Rating: 4.8, DateAdded: 2024-01-10",
    "ProductID: P006, ProductName: LED Desk Lamp, Category: Home, Price: 24.99, QuantityInStock: 200, Rating: 4.4, DateAdded: 2024-01-12",
    "ProductID: P007, ProductName: Wireless Earbuds, Category: Electronics, Price: 49.99, QuantityInStock: 100, Rating: 4.2, DateAdded: 2024-01-15",
    "ProductID: P008, ProductName: Fitness Tracker, Category: Health, Price: 99.99, QuantityInStock: 50, Rating: 4.9, DateAdded: 2024-01-18",
    "ProductID: P009, ProductName: Running Shoes, Category: Fashion, Price: 79.99, QuantityInStock: 120, Rating: 4.6, DateAdded: 2024-01-20",
    "ProductID: P010, ProductName: Smartphone Case, Category: Accessories, Price: 14.99, QuantityInStock: 400, Rating: 4.1, DateAdded: 2024-01-25",
    "PatientID: P001, Name: John Doe, Age: 34, Gender: Male, Condition: Hypertension, Medication: Amlodipine, AppointmentDate: 2025-02-10, Doctor: Dr. Smith, Contact: 555-1234",
    "PatientID: P002, Name: Jane Smith, Age: 28, Gender: Female, Condition: Diabetes, Medication: Metformin, AppointmentDate: 2025-02-12, Doctor: Dr. Johnson, Contact: 555-5678",
    "PatientID: P003, Name: Ali Khan, Age: 45, Gender: Male, Condition: Asthma, Medication: Salbutamol, AppointmentDate: 2025-02-15, Doctor: Dr. Lee, Contact: 555-9101",
    "PatientID: P004, Name: Maria Gonzalez, Age: 52, Gender: Female, Condition: Arthritis, Medication: Ibuprofen, AppointmentDate: 2025-02-20, Doctor: Dr. Brown, Contact: 555-1122",
    "PatientID: P005, Name: Liam O'Brien, Age: 37, Gender: Male, Condition: Flu, Medication: Paracetamol, AppointmentDate: 2025-01-18, Doctor: Dr. White, Contact: 555-3344",
    "PatientID: P006, Name: Emma Davis, Age: 62, Gender: Female, Condition: Osteoporosis, Medication: Alendronate, AppointmentDate: 2025-03-01, Doctor: Dr. Green, Contact: 555-5566",
    "PatientID: P007, Name: Ravi Patel, Age: 50, Gender: Male, Condition: Hypertension, Medication: Lisinopril, AppointmentDate: 2025-02-25, Doctor: Dr. Smith, Contact: 555-7788",
    "PatientID: P008, Name: Chen Wei, Age: 30, Gender: Female, Condition: Allergy, Medication: Loratadine, AppointmentDate: 2025-03-05, Doctor: Dr. Johnson, Contact: 555-9900",
    "PatientID: P009, Name: Aisha Mohammed, Age: 43, Gender: Female, Condition: Thyroid Disorder, Medication: Levothyroxine, AppointmentDate: 2025-02-18, Doctor: Dr. Brown, Contact: 555-2233",
    "PatientID: P010, Name: Michael Brown, Age: 39, Gender: Male, Condition: Migraine, Medication: Sumatriptan, AppointmentDate: 2025-01-25, Doctor: Dr. Green, Contact: 555-4455",
    "CustomerID: C001, Name: John Doe, Gender: Male, Age: 35, AccountType: Savings, AccountBalance: 15000.50, Branch: New York, TransactionID: T001, TransactionType: Deposit, TransactionAmount: 2000.00, TransactionDate: 2023-12-01",
    "CustomerID: C002, Name: Jane Smith, Gender: Female, Age: 28, AccountType: Checking, AccountBalance: 8500.75, Branch: Los Angeles, TransactionID: T002, TransactionType: Withdrawal, TransactionAmount: 500.00, TransactionDate: 2023-12-02",
    "CustomerID: C003, Name: Robert Brown, Gender: Male, Age: 42, AccountType: Savings, AccountBalance: 20000.00, Branch: Chicago, TransactionID: T003, TransactionType: Deposit, TransactionAmount: 1000.00, TransactionDate: 2023-12-03",
    "CustomerID: C004, Name: Emily Davis, Gender: Female, Age: 31, AccountType: Checking, AccountBalance: 12500.20, Branch: Houston, TransactionID: T004, TransactionType: Withdrawal, TransactionAmount: 300.00, TransactionDate: 2023-12-04",
    "CustomerID: C005, Name: Michael Wilson, Gender: Male, Age: 29, AccountType: Savings, AccountBalance: 18000.00, Branch: San Francisco, TransactionID: T005, TransactionType: Deposit, TransactionAmount: 1500.00, TransactionDate: 2023-12-05",
    "CustomerID: C006, Name: Sophia Taylor, Gender: Female, Age: 34, AccountType: Checking, AccountBalance: 9500.50, Branch: Miami, TransactionID: T006, TransactionType: Withdrawal, TransactionAmount: 700.00, TransactionDate: 2023-12-06",
    "CustomerID: C007, Name: James Anderson, Gender: Male, Age: 45, AccountType: Savings, AccountBalance: 22000.75, Branch: Dallas, TransactionID: T007, TransactionType: Deposit, TransactionAmount: 2500.00, TransactionDate: 2023-12-07",
    "CustomerID: C008, Name: Olivia Martinez, Gender: Female, Age: 27, AccountType: Checking, AccountBalance: 10500.90, Branch: Seattle, TransactionID: T008, TransactionType: Withdrawal, TransactionAmount: 450.00, TransactionDate: 2023-12-08",
    "CustomerID: C009, Name: Liam Johnson, Gender: Male, Age: 33, AccountType: Savings, AccountBalance: 19500.60, Branch: Denver, TransactionID: T009, TransactionType: Deposit, TransactionAmount: 1800.00, TransactionDate: 2023-12-09",
    "CustomerID: C010, Name: Ava White, Gender: Female, Age: 30, AccountType: Checking, AccountBalance: 9700.40, Branch: Atlanta, TransactionID: T010, TransactionType: Withdrawal, TransactionAmount: 600.00, TransactionDate: 2023-12-10"

]

# Convert text into vectors
vectors = embed_model.encode(docs)
dimension = vectors.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))

# Save the index and documents for later use
faiss.write_index(index, "faiss_index.bin")
with open("documents.pkl", "wb") as f:
    pickle.dump(docs, f)

print("Knowledge base stored in FAISS successfully!")

def retrieve_docs(query, top_k=3):
    # Load FAISS index and documents
    index = faiss.read_index("faiss_index.bin")
    with open("documents.pkl", "rb") as f:
        docs = pickle.load(f)

    # Convert query to vector
    query_vec = embed_model.encode([query]).astype(np.float32)

    # Search in FAISS
    distances, indices = index.search(query_vec, top_k)

    # Retrieve relevant docs
    retrieved_texts = [docs[i] for i in indices[0]]
    return retrieved_texts
