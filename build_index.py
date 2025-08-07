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
    "ProductID: P002, ProductName: Laptop, Category: Electronics, Price: 1200.00, QuantityInStock: 50, Rating: 4.8, DateAdded: 2024-01-15",
    "ProductID: P003, ProductName: Coffee Maker, Category: Home Appliances, Price: 89.99, QuantityInStock: 100, Rating: 4.6, DateAdded: 2024-02-20",
    "ProductID: P004, ProductName: Desk Chair, Category: Office Supplies, Price: 150.50, QuantityInStock: 75, Rating: 4.7, DateAdded: 2024-03-10",
    "ProductID: P005, ProductName: Smartphone, Category: Electronics, Price: 799.99, QuantityInStock: 200, Rating: 4.9, DateAdded: 2024-04-05",
    "ProductID: P006, ProductName: Blender, Category: Home Appliances, Price: 45.00, QuantityInStock: 150, Rating: 4.4, DateAdded: 2024-05-22",
    "ProductID: P007, ProductName: Keyboard, Category: Electronics, Price: 75.00, QuantityInStock: 300, Rating: 4.5, DateAdded: 2024-06-18",
    "ProductID: P008, ProductName: Monitor, Category: Electronics, Price: 300.00, QuantityInStock: 120, Rating: 4.7, DateAdded: 2024-07-29",
    "ProductID: P009, ProductName: Webcam, Category: Electronics, Price: 55.00, QuantityInStock: 180, Rating: 4.3, DateAdded: 2024-08-11",
    "ProductID: P010, ProductName: Mousepad, Category: Office Supplies, Price: 15.00, QuantityInStock: 400, Rating: 4.2, DateAdded: 2024-09-01"
]

medical_docs = [
    "PatientID: P001, Name: John Doe, Age: 34, Gender: Male, Condition: Hypertension, Medication: Amlodipine, AppointmentDate: 2025-02-10, Doctor: Dr. Smith, Contact: 555-1234",
    "PatientID: P002, Name: Jane Smith, Age: 45, Gender: Female, Condition: Diabetes, Medication: Metformin, AppointmentDate: 2025-03-15, Doctor: Dr. Jones, Contact: 555-5678",
    "PatientID: P003, Name: Robert Johnson, Age: 56, Gender: Male, Condition: Arthritis, Medication: Ibuprofen, AppointmentDate: 2025-04-20, Doctor: Dr. Williams, Contact: 555-8765",
    "PatientID: P004, Name: Emily Davis, Age: 29, Gender: Female, Condition: Asthma, Medication: Albuterol, AppointmentDate: 2025-05-25, Doctor: Dr. Brown, Contact: 555-4321",
    "PatientID: P005, Name: Michael Wilson, Age: 67, Gender: Male, Condition: High Cholesterol, Medication: Atorvastatin, AppointmentDate: 2025-06-30, Doctor: Dr. Miller, Contact: 555-9876",
    "PatientID: P006, Name: Jessica Moore, Age: 38, Gender: Female, Condition: Migraine, Medication: Sumatriptan, AppointmentDate: 2025-07-05, Doctor: Dr. Taylor, Contact: 555-1122",
    "PatientID: P007, Name: Christopher Anderson, Age: 51, Gender: Male, Condition: Back Pain, Medication: Acetaminophen, AppointmentDate: 2025-08-10, Doctor: Dr. Thomas, Contact: 555-3344",
    "PatientID: P008, Name: Amanda Martinez, Age: 42, Gender: Female, Condition: Depression, Medication: Sertraline, AppointmentDate: 2025-09-15, Doctor: Dr. Hernandez, Contact: 555-5566",
    "PatientID: P009, Name: Daniel Garcia, Age: 60, Gender: Male, Condition: Allergies, Medication: Loratadine, AppointmentDate: 2025-10-20, Doctor: Dr. Rodriguez, Contact: 555-7788",
    "PatientID: P010, Name: Sarah Lopez, Age: 33, Gender: Female, Condition: Insomnia, Medication: Zolpidem, AppointmentDate: 2025-11-25, Doctor: Dr. Perez, Contact: 555-9900"
]

banking_docs = [
    "CustomerID: C001, Name: John Doe, Gender: Male, Age: 35, AccountType: Savings, AccountBalance: 15000.50, Branch: New York, TransactionID: T001, TransactionType: Deposit, TransactionAmount: 2000.00, TransactionDate: 2023-12-01",
    "CustomerID: C002, Name: Jane Smith, Gender: Female, Age: 45, AccountType: Checking, AccountBalance: 5000.75, Branch: Los Angeles, TransactionID: T002, TransactionType: Withdrawal, TransactionAmount: 500.00, TransactionDate: 2023-12-05",
    "CustomerID: C003, Name: Robert Johnson, Gender: Male, Age: 56, AccountType: Savings, AccountBalance: 25000.00, Branch: Chicago, TransactionID: T003, TransactionType: Deposit, TransactionAmount: 10000.00, TransactionDate: 2023-12-10",
    "CustomerID: C004, Name: Emily Davis, Gender: Female, Age: 29, AccountType: Checking, AccountBalance: 2500.25, Branch: Houston, TransactionID: T004, TransactionType: Withdrawal, TransactionAmount: 200.00, TransactionDate: 2023-12-12",
    "CustomerID: C005, Name: Michael Wilson, Gender: Male, Age: 67, AccountType: Savings, AccountBalance: 150000.00, Branch: Phoenix, TransactionID: T005, TransactionType: Deposit, TransactionAmount: 50000.00, TransactionDate: 2023-12-15",
    "CustomerID: C006, Name: Jessica Moore, Gender: Female, Age: 38, AccountType: Checking, AccountBalance: 7500.50, Branch: Philadelphia, TransactionID: T006, TransactionType: Withdrawal, TransactionAmount: 1000.00, TransactionDate: 2023-12-18",
    "CustomerID: C007, Name: Christopher Anderson, Gender: Male, Age: 51, AccountType: Savings, AccountBalance: 50000.00, Branch: San Antonio, TransactionID: T007, TransactionType: Deposit, TransactionAmount: 5000.00, TransactionDate: 2023-12-20",
    "CustomerID: C008, Name: Amanda Martinez, Gender: Female, Age: 42, AccountType: Checking, AccountBalance: 12000.00, Branch: San Diego, TransactionID: T008, TransactionType: Withdrawal, TransactionAmount: 1500.00, TransactionDate: 2023-12-22",
    "CustomerID: C009, Name: Daniel Garcia, Gender: Male, Age: 60, AccountType: Savings, AccountBalance: 75000.00, Branch: Dallas, TransactionID: T009, TransactionType: Deposit, TransactionAmount: 25000.00, TransactionDate: 2023-12-28",
    "CustomerID: C010, Name: Sarah Lopez, Gender: Female, Age: 33, AccountType: Checking, AccountBalance: 3000.00, Branch: San Jose, TransactionID: T010, TransactionType: Withdrawal, TransactionAmount: 300.00, TransactionDate: 2023-12-30"
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