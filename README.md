# 🤖 Suvidha - Domain-Specific AI Chatbot

Suvidha is a domain-aware AI chatbot that uses **Retrieval-Augmented Generation (RAG)** with **FAISS vector search** and **Gemini 1.5 Flash** LLM to deliver real-time customer support across various sectors. Integrated with a **Flutter** frontend for a smooth, mobile-friendly chat experience.

---

## 🔧 Features

- 🔍 Domain-Specific Retrieval (Banking, Ecommerce, Medical)
- 🧠 Gemini LLM Integration via Google Generative AI API
- ⚡ Fast FAISS-Based Vector Search
- 💬 Chat Interface with Flutter
- 🌐 Deployed Backend using Flask on Render
- 📊 Performance Monitoring (memory, latency logs)

---

## 🏗️ Tech Stack

| Layer       | Tech                        |
|------------|-----------------------------|
| Frontend   | Flutter, Dart                |
| Backend    | Python, Flask, FAISS         |
| LLM        | Gemini 1.5 Flash (Google)    |
| Embeddings | SentenceTransformers (MiniLM)|
| Deployment | Render                       |

---

## 🧠 Architecture

![Suvidha Architecture](assets/SuvidhaModel.png)



## 🚀 Running Locally

### Backend

```bash
cd backend/
pip install -r requirements.txt
python build_index.py
python app.py
```

### Frontend

```bash
cd frontend/
flutter pub get
flutter run
```

---

## 📸 Screenshots

_Add screenshots or a demo video link here._

---

## 🧪 Sample API Usage

**Endpoint:** `POST /chat`  
**Request Body:**
```json
{
    "domain": "ecommerce",
    "query": "What is the price of a wireless mouse?"
}
```

**Response:**
```json
{
    "response": "The price of the Wireless Mouse (ProductID: P001) is $19.99.\n"
}
```

---

## ✍️ Author

**Abhrajit Ghosh**  
B.Tech CSE @ KIIT | AI Enthusiast  
[LinkedIn](https://www.linkedin.com/in/abhrajitghosh/) | [GitHub](https://github.com/u4ia-abhra)

