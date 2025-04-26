from flask import Flask, request, jsonify
import faiss  # Assuming you're using FAISS for vector search

app = Flask(__name__)

# Global variable for FAISS index, initially set to None
faiss_index = None

# Lazy load FAISS index
def load_faiss():
    global faiss_index
    if faiss_index is None:
        # Only load FAISS when needed
        faiss_index = faiss.read_index("your_index_file.index")  # Path to your FAISS index
    return faiss_index

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('query')
    
    # Lazy load FAISS only when required
    index = load_faiss()
    
    # Your RAG logic using FAISS + Gemini API
    # For example: fetch relevant documents using FAISS, then query Gemini

    # Dummy response for now
    response = {"response": f"Processing query: {user_query}"}
    return jsonify(response)

@app.route('/ping', methods=['GET'])
def ping():
    return "Server is live!", 200

if __name__ == "__main__":
    app.run()
