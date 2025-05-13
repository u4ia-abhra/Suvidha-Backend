from flask import Flask, request, jsonify
from flask_cors import CORS
import generation
import vector
import os
import time
import psutil

app = Flask(__name__)
CORS(app, supports_credentials=True)

def log_memory(label=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[{label}] Memory usage: {mem_mb:.2f} MB")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        start_time = time.time()
        log_memory("START")

        print("⏳ Receiving request...")
        data = request.json
        domain = data.get('domain', '').strip()
        query = data.get('query', '').strip()

        if not domain:
            print("❌ Missing domain")
            return jsonify({"error": "Domain is required."}), 400
        if not query:
            print("❌ Missing query")
            return jsonify({"error": "Query is required."}), 400

        print(f"📚 Retrieving documents for domain: {domain}")
        retrieved_docs = vector.retrieve_docs(domain, query)
        if not retrieved_docs:
            return jsonify({"error": "No relevant documents found or retrieval failed."}), 404

        log_memory("AFTER DOC RETRIEVAL")

        context = "
".join(retrieved_docs)
        if len(context) > 2000:
            context = context[:2000] + "..."

        full_prompt = f"""You are Suvidha, a helpful customer support chatbot for the {domain} domain. 
Use the following knowledge base to answer the user's question accurately:

{context}

User: {query}
Assistant:"""

        print("🧠 Sending prompt to Gemini...")
        answer = generation.generate_response(domain, full_prompt)
        log_memory("AFTER GEMINI RESPONSE")

        print("✅ Finished generation.")
        print(f"⏱️ Total time: {time.time() - start_time:.2f} seconds")

        return jsonify({"response": answer})

    except Exception as e:
        print(f"💥 Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return "Server is live!", 200

@app.route('/', methods=['GET'])
def root():
    return "Chatbot server running.", 200

@app.route('/debug', methods=['GET'])
def debug():
    api_key = os.getenv("GOOGLE_API_KEY")
    data_path = os.path.exists("data/ecommerce_index.bin") and os.path.exists("data/ecommerce_documents.pkl")
    return jsonify({
        "google_api_key_found": bool(api_key),
        "faiss_index_found": data_path
    })

# Warm-up embedding model and Gemini model
_ = vector.embed_model  # Preload embedding model
# generation.py already loads the Gemini model at module level

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
