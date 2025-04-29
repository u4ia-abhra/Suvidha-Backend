from flask import Flask, request, jsonify
import generation
import vector

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json

        # Get domain and query from request
        domain = data.get('domain', '').strip()
        query = data.get('query', '').strip()

        if not domain:
            return jsonify({"error": "Domain is required."}), 400
        if not query:
            return jsonify({"error": "Query is required."}), 400

        # Retrieve relevant documents
        retrieved_docs = vector.retrieve_docs(domain, query)

        if not retrieved_docs:
            return jsonify({"error": "No relevant documents found."}), 404

        context = "\n".join(retrieved_docs)
        full_prompt = f"""You are a helpful customer support chatbot for the {domain} domain. 
Use the following knowledge base to answer the user's question accurately:

{context}

User: {query}
Assistant:"""

        # Generate response using Gemini
        answer = generation.generate_response(domain, full_prompt)

        return jsonify({"response": answer})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Health check route
@app.route('/ping', methods=['GET'])
def ping():
    return "Server is live!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render injects PORT variable
    app.run(host="0.0.0.0", port=port)