from flask import Flask, request, jsonify
import generation
import vector
import os

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        domain = data.get('domain', '').strip()
        query = data.get('query', '').strip()

        if not domain:
            return jsonify({"error": "Domain is required."}), 400
        if not query:
            return jsonify({"error": "Query is required."}), 400

        retrieved_docs = vector.retrieve_docs(domain, query)
        if not retrieved_docs:
            return jsonify({"error": "No relevant documents found."}), 404

        context = "\n".join(retrieved_docs)
        full_prompt = f"""You are a helpful customer support chatbot for the {domain} domain. 
Use the following knowledge base to answer the user's question accurately:

{context}

User: {query}
Assistant:"""

        print("Starting generation...")
        answer = generation.generate_response(domain, full_prompt)
        print("Finished generation.")

        return jsonify({"response": answer})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return "Server is live!", 200