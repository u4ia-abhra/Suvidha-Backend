from flask import Flask, request, jsonify
import generation
import vector
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('query', '')

        if not query:

            return jsonify({"error": "Query is required."}), 400

        # Retrieve relevant documents
        retrieved_docs = vector.retrieve_docs(query)

        # Construct prompt for Gemini
        context = "\n".join(retrieved_docs)
        full_prompt = f"""You are a helpful customer support chatbot. Use the following knowledge base to answer the user's question accurately:
        
        {context}

        User: {query}
        Assistant:"""

        # Generate a response
        answer = generation.generate_response(full_prompt)

        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
