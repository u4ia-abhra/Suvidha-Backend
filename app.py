from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import generation
from vector import VectorSearcher
import os
from config import SUPPORTED_DOMAINS
from logger import logger
import asyncio
from cachetools import TTLCache

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load vector searchers for all domains at startup
VECTOR_SEARCHERS = {domain: VectorSearcher(domain) for domain in SUPPORTED_DOMAINS}

# Cache for storing responses
cache = TTLCache(maxsize=100, ttl=300) # Cache up to 100 items for 5 minutes

@app.route('/chat', methods=['POST'])
async def chat() -> Response:
    """
    Handles chat requests from the user.

    Returns:
        A JSON response with the chatbot's answer.
    """
    try:
        logger.info("⏳ Receiving request...")
        data = request.json
        domain = data.get('domain', '').strip()
        query = data.get('query', '').strip()

        if not domain:
            logger.warning("❌ Missing domain")
            return jsonify({"error": "Domain is required."}), 400
        if not query:
            logger.warning("❌ Missing query")
            return jsonify({"error": "Query is required."}), 400
        
        if domain not in VECTOR_SEARCHERS:
            logger.warning(f"❌ Invalid domain: {domain}")
            return jsonify({"error": f"Invalid domain specified. Available domains: {list(VECTOR_SEARCHERS.keys())}"}), 400

        # Check cache first
        cache_key = f"{domain}:{query}"
        if cache_key in cache:
            logger.info(f"✅ Cache hit for key: {cache_key}")
            return jsonify({"response": cache[cache_key]})

        logger.info(f"📚 Retrieving documents for domain: {domain}")
        searcher = VECTOR_SEARCHERS[domain]
        retrieved_docs = searcher.retrieve_docs(query)
        if not retrieved_docs:
            logger.warning("No relevant documents found or retrieval failed.")
            return jsonify({"error": "No relevant documents found or retrieval failed."}), 404

        context = "\n".join(retrieved_docs)

        full_prompt = f"Answer the user's query based on the following context:\n\n{context}\n\nUser: {query}"

        logger.info("🧠 Sending prompt to Gemini...")
        try:
            # Set timeout to 2.8 seconds to ensure response < 3 seconds
            answer = await asyncio.wait_for(
                generation.generate_response_async(domain, full_prompt),
                timeout=2.8
            )
        except asyncio.TimeoutError:
            logger.error("💥 Generation timed out (>2.8s)")
            return jsonify({"error": "Response took too long. Please try again."}), 504
        
        # Store response in cache
        cache[cache_key] = answer

        logger.info("✅ Finished generation.")

        return jsonify({"response": answer})

    except Exception as e:
        logger.error(f"💥 Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping() -> Response:
    """A simple endpoint to check if the server is live."""
    return "Server is live!", 200

@app.route('/', methods=['GET'])
def root() -> Response:
    """The root endpoint of the server."""
    return "Chatbot server running.", 200

@app.route('/debug', methods=['GET'])
def debug() -> Response:
    """An endpoint to debug the server's state."""
    api_key = os.getenv("GOOGLE_API_KEY")
    # Check if all domain data is loaded
    all_data_loaded = all(
        searcher.index is not None and searcher.docs is not None 
        for searcher in VECTOR_SEARCHERS.values()
    )
    return jsonify({
        "google_api_key_found": bool(api_key),
        "all_domain_data_loaded": all_data_loaded,
        "loaded_domains": list(VECTOR_SEARCHERS.keys())
    })

# The vector searchers are already initialized at the top

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
