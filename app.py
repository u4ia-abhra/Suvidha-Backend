from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import generation
from vector import VectorSearcher
import os
from config import SUPPORTED_DOMAINS
from logger import logger

from cachetools import TTLCache
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load vector searchers for all domains at startup
VECTOR_SEARCHERS = {domain: VectorSearcher(domain) for domain in SUPPORTED_DOMAINS}

# Cache for storing responses
cache = TTLCache(maxsize=100, ttl=300) # Cache up to 100 items for 5 minutes

def generate_audio(text):
    """Generates audio from text using gTTS."""
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        logger.error(f"💥 Error in gTTS generation: {str(e)}")
        return None

@app.route('/chat', methods=['POST'])
def chat() -> Response:
    """
    Handles chat requests from the user.
    Can return a JSON response with the chatbot's answer or an audio file.
    """
    try:
        logger.info("⏳ Receiving request...")
        data = request.json
        if not data:
            logger.warning("❌ Missing request body")
            return jsonify({"error": "Request body is missing."}), 400

        domain = data.get('domain', '').strip()
        query = data.get('query', '').strip()
        speak = data.get('speak', False)

        if not domain:
            logger.warning("❌ Missing domain")
            return jsonify({"error": "Domain is required."}), 400
        if not query:
            logger.warning("❌ Missing query")
            return jsonify({"error": "Query is required."}), 400
        
        if domain not in VECTOR_SEARCHERS and domain != "combined":
            logger.warning(f"❌ Invalid domain: {domain}")
            valid_domains = list(VECTOR_SEARCHERS.keys()) + ["combined"]
            return jsonify({"error": f"Invalid domain specified. Available domains: {valid_domains}"}), 400

        # Check cache first for text response
        cache_key = f"{domain}:{query}"
        if not speak and cache_key in cache:
            logger.info(f"✅ Cache hit for key: {cache_key}")
            return jsonify({"response": cache[cache_key]})

        # If audio is requested, we might still have the text cached
        if speak and cache_key in cache:
            answer = cache[cache_key]
            logger.info(f"✅ Cache hit for key: {cache_key}. Generating audio.")
        else:
            if domain == "combined":
                logger.info(f"📚 Retrieving documents for all domains")
                all_docs = []
                for domain_name, searcher in VECTOR_SEARCHERS.items():
                    retrieved_docs = searcher.retrieve_docs(query)
                    if retrieved_docs:
                        all_docs.extend(retrieved_docs)
                retrieved_docs = all_docs
            else:
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
                answer = generation.generate_response(domain, full_prompt)
            except Exception as e:
                logger.error(f"💥 Generation error: {e}")
                return jsonify({"error": "An error occurred during response generation."}), 500
            
            # Store text response in cache
            cache[cache_key] = answer
            logger.info("✅ Finished generation.")

        if speak:
            logger.info("🎤 Generating audio...")
            audio_fp = generate_audio(answer)
            if audio_fp:
                logger.info("✅ Audio generated.")
                return Response(audio_fp, mimetype="audio/mpeg")
            else:
                # Fallback to returning the text response if audio generation fails
                return jsonify({"response": answer, "error": "Failed to generate audio."})

        return jsonify({"response": answer})

    except Exception as e:
        logger.error(f"💥 Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping() -> Response:
    """A simple endpoint to check if the server is live."""
    return "Server is live!", 200

@app.route('/transcribe', methods=['POST'])
def transcribe_audio_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                return jsonify({"transcript": text})
            except sr.UnknownValueError:
                return jsonify({"error": "Unable to recognize speech."}), 500
            except sr.RequestError:
                return jsonify({"error": "Could not request results from speech recognition service."}), 500

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
