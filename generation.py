import os
import google.generativeai as genai
from dotenv import load_dotenv
from logger import logger

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def generate_response(domain: str, prompt: str) -> str:
    """
    Generates a response from the Gemini model.

    Args:
        domain: The domain of the request.
        prompt: The prompt to send to the model.

    Returns:
        The generated response.
    """
    try:
        logger.info(f"🚀 Starting generation for domain: {domain}")

        # Truncate prompt/context to 1000 chars for speed
        prompt = prompt[:1000]

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        logger.info(f"✅ Generation complete.")

        return response.text

    except Exception as e:
        logger.error(f"💥 Gemini Generation Error: {e}")
        return "Sorry, something went wrong generating the response."

async def generate_response_async(domain: str, prompt: str) -> str:
    """
    Generates a response from the Gemini model asynchronously.

    Args:
        domain: The domain of the request.
        prompt: The prompt to send to the model.

    Returns:
        The generated response.
    """
    try:
        logger.info(f"🚀 Starting async generation for domain: {domain}")

        # Truncate prompt/context to 1000 chars for speed
        prompt = prompt[:1000]

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = await model.generate_content_async(prompt)

        logger.info(f"✅ Async generation complete.")

        return response.text

    except Exception as e:
        logger.error(f"💥 Gemini Async Generation Error: {e}")
        return "Sorry, something went wrong generating the response."
