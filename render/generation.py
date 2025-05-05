# generates a response by sending the retrieved data to the LLM
import os
import time
import psutil
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def log_memory(label=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[{label}] Memory usage: {mem_mb:.2f} MB")

def generate_response(domain, prompt):
    try:
        print(f"🚀 Starting generation for domain: {domain}")
        start_time = time.time()
        log_memory("GENERATION START")

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        log_memory("GENERATION END")
        print(f"✅ Generation complete in {time.time() - start_time:.2f} seconds")

        return response.text

    except Exception as e:
        print(f"💥 Gemini Generation Error: {e}")
        return "Sorry, something went wrong generating the response."
