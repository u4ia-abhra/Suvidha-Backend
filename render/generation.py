#generates a response by seding the retrieved data to the LLM
import os
import google.generativeai as genai
import vector
from dotenv import load_dotenv

load_dotenv()
# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
def generate_response(domain, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Sorry, something went wrong generating the response."