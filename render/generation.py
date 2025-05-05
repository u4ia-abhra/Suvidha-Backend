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
    response = model.generate_content(prompt)
    return response.text