#generates a response by seding the retrieved data to the LLM
import os
import google.generativeai as genai
import vector
from dotenv import load_dotenv

load_dotenv()
# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
def generate_response(domain,query):
    retrieved_docs = vector.retrieve_docs(domain,query)
    context = "\n".join(retrieved_docs)

    prompt = f"""You are a customer support assistant. Use the following knowledge base to answer the query:
    {context}

    User: {query}
    Assistant:"""

    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)

    return response.text
