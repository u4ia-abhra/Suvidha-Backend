import google.generativeai as genai
import vector

# Configure Gemini API
genai.configure(api_key="AIzaSyDG5UnxQXW6JFUaLPQiGWiJ9w6ezKDHDwc")

def generate_response(query):
    retrieved_docs = vector.retrieve_docs(query)
    context = "\n".join(retrieved_docs)

    prompt = f"""You are a customer support assistant. Use the following knowledge base to answer the query:
    {context}

    User: {query}
    Assistant:"""

    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)

    return response.text
