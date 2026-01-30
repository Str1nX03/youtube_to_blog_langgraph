from langchain_groq.chat_models import ChatGroq
import os

def get_llm():
    
    llm = ChatGroq(
        model_name = "llama-3.1-8b-instant",
        temperature = 0.2,
        api_key = os.getenv("GROQ_API_KEY")
    )
    return llm