from langchain_groq.chat_models import ChatGroq
from src.exception import CustomException
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    
    try:

        llm = ChatGroq(
            model_name = "llama-3.3-70b-versatile",
            temperature = 0.2
        )

        return llm
    
    except Exception as e:

        raise CustomException(e, sys)