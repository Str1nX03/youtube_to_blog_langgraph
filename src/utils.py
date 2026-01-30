from langchain_groq.chat_models import ChatGroq
from src.exception import CustomException
from src.logger import logging
import os
import sys

def get_llm():
    
    try:

        logging.info("Getting LLM")

        llm = ChatGroq(
            model_name = "llama-3.1-8b-instant",
            temperature = 0.2,
            api_key = os.getenv("GROQ_API_KEY")
        )

        logging.info("LLM retrieved successfully")

        return llm
    
    except Exception as e:

        raise CustomException(e, sys)