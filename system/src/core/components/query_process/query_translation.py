from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
import sys
from dotenv import load_dotenv

load_dotenv(dotenv_path="system/src/core/config/.env")

# === fix import module
project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import cohere


class QueryTranslator:
    def __init__(self, temperature: float=0.5, max_tokens: int=1000) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.llm = cohere.Client(
            api_key= os.getenv("COHERE_API_KEY"),
        )

    def translate_english_to_vietnamese(self, text: str) -> str:
        try:
            response = self.llm.chat(
                message= text,
                chat_history=[
                        {
                        "role": "SYSTEM",
                        "message": 
                            """
                            You are a professional translator. 
                            Translate the following text from English to Vietnamese. 
                            Ensure the translation is accurate, natural, and keeps the original tone. 
                            Do not explain or add any additional information or context to the translation.
                            """
                    }
                ]
                
            )
            return response.text
        except Exception as e:
            print(f"Error translating text: {e}")
            return None
    def translate_vietnamese_to_english(self, text: str) -> str:
        try:
            response = self.llm.chat(
                message= text,
                chat_history=[
                        {
                        "role": "SYSTEM",
                        "message": 
                            """
                            You are a professional translator. 
                            Translate the following text from Vietnamese to English. 
                            Ensure the translation is accurate, natural, and keeps the original tone. 
                            Do not explain or add any additional information or context to the translation.
                            """
                    }   
                ]
                
            )
            return response.text
        except Exception as e:
            print(f"Error translating text: {e}")
            return None
        
from system.src.core.components.api.api_keys import APIKeyManager

class QueryTranslator_GEMINI:
    def __init__(self, model: str, api_key: APIKeyManager,
                 temperature: float=0.5, max_tokens: int=1000) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            api_key=self.api_key.get_api_key(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
    def get_english_to_vietnamese(self) -> ChatPromptTemplate:
        prompt =ChatPromptTemplate.from_messages(
            messages=[
                ("system",
                """
                You are a professional translator. 
                Translate the following text from English to Vietnamese. 
                Ensure the translation is accurate, natural, and keeps the original tone:
                """),
                ("human", "Text to translate: {text}. DO NOT add any additional information or context to the translation."),
            ]
        )
        return prompt
    
    def get_vietnamese_to_english(self) -> ChatPromptTemplate:
        prompt =ChatPromptTemplate.from_messages(
            messages=[
                ("system",
                """
                You are a professional translator. 
                Translate the following text from Vietnamese to English. 
                Ensure the translation is accurate, natural, and keeps the original tone:
                """),
                ("human", "Text to translate: {text}. DO NOT add any additional information or context to the translation."),
            ]
        )
        return prompt

    def translate_english_to_vietnamese(self, text: str) -> str:
        try:
            prompt = self.get_english_to_vietnamese()
            generater = prompt | self.llm | StrOutputParser()
            response = generater.invoke({"text": text})
            print(f"Translated text: {response}")
            return response
        except Exception as e:
            print(f"Error translating text: {e}")
            return None
    def translate_vietnamese_to_english(self, text: str) -> str:
        try:
            prompt = self.get_vietnamese_to_english()
            generater = prompt | self.llm | StrOutputParser()
            response = generater.invoke({"text": text})
            print(f"Translated text: {response}")
            return response
        except Exception as e:
            print(f"Error translating text: {e}")
            return None