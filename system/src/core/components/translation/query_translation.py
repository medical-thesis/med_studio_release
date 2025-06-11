from langchain_community.llms import Cohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv(dotenv_path="system/core/src/config/.env")

from api import APIKeyManager

class QueryTranslator:
    def __init__(self, model: str, api_key: APIKeyManager, temperature: float=0.5, max_token: int=1000) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_token = max_token
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            api_key=self.api_key.get_api_key(),
            temperature=self.temperature,
            max_tokens=self.max_token,
        )
    def get_english_to_vietnamese(self) -> ChatPromptTemplate:
        prompt =ChatPromptTemplate.from_messages(
            messages=[
                ("system",
                """
                Please translate the following English text into Vietnamese.
                Make sure the translation is accurate, natural, and contextually appropriate for native Vietnamese speakers.
                Preserve the meaning and context of the original text.
                Do not add any additional information or change the meaning of the text.
                """),
                ("human", "Text to translate: {text}"),
            ]
        )
        return prompt
    
    def get_vietnamese_to_english(self) -> ChatPromptTemplate:
        prompt =ChatPromptTemplate.from_messages(
            messages=[
                ("system",
                """
                Please translate the following Vietnamese text into English.
                Make sure the translation is accurate, natural, and contextually appropriate for native English speakers.
                Preserve the meaning and context of the original text.
                Do not add any additional information or change the meaning of the text.
                """),
                ("human", "Text to translate: {text}"),
            ]
        )
        return prompt

    def translate_english_to_vietnamese(self, text: str) -> str:
        try:
            prompt = self.get_english_to_vietnamese()
            generater = prompt | self.llm | StrOutputParser()
            response = generater.invoke({"text": text})
            return response
        except Exception as e:
            print(f"Error translating text: {e}")
            return None
    def translate_vietnamese_to_english(self, text: str) -> str:
        try:
            prompt = self.get_vietnamese_to_english()
            generater = prompt | self.llm | StrOutputParser()
            response = generater.invoke({"text": text})
            return response
        except Exception as e:
            print(f"Error translating text: {e}")
            return None