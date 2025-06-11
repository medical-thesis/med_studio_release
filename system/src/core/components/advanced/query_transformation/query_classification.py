import os
import sys
from dotenv import load_dotenv

load_dotenv(dotenv_path="system/core/src/config/.env")

# === fix import module
project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from system.src.core.components.api.api_keys import APIKeyManager

class QueryClassification:
    def __init__(self, model: str, api_key: APIKeyManager, temperature: float= 0.5, max_tokens: int= 1000) -> None:
        try:
            self.api_key = api_key
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            
            # self.llm = ChatGoogleGenerativeAI(
            #     model= self.model,
            #     api_key= self.api_key.get_api_key(),
            #     temperature= self.temperature,
            #     max_tokens= self.max_tokens,
            # )
            from langchain_community.llms import Cohere
            self.llm = Cohere(
                model= "command",
                cohere_api_key= os.getenv("COHERE_API_KEY"),
                temperature= self.temperature,
                max_tokens= self.max_tokens,
            )
        except Exception as e:
            print(f"Error initializing QueryClassifier: {e}")
            
    def classify_query(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("system", 
                 """
                 You are a medical query classifier. Classify the type of question.
                 Categories:
                 - If it is about a specific disease or condition (like 'What is glaucoma?'), respond with: disease_info
                 - If it is asking for diagnosis or possible diseases from symptoms (like 'I have a headache, what could it be?'), respond with: diagnosis_query
                 - If it is a query about common information (like greetings, aking for the assistant: "What is your name?"), respond with: general
                 """),
                ("human", "Classify the type of question: {query} "),
            ]
        )
        return prompt
    
    def classify(self, query: str) -> str:
        try: 
            prompt = self.classify_query()

            generater = prompt | self.llm | StrOutputParser()
            response = generater.invoke({"query": query})
            return response
        except Exception as e:
            print(f"Error classifying query: {e}")
            return None