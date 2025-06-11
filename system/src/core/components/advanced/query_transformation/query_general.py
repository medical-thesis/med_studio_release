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

class QueryGeneral:
    def __init__(self, model: str, api_key: APIKeyManager, temperature: float= 0.5, max_tokens: int= 1000) -> None:
        try:
            self.model = model            
            self.api_key = api_key
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
            print(f"Error initializing QueryGeneral: {e}")
            
    def generate_query(self) -> ChatPromptTemplate:
        prompt= ChatPromptTemplate.from_messages(
            messages=[
                ("system",
                """
                Please classify the query into one of the following categories and provide the exact corresponding response: 

                - If the query is related to a greeting, respond only with: "Hi, I'm a medical assistant. How can I help you today?" 
                - If the query is related to goodbyes, respond only with: "Goodbye! Have a great day!" 
                - If the query is expresses gratitude, respond only with: "You're welcome! I'm here to help." 
                - If the query is about the chatbot, respond only with: "I'm a medical assistant here to provide health advice. I am trained to be a helpful and supportive AI assistant, so feel comfortable asking me any questions." 
                - Otherwise, respond only with: "I couldn't find the relevant information." 

                Ensure that your response strictly follows the given options without any additional text.
                """),
                ("human", "Classify this query: {query}"),
            ]
        )
        return prompt
    
    def generate(self, query: str) -> str:
        try: 
            prompt = self.generate_query()

            generater = prompt | self.llm | StrOutputParser()
            
            response = generater.invoke({"query": query})
            return response
        except Exception as e:
            print(f"Error generating query: {e}")
            return None