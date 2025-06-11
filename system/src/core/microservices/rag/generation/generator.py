import os
import sys
from dotenv import load_dotenv

load_dotenv(dotenv_path="system/core/src/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from system.src.core.components.api.api_keys import APIKeyManager

class Generator:
    def __init__(self, model: str, api_key: APIKeyManager, max_tokens: int= 1000, temperature: float= 0.5) -> None:
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
            print(f"Error initializing QueryGenerator: {e}")
            
    def generate_query(self) -> ChatPromptTemplate:
        prompt= ChatPromptTemplate.from_messages(
            messages=[
                ("system",
                """
                You are a medical assistant providing concise and responsible health advice.
                The following context has been retrieved and reranked based on relevance. Use only this context to answer the question.
                If the context does not provide enough information, respond with: "I couldn't find the relevant information. Please provide more details."
                """),
                ("human", 
                """
                Context: {context}
                Generate a query based on the context: {query} """),
            ]
        )
        return prompt
    
    def generate(self, query: str, context: str) -> str:
        try: 
            prompt = self.generate_query()
            
            context = context if context else "No context provided."
            generater = prompt | self.llm | StrOutputParser()
            
            response = generater.invoke({"query": query,"context": context})
            return response
        except Exception as e:
            print(f"Error generating query: {e}")
            return None