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
from system.src.core.components.conversational.memory.chat_history import ChatHistory

class QueryNormalization:
    def __init__(self, model: str, api_key: APIKeyManager, chathistory= ChatHistory, temperature: float= 0.5,max_tokens: int= 1000) -> None:
        try:
            self.api_key = api_key
            self.model = model
            self.chathistory = chathistory
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
            print(f"Error initializing QueryNormalization: {e}")
    def normalize_query(self, query: str) -> ChatPromptTemplate:
        if self.chathistory.get_latest_history() is not None:
            context = self.chathistory.get_context_chathistory()
            prompt= ChatPromptTemplate.from_messages(
                messages=[
                    ("system", 
                    """You are a medical query normalizer. Your task is to normalize the query into a more specific and clear form.
                    If the original query information is unclear or related to the conversation history, use the conversation history to normalize the query.
                    Rephrase if needed and remove any irrelevant information in the original query.
                    Focus on maintaining the core meaning and intent of the query.
                    Avoid adding any new information or context that is not present in the original query.
                    Only return the normalized query itself, without add any explanations, descriptions, or additional text.
                    
                    
                    Example 1:
                        Original query: "I've been feeling dizzy and nauseous lately, is it due to anemia?"
                        Normalized Query: "Are dizziness and nausea related to anemia?"
                        
                    Example 2:
                        Original query: "Can medicine A be used to treat sore throat? I see different information online."
                        Normalized Query: "Can medicine A treat sore throat?"
                        
                    Example 3:
                        Original query: "Hi, I think my child might have bronchiolitis. Can you provide more information on this condition?"
                        Normalized Query: "what is bronchiolitis and its symptoms in child?" 
                        
                    Example 4:
                        Original query: "What is Glaucoma?"
                        Normalized Query: "What is Glaucoma?"
                    
                    Example 5:
                        Original query: "What causes Glaucoma?"
                        Normalized Query: "What causes Glaucoma?"
                        
                    Example 6:
                        Original query: "What diseases can headaches be a sign of?"
                        Normalized Query: "What diseases can headaches be a sign of?"
                        
                    Example 7:
                        Original query: "I have frequent headaches, is it due to anemia?"
                        Normalized Query: "Is frequent headache related to anemia?"  
                        
                    Example 8:
                        Conversation history:
                            - Query: What are the side effects of drug A? 
                            - Response: Drug A can cause drowsiness and dry mouth.
                            - Query: I am allergic to antihistamines, can I use drug A? 
                            - Response: Drug A belongs to the antihistamine group, so you should consult your doctor before using it.

                        Original Query: "I have allergic rhinitis, can I use medicine A?"
                        Normalized Query: "Can I use medicine A for allergic rhinitis?"
                    
                    Example 9:
                        Conversation history:
                            - Query: What diseases can headaches be a sign of? 
                            - Response: Headaches can be caused by many things such as stress, lack of sleep or serious illnesses such as high blood pressure.
                            - Query: I have frequent headaches, is it due to anemia? 
                            - Response: Anemia can cause headaches, but tests are needed to determine exactly.

                        Original Query: "So if a headache is accompanied by dizziness, what diseases can it be?"
                        Normalized Query: "What diseases can a headache with dizziness be?"
                    """),
                    ("human", f"""Given the conversation history: 
                                {context}
                                Normalize the query: {query}. Don't use any additional information or context.""")
                ]
            )
        else:
            prompt= ChatPromptTemplate.from_messages(
                messages=[
                    ("system", 
                    """You are a medical query normalizer. Your task is to normalize the query into a more specific and clear form.
                    Rephrase if needed and remove any irrelevant information in the original query.
                    Focus on maintaining the core meaning and intent of the query.
                    Avoid adding any new information or context that is not present in the original query.
                    Only return the normalized query itself, without add any explanations, descriptions, or additional text.
                    
                    Example 1:
                        Original query: "I've been feeling dizzy and nauseous lately, is it due to anemia?"
                        Normalized Query: "Are dizziness and nausea related to anemia?"
                        
                    Example 2:
                        Original query: "Can medicine A be used to treat sore throat? I see different information online."
                        Normalized Query: "Can medicine A treat sore throat?"    
                    
                    Example 3:
                        Original query: "Hi, I think my child might have bronchiolitis. Can you provide more information on this condition?"
                        Normalized Query: "what is bronchiolitis and its symptoms in child?" 
                    
                    Example 4:
                        Original query: "What is Glaucoma?"
                        Normalized Query: "What is Glaucoma?"
                    
                    Example 5:
                        Original query: "What causes Glaucoma?"
                        Normalized Query: "What causes Glaucoma?"
                        
                    Example 6:
                        Original query: "What diseases can headaches be a sign of?"
                        Normalized Query: "What diseases can headaches be a sign of?"
                        
                    Example 7:
                        Original query: "I have frequent headaches, is it due to anemia?"
                        Normalized Query: "Is frequent headache related to anemia?"
                    """),
                    ("human", f"Normalize the query: {query}. Don't use any additional information or context."),
                ]
            )
        return prompt
    
    def normalize(self, query: str) -> str:
        try: 
            prompt = self.normalize_query(query=query)
            
            generater = prompt | self.llm | StrOutputParser()
            response = generater.invoke({})
            return response
        except Exception as e:
            print(f"Error normalizing query: {e}")
            return None