from data_preparation import DataPreparation
from vector_store import VectorStore
from service import VectorStoreService

import os
from dotenv import load_dotenv
load_dotenv("system/core/src/config/.env")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from embeddings import Embedding
import pandas as pd
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

class ListRetriever:
    def __init__(self):
        self.outputs = []
    
    def add_output(self, id, question, answer,source, category, score):
        output = {
            "ID": id,
            "Question": question,
            "Answer": answer,
            "Source": source,
            "Category": category,
            "Score": score
        }
        self.outputs.append(output)
    
        
    def handle_query(self, query, vector_store_service):
        retrieved_docs = vector_store_service.search_similarity(
            query=query,
            k=10,
        )
        
        for doc, score in retrieved_docs:
            print("=" * 20)
            print(f"ID: {doc.id}")
            print(f"Question: {doc.metadata['question']}")
            print(f"Answer: {doc.metadata['answer']}")
            print(f"Source: {doc.metadata['source']}")
            print(f"category: {doc.metadata['focus_area']}")
            print(f"Score: {score}")
            
            self.add_output(
                id=doc.id,
                question=doc.metadata['question'],
                answer=doc.metadata['answer'],
                source=doc.metadata['source'],
                category=doc.metadata['focus_area'],
                score=score
            )
            
        return self.outputs
        
        
if __name__ == "__main__":
    
    # Initialize embedding model
    embedding = Embedding(embedding_model_name=EMBEDDING_MODEL)

    # Load the data
    dt = pd.read_csv("system/core/dataset/processed/medquad_qa_pairs.csv")
    
    list_retriever = ListRetriever()
    
    data_preparation = DataPreparation(data=dt)
    vector_store = VectorStore(data_preparation=data_preparation, embedding_name=EMBEDDING_MODEL)
    vector_store_service = VectorStoreService(vector_store=vector_store)
    
    path_vector_store = "system/core/FAISS/faiss_index"
    if not os.path.exists(path_vector_store):
        print(f"Creating vector store at {path_vector_store}")
        vector_store.create_vector_store()
        vector_store.save_vector_store(path=path_vector_store)
    else:
        print(f"Loading vector store from {path_vector_store}")
        vector_store.load_vector_store(path=path_vector_store)
        
    query = "How to prevent Hearing Loss?"
    
    list_retriever.handle_query(query, vector_store_service)