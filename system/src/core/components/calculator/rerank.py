import torch
import numpy as np
from sentence_transformers import util

import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.src.core.components.embeddings.embedding import Embedding   
class Rerank:
    def __init__(self, embedding: Embedding) -> None:
        self.embedding = embedding
    
    def rerank(self, query: str, documents: list, threshold= 0.5, top_k= 5) -> list:
        try:
            query_embedding = self.embedding.get_embed(query)
            documents_embeddings = self.embedding.get_embed(documents)
            
            simmilarity_scores = self.embedding.get_similarities(query_embedding, documents_embeddings)
            
            max_score = torch.max(simmilarity_scores).item()

            if max_score < threshold:
                return None
            
            sorted_indices = torch.argsort(simmilarity_scores.squeeze(), descending=True)[:top_k]
            
            reranked_documents = [documents[i] for i in sorted_indices]
            
            return reranked_documents
            
        except Exception as e:
            print("Rerank ERROR")
            print(f"Error during reranking: {e}")
            return None