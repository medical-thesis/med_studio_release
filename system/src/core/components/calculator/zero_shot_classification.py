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

class ZeroShotClassifier:
    def __init__(self, embedding: Embedding = None) -> None:
        self.embedding = embedding
        if embedding is None:
            self.embedding = Embedding(
                embedding_model_name=os.getenv("EMBEDDING_MODEL")
            )
    
    def zero_shot(self, query: str, documents: list, threshold= 0.5):
        try:
            query_embedding = self.embedding.get_embed(query)
            documents_embeddings = self.embedding.get_embed(documents)
            
            simmilarity_scores = self.embedding.get_similarities(query_embedding, documents_embeddings)
            
            max_score_index = torch.argmax(simmilarity_scores).item()
            
            return documents[max_score_index]
            
        except Exception as e:
            print(f"Error during zero-shot classification: {e}")
            return None