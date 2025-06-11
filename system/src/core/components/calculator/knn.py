import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import sys
from dotenv import load_dotenv

load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.src.core.components.embeddings.embedding import Embedding
    
class KNNClassifier:
    def __init__(self, embedding: Embedding, data: pd.DataFrame, k: int=5) -> None:
        self.embedding = embedding
        self.data = data
        self.k = k 
        self.fit()
        
    def fit(self) -> None:
        self.vectorizer = self.embedding.get_embed(self.data['question'].to_list())
        self.labels = self.data['focus_area'].to_list()

        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(self.vectorizer, self.labels)
        print("KNN Classifier has been fitted.")
        print(f"Number of classes: {self.k}")
        
    def predict(self, text: str) -> str:
        vector = np.array(self.embedding.get_embed(text)).reshape(1, -1)
        prediction = self.knn.predict(vector)
        return prediction[0]