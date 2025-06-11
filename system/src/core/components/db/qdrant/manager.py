import os
import sys
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from dotenv import load_dotenv

load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.src.core.components.embeddings.embedding import Embedding

class QdrantManager:
    def __init__(
            self,
            embedding: Embedding,
            url: str = os.getenv("QDRANT_URL") if os.getenv(
                "QDRANT_URL") else "",
            api_key: str = os.getenv("QDRANT_API_KEY") if os.getenv(
                "QDRANT_API_KEY") else "",
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.client = None
        self.embedding = embedding

        self.connect()

    def connect(self):
        try:
            if self.client is None:
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                )
            print("Connected to Qdrant successfully.")

        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")

    def close_db(self) -> None:
        try:
            if self.client is not None:
                self.client.close()
                print("Qdrant connection closed.")
            else:
                print("No active Qdrant connection to close.")
        except Exception as e:
            print(f"Failed to close the Qdrant connection: {e}")


    def has_collection(self, collection_name: str) -> bool:
        collections = self.client.get_collections().collections
        return any(
            collection.name == collection_name for collection in collections
        )

    def create_collection(self, collection_name: str) -> None:
        if self.has_collection(collection_name):
            print(f"Collection '{collection_name}' already exists.")
            return

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding.embed_model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                )
            )
            print(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            print(f"Failed to create collection '{collection_name}': {e}")

    def delete_collection(self, collection_name: str) -> None:
        if not self.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return

        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            print(f"Failed to delete collection '{collection_name}': {e}")

    def add_database(self, collection_name: str, data: pd.DataFrame, batch_size: int = 500) -> None:
        if not self.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return

        data["content"] = data["question"] + " " + data["answer"]
        data["embeddings"] = data["content"].apply(
            lambda x: self.embedding.get_embed(x, batch_size=batch_size)
        )

        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=i + j,
                        vector=row['embeddings'],
                        payload={
                            "question": row['question'],
                            "answer": row['answer'],
                            "focus_area": row['focus_area'],
                            "source": row['source'],
                        }
                    )
                    for j, row in batch.iterrows()
                ],
                wait=True,
            )

    def delete_data(self, collection_name: str, ids: list) -> None:
        if not self.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=ids,
                wait=True,
            )
            print(
                f"Data with IDs {ids} deleted successfully from collection '{collection_name}'.")
        except Exception as e:
            print(
                f"Failed to delete data from collection '{collection_name}': {e}")

    def semantic_search(self, collection_name: str, query: str, top_k: int = 5, filter: dict = None) -> list:
        if not self.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return []

        if filter:
            query_filter = {
                "must": [
                    {
                        "key": key,
                        "match": {
                            "value": value
                        }
                    }
                    for key, value in filter.items()
                ]
            }
        else:
            query_filter = None

        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=self.embedding.get_embed(query),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return search_result
