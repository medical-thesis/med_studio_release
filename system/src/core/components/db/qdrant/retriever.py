import pandas as pd
import sys
import os
from dotenv import load_dotenv
load_dotenv("system/core/src/config/.env")


load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.src.core.components.embeddings.embedding import Embedding
from system.src.core.components.db.qdrant.manager import QdrantManager


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

class QdrantRetriever:
    def __init__(self):
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.outputs = []
        self.embedding = Embedding(embedding_model_name=EMBEDDING_MODEL)
        
        self.client = QdrantManager(
            embedding=self.embedding,
            url=self.QDRANT_URL,
            api_key=self.QDRANT_API_KEY,
        )


    def add_output(self, index, id, question, answer, source, category, score):
        output = {
            "Index": index,
            "ID": id,
            "Question": question,
            "Answer": answer,
            "Source": source,
            "Category": category,
            "Score": score
        }
        self.outputs.append(output)

    def handle_query(self, query):
        retrieved_docs = self.client.semantic_search(
            collection_name="test_",
            query=query,
            top_k=10,
        )
        index = 0

        for doc in retrieved_docs:
            index += 1
            print("\n\n", "=" * 20)
            print(f"Index: {index}")
            print(f"ID: {doc.id}")
            print(f"Question: {doc.payload['question']}")
            print(f"Answer: {doc.payload['answer']}")
            print(f"Source: {doc.payload['source']}")
            print(f"Category: {doc.payload['focus_area']}")
            print(f"Score: {doc.score}")
            

            self.add_output(
                index=index,
                id=doc.id,
                question=doc.payload['question'],
                answer=doc.payload['answer'],
                source=doc.payload['source'],
                category=doc.payload['focus_area'],
                score=doc.score
            )

        return self.outputs


if __name__ == "__main__":
    qdrant_retriever = QdrantRetriever()

    query = "How to prevent Hearing Loss?"

    qdrant_retriever.handle_query(query=query)
