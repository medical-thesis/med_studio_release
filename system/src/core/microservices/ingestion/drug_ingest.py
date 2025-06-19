from datetime import datetime
import os
import sys
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import pandas as pd

from dotenv import load_dotenv
load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")
if project_root not in sys.path:
    sys.path.insert(0, project_root)


if True:
    USING_ELASTIC_CLOUD = (os.getenv("USING_ELASTIC_CLOUD") == "1")
    print(USING_ELASTIC_CLOUD)
    print(type(USING_ELASTIC_CLOUD))

    ES_URI = None
    ES_USER = None
    ES_PASSWORD = None
    CA_CERTS = None

    if USING_ELASTIC_CLOUD:
        ES_ENDPOINT = os.getenv("ES_ENDPOINT")
        ES_API_KEY = os.getenv("ES_API_KEY")

    if not USING_ELASTIC_CLOUD:
        ES_URI = os.getenv("ES_URI")
        ES_USER = os.getenv("ES_USER")
        ES_PASSWORD = os.getenv("ES_PASSWORD")
        CA_CERTS = os.getenv("ES_CA_CERTS")

    INDEX_NAME = "drug_records"

    INDEX_CONFIG = {
        "mappings": {
            "properties": {
                "drug_name": {"type": "text"},
                "drug_description": {"type": "text"},
                "origin_source": {"type": "text"},
                "category": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": "true",
                    "similarity": "cosine",
                }
            }
        }
    }

    if True:
        client = Elasticsearch(
            hosts=ES_ENDPOINT,
            api_key=ES_API_KEY,
            request_timeout=120,
            retry_on_timeout=True,
            max_retries=10
        )

        if client.indices.exists(index=INDEX_NAME):
            client.indices.delete(index=INDEX_NAME, ignore_unavailable=True)

        client.indices.create(index=INDEX_NAME, body=INDEX_CONFIG)
        print("Index created successfully!")

        df = pd.read_csv(
            'vietmed_dataset_drugs_vi.csv')
        print(f"\n ---- Dataset overview: \n{df.describe()} ---- \n\n")

        # embed_model = Embedding(embedding_model_name="all-MiniLM-L6-v2")
        embed_model = SentenceTransformer('keepitreal/vietnamese-sbert')
        for _, row in df.iterrows():
            data = None
            data = {
                "drug_name": str(row['name']),
                "drug_description": str(row['description']),
                "origin_source": str(row['origin_source']),
                "category": str(row['category']),
                "embedding": embed_model.encode(str(row['description'])),
            }

            if data:
                client.index(index=INDEX_NAME, document=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(f"[{current_time}] Insert success ...! \n\n")
