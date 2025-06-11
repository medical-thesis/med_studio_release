import os
import sys

import pandas as pd
from datetime import datetime

from elasticsearch import Elasticsearch

from dotenv import load_dotenv
load_dotenv(dotenv_path="system/src/core/config/.env")

# === fix import module
project_root = os.getenv("PROJECT_ROOT")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.src.core.components.db.elasticsearch.manager import ElasticsearchManager
from system.src.core.components.embeddings.embedding import Embedding

USING_QDRANT = (os.getenv("USING_QDRANT") == "1")

if USING_QDRANT:
    from system.src.core.components.db.qdrant.manager import QdrantManager
    print(USING_QDRANT)
    
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    embedding = Embedding(embedding_model_name="all-MiniLM-L6-v2")
    
    client = QdrantManager(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        embedding=embedding
    )
    
    collection_name = "db_1" 
   
    ck = client.has_collection(  
        collection_name=collection_name,
    )

    if ck:
        print(f"Collection {collection_name} already exists.")
    else:
        from qdrant_client.models import PointStruct
        
        client.create_collection(
            collection_name=collection_name,
        )
        
        print("Collection created successfully!")
        
        dt = pd.read_csv('system/dataset/processed/medquad_qa_pairs.csv')
        
        batch_size=500
        dt["content"] = dt["question"] + " " + dt["answer"]

        dt["embeddings"] = embedding.get_embed(
            content=dt["content"].tolist(), batch_size=batch_size
            )
        
        for i in range(0, len(dt), batch_size):
            batch = dt.iloc[i:i + batch_size]
            client.client.upsert(
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
        
        print("Data indexed successfully!")
        
if not USING_QDRANT:
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

    INDEX_NAME = "medical_records"

    INDEX_CONFIG = {
        "mappings": {
            "properties": {
                "question": {"type": "text"},
                "answer": {"type": "text"},
                "embeding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": "true",
                    "similarity": "cosine",
                }
            }
        }
    }

    if USING_ELASTIC_CLOUD:
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

        # (hiện tại là setup tạm thời, sau này hoàn thiện module này chỗ nào có pd đọc sẽ đọc data từ mongodb database source thay vì csv bằng pandas như hiện tại)
        df = pd.read_csv(
            'E:/source_code/nlp/med_studio/system/dataset/processed/medquad_qa_pairs.csv')
        print(f"\n ---- Dataset overview: \n{df.describe()} ---- \n\n")

        embed_model = Embedding(embedding_model_name="all-MiniLM-L6-v2")
        for _, row in df.iterrows():
            data = None
            data = {
                "question": str(row['question']),
                "answer": str(row['answer']),
                "metadata": {
                    "source": str(row["source"]),
                    "category": str(row["focus_area"]),
                },
                "embedding": embed_model.get_embed(content=str(row['answer'])),
            }

            if data:
                client.index(index=INDEX_NAME, document=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(f"[{current_time}] Insert success ...! \n\n")


    if not USING_ELASTIC_CLOUD:
        # Task 01: Create elasticsearch connection and index
        print("\n\n\n ---- Elasticsearch database service tasks: ---- \n")
        print("\n ---- Task 01: Create elasticsearch connection and index ---- \n")

        client = ElasticsearchManager(
            uri=ES_URI,
            es_user=ES_USER,
            es_password=ES_PASSWORD,
            ca_certs=CA_CERTS
        )

        client.create_index(
            index_name=INDEX_NAME,
            index_config=INDEX_CONFIG
        )

        #
        # ========
        # Task 2: Read data from source, embedding data and index all data (documents)
        # (hiện tại là setup tạm thời, sau này hoàn thiện module này chỗ nào có pd đọc sẽ đọc data từ mongodb database source thay vì csv bằng pandas như hiện tại)
        print("\n ---- Task 2: Read data from source, embedding data and index all data (documents) ---- \n")

        df = pd.read_csv(
            'E:/source_code/nlp/med_studio/system/dataset/processed/medquad_qa_pairs.csv')

        print(f"\n ---- Dataset overview: \n{df.describe()} ---- \n\n")

        embed_model = Embedding(embedding_model_name="all-MiniLM-L6-v2")

        for _, row in df.iterrows():
            data = {}
            data = {
                "question": str(row['question']),
                "answer": str(row['answer']),
                "metadata": {
                    "source": str(row["source"]),
                    "category": str(row["focus_area"]),
                },
                "embedding": embed_model.get_embed(content=str(row['answer'])),
            }
            if data:
                client.insert_document(document=data)

                current_time = datetime.now().isoformat(timespec='seconds')
                print(f"[{current_time}] Insert success ...! \n\n")
