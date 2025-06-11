import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from typing import Dict, Any, List

load_dotenv(dotenv_path="system/src/core/config/.env")
uri = os.getenv("ES_URI")
es_user = os.getenv("ES_USER")
es_password = os.getenv("ES_PASSWORD")
ca_certs = os.getenv("ES_CA_CERTS")


class ElasticsearchManager:
    def __init__(self,
                 uri: str = uri if uri else '',
                 es_user: str = es_user if es_user else '',
                 es_password: str = es_password if es_password else '',
                 ca_certs: str = ca_certs if ca_certs else ''):
        self.uri = uri
        self.es_user = es_user
        self.es_password = es_password
        self.ca_certs = ca_certs

        self.index_name = None
        self.index_config = None
        self.es = None

        self.connect(
            uri=self.uri,
            es_user=self.es_user,
            es_password=es_password,
            ca_certs=self.ca_certs
        )

    def set_uri(self, uri: str): self.uri = uri

    def get_uri(self) -> str: return self.uri

    def set_db_user(self, es_user: str, es_password: str):
        self.es_user = es_user
        self.es_password = es_password

    def get_db_user(self) -> List[Any]: return [self.es_user, self.es_password]

    def connect(
            self, uri: str, es_user: str, es_password: str, ca_certs: str
    ) -> Elasticsearch:
        if uri:
            self.uri = uri
        if es_user:
            self.es_user = es_user
        if es_password:
            self.es_password = es_password
        if ca_certs:
            self.ca_certs = ca_certs

        try:
            self.es = Elasticsearch(hosts=self.uri,
                                    basic_auth=(
                                        self.es_user, self.es_password),
                                    verify_certs=True,
                                    ca_certs=self.ca_certs,
                                    request_timeout=10)

            if self.es.ping():
                print("Connected to Elasticsearch server.")
                client_info = self.es.info()
                print('Connected to Elasticsearch!')
                print(client_info.body)
            else:
                print("Failed to connect to Elasticsearch server.")
                self.es = None

        except Exception as e:
            print(f"Connection failed with error: {str(e)}")
            self.es = None

        return self.es

    def set_index(self, index_name: str, index_config: str):
        self.index_name = index_name
        self.index_config = index_config

    def get_indices(self) -> Dict[str, Any] | None:
        """
        beta
        """
        try:
            indices = self.es.get(index="*", id="1")
            return indices
        except Exception as e:
            return None

    def create_index(self, index_name: str, index_config: dict):
        if index_name:
            self.index_name = index_name
        if index_config:
            self.index_config = index_config

        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name,
                                       ignore_unavailable=True)

            self.es.indices.create(index=self.index_name,
                                   body=self.index_config)
            print("Index created successfully!")
        except Exception as e:
            print(f"Creating index failed with error: {str(e)}")

    def insert_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.es.index(index=self.index_name, document=document)
        except Exception as e:
            print(f"Insert document failed with error: {str(e)}")
            return None
