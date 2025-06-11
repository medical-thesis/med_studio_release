import sys
import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# === fix import module
load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.src.core.components.embeddings.embedding import Embedding

class ElasticsearchRetriever:
    def __init__(self):
        self.client = None
        self.embed = Embedding(embedding_model_name="all-MiniLM-L6-v2")

        self.connect_elastic_cloud(
            endpoint= os.getenv("ES_ENDPOINT"),
            api_key= os.getenv("ES_API_KEY")
        )

    # def connect_elastic(
    #         self,
    #         uri: str = os.getenv("ES_URI") if os.getenv("ES_URI") else "",
    #         es_user: str = os.getenv(
    #             "ES_USER") if os.getenv("ES_USER") else "",
    #         es_password: str = os.getenv(
    #             "ES_PASSWORD") if os.getenv("ES_PASSWORD") else "",
    #         ca_certs: str = os.getenv("ES_CA_CERTS") if os.getenv(
    #             "ES_CA_CERTS") else "",
    # ):
    #     self.client = Elasticsearch(hosts=uri,
    #                                 basic_auth=(
    #                                     es_user, es_password),
    #                                 verify_certs=True,
    #                                 ca_certs=ca_certs,
    #                                 request_timeout=10)
    #     if self.client.ping():
    #         print("Connected to Elasticsearch server.")
    #         client_info = self.client.info()
    #         print('Connected to Elasticsearch!')
    #         print(client_info.body)
    #     else:
    #         print("Failed to connect to Elasticsearch server.")


    def connect_elastic_cloud(
            self,
            endpoint: str = os.getenv("ES_ENDPOINT"),
            api_key: str = os.getenv("ES_API_KEY")
    ):
        self.client = Elasticsearch(
            hosts=endpoint,
            api_key=api_key,
            request_timeout=120,
            retry_on_timeout=True,
            max_retries=10
        )
        print("\n\n" \
        "Connecting elastic cloud ... \n\n")

        if self.client.ping():
            print("Connected to Elasticsearch cloud server.")
            client_info = self.client.info()
            print('Connected to Elasticsearch cloud!')
            print(client_info.body)
        else:
            print("Failed to connect to Elasticsearch cloud server.")

    def pretty_response(self, response, index=0):
        if len(response["hits"]["hits"]) == 0:
            print("Your search returned no results.")
        else:
            for hit in response["hits"]["hits"]:
                index += 1
                id = hit["_id"]
                score = hit["_score"]
                question = hit["_source"]["question"]
                answer = hit["_source"]["answer"]
                pretty_output = f"\nIndex: {index} \nID: {id}\nQuestion: {question}\nAnswer: {answer}\nScore: {score}\n"
                print(pretty_output)

    def create_pretty_response(self, response, index=0):
        outputs = []

        if len(response["hits"]["hits"]) == 0:
            print("Your search returned no results.")
        else:
            for hit in response["hits"]["hits"]:
                index += 1

                id = hit["_id"]
                score = hit["_score"]
                question = hit["_source"]["question"]
                answer = hit["_source"]["answer"]
                source = hit["_source"]["metadata"]["source"]
                catogory = hit["_source"]["metadata"]["category"]

                output = {
                    "Index": index,
                    "ID": id,
                    "Question": question,
                    "Answer": answer,
                    "Source": source,
                    "Category": catogory,
                    "Score": score
                }

                outputs.append(output)

        return outputs

    def semantic_search(self, index_name: str, query: str, text: str = "", content: str = ""):
        response = self.client.search(
            index=index_name,
            knn={
                "field": "embedding",
                "query_vector": self.embed.get_embed(query),
                "k": 10,
                "num_candidates": 100,
            }
        )

        return response

    def handle_query(self, query: str):
        index_name = "medical_records"
        response = self.semantic_search(index_name=index_name, query=query)

        results = self.create_pretty_response(response=response)
        return results


if __name__ == "__main__":
    print("\n\n\n ---- Retriever example tasks: ---- \n\n")

    retriever = ElasticsearchRetriever()

    index_name = "medical_records"

    test_data = ["How to prevent Hearing Loss", "Hearing Loss",
                 "What are the treatments for Hearing Loss ?"]
    for i in test_data:
        print(i)
        response = retriever.semantic_search(index_name=index_name,
                                             query=i)

        # print(response)
        print("\n\n test: \n")
        print("===============")
        print(f"for sample: \[\"{i}\"\]")
        retriever.pretty_response(response)
    # print(type(retriever.client))
    # print(type(response))
