import ollama
import streamlit as st
import time
import sys
import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# === fix import module
load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.src.core.components.advanced.query_optimization.query_expanison import QueryExpansion
from system.src.core.microservices.rag.generation.generator import Generator
from system.src.core.components.db.elasticsearch.retriever import ElasticsearchRetriever
from system.src.core.components.api.api_keys import APIKeyManager

from system.src.core.components.db.qdrant.manager import QdrantManager
from system.src.core.components.db.qdrant.retriever import QdrantRetriever
from system.src.core.components.conversational.memory.chat_history import ChatHistory

chathistory = ChatHistory()
from system.src.core.components.embeddings.embedding import Embedding 

embedding = Embedding(
    embedding_model_name=os.getenv("EMBEDDING_MODEL"),
)

manager = QdrantManager(
    embedding=embedding,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

qdrant_retriever = QdrantRetriever()

from system.src.core.components.api.api_keys import APIKeyManager

api_key = APIKeyManager(name="GEMINI_API_KEY")
model= 'gemini-1.5-pro'

from system.src.core.microservices.rag.generation.generator import Generator
query_generator = Generator(
    model=model,
    api_key=api_key
)

from system.src.core.components.advanced.query_transformation.query_normalization import QueryNormalization   
query_normalization = QueryNormalization(
    model=model,
    api_key=api_key,
    chathistory= chathistory,
)

from system.src.core.components.advanced.query_transformation.query_classification import QueryClassification
query_classifier = QueryClassification(
    model=model,
    api_key=api_key
)

from system.src.core.components.advanced.query_transformation.query_general import QueryGeneral
query_general = QueryGeneral(
    model=model,
    api_key=api_key
)

from system.src.core.components.advanced.query_routing.logical_routing import LogicalRouting

cate_classifier = LogicalRouting()

import pandas as pd
class MessageController:
    def __init__(self):
        pass

    def get_responses(self, input_query: str = "How to prevent Hearing Loss?") -> str:
        LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

        step_box = st.empty()
        step_box.info("Normalize query ...")
        time.sleep(1.5)

        step_box.empty()
        normalized_query = query_normalization.normalize(input_query)
        print(normalized_query)
        step_box.info(f"{normalized_query}")
        time.sleep(1.5)

        step_box.empty()
        step_box.info("Classify query ...")
        time.sleep(1.5)

        step_box.empty()
        classified_query = query_classifier.classify(normalized_query)
        step_box.info(f"Classified query: {classified_query}")
        time.sleep(1.5)

        step_box.empty()
        if classified_query == "general":
            step_box.info("General query detected, so generating response...")
            response = query_general.generate(normalized_query)
            return response

        if classified_query == "disease_info":
            final_queries = []
            try:
                expander = QueryExpansion()
                queries = expander.query_expansion(normalized_query)
                index = 0
                for query in queries:
                    final_queries.append(query)
                    index += 1
                    print(f"item {index}: ", query)
                step_box.empty()
                step_box.info("Expanding and analyzing query ...")
                time.sleep(1.5)
                result_box = st.empty()
                text = f"Origin query: {input_query}\n"
                for query in queries:
                    new_query = f"- Subquery is created: {query}\n\n"
                    text += new_query

                    result_box.error(
                        text,
                        # unsafe_allow_html=True
                    )

                    # text = ""
                    time.sleep(1.5)
                result_box.empty()
                if len(final_queries) < 2:
                    final_queries = []
                    final_queries.append(normalized_query)
            except Exception as err:
                print(err)
                final_queries.append(normalized_query)

            step_box.empty()
            step_box.info("Retrieving relevant documents...")
            time.sleep(1.5)

            # retriever = ElasticsearchRetriever()
            # index_name = "medical_records"
            # context = retriever.handle_query(query=normalized_query)
            # context = context[0]["Answer"]

            # ====
            retriever = ElasticsearchRetriever()
            index_name = "medical_records"

            context = []
            for query in final_queries:
                results = retriever.handle_query(query=query)
                results = results[:4]
                context.extend(results) 
            print("\n\nlen context: ", len(context))

            # ===

        subheader = st.empty()
        subheader.markdown("📚 The context retrieved is:")
        time.sleep(1.5)

        result_box = st.empty()
        text = ""
        for item in context:
            new_line = f"- {item['Question']}\n- {item['Answer']}\n\n"
            text += new_line

            result_box.markdown(
                text,
                unsafe_allow_html=True
            )
            text = ""
            time.sleep(1.5)
        result_box.empty()
        subheader.empty()

        step_box.info("Generating response ...")

        context = "\n".join(
            [f"- {item['Question']} {item['Answer']}" for item in context]
        )

        instruction_prompt = f'''You are a helpful chatbot. Use only the following pieces of context to answer the question. Don't make up any new information: {context}.'''
        # print(instruction_prompt)

        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query},
            ],
            stream=True,
        )

        # print the response from the chatbot in real-time
        response = ""
        print('Chatbot response:')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
            response += chunk['message']['content']

        print("\n\n\ Generating response ... donen, prepapre return \n\n\n")
        return str(response), results

    def get_response_gemini(self, input_query: str) -> str:

        step_box = st.empty()

        step_box.info("Normalize query...")
        normalized_query = query_normalization.normalize(input_query)
        step_box.markdown(f"Normalized query: {normalized_query}")
        print(f"Normalized query: {normalized_query}")
        time.sleep(1.5)

        step_box.empty()  # Clear the step box
        step_box.info("Classify query...")
        classified_query = query_classifier.classify(normalized_query)
        step_box.markdown(f"Classified query: {classified_query}")
        print(f"Classified query: {classified_query}")
        time.sleep(1.5)

        step_box.empty()  # Clear the step box
        if classified_query == "general":
            step_box.info("General query detected. Generating response...")
            response = query_general.generate(normalized_query)
            return response

        if classified_query == "disease_info":
            step_box.info("Detecting category...")
            dt = pd.read_csv("system/dataset/processed/medquad_qa_pairs.csv")
            categories = dt['focus_area'].unique().tolist()

            cate = cate_classifier.classify(
                query=normalized_query,
                documents=categories,
            )

            print(f"Detected category: {cate}")
            step_box.markdown(f"Detected category: {cate}")

            time.sleep(1.5)

            step_box.empty()
            step_box.info("Retrieving relevant documents...")
            filter_docs = {
                "must": [
                    {
                        "key": "focus_area",
                        "match": {
                            "value": cate
                        }
                    }
                ]
            }
            retriever = manager.client.search(
                collection_name='db_1',
                query_vector=embedding.get_embed(normalized_query),
                query_filter=filter_docs,
                limit=5,
                with_payload=True,
            )
            
            # retriever = qdrant_retriever.handle_query(
            #     query= normalized_query
            # )
        else:
            step_box.info("Retrieving relevant documents...")
            retriever = manager.semantic_search(
                collection_name="db_1",
                query=normalized_query,
                top_k=5,
            )
            
        ######
        
        lst_docs = []
        
        index = 0

        for doc in retriever:
            index += 1
            print("\n\n", "=" * 20)
            print(f"Index: {index}")
            print(f"ID: {doc.id}")
            print(f"Question: {doc.payload['question']}")
            print(f"Answer: {doc.payload['answer']}")
            print(f"Source: {doc.payload['source']}")
            print(f"Category: {doc.payload['focus_area']}")
            print(f"Score: {doc.score}")
            

            lst_docs.append({
                "Index": index,
                "ID": doc.id,
                "Question": doc.payload['question'],
                "Answer": doc.payload['answer'],
                "Source": doc.payload['source'],
                "Category": doc.payload['focus_area'],
                "Score": doc.score
            })
        
        print("\n\n", "=" * 20)
        print(lst_docs)
        print("\n\n", "=" * 20)
        
        ######
        
        context = "\n".join(
            [f"- {doc.payload['question']} {doc.payload['answer']}" for doc in retriever]
        )
        subheader = st.empty()
        subheader.subheader("Retriever:")

        result_box = st.empty()
        text = ""
        for doc in retriever:
            new_line = f"- {doc.payload['question']}\n- {doc.payload['answer']}\n\n"
            text += new_line

            result_box.markdown(
                text,
                unsafe_allow_html=True
            )
            text = ""
            time.sleep(1.5)
        result_box.empty()
        subheader.empty()

        step_box.info("Generating...")

        response = query_generator.generate(
            query=normalized_query, context=context)
        chathistory.add_message({
            "query": input_query,
            "response": response
        })
        return response, lst_docs
