from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any


class SemanticRouter:
    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.reference_embeddings = {}

    def remove_duplicates(self, l):
        return list(set(l))

    def pre_process_data(self) -> Dict[str, Any]:
        routes = ['treatments',
                  'symptoms',
                  'concepts',
                  'complications',
                  'genetic_changes',
                  'outlooks',
                  'stages']

        intent = {
            'treatments': None,
            'symptoms': None,
            'concepts': None,
            'complications': None,
            'genetic_changes': None,
            'outlooks': None,
            'stages': None
        }

        for route in routes:
            with open(file=f"E:/source_code/nlp/med_studio/system/dataset/semantic_router_data/{route}.txt", mode="r") as file:
                examples = [item.strip() for item in file.readlines()]
                print(route)
                print(len(examples))
                # print(remove_duplicates(examples))
                intent[route] = self.remove_duplicates(examples)
                print(len(self.remove_duplicates(examples)))
                print()

        print(type(examples))

        return intent

    def buid_ref_embed_matrix(self):
        intents = self.pre_process_data()

        self.reference_embeddings = {}
        for intent, examples in intents.items():
            print('intent: ', intent)
            # print('examples: ', examples)
            embeddings = self.embed_model.encode(examples)
            self.reference_embeddings[intent] = np.mean(embeddings, axis=0)
        print()
        # print("reference_embeddings: ", self.reference_embeddings)

    def route_query(self, query):
        query_embedding = self.embed_model.encode([query])[0]

        similarities = {}
        for intent, ref_embedding in self.reference_embeddings.items():
            similarity = cosine_similarity(
                [query_embedding], [ref_embedding])[0][0]
            similarities[intent] = similarity

        print(f"Query: {query}")
        print(f"Similarities: {similarities}")
        best_intent = max(similarities, key=similarities.get)
        return best_intent, similarities[best_intent]


if __name__ == "__main__":
    # test code
    semanticRouter = SemanticRouter()
    semanticRouter.buid_ref_embed_matrix()

    query = "Hôm nay trời thế nào?"
    intent, score = semanticRouter.route_query(query)
    print(f"Intent: {intent}, Confidence: {score}")

    print("\n\n====\n\n")

    query = "Dự báo thời tiết hôm nay thế nào?"
    intent, score = semanticRouter.route_query(query)
    print(f"Intent: {intent}, Confidence: {score}")

    print("\n\n====\n\n")

    query = "What are the complications of Mineral and Bone Disorder in Chronic Kidney Disease ?"
    intent, score = semanticRouter.route_query(query)
    print(f"Intent: {intent}, Confidence: {score}")

    print("\n\n====\n\n")

    query = "What is the outlook for Neuroacanthocytosis ?"
    intent, score = semanticRouter.route_query(query)
    print(f"Intent: {intent}, Confidence: {score}")

    print("\n\n====\n\n")

    query = "What are the treatments for Hemolytic Uremic Syndrome in Children"
    intent, score = semanticRouter.route_query(query)
    print(f"Intent: {intent}, Confidence: {score}")

    print("\n\n====\n\n")

    query = "can you tell me about the treatments for Hemolytic Uremic Syndrome in Children."
    intent, score = semanticRouter.route_query(query)
    print(f"Intent: {intent}, Confidence: {score}")

    if intent == "check_weather":
        pass
        # check_weather_module(query)
    elif intent == "book_ticket":
        pass
        # book_ticket_module(query)
    elif intent == "cancel_order":
        pass
        # cancel_order_module(query)
