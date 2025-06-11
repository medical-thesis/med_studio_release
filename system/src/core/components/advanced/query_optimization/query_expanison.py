import ast
import ollama
from typing import List

class QueryExpansion:
    def __init__(self, model: str = 'gemma3:1b'):
        self.model = model

    def query_augmentation(self, query: str = "What causes diabetes?"):
        """
        Generate five alternative phrasings of a given question to improve vector search results.
        """
        prompt = f"""
            You are a helpful language model. Your task is to generate four to five alternative phrasings of the given user question.
            Each version should provide a slightly different perspective to improve document retrieval using vector similarity search.

            Example:
            Input: "What is thalassemia?"
            Output:
            [
            "What is thalassemia?",
            "What are the symptoms of thalassemia?",
            "How is thalassemia treated?",
            "What is the treatment for thalassemia?"
            ]

            Respond ONLY with a Python-style list of question strings.
            Some example output: ['What is the cause of diabetes?', 'What triggers diabetes?', 'Can you explain how diabetes develops?', 'How does diabetes come to be?', 'Why does diabetes occur?']
            
            Original question: {query}
        """
        return self._get_response(prompt)

    def query_expansion(self, query: str = "Can you explain what causes migraine, how it can be diagnosed, and what complications it leads to?") -> List[str]:
        """
        Decompose a complex medical question into simpler, focused questions for more effective semantic search.
        """
        prompt = f"""
            You are a medical language model assistant. Your task is to decompose a complex, multi-part medical question into a list of simpler, single-focus questions. 
            This will help improve semantic search based on vector databases.

            Here are some examples:

            Example 1:
            Input: "What is thalassemia, what are the symptoms, and how is it treated?"
            Output:
            [
            "What is thalassemia?",
            "What are the symptoms of thalassemia?",
            "How is thalassemia treated?",
            "What is the treatment for thalassemia?"
            ]

            Example 2:
            Input: "Can you explain what causes diabetes, how it can be diagnosed, and what complications it leads to?"
            Output:
            [
            "What causes diabetes?",
            "How is diabetes diagnosed?",
            "What are the complications of diabetes?"
            ]

            Respond ONLY with a Python-style list of question strings. Example: []. Don't add ```python or ``` like ```python[]```.

            Original question: {query}
        """
        return self._get_response(prompt, system_role="You are a helpful medical assistant.")

    def _get_response(self, user_prompt: str, system_role: str = "You are a helpful assistant.") -> List[str]:
        """
        Send a prompt to the model and print the response in real-time.
        """
        stream = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_role},
                {'role': 'user', 'content': user_prompt},
            ],
            stream=True,
        )

        response = ""
        print("Chatbot response:")
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
            response += chunk['message']['content']

        print("\n\n=====\n[Converted to array] Final response 2:\n", ast.literal_eval(response), "\n\n")
        return ast.literal_eval(response)


if __name__ == "__main__":
    expander = QueryExpansion()
    
    try:

        queries = expander.query_expansion("What is dengue fever, what are its symptoms, how can it be diagnosed, what is the treatment, and what are the complications it leaves behind?")
        index = 0
        for query in queries:
            index += 1
            print(f"item {index}: ", query)
    except Exception as e:
        print("\n\nerror in query_expansion: \n\n", {e})





# ====== test data =====
# - What is high blood pressure, what are its causes, how can it be detected early, and what are the effective treatment methods available today?
# - How does type 2 diabetes affect the body, what are its characteristic symptoms, and what preventive measures can reduce the risk of developing the disease?
# - What are the signs of lung cancer, how can it be diagnosed, and what are the modern treatment methods being applied?
# - What is cardiovascular disease, what are the risk factors, and how can we improve heart health?
# - Is depression a mental illness, what are its main symptoms, and what treatment options are considered most effective?
# - What is the difference between hepatitis B and C, how are they diagnosed and treated, and are there preventive measures to avoid transmission?
# - What is asthma, what factors trigger an asthma attack, and how can the disease be managed to prevent recurrent attacks?
# - What is kidney failure, what causes chronic kidney disease, and are treatments like dialysis effective?
# - What is Alzheimer’s disease, how can early signs be recognized, and are there ways to delay its progression?
# - How is tuberculosis transmitted, how can it be diagnosed and treated, and what are the effective preventive measures?