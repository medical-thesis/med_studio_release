import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from typing import List, Union
import torch
from dotenv import load_dotenv

load_dotenv(dotenv_path="system/src/core/config/.env")

class Embedding:
    def __init__(self, embedding_model_name: str = os.getenv("EMBEDDING_MODEL")):
        self.embed_model_name = embedding_model_name
        print(f"You are using the Embedding model: {self.embed_model_name}")

        self.embed_model = SentenceTransformer(
            model_name_or_path=embedding_model_name, 
            use_auth_token=os.getenv("HF_AUTH_TOKEN"))
        print(f"The model has been set up, some info: {self.embed_model}")

    def get_embed(self,
                  content: Union[str, List[str]],
                  batch_size: int = 32
                  ) -> Union[List[float], List[List[float]]]:

        try:
            if isinstance(content, str):
                embedding = self.embed_model.encode(
                    content, batch_size=batch_size)
                print(
                    f"[Embedding] Input: 1 string → Output: vector size {len(embedding)}")
                return embedding.tolist()
            elif isinstance(content, list):
                embeddings = self.embed_model.encode(
                    content, batch_size=batch_size)
                print(
                    f"[Embedding] Input: {len(content)} strings → Output: {len(embeddings)} vectors")
                return [embed.tolist() for embed in embeddings]
            else:
                raise TypeError("Input must be a string or a list of strings.")

        except Exception as e:
            print(f"Full error details: {str(e)}")

    def get_similarities(self, first_embed, second_embed):
        try:
            if not isinstance(first_embed, torch.Tensor):
                first_embed = torch.tensor(first_embed)
            if not isinstance(second_embed, torch.Tensor):
                second_embed = torch.tensor(second_embed)
            return util.pytorch_cos_sim(first_embed, second_embed)
        except Exception as e:
            print(f"Full error details: {str(e)}")


if __name__ == "__main__":
    print("\n\n\n ---- Embedding example tasks: ---- \n\n")
    # 1. Load a pretrained Sentence Transformer model
    model = Embedding("all-MiniLM-L6-v2")

    # The sentences to get_embed
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]

    # 2. Calculate embeddings by calling model.get_embed()
    embeddings = model.get_embed(sentences)
    print("\n ---- Results: ---- \n")

    print(f"The embedding for \"{sentences[0]}\": {embeddings[0][:5]} \n", )
    # [3, 384]

    # 3. Calculate the embedding similarities
    similarities = model.get_similarities(embeddings, embeddings)
    print(
        f"Calculate the embedding similarities:\n{similarities.tolist()}\n\n", )
    # tensor([[1.0000, 0.6660, 0.1046],
    #         [0.6660, 1.0000, 0.1411],
    #         [0.1046, 0.1411, 1.0000]])
