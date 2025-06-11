from vector_store import VectorStore

class VectorStoreService:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
    
    def search_similarity(self, query: str, filter: dict= None, k: int = 5) -> list:
        results = self.vector_store.vector_store.similarity_search_with_score(
            query, 
            k=k,
            filter=filter)
        return results
    

