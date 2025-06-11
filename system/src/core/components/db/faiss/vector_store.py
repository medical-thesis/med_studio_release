from data_preparation import DataPreparation
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStore:
    def __init__(self, data_preparation: DataPreparation, embedding_name: str= None, api_key: str= None) -> None:
        self.data_preparation = data_preparation
        self.embedding = None
        self.initialize_embeddings(embedding_name, api_key)
        self.vector_store = None

    def initialize_embeddings(self, embedding_name: str, api_key: str= None) -> None:
        try:
            if embedding_name:
                if api_key:
                    self.embedding = HuggingFaceEmbeddings(model_name=embedding_name, api_key=api_key)
                else:
                    self.embedding = HuggingFaceEmbeddings(model_name=embedding_name)
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            
    def create_vector_store(self) -> FAISS:
        """Create a vector store from the documents."""
        self.vector_store = FAISS.from_documents(
            self.data_preparation.documents, 
            self.embedding, 
            )
        return self.vector_store
    
    def save_vector_store(self, path: str) -> None:
        """Save the vector store to a file."""
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}.")
        
    def load_vector_store(self, path: str) -> FAISS:
        """Load the vector store from a file."""
        self.vector_store = FAISS.load_local(path, self.embedding, allow_dangerous_deserialization=True)
        return self.vector_store        
