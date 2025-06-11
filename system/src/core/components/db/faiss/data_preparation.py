import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
class DataPreparation:
    def __init__(self, data: pd.DataFrame, chunk_size: int= 2000, overlap=200) -> None:
        self.data = data
        self.documents = []
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        self.focus_areas = []

        self.prepare_data()
        
    def prepare_data(self) -> None:
        self.focus_areas = self.get_focus_areas()
        self.documents = self.get_documents()

    def split_text(self, text: str) -> list:
        """Split the text into chunks."""
        return self.text_splitter.split_text(text)

    def get_documents(self) -> list:
        for _, row in self.data.iterrows():
            content = row["question"] + " " +row["answer"]
            metadata = {
                "question": row["question"],
                "answer": row["answer"],
                "source": row["source"],
                "focus_area": row["focus_area"]
                }
            
            chunks = self.split_text(content)
            
            for chunk in chunks:
                self.documents.append(
                    Document(
                        page_content=chunk, 
                        metadata=metadata
                        )
                    )
        return self.documents

    def get_focus_areas(self) -> list:
        return self.data["focus_area"].unique().tolist()