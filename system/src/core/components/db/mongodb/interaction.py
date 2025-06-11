from typing import Dict, Any
from pymongo import MongoClient, database, collection
from pymongo.database import Database
from pymongo.collection import Collection

class MongoDBManager:
    def __init__(self, db: database.Database):
        self.db = db
        self.collection = None

    def create_collection(self, collection_name: str) -> Collection:
        self.collection = self.db[collection_name]
        print(f"Collection '{collection_name}' created or retrieved successfully!")
        return self.collection
    
    def set_collection(self, collection: Collection):
        self.collection = collection

    def insert_data(self, data: Dict[str, Any], collection_name: str):
        self.collection = self.db[collection_name]
        
        try:
            if data:
                if isinstance(data, list):
                    self.collection.insert_many(data)
                    print(f"[MongoDBManager, insert_many] - Insert data successfully.")
                else:
                    self.collection.insert_one(data)
                    print(f"[MongoDBManager, insert_one] - Insert data successfully.")
            else: 
                print(f"Failed to insert data because data is empty.")
        except Exception as e:
            print(f"Failed to insert data into MongoDB database: {e}")



            