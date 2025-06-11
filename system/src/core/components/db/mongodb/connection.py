from pymongo import MongoClient, database

class MongoDBConnection:
    def __init__(self, uri: str, db_name: str):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None

        self.connect()
        
    def set_uri(self, uri: str):
        self.uri = uri
    
    def set_db(self, db_name: str):
        self.db_name = db_name
        
    def connect(self) -> database.Database:
        try:
            self.client = MongoClient(host=self.uri)
            self.db = self.client[self.db_name]
            self.db.command('ping')
            self.client.admin.command('ping')
            
            print("Pinged your deployment. You successfully connected to MongoDB!")
            print(f"Successfully connected to MongoDB database: {self.db_name}")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            self.db = None
        
        return self.db

    def get_db(self) -> database.Database:
        return self.db  

    def close_connection(self):
        try:
            if self.client:
                self.client.close()
                print("MongoDB connection closed.")
            else: 
                print("No active MongoDB connection to close.")
        except Exception as e:
            print(f"Failed to close the MongoDB connection.")
            print(f"Full error details: {str(e)}")
            
            