from dotenv import load_dotenv
import os
import sys
load_dotenv(dotenv_path="system/core/src/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
class APIKeyManager:
    """
    Class to manage API keys for different services.
    """
    def __init__(self, name: str) -> None:
        """
        Initialize the APIKeyManager and load API keys from environment variables.
        """  
        try: 
            keys = os.getenv(name)
            self.api_keys = [
                key.strip() for key in keys.split(",") if key.strip()
            ]
            self.current_key_index = 0
        except Exception as e:  
            print(f"Error loading API keys: {e}")
    def get_api_key(self) -> str:
        """
        Get the current API key and update the index for the next call.
        """
        if not self.api_keys:
            return "No API keys available."
        
        api_key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return api_key
    
    def reset_api_key_index(self) -> None:
        """
        Reset the API key index to the first key.
        """
        self.current_key_index = 0
        
    def add_api_key(self, api_key: str) -> None:
        """
        Add a new API key to the list of API keys.
        """
        if api_key not in self.api_keys:
            self.api_keys.append(api_key)
        else:
            print("API key already exists.")
        