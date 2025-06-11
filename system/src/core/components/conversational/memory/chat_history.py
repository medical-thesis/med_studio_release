class ChatHistory:
    def __init__(self, max_history: int= 5):
        """
        initialize the ChatHistory class to manage chat history.
        Parameters:
        - max_history: The maximum number of chat history entries to keep.
        """
        self.history = []
        self.max_history = max_history
        
    def add_message(self, context: dict) -> None:
        """
        Add a new conversation to the chat history.
        """
        if not isinstance(context, dict) or 'query' not in context or 'response' not in context:
            raise ValueError("Context must be a dictionary with 'query' and 'response' keys.")
        self.history.append(context)
        
    def clear_history(self) -> None:
        """
        Clear the chat history.
        """
        self.history = []
        
    def get_latest_history(self) -> dict:
        """
        Get the latest entry in the chat history.
        Retries the last entry if it exists, otherwise returns None.
        """
        return self.history[-1] if self.history else None
    
    def get_context_chathistory(self) -> str:
        """
        Get the context of the chat history.
        This method retrieves the last 'max_history' entries from the chat history.
        If the history is shorter than 'max_history', it retrieves all entries.
        
        Returns:
        - str: The context of the chat history formatted as a string.
        """
        if not self.history:
            return "No chat history available."
        
        chathistory = self.history[-self.max_history:] if len(self.history) >= self.max_history else self.history
        context = "\n".join([f"- Query: {chat['query']}\n- Response: {chat['response']}" for chat in chathistory])
        return context
    
    def limit_chat_history(self, k: int= 5) -> None:
        """
        Limit the chat history to the last 'k' entries.
        This method modifies the chat history to keep only the last 'k' entries.
        If 'k' is greater than the current length of the history, it does nothing.
        Parameters:
        - k: The number of entries to keep in the chat history.
        """ 
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        
        self.history = self.history[-k:] if len(self.history) >= k else self.history
        
    def get_chat_history(self) -> list:
        """
        Get the entire chat history.
        Returns:
        - list: The chat history as a list of dictionaries.
        """
        return self.history