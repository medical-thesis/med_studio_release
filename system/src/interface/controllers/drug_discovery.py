import os

class DrugDiscoveryController:
    def __init__(self):
        self.data_dir = "system/src/core/processed_datasets"
        self.items_per_page = 18 
    
    def get_medicine_list(self):
        return sorted([f.replace(".md", "") for f in os.listdir(self.data_dir) if f.endswith(".md")])

    def load_medicine_info(self, medicine_name):
        file_path = os.path.join(self.data_dir, f"{medicine_name}.md")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()