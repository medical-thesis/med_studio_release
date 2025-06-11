import pandas as pd
from typing import List, Dict, Any

class PreProcess():
    """
    # Sample structure data: text chunks with metadata
        documents = [
            {
                "text": "",
                "metadata": {"category": "", "sub_category": "", "date": ".."}
            },
            {
                "text": "",
                "metadata": {"category": "", "sub_category": "", "date": ".."}
            }
        ]
    """

    def __init__(self, in_path='../../dataset/raw/medquad_qa_pairs.csv', out_path='../../dataset/processed/medquad_qa_pairs.csv'):
        self.in_path = in_path
        self.out_path = out_path

    def clean_and_pre_process_data(self):
        df = pd.read_csv(self.in_path)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        df['source'].value_counts()
        df['source'] = df['source'].replace('NIHSeniorHealth', 'Senior Health')
        df['source'] = df['source'].replace('CancerGov', 'Comprehensive Cancer')
        df['source'] = df['source'].replace('MPlusHealthTopics', 'Medline Plus, Health Topics')
        df['source'] = df['source'].replace('GARD', 'Genetic and Rare Diseases (GARD)')
        df['source'] = df['source'].replace('CDC', 'Disease Control and Prevention (CDC)')
        df['source'] = df['source'].replace('NHLBI', 'Heart, Lung, and Blood (NHLBI)')
        df['source'] = df['source'].replace('NINDS', 'Neurological Disorders and Stroke (NINDS)')
        df['source'] = df['source'].replace('GHR', 'Growth Hormone Receptor (GHR)')
        df['source'] = df['source'].replace('NIDDK', 'Diabetes and Digestive and Kidney Diseases (NIDDK)')

        unique_source = df['source'].unique()
        print('\n\n ========== \n')
        print('Dataset after pre-processed:')
        print('List source database: ', unique_source)

        for i in unique_source:
            print(f'Type {i}, number of record: ', df[df['source'] == i]['source'].count())

        df.to_csv(self.out_path, index=False)

    def commit_processed_data(self) -> List[Dict[str, Any]]:
        df = pd.read_csv(self.in_path)

        documents = []

        for _, row in df.iterrows():
            doc = {
                "text": str(row["title"]) + " " + str(row["content"]),
                
                "metadata": {
                    "category": row["category"],
                    "sub_category": row["sub_category"]
                }
            }

            documents.append(doc)

        print("\n\n === Sample dataset ===")
        print(documents)

        return documents