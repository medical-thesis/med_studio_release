import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from system.src.core.components.calculator.zero_shot_classification import ZeroShotClassifier
from system.src.core.components.calculator.svm import SVMClassifier
from system.src.core.components.calculator.logistic_regression import LogisticRegression
from system.src.core.components.calculator.knn import KNNClassifier
import pandas as pd

class EvaluateCategory:
    def __init__(self, model_classifier: ZeroShotClassifier | KNNClassifier | LogisticRegression | SVMClassifier,
                 data: pd.DataFrame) -> None:
        self.model_classifier = model_classifier
        self.data = data
        self.results = pd.DataFrame(columns=['question', 'predicted_category', 'actual_category'])
        self.categories = self.data['focus_area'].unique().tolist()
    
    def predict(self, question: str) -> str:
        if not isinstance(self.model_classifier, ZeroShotClassifier):
            return self.model_classifier.predict(question)
        else:
            return self.model_classifier.zero_shot(question, self.categories)
        
    def evaluate(self) -> list:
        for _, row in self.data.iterrows():
            question = row['question']
            pred = self.predict(question)
            actual = row['focus_area']
            self.results = self.results._append({
                'question': question,
                'predicted_category': pred,
                'actual_category': actual,
                'correct': True if pred == actual else False
            }, ignore_index=True)
        return self.results
    
    def get_accuracy(self) -> pd.DataFrame:
        len_results = len(self.results)
        correct_count = self.results['correct'].sum()
        accuracy = correct_count / len_results if len_results > 0 else 0
        return accuracy
    
if __name__ == "__main__":
    data = pd.read_csv("system/core/dataset/processed/medquad_qa_pairs.csv")
    print(data.size)
    dt = data.sample(1000, random_state=42)
    from embeddings import Embedding
    from calculator import ZeroShot
    
    Embedding = Embedding(embedding_model_name="all-MiniLM-L6-v2")
    # model_classifier = ZeroShotClassifier(embedding= Embedding)
    # model_classifier = KNNClassifier(embedding= Embedding, data= data)
    # model_classifier = LogisticRegression(embedding= Embedding, data= data)
    model_classifier = SVMClassifier(embedding= Embedding, data= data)
    
    evaluator = EvaluateCategory(model_classifier= model_classifier, data= dt)
    results = evaluator.evaluate()
    results = evaluator.get_accuracy()
    print("Evaluation Results:")
    print(results)