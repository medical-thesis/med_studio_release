from dotenv import load_dotenv
import os
import sys

load_dotenv(dotenv_path="system/src/core/config/.env")
project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from system.src.core.components.calculator.knn import KNNClassifier
from system.src.core.components.calculator.zero_shot_classification import ZeroShotClassifier
from system.src.core.components.calculator.svm import SVMClassifier
from system.src.core.components.calculator.logistic_regression import LogisticRegression

class LogicalRouting:
    def __init__(self, classifier: KNNClassifier |
                                         SVMClassifier |
                                         LogisticRegression |
                                         ZeroShotClassifier = None) -> None:
        self.classifier = classifier
        if self.classifier is None:
            self.classifier = ZeroShotClassifier()
            
    def classify(self, query: str, documents: list) -> str:
        if isinstance(self.classifier, ZeroShotClassifier):
            return self.classifier.zero_shot(query, documents)
        else:
            return self.classifier.predict(query)