import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class BaselineDetector:
    """
    Baseline detector wrapper providing model persistence.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        print(f"Initializing baseline pipeline with random_state={self.random_state}")
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('clf', LogisticRegression(random_state=self.random_state, max_iter=1000))
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)
        
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def save_model(self, filepath: str):
        """Saves the trained pipeline (vectorizer + model) to disk."""
        joblib.dump(self.pipeline, filepath)
        
    @classmethod
    def load_model(cls, filepath: str) -> 'BaselineDetector':
        """Loads a trained pipeline from disk into a new BaselineDetector instance."""
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance

def create_baseline_pipeline(random_state: int = 42):
    # Maintained for backwards compatibility in main.py
    return BaselineDetector(random_state=random_state)
