from pymorphy3 import MorphAnalyzer
import nltk
from nltk.corpus import stopwords
import joblib
import re
import logging

logger = logging.getLogger(__name__)

nltk.download('stopwords')

class PromptClassifier:
    def __init__(self):
        self.russian_stopwords = stopwords.words('russian')
        self.morph = MorphAnalyzer()
        self.model = joblib.load('resources/classifier.joblib')
        self.vectorizer = joblib.load('resources/vectorizer.joblib')
        logger.info("Prompt classifier initialized")

    def is_image_request(self, text: str) -> bool:
        processed_text = self._preprocess_text(text)
        features = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(features)[0]
        return bool(prediction)

    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [self.morph.parse(token)[0].normal_form for token in tokens]
        return ' '.join([t for t in tokens if t not in self.russian_stopwords])