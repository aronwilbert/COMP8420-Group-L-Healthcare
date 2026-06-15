import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from typing import Callable, Any

from unidecode import unidecode
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Callable, Any

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_medical_transcript(text: str) -> str:

    text = unidecode(text)

    # 1. Standardize case and strip line breaks/tabs
    text = text.lower()
    text = re.sub(r'[\r\n\t]', ' ', text)
    
    # 2. Remove universal formatting noise (duplicate commas, list numbers like "1. ", "2. ")
    text = re.sub(r',\s*,', ' ', text)
    text = re.sub(r'\b\d+\.\s+', ' ', text)

    # 3. Standardize symbols that carry universal clinical meaning
    text = text.replace('%', ' percent ')
    
    # 4. Abstract ALL numerical values (weights, dosages, dimensions, vitals)
    # This prevents your TF-IDF vocabulary from bloating with thousands of unique numbers
    text = re.sub(r'\b\d+(?:\.\d+)?\b', ' [num] ', text)

    # 5. Universal Punctuation Stripper
    # Keeps only lowercase letters, numbers, spaces, and our placeholder brackets.
    # This naturally handles hyphens (e.g., "beta-blocker" -> "beta blocker", "2-d" -> "2 d")
    text = re.sub(r'[^a-z0-9\[\]\s]', ' ', text)
    
    # 6. Collapse duplicate spaces
    text = " ".join(text.split())

    return text

def remove_stopwords(tokens: list[str]) -> list[str]:
    return list(
        filter(
            lambda t : t not in stop_words and len(t) > 1,
            tokens
        )
    )

def lemmatize(tokens: list[str]) -> list[str]:
    return list(
        map(
            lambda t: lemmatizer.lemmatize(t),
            tokens
        )
    )

class TextPreprocessor:
    transformations: list[Callable]

    def __init__(self) -> None:
        self.transformations = []

    def add(self, tfomer: Callable):
        self.transformations.append(tfomer)
        return self

    def __call__(self, sample: str) -> Any:
        for tform in self.transformations:
            sample = tform(sample)
        return sample

class SpecialtyClassifierModule:
    classifier: LinearSVC
    vectorizer: TfidfVectorizer
    classes:    list[str]

    def __init__(self, serialized_dict_path: str) -> None:
        sklearn_dict = joblib.load(serialized_dict_path)

        self.classifier = sklearn_dict['classifier']
        self.classes = sklearn_dict['classes']
        self.vectorizer = sklearn_dict['vectorizer']

    def classify(self, text: str):
        vector = self.vectorizer.transform([text])
        class_index = self.classifier.predict(X = vector)[0]
        return {
            'prediction': self.classes[class_index]
        }

    def __call__(self, text: str) -> Any:
        return self.classify(text)