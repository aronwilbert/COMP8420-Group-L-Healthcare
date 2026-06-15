import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from typing import Callable, Any

from unidecode import unidecode
import re
import html

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Callable, Any

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = unidecode(text)

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)

    # text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

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

class SentimentAnalysisModule:
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
        return (self.classes[class_index], )

    def __call__(self, text: str) -> Any:
        return self.classify(text)