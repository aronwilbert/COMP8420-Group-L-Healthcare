from transformers import pipeline, TextClassificationPipeline

MAPPING = {
    'LABEL_0': 'NEGATIVE',
    'LABEL_1': 'POSITIVE'
}

class SentimentAnalysisModule:
    model: TextClassificationPipeline

    def __init__(self) -> None:
        self.model = pipeline('text-classification', model = 'phanerozoic/BERT-Sentiment-Classifier', device_map = 'auto')

    def classify(self, text: str):
        result = self.model(text)

        if not result:
            return 'NEUTRAL', 0.5

        result = result[0]

        label, score = result['label'], result['score']

        return MAPPING.get(label, 'NEUTRAL'), score

    def __call__(self, text: str):
        return self.classify(text)