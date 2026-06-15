from transformers import pipeline, TokenClassificationPipeline

ALLOWED_GROUPS = { 'age', 'gender' }

class CensorshipModule:
    model: TokenClassificationPipeline
    placeholder: str

    def __init__(self, placeholder: str = '\\*') -> None:
        self.placeholder = placeholder
        self.model = pipeline('ner', model = 'openmed/OpenMed-PII-ClinicalE5-Small-33M-v1', aggregation_strategy = 'simple') # type: ignore

    def redact(self, text: str, entities: list[dict]):
        sorted_entities = sorted(entities, key = lambda x : x['start'], reverse = True)
        redacted = text
        for ent in sorted_entities:
            if ent['entity_group'] in ALLOWED_GROUPS:
                continue
            redacted = redacted[:ent['start']] + self.placeholder + redacted[ent['end']:]
        return redacted

    def extract(self, text: str):
        entities = self.model(text)
        return entities

    def extract_and_redact(self, text: str):
        entities = self.extract(text)
        text = self.redact(text, entities)
        return text

    def __call__(self, text: str):
        return self.extract_and_redact(text)