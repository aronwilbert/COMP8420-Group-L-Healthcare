import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dataclasses import dataclass

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

import medspacy # even though unused, this needed to function
from loguru import logger

from bs4 import BeautifulSoup


@dataclass
class Entity:
    text:  str
    score: float
    start: int
    end:   int

class EntityExtractor:
    tokenizer: AutoTokenizer
    model:     AutoModelForTokenClassification

    def __init__(self, classifier_key: str, device_map: str | None = None) -> None:
        tokenizer = AutoTokenizer.from_pretrained(classifier_key, use_fast = True)
        model     = AutoModelForTokenClassification.from_pretrained(classifier_key, device_map = device_map)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text: str):
        encoded_tokens = self.tokenizer(text, return_tensors = 'pt', return_offsets_mapping = True, truncation = True) # type: ignore
        token_spans    = encoded_tokens.pop('offset_mapping')[0]
        token_count    = encoded_tokens['input_ids'].shape[1]

        with torch.no_grad():
            outputs = self.model(**encoded_tokens) # type: ignore

        logits     = outputs.logits[0]
        pred_ids   = logits.argmax(dim = -1)
        pred_probs = logits.softmax(dim = -1)
        tokens     = self.tokenizer.convert_ids_to_tokens(encoded_tokens['input_ids'][0]) # type: ignore

        # filter pass
        subword_entities: list[tuple[Entity, int]] = []
        for i in range(token_count):
            toktype = pred_ids[i]
            if toktype == 0:
                continue
            subword_entities.append(
                (
                    Entity(
                        text  = tokens[i],
                        score = pred_probs[i, toktype].item(),
                        start = token_spans[i, 0].item(),
                        end   = token_spans[i, 1].item()
                    ),
                    toktype.item()
                )
            )


        # stitch pass
        stitched_entities: list[tuple[Entity, int]] = []
        stitch_buffer: list[tuple[Entity, int]] = []
        def stitch():
            nonlocal stitch_buffer
            if len(stitch_buffer) == 0:
                return
            stitched_entities.append(
                (
                    Entity(
                        text  = ''.join(map(lambda item : item[0].text.replace('##', ''), stitch_buffer)),
                        score = stitch_buffer[0][0].score,
                        start = stitch_buffer[0][0].start,
                        end   = stitch_buffer[-1][0].end,
                    ),
                    stitch_buffer[0][1]
                )
            )
            stitch_buffer = []

        i = 0
        while i < len(subword_entities):
            assert not subword_entities[i][0].text.startswith('##'), 'Token at this index should not start with ##'
            stitch_buffer.append(subword_entities[i])
            j = i + 1
            increment = 1
            while j < len(subword_entities) and subword_entities[j][0].text.startswith('##'):
                stitch_buffer.append(subword_entities[j])
                j += 1
                increment += 1
            stitch()
            i += increment
        stitch()


        # word chaining pass
        chained_entities: list[Entity] = []
        chain_buffer: list[Entity] = []
        def chain():
            nonlocal chain_buffer
            if len(chain_buffer) == 0:
                return
            chained_entities.append(
                Entity(
                    # text  = ' '.join(map(lambda item : item.text, chain_buffer)),
                    text  = text[chain_buffer[0].start:chain_buffer[-1].end],
                    score = chain_buffer[0].score,
                    start = chain_buffer[0].start,
                    end   = chain_buffer[-1].end,
                )
            )
            chain_buffer = []

        i = 0
        while i < len(stitched_entities):
            assert stitched_entities[i][1] != 2, 'Sub-entity at this index should not be of type 2'
            chain_buffer.append(stitched_entities[i][0])
            j = i + 1
            increment = 1
            while j < len(stitched_entities) and stitched_entities[j][1] == 2:
                chain_buffer.append(stitched_entities[j][0])
                j += 1
                increment += 1
            chain()
            i += increment
        chain()

        return chained_entities

@Language.factory('medical_ner')
class CustomNER:
    disease_extractor:  EntityExtractor
    medicine_extractor: EntityExtractor

    def __init__(self, nlp: Language, name: str):
        self.disease_extractor  = EntityExtractor('OpenMed/OpenMed-NER-DiseaseDetect-BioClinical-108M')
        self.medicine_extractor = EntityExtractor('OpenMed/OpenMed-NER-PharmaDetect-BioPatient-108M')
        pass

    def merge_entity_sequences(self, sequences: list[list[Entity]], types: list[str]):
        assert len(sequences) == len(types), "Length of sequence list and type list must match"

        merged_sequence: list[tuple[Entity, str]] = []

        for sequence, seq_type in zip(sequences, types):
            merged_sequence.extend(zip(
                sequence,
                [ seq_type ] * len(sequence)
            ))

        merged_sequence.sort(key = lambda e_t : e_t[0].start)

        return merged_sequence

    def __call__(self, doc: Doc):
        diseases  = self.disease_extractor(doc.text)
        medicines = self.medicine_extractor(doc.text)

        spacy_spans: list[Span] = []

        for entity, entype in self.merge_entity_sequences([ medicines, diseases ], [ 'MEDICINE', 'DISEASE' ]):
            span = doc.char_span(entity.start, entity.end, label = entype)
            if span is not None:
                spacy_spans.append(span)

        doc.ents = spacy_spans

        return doc

NLP_NER_PIPELINE = spacy.blank("en")
NLP_NER_PIPELINE.add_pipe('medspacy_pyrush')  # from medspacy
NLP_NER_PIPELINE.add_pipe('medical_ner')      # our solution
NLP_NER_PIPELINE.add_pipe('medspacy_context') # from medspacy

logger.remove() # Make medspacy PyRUSH silent, otherwise logs too much.

COLOR_MAP  = {
    'DISEASE':            {'bg': "#6e2b37", 'fg': "#a3a3a3"},
    'MEDICINE':           {'bg': "#156fac", 'fg': "#d8d8d8"},
    'POSSIBLE_EXISTENCE': {'bg': "#8f7302", 'fg': "#b1b1b1"},
    'NEGATED_EXISTENCE':  {'bg': "#501b57", 'fg': "#969696"},
    'HYPOTHETICAL':       {'bg': "#03775e", 'fg': "#a2bdba"}
}

DEFAULT_COLOR = {'bg': '#e2e8f0', 'fg': '#1e293b'}

def highlight_entities(text: str, entities: list) -> str:
    # Sort entities in reverse order by their start index to protect string indices
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse = True)
    html_text = text

    for ent in sorted_entities:
        start = ent['start']
        end   = ent['end']
        label = ent['label']

        colors = COLOR_MAP.get(label, DEFAULT_COLOR)

        entity_text = html_text[start:end]

        # Modern inline HTML structure using flexbox for perfect alignment
        html_tag = (
            f'<mark style="background: {colors["bg"]}; color: {colors["fg"]}; '
            f'padding: 2px 6px; margin: 0 3px; border-radius: 4px; '
            f'font-weight: 500; display: inline-flex; align-items: center; '
            f'gap: 6px; font-size: 0.95em;">'
            f'{entity_text}'
            f'<span style="font-size: 0.65em; font-weight: 700; '
            f'background: rgba(0, 0, 0, 0.08); padding: 1px 4px; '
            f'border-radius: 3px; text-transform: uppercase; letter-spacing: 0.5px;">'
            f'{label}</span></mark>'
        )

        # Splice the HTML tag into the original text
        html_text = html_text[:start] + html_tag + html_text[end:]

    return html_text

# :violet-badge[:material/star: Favorite]
MD_BADGE_SETTINGS = {
    'DISEASE':            { 'color': 'red',    'icon': 'material/star' },
    'MEDICINE':           { 'color': 'blue',   'icon': 'material/star' },
    'POSSIBLE_EXISTENCE': { 'color': 'green',  'icon': 'material/star' },
    'NEGATED_EXISTENCE':  { 'color': 'violet', 'icon': 'material/star' },
    'HYPOTHETICAL':       { 'color': 'yellow', 'icon': 'material/star' }
}

def highlight_entities_markdown(text: str, entities: list) -> str:
    # Sort entities in reverse order by their start index to protect string indices
    sorted_entities = sorted(entities, key = lambda x: x['start'], reverse = True)
    md_text = text

    for ent in sorted_entities:
        start = ent['start']
        end   = ent['end']
        label = ent['label']

        bset = MD_BADGE_SETTINGS.get(label, MD_BADGE_SETTINGS['NEGATED_EXISTENCE'])

        entity_text = md_text[start:end]

        md_badge = f':{bset["color"]}-badge[:{bset["icon"]}: {entity_text} ({label})]'

        md_text = md_text[:start] + md_badge + md_text[end:]

    return md_text

def revert_highlights_bs4(html_text: str) -> str:
    soup = BeautifulSoup(html_text, 'html.parser')

    # .decompose() completely deletes the tag AND its inner label text
    for span in soup.find_all('span'):
        span.decompose()

    # .get_text() returns the remaining text, automatically stripping the <mark> tags
    return soup.get_text()

class MedicalNERModule:
    def __init__(self) -> None:
        pass

    def extract_entities_and_modifiers(self, text: str):
        doc = NLP_NER_PIPELINE(text) # nlp is global variable of this file
        entities:  list[dict] = []
        modifiers: list[dict] = []

        visualized_modifiers = set()

        for target in doc.ents:
            entities.append({
                'start': target.start_char,
                'end':   target.end_char,
                'label': target.label_.upper()
            })

            for modifier in target._.modifiers:
                if modifier in visualized_modifiers:
                    continue
                span = doc[modifier.modifier_span[0]: modifier.modifier_span[1]]
                modifiers.append({
                    'start': span.start_char,
                    'end':   span.end_char,
                    'label': modifier.category
                })

        return entities, modifiers, doc

    def apply_highlighting(self, text: str, entities: list[dict], modifiers: list[dict]):
        return highlight_entities_markdown(text, entities + modifiers)

    def remove_highlighting(self, text: str):
        return revert_highlights_bs4(text)
