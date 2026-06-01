# =============================================================================
# Advanced RAG Pipeline
# =============================================================================

import os
import json
import torch
import numpy as np
import pandas as pd
import faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =============================================================================
# 1. Hierarchical Chunking
# =============================================================================

class HierarchicalMedicalChunker:
    def __init__(self, parent_size=300, child_size=100, overlap=20):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def split_text(self, text, chunk_size):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - self.overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def process_corpus(self, dataframe):
        hierarchy_mapping = {}
        child_chunks_list = []

        for idx, row in dataframe.iterrows():
            parent_text = row['full_medical_record']
            parent_id = f"parent_{idx}"
            parents = self.split_text(parent_text, self.parent_size)

            for p_sub_idx, parent_chunk in enumerate(parents):
                p_id = f"{parent_id}_sub_{p_sub_idx}"
                hierarchy_mapping[p_id] = parent_chunk
                children = self.split_text(parent_chunk, self.child_size)
                for c_idx, child_chunk in enumerate(children):
                    child_chunks_list.append({
                        "child_id": f"{p_id}_child_{c_idx}",
                        "parent_id": p_id,
                        "text": child_chunk
                    })
        return child_chunks_list, hierarchy_mapping


# =============================================================================
# 2. Query Expansion
# =============================================================================

def expand_medical_query(user_symptom_query):
    """
    Simulates clinical query expansion by appending generalized medical synonyms
    to maximize search recall in dense indices.
    """
    expansion_library = {
        "cough": ["respiratory distress", "bronchial irritation", "coughing episodes"],
        "fever": ["febrile condition", "elevated body temperature", "pyrexia"],
        "chest pain": ["angina", "thoracic pain", "cardiovascular symptom"],
        "headache": ["migraine", "cephalalgia", "cranial throbbing"],
        "urination": ["polyuria", "renal output", "urinary frequency"]
    }

    expanded_queries = [user_symptom_query]
    lowered_query = user_symptom_query.lower()

    for key, synonyms in expansion_library.items():
        if key in lowered_query:
            expanded_queries.extend(synonyms)

    return list(set(expanded_queries))[:3]


# =============================================================================
# 3. Two-Stage Retrieval & Cross-Encoder Re-ranking
# =============================================================================

def execute_advanced_retrieval(raw_query, embedding_model, faiss_index, child_chunks, parent_map,
                                rerank_tokenizer, rerank_model, top_k_vectors=5, final_top_n=2):
    """
    Performs two-stage retrieval: vector search with query expansion followed by
    cross-encoder re-ranking to return the most relevant parent context chunks.
    """
    queries_to_search = expand_medical_query(raw_query)

    candidate_child_indices = set()
    for q in queries_to_search:
        q_vec = embedding_model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        scores, indices = faiss_index.search(q_vec, top_k_vectors)
        for idx in indices[0]:
            if idx != -1:
                candidate_child_indices.add(idx)

    unique_parent_chunks = {}
    for idx in candidate_child_indices:
        parent_id = child_chunks[idx]['parent_id']
        if parent_id in parent_map:
            unique_parent_chunks[parent_id] = parent_map[parent_id]

    parent_contexts = list(unique_parent_chunks.values())
    if not parent_contexts:
        return []

    pairs = [[raw_query, ctx] for ctx in parent_contexts]
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = rerank_model(**inputs).logits.view(-1).float().tolist()

    ranked_results = sorted(zip(scores, parent_contexts), key=lambda x: x[0], reverse=True)
    return [context for score, context in ranked_results[:final_top_n]]


# =============================================================================
# 4. Context Orchestration & Prompt Assembly
# =============================================================================

def orchestrate_rag_payload(patient_symptoms_input, embedding_model, faiss_index, child_chunks,
                             parent_map, rerank_tokenizer, rerank_model):
    """
    Accepts patient symptoms, orchestrates advanced retrieval, handles grounding data extraction,
    and packages payload for external LLM ingestion.
    """
    discovered_grounding_contexts = execute_advanced_retrieval(
        patient_symptoms_input, embedding_model, faiss_index,
        child_chunks, parent_map, rerank_tokenizer, rerank_model, final_top_n=2
    )
    context_str = "\n---\n".join(discovered_grounding_contexts)

    system_instruction = (
        "You are an advanced, specialized medical assistance LLM. Your objective is to read the patient's "
        "reported symptom descriptions, analyze verified medical clinical case details provided strictly in the "
        "grounding context, and render an indicative diagnosis followed by evidence-based treatment recommendations. "
        "Always flag that your outputs are advisory and require professional medical oversight to uphold patient safety compliance. \n"
    )

    formatted_llm_prompt = (
        f"{system_instruction}\n"
        f"CONTEXT KNOWLEDGE PLATES:\n{context_str}\n\n"
        f"PATIENT SYMPTOM INPUT CASE:\n{patient_symptoms_input}\n\n"
        f"STRICT INFERENCE ASSIGNMENT:\n"
        f"1. Diagnose the condition based explicitly on matching traits in the Context.\n"
        f"2. Suggest detailed therapeutic treatments and immediate recovery protocols.\n"
        f"3. Highlight specific clinical risks or diagnostic logic.\n\n"
        f"DETAILED CLINICAL ANALYSIS:"
    )

    return {
        "status": "success",
        "original_query": patient_symptoms_input,
        "retrieved_evidence_count": len(discovered_grounding_contexts),
        "injected_prompt_payload": formatted_llm_prompt
    }


# =============================================================================
# 5. Setup Helpers
# =============================================================================

def load_dataset_df(max_records=5000):
    """Loads and preprocesses the medical QA dataset from Hugging Face."""
    print("--- Loading Hugging Face Medical QA Dataset ---")
    try:
        raw_dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
        df = pd.DataFrame(raw_dataset)
        df = df[['Question', 'Answer']].rename(columns={'Question': 'symptom_description', 'Answer': 'diagnosis_treatment'})
        df = df.head(max_records)
    except Exception as e:
        print(f"Dataset download failed or timed out: {e}\nFalling back to synthetic robust medical dataframe.")
        fallback_data = {
            "symptom_description": [
                "Patient presents with persistent dry cough, high fever, shortness of breath, and loss of taste.",
                "Patient exhibits severe chest pain radiating to the left arm, acute sweating, and nausea.",
                "Frequent urination, excessive thirst, unexplained weight loss, and chronic fatigue noted.",
                "Acute throbbing unilateral headache accompanied by sensitivity to light, nausea, and visual aura."
            ],
            "diagnosis_treatment": [
                "Diagnosis: Suspected Respiratory Viral Infection (e.g., COVID-19 / Influenza). Treatment: Isolate, monitor oxygen saturation, prescribe antipyretics, and maintain aggressive hydration.",
                "Diagnosis: Suspected Acute Myocardial Infarction (Heart Attack). Treatment: Emergency administration of Aspirin, oxygen therapy, and immediate transfer to cardiac catheterization lab.",
                "Diagnosis: Suspected Type 2 Diabetes Mellitus. Treatment: Initiate Metformin therapy, perform HbA1c screening, mandate strict low-glycemic dietary modifications.",
                "Diagnosis: Classic Migraine Episode. Treatment: Administer Triptans (e.g., Sumatriptan), prescribe NSAIDs, advise resting in a dark, noise-isolated environment."
            ]
        }
        df = pd.DataFrame(fallback_data)

    df['full_medical_record'] = "Symptom/Case: " + df['symptom_description'] + "\n" + df['diagnosis_treatment']
    print(f"Successfully processed {len(df)} medical knowledge records.")
    return df


def build_or_load_index(child_chunks, parent_map, embedding_model,
                         faiss_index_path="medical_rag_data.faiss",
                         metadata_path="medical_rag_data_metadata.json"):
    """Builds a FAISS index from scratch or loads a cached one from disk."""
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        print("--- Found stable cache. Loading database from disk ---")
        faiss_index = faiss.read_index(faiss_index_path)
        with open(metadata_path, 'r') as f:
            loaded_data = json.load(f)
        child_chunks = loaded_data['child_chunks']
        parent_map = loaded_data['parent_map']
        print(f"FAISS Vector database restored with {faiss_index.ntotal} records.")
    else:
        print("--- Cache not found. Initiating vectorization and saving stage ---")
        child_texts = [item['text'] for item in child_chunks]
        print("Vectorizing medical knowledge base chunks...")
        child_embeddings = embedding_model.encode(child_texts, show_progress_bar=True, convert_to_numpy=True)
        embedding_dim = child_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(child_embeddings)
        faiss_index.add(child_embeddings)
        print("Exporting data to persistent storage...")
        faiss.write_index(faiss_index, faiss_index_path)
        with open(metadata_path, 'w') as f:
            json.dump({"child_chunks": child_chunks, "parent_map": parent_map}, f, indent=4)
        print(f"Successfully created: '{faiss_index_path}' and '{metadata_path}'.")

    return faiss_index, child_chunks, parent_map


def load_models():
    """Loads the embedding model and cross-encoder reranker."""
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker_name = "BAAI/bge-reranker-base"
    rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
    rerank_model.eval()
    return embedding_model, rerank_tokenizer, rerank_model


# =============================================================================
# 6. Main
# =============================================================================

if __name__ == "__main__":
    # Load dataset
    df = load_dataset_df(max_records=5000)

    # Chunk corpus
    chunker = HierarchicalMedicalChunker()
    child_chunks, parent_map = chunker.process_corpus(df)
    print(f"Generated {len(parent_map)} Parent Context Chunks and {len(child_chunks)} Child Embedding Chunks.")

    # Load models
    embedding_model, rerank_tokenizer, rerank_model = load_models()

    # Build or load FAISS index
    faiss_index, child_chunks, parent_map = build_or_load_index(child_chunks, parent_map, embedding_model)

    # Test query expansion
    sample_query = "Patient feels dizzy and has sharp chest pain"
    print(f"Original Input: {sample_query}")
    print(f"Expanded Search Vectors: {expand_medical_query(sample_query)}")

    # Test retrieval
    retrieved_contexts = execute_advanced_retrieval(
        "I am noticing extreme thirst and going to the bathroom too often.",
        embedding_model, faiss_index, child_chunks, parent_map,
        rerank_tokenizer, rerank_model, final_top_n=1
    )
    print("\n--- Advanced Top Retrieved Context ---")
    print(retrieved_contexts[0] if retrieved_contexts else "No context found.")

    # Test full pipeline
    test_symptoms = "Patient has a high fever, severe dry cough and cannot smell anything."
    pipeline_payload = orchestrate_rag_payload(
        test_symptoms, embedding_model, faiss_index,
        child_chunks, parent_map, rerank_tokenizer, rerank_model
    )
    print(pipeline_payload["injected_prompt_payload"])