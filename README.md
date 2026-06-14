# COMP8420-Group-L-Healthcare: MediAssist AI

MediAssist AI is a modular, privacy-preserving clinical chatbot pipeline. This project integrates Sentiment Analysis, Named Entity Recognition (NER), and a Retrieval-Augmented Generation (RAG) system with fine-tuned Generative LLMs to provide safe, fact-grounded medical guidance.

## 📁 Repository Structure

* **`LLM/`** - Training, fine-tuning (LoRA), and evaluation scripts for the Generative LLMs (BioGPT, Qwen3).
* **`drug-disease-mapping/`** - Mechanism to map extracted drugs to their corresponding diseases.
* **`medical-ner/`** - Custom spaCy and Medspacy pipeline using OpenMed transformers for entity extraction and clinical negation detection.
* **`medical-specialties/`** - Logic for categorizing queries into specific medical specialties.
* **`rag-pipeline/`** - Hierarchical chunking, FAISS vector index, and cross-encoder re-ranking implementation grounded on the MedQuAD dataset.
* **`sentiment-analysis/`** - LinearSVC model for predicting patient sentiment from drug reviews.
* **`web_app/`** - Initial prototype for the chatbot UI.
* **`web_app-integrated/`** - The final, production-ready Streamlit application that integrates all modular pipelines (NER, RAG, Sentiment, LLM).
* **`COMP8420_Group_L_Healthcare.ipynb`** - Master Google Colab notebook containing combined project experiments.

## 🚀 Quick Start (Running the App)

**1. Clone the repository**
```bash
git clone [https://github.com/aronwilbert/COMP8420-Group-L-Healthcare.git](https://github.com/aronwilbert/COMP8420-Group-L-Healthcare.git)
cd COMP8420-Group-L-Healthcare
```

**2. Install dependencies
```bash
pip install -r requirements.txt 
# Or manually install: torch transformers peft spacy medspacy streamlit faiss-cpu sentence-transformers textblob
```

**3. Launch the Integrated UI
Navigate to the integrated web app folder and run the Streamlit server:
```bash
cd web_app-integrated
streamlit run app.py  # Replace 'app.py' with your actual main script name (e.g., medi_chat.py)
```
