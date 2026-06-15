# COMP8420-Group-L-Healthcare: MediAssist AI

# COMP8420-Group-L-Healthcare: MediAssist AI

MediAssist AI is a modular, privacy-preserving clinical chatbot pipeline. This project integrates Sentiment Analysis, Named Entity Recognition (NER), and a Retrieval-Augmented Generation (RAG) system with fine-tuned Generative LLMs to provide safe, fact-grounded medical guidance.

## 📁 Repository Structure

* **`LLM/`** - Contains the LoRA fine-tuned adapters (`biogpt_familydoctor_final`, `qwen3_1_7b_familydoctor_final`) and evaluation notebooks.
* **`drug-disease-mapping/`** - Datasets and mapping mechanism (`medic-mapping.ipynb`).
* **`medical-ner/`** - Custom spaCy pipeline experiments (`medical-ner-experiment.ipynb`).
* **`medical-specialties/`** - Specialty classification models (`specialties.ipynb`).
* **`rag-pipeline/`** - Vector index (`medical_rag_data.faiss`) and pipeline code.
* **`sentiment-analysis/`** - LinearSVC model experiments (`drugs-review-sentiment.ipynb`).
* **`web_app/`** - Initial UI prototype and backend services (`llm_service.py`).
* **`web_app-integrated/`** - **[MAIN DELIVERABLE]** The final, production-ready Streamlit application. Contains the main UI (`medi_chat.py`) and all modular components (`ner.py`, `retriever.py`, etc.).
* **`COMP8420_Group_L_Healthcare.ipynb`** - Master notebook containing combined project experiments.

## 🚀 Quick Start (Running the App)

**1. Clone the repository**
```bash
git clone [https://github.com/aronwilbert/COMP8420-Group-L-Healthcare.git](https://github.com/aronwilbert/COMP8420-Group-L-Healthcare.git)
cd COMP8420-Group-L-Healthcare
```

**2. Install dependencies
Navigate to the UI folder to access the requirements file
```bash
cd web_app/ui
pip install -r requirements.txt
```

**3. Launch the Integrated UI
Navigate to the integrated web app folder and run the Streamlit server:
```bash
cd ../../web_app-integrated
streamlit run medi_chat.py
```


