# COMP8420-Group-L-Healthcare: MediAssist AI

MediAssist AI is a modular, privacy-preserving clinical chatbot pipeline. This project integrates Sentiment Analysis, Named Entity Recognition (NER), and a Retrieval-Augmented Generation (RAG) system with fine-tuned Generative LLMs to provide safe, fact-grounded medical guidance.

## Repository Structure

* **`LLM/`** - Contains the LoRA fine-tuned adapters (`biogpt_familydoctor_final`, `qwen3_1_7b_familydoctor_final`) and evaluation notebooks.
* **`medical-ner/`** - Custom spaCy pipeline experiments (`medical-ner-experiment.ipynb`).
* **`medical-specialties/`** - Specialty classification model experiments (`specialties.ipynb`).
* **`rag-pipeline/`** - Vector index (`medical_rag_data.faiss`) and pipeline code experiments.
* **`sentiment-analysis/`** - LinearSVC model experiments (`drugs-review-sentiment.ipynb`).
* **`web_app-integrated/`** - The final, production-ready Streamlit application. Contains the main UI (`medi_chat.py`) and all modular components (e.g. `ner.py`, `retriever.py`, etc.).
* **`dataset-samples/`** - Contains excerpt samples from the datasets we have used to develop this system.

## Quick Start (Running the App)

**1. Clone the repository**
```bash
git clone https://github.com/aronwilbert/COMP8420-Group-L-Healthcare.git
cd COMP8420-Group-L-Healthcare
```

**2. Install dependencies
Navigate to the UI folder to access the requirements file
```bash
cd web_app-integrated
pip install -r requirements.txt
```

**3. Launch the Integrated UI
Navigate to the integrated web app folder and run the Streamlit server:
```bash
# from inside the `web_app-integrated` folder
streamlit run medi_chat.py
```

## Development
To make changes to resource files like serialised RAG database, adapter (LoRA) weights, classifier and vectoriser files (in joblib format), run the necessary ipynb notebook files generate updated version of these files.


## Individual Contributions
Individual contributions of each group member on this repository can be found under the contributors tab on the right.

| Name | Github handle |
|------|---------------|
| Md Juhaer Adittya Pasha | pasha-292 |
| Abrar Mahmud | AbrarMahmudMq |
| Siyong Feng | SiyongFeng |
| Aaron Wilbert Kosidin | aronwilbert |
| Stanley Wijaya | stanleywijayamq, tonehparkah |