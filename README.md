# **PsyBot – Local LLM Clinical Assistant**
**PsyBot** is a local AI assistant designed for interpreting psychometric questionnaires, retrieving ICD-11 diagnostic descriptions, and providing structured clinical insights — all without calling external LLM APIs.
This project was developed as part of a research internship inspired by the MMASH dataset, which explores physiological and psychological signals (including wearable devices) to support the analysis of mental disorders. PsyBot integrates some of the same psychometric tools used in MMASH and extends them into a structured clinical chatbot capable of providing:
- Definitions of disorders and psychometric tests
- Comparisons between disorders
- Diagnostic suggestions based on user-submitted questionnaire scores

## **Main Features**
- Local LLM (**Meta-Llama-3-8B-Instruct.Q5_K_M.gguf** via **llama-cpp-python**)
- RAG engine using scientific PubMedBERT embeddings (**pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb**)
- ICD-11 API integration for structured diagnosis and hierarchy
- YAML-based questionnaires 
- YAML-based rules for mapping psychometric scores to disorders
- Fully offline inference, no external LLMs used
- **Gradio**-based UI

## **Installation**
1. Clone the repository
```bash
git clone https://github.com/yourusername/psybot.git
cd psybot
```

2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

## **Configuration**
Review the parameters in config.py to match your hardware setup, particularly:



## **Running the App**
```bash
python src/main.py
```

## **Project Structure**
PsyBot/
├── docs/             # YAML and Markdown documents
├── models/           # Quantized LLaMA GGUF models
├── prompts/          # Prompt templates
├── rag_index/        # RAG chunks and FAISS index
├── src/              # Source code
├── .env              # ICD API credentials
├── log.csv           # Interaction logging
└── requirements.txt  # Python dependencies

### **Example Input**
```yaml
A patient obtained these scores: STAI-Y: 72, PSQI: 13, BIS: 28, BAS Drive: 14, MEQ: 50.
Which disorder could it be?
```

### **Notes**
* ICD-11 definitions are retrieved directly from the WHO public API.
* PubMedBERT is used for domain-specific RAG over scientific documents.
* LLaMA 3 is used locally via llama-cpp with GGUF quantization.

### **Disclaimer**
This project is intended for research and educational purposes only. It does not replace clinical evaluation, diagnosis, or professional mental health treatment. Always consult a qualified clinician.