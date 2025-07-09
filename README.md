![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![LLM](https://img.shields.io/badge/Model-LLaMA3_8B_Q5_K_M-purple)

# **PsyBot – Local LLM Clinical Assistant**
PsyBot is a local AI assistant designed for interpreting psychometric questionnaires, retrieving ICD-11 diagnostic descriptions, and providing structured clinical insights — all without relying on external LLM inference APIs. The core language model runs entirely on-device.
This project was developed as part of a research internship inspired by the MMASH dataset, which explores physiological and psychological signals (including data from wearable devices) to support the analysis of mental disorders. PsyBot integrates psychometric tools similar to those used in MMASH and extends them into a structured clinical chatbot capable of:
- Providing definitions of psychological disorders and psychometric assessments
- Comparing diagnostic categories and clinical criteria
- Offering diagnostic suggestions based on user-submitted questionnaire scores
- Interpreting symptom descriptions or transcripts from psychological sessions

## **Main Features**
- Local LLM (`Meta-Llama-3-8B-Instruct.Q5_K_M.gguf` via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python))
- RAG engine using scientific PubMedBERT embeddings (```pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb```)
- [ICD-11 API](https://icd.who.int/icdapi) integration for structured diagnosis and hierarchy 
- [YAML-based questionnaires](docs/questionnaires/)
- [YAML-based mapping rules](docs/mapping.yaml) for psychometric scores to disorders
- Fully offline inference, no external LLMs used
- Gradio-based UI

## **Installation**
1. Clone the repository
```bash
git clone https://github.com/AlesssandroTinti/PsyBot.git
cd PsyBot
```

2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
```

3. Install [dependencies](requirements.txt)
```bash
pip install -r requirements.txt
```

4. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

## **Configuration**
Review the parameters in [src/config.py](src/config.py) to match your hardware setup, particularly:
```python
N_CTX = 8192       # Max context window (number of tokens)
N_BATCH = 512      # Batch size for token generation
N_THREADS = 8      # Number of CPU threads 
MAX_TOKENS = 1024  # Max length of the generated output (prevents excessive verbosity)
```

## **Running the App**
```bash
python src/llm.py 
```

## **Project Structure**
```markdown
PsyBot/
├── docs/             # YAML and Markdown documents
├── models/           # Quantized LLaMA GGUF models
├── prompts/          # Prompt templates
├── rag_index/        # RAG chunks and FAISS index
├── src/              # Source code
├── .env              # ICD API credentials
├── log.csv           # Interaction logging
└── requirements.txt  # Python dependencies
```

## **Example Input**
```yaml
DEFINITION:
what can you tell me about Bulimia Nervosa?

COMPARISON:
can you compare Social Anxiety Disorder, Generalized Anxiety Disorder and Separation Anxiety Disorder?

DIAGNOSIS:
a patient obtained these scores: MEQ: 50, PSQI: 13, PANAS PA: 22, PANAS NA: 38, BIS/BAS: BIS: 28, BAS Drive: 14, BAS  Fun Seeking: 12, BAS Reward Resp.: 13, STAI-Y: 72. Which kind of disorder could it be?

CLINICAL:
List the 3 suggested disorders for this patient: MEQ: 50, PSQI: 13, PANAS PA: 22, PANAS NA: 38, BIS/BAS: BIS: 28, BAS Drive: 14, BAS Fun Seeking: 12, BAS Reward Resp.: 13, STAI-Y: 72

INTERPRETATION:
A patient has a significantly low body weight, with extreme fear of gain weight. How can we interpret these symptoms?
```

## **Notes**
* Project inspired by the [MMASH dataset](https://physionet.org/content/mmash/1.0.0/) for mental and physiological signal analysis.
* [ICD-11](https://icd.who.int/browse/2025-01/mms/en#1516623224) definitions are retrieved directly from the WHO public API 
* [PubMedBERT](https://huggingface.co/pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb) is used for domain-specific RAG over scientific documents 
* [LLaMa 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) is used locally via llama-cpp with GGUF quantization 

## **Disclaimer**
This project is intended for research and educational purposes only. It does not replace clinical evaluation, diagnosis, or professional mental health treatment. Always consult a qualified clinician.

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.