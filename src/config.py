import os
from dotenv import load_dotenv

# Embedding & RAG settings
LLAMA_3 = "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"                                # Local large language model (quantized, 5-bit)
EMBEDDING_MODEL = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"  # Sentence embedding model for RAG retrieval
TOP_K = 1                                                                       # Number of top documents to retrieve
TEMPERATURE = 0.5                                                               # LLM creativity level ("empathy" for clinical tone)
SIMILARITY = 0.80                                                               # Minimum similarity threshold for RAG results

# LLM runtime settings
N_CTX = 8192       # Max context window (number of tokens)
N_BATCH = 512      # Batch size for token generation
N_THREADS = 8      # Number of CPU threads (e.g., i7 quad-core with hyperthreading)
MAX_TOKENS = 1024  # Max length of the generated output (prevents excessive verbosity)

# ICD-11 API (loaded from environment variables)
load_dotenv()  # Load from .env file
ICD_CLIENT_ID = os.getenv("ICD_CLIENT_ID")
ICD_CLIENT_SECRET = os.getenv("ICD_CLIENT_SECRET")
AUTH_URL = os.getenv("AUTH_URL")
SEARCH_URL = os.getenv("SEARCH_URL")

# Absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root directory of the application
DOCS_PATH = os.path.join(BASE_DIR, "docs")                              # Directory containing medical or RAG documents
SRC_PATH = os.path.join(BASE_DIR, "src")                                # Directory for Python source files
LLAMA_MODEL_PATH = os.path.join(BASE_DIR, "models", LLAMA_3)            # Path to the GGUF LLaMA model file
CSV_LOG_PATH = os.path.join(BASE_DIR, "log.csv")                        # Path to the interaction log (CSV)
PROMPT_PATH = os.path.join(BASE_DIR, "prompts")                         # Directory for prompt templates
RAG_INDEX_DIR = os.path.join(BASE_DIR, "rag_index")                     # Directory for vector index data
CHUNKS_PATH = os.path.join(RAG_INDEX_DIR, "chunks.json")                # Path to the serialized document chunks
INDEX_PATH = os.path.join(RAG_INDEX_DIR, "index.npz")                   # Path to the embedding index
MAPPING_PATH = os.path.join(DOCS_PATH, "mapping.yaml")                  # Path to YAML file for questionnaire–disorder mapping