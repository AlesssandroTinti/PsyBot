import os                                               # For environment variables and system operations
import yaml                                             # For parsing YAML files
import glob                                             # For recursive file pattern matching
import json                                             # For saving and loading metadata
import torch                                            # For checking CUDA availability
import logging                                          # For logging events and debug/error messages
import numpy as np                                      # For numerical operations on embeddings
from typing import Any, TypedDict                       # For defining structured dictionaries with type hints
from sentence_transformers import SentenceTransformer   # Model for encoding text
from sklearn.metrics.pairwise import cosine_similarity  # To compute vector similarities
from config import INDEX_PATH, CHUNKS_PATH, TOP_K       # Predefined constants

# Configure logging format and level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG")

# Type Definitions 
class ChunkEntry(TypedDict):
    text: str
    source: str

class QueryResult(TypedDict):
    score: float
    text: str
    source: str

class QuestionnaireEntry(TypedDict, total=False):
    questionnaire: str
    description: str
    measures: dict
    score_range: dict
    reference_values: dict
    path: str

def format_value(value: Any, indent: int = 0) -> str:
    """
    Recursively formats nested YAML data into markdown-style strings suitable for LLM input.

    Args:
        key (str): Key being processed.
        value: Associated value (could be dict, list, or primitive).
        indent (int): Indentation level for formatting.

    Returns:
        str: Pretty-formatted text block.
    """
    spacer = "  " * indent  # Adds indentation based on depth
    if isinstance(value, dict):  # If the value is a dictionary
        lines = []
        for subkey, subval in value.items():
            if isinstance(subval, dict) and "minimum" in subval and "maximum" in subval:
                # Special formatting for min-max ranges
                lines.append(f"{spacer}- **{subkey}**: {subval['minimum']}-{subval['maximum']}")
            elif isinstance(subval, dict):
                # Recursive formatting for nested dictionaries
                lines.append(f"{spacer}- **{subkey}**:\n{format_value(subval, indent + 1)}")
            else:
                # Key-value format for primitive types
                lines.append(f"{spacer}- **{subkey}**: {subval}")
        return "\n".join(lines)
    if isinstance(value, list):
        # Format each item in the list
        return "\n".join(f"{spacer}- {item}" for item in value)
    # Base case: return string representation
    return spacer + str(value)

class YAMLRAGLoader:
    """
    Loads YAML documents and formats them into text chunks for embedding and retrieval.
    """

    def __init__(self, docs_path: str) -> None:
        """
        Initializes the loader with a path to YAML files.

        Args:
            docs_path (str): Directory containing YAML documents.
        """
        self.docs_path = docs_path     # Directory where YAMLs reside
        self.entries: list[dict] = []  # Container for parsed questionnaire data

    def load_documents(self) -> None:
        """
        Loads all valid YAML files and extracts fields for downstream processing.
        """
        logger.info("Scanning for YAML files...")  # Log progress
        yaml_files = glob.glob(os.path.join(self.docs_path, "**", "*.yaml"), recursive=True)  # Find all YAML files
        logger.info(f"Found {len(yaml_files)} YAML files.")  # Log result

        for path in yaml_files:
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)  # Parse YAML
                except yaml.YAMLError as e:
                    logger.error(f"YAML error in {path}: {e}")  # Handle errors gracefully
                    continue

                if not data or "questionnaire" not in data:
                    continue  # Skip incomplete files

                # Store the parsed data as a dictionary
                doc = {
                    "questionnaire": data.get("questionnaire"),
                    "description": data.get("description"),
                    "measures": data.get("measures"),
                    "score_range": data.get("score_range"),
                    "reference_values": data.get("reference_values"),
                    "path": path  # Save file path for traceability
                }
                self.entries.append(doc)

    def get_text_chunks(self) -> list[ChunkEntry]:
        """
        Converts loaded YAML entries into structured markdown-formatted strings.

        Returns:
            List[Dict[str,str]]: List of formatted text blocks and metadata.
        """
        chunks = []
        for entry in self.entries:
            text = f"{entry.get('questionnaire', 'N/A')}"  # Start with questionnaire title

            # Iterate over known metadata sections
            for key in ['description', 'measures', 'score_range', 'reference_values']:
                if entry.get(key):
                    text += f"\n\n**{key.replace('_', ' ').title()}:**\n{format_value(entry[key])}"

            # Append the result as a chunk
            chunks.append({
                "text": text.strip(),  # Remove trailing whitespace
                "source": entry["path"]  # Attach the file path for traceability
            })

        return chunks

class SimpleRAG:
    """
    A minimal Retrieval-Augmented Generation engine using sentence embeddings and cosine similarity.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes the SentenceTransformer and preallocates space for corpus structures.

        Args:
            model_name (str): Name of the SentenceTransformer model.
        """
        logger.info("Loading embedding model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"      # Check GPU availability
        logger.info(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        self.model = SentenceTransformer(model_name, device=device)  # Load model on appropriate device
        self.corpus_embeddings = None                                # Embedding matrix
        self.corpus_chunks: list[ChunkEntry] = []                    # Text and metadata
        self.questionnaires: list = []                               # Index + questionnaire label pairs

    def build_index(self, chunks: list[ChunkEntry], entries: list[QuestionnaireEntry]) -> None:
        """
        Builds the document embedding index from preprocessed text chunks.

        Args:
            chunks (List): List of formatted text documents.
            entries (List): Original metadata entries.
        """
        logger.info("Encoding chunks...")  
        self.corpus_chunks = chunks  # Store documents
        texts = [f"passage: {chunk['text']}" for chunk in chunks]  # Add prefix expected by certain models

        # Generate dense vector embeddings
        self.corpus_embeddings = self.model.encode(
            texts,
            batch_size=8,
            convert_to_tensor=True
        ).cpu().numpy()

        # Map index to questionnaire label
        for i, entry in enumerate(entries):
            if "questionnaire" in entry:
                self.questionnaires.append((i, entry["questionnaire"]))

    def save_index(self) -> None:
        """
        Saves embeddings and metadata to disk.
        """
        np.savez(INDEX_PATH, embeddings=self.corpus_embeddings)  # Save vector matrix
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": self.corpus_chunks,
                "questionnaires": self.questionnaires
            }, f, ensure_ascii=False, indent=2)  # Save metadata

    def load_index(self) -> None:
        """
        Loads precomputed embeddings and metadata from disk.
        """
        data = np.load(INDEX_PATH)  # Load vector matrix
        self.corpus_embeddings = data["embeddings"]
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)  # Load chunk metadata
            self.corpus_chunks = saved.get("chunks", [])
            self.questionnaires = saved.get("questionnaires", [])

    def query(self, question: str) -> list[QueryResult]:
        """
        Retrieves the most relevant documents for a given question.

        Args:
            question (str): The user's input query.

        Returns:
            List[Dict[str,str]]: Ranked documents with semantic similarity scores.
        """
        logger.info("Encoding query...")  # Track progress

        # Encode query as dense vector
        query_embedding = self.model.encode(
            f"query: {question}",  # Use prefix to match training scheme
            batch_size=8,
            show_progress_bar=False,
            convert_to_tensor=True
        ).cpu().numpy()

        # Calculate cosine similarity to all documents
        similarities = cosine_similarity([query_embedding], self.corpus_embeddings)[0]

        # Rank documents by semantic similarity
        top_indices = np.argsort(similarities)[::-1][:TOP_K]

        # List of dicts containing 'score', 'text', and 'source'.
        return [
            {
                "score": float(similarities[i]),           # Convert numpy float to native type
                "text": self.corpus_chunks[i]["text"],     # Return chunk
                "source": self.corpus_chunks[i]["source"]  # Return metadata
            }
            for i in top_indices
        ]
    