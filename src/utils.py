import os                        # For environment variables and system operations
import re                        # For regular expressions and pattern matching
import yaml                      # For parsing YAML files
import spacy                     # For NLP entity recognition 
from typing import TypedDict     # For defining structured dictionaries with type hints
from collections import Counter  # For count occurrences 
from config import PROMPT_PATH, CSV_LOG_PATH, MAPPING_PATH, TOP_K
from keywords import ALIASES, DISEASE_KEYWORDS, QUESTIONNAIRES_KEYWORDS, KEYWORD_GROUPS

# Type Definitions 
class QuestionnaireScores(TypedDict, total=False):
    STAI_Y: int
    PSQI: int
    BIS: int
    BAS_Drive: int
    BAS_Fun_Seeking: int
    BAS_Reward_Resp: int
    MEQ: int
    PANAS_PA: int
    PANAS_NA: int

def detect_disorders(user_input: str) -> list:
    """
    Analyze user input text to extract potential disorder-related expressions.

    Combines named entity recognition, noun phrase matching, and lexical heuristics.

    Args:
        user_input (str): The user's question or statement.

    Returns:
        list: Unique disorder-related terms found in the input.
    """
    nlp = spacy.load("en_core_web_sm")  # Load English spaCy model
    doc = nlp(user_input)  # Process text
    terms = set()

    # Named entities with disease/condition labels
    for ent in doc.ents:
        if ent.label_ in {"DISEASE", "CONDITION"}:
            terms.add(ent.text.strip())

    # Noun chunks containing disease keywords
    for chunk in doc.noun_chunks:
        lower = chunk.text.lower()
        if any(kw in lower for kw in DISEASE_KEYWORDS):
            terms.add(chunk.text.strip())

    # Bigrams formed from relevant parts of speech
    tokens = [t.text for t in doc if t.pos_ in {"NOUN", "ADJ", "PROPN"}]
    for a, b in zip(tokens, tokens[1:]):
        combined = f"{a} {b}"
        if any(kw in combined.lower() for kw in DISEASE_KEYWORDS):
            terms.add(combined.strip())

    # Fallback: title-cased nouns or proper nouns
    if not terms:
        for token in doc:
            if token.text.istitle() and token.pos_ in {"NOUN", "PROPN"}:
                if token.text.lower() not in {"the", "a", "an", "of", "and", "or"}:
                    terms.add(token.text.strip())

    # Filter to avoid substrings being repeated
    final_terms = []
    for t in sorted(terms, key=len, reverse=True):
        if not any(t in other for other in final_terms):
            final_terms.append(t)

    return final_terms

def log_to_csv(question: str, rag_sources: list, mistral_response: str) -> None:
    """
    Save the question, associated references, and model output to a CSV log file.

    Extracts document names or ICD-11 headers from the RAG results.

    Args:
        question (str): The user input.
        rag_sources (list): Documents retrieved and used for RAG.
        mistral_response (str): The model's textual output.
    """
    # Branch for interpretation requests
    if rag_sources == "RAG Disabled":
        rag_files = "RAG Disabled"
    # Branch for RAG requests
    else:
        source_set = set()  # Build a set to collect unique, cleaned source references 
        for r in rag_sources:
            raw_text = r.get("text", "")
            # If it's an ICD chunk, clean the first line
            if "\n" in raw_text:
                line = raw_text.split("\n")[0].strip()
                line = re.sub(r"^###\s*", "", line)
                line = re.sub(r"\s*\(ICD-11 Code:\s*", " (", line)
                source = line
            # Otherwise, treat it as a YAML filename or flat source string
            else:
                source = os.path.basename(r.get("source", ""))
            source_set.add(source)

        # Sort and join the unique sources
        normalized_sources = sorted(source_set)
        rag_files = ", ".join(normalized_sources)

    # Clean input and output text for CSV formatting
    q_text = question.replace('\n', ' ').replace('"', "'")
    m_text = mistral_response.replace('\n', ' ').replace('"', "'")

    # Write formatted row to the log file
    with open(CSV_LOG_PATH, "a", encoding="utf-8") as csvfile:
        csvfile.write(f'"{q_text}","{m_text}","{rag_files}"\n\n')

def load_prompt_template(name: str) -> str:
    """
    Retrieve a text prompt template by filename prefix.

    Searches predefined extensions and loads the first match.

    Args:
        name (str): The base name of the prompt file (without extension).

    Returns:
        str: The contents of the matched file as a string.
    """
    valid_extensions = [".prompt", ".txt", ".md"]
    found = None

    for ext in valid_extensions:
        path = os.path.join(PROMPT_PATH, name + ext)
        if os.path.exists(path):
            found = path
            break

    # Error during loading
    if not found:
        raise FileNotFoundError(f"Prompt template '{name}' not found in {PROMPT_PATH}")

    # Prompt found
    with open(found, "r", encoding="utf-8") as f:
        return f.read()

def detect_question_type(user_input: str) -> str:
    """
    Classify the type of user inquiry based on its contents.

    Uses keyword detection to return a tag for routing.

    Args:
        user_input (str): The user-provided natural language query.

    Returns:
        str: One of "definition", "comparison", "interpretation", "diagnosis" or "clinical".
    """
    lowered = user_input.lower()

    # Detects requests based on keywords
    for label, keywords in KEYWORD_GROUPS.items():
        if any(kw in lowered for kw in keywords):
            return label
        
    return "definition"  # Base description 

def extract_scores(user_input: str) -> tuple[list[str], QuestionnaireScores]:
    """
    Identify questionnaire results in the user text.

    Supports aliases and flexible formatting with colons, equals, or spaces.

    Args:
        user_input (str): Natural language text with embedded scores.

    Returns:
        tuple[list,dict]: A list of unique normalized test names, and a dict of raw scores.
    """
    results = {}
    questionnaires = set()

    for test in QUESTIONNAIRES_KEYWORDS:
        escaped_test = re.escape(test).replace(r"\ ", r"\s+")  # Convert test name to regex-friendly pattern
        pattern = rf"{escaped_test}\s*[:=]?\s*(\d{{1,3}})"  # Allow ':' or '=' or whitespace before number
        match = re.search(pattern, user_input)
        if match:
            try:
                score = int(match.group(1))
                results[test] = score
                normalized = ALIASES.get(test, test)
                questionnaires.add(normalized)
            except ValueError:
                continue

    return list(questionnaires), results

def score_to_disorders(score_dict: dict, question_type: str) -> list[str]:
    """
    Evaluate questionnaire scores using a YAML-defined mapping schema.

    If question_type is "clinical", return the top disorders based on frequency,
    including ties at the cutoff. Otherwise, return all in descending frequency.

    Args:
        score_dict (dict): Questionnaire-to-score dictionary.
        question_type (str): Type of the user question (e.g., "clinical").

    Returns:
        list[str]: Sorted disorder labels by frequency.
    """
    with open(MAPPING_PATH, "r") as f:
        mapping = yaml.safe_load(f)

    counter = Counter()

    for test, score in score_dict.items():
        config = mapping.get(test)
        if not config:
            continue

        direction = config.get("direction")
        threshold = config.get("threshold")
        thresholds = config.get("thresholds")

        triggered = False
        if direction == "gte" and score >= threshold:
            triggered = True
        elif direction == "gt" and score > threshold:
            triggered = True
        elif direction == "lt" and score < threshold:
            triggered = True
        elif direction == "outside" and thresholds:
            low, high = thresholds
            if score <= low or score >= high:
                triggered = True

        if triggered:
            counter.update(config.get("disorders", []))

    # Sort by frequency DESC, then alphabetically
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if question_type == "clinical":
        if not sorted_items:
            return []

        # Get count of the third most frequent disorder (or fewer if not enough)
        top3 = sorted_items[:3]
        min_count = top3[-1][1]

        # Include all with count >= min_count
        result = [d for d, count in sorted_items if count >= min_count]
        return result

    else:
        top10 = sorted_items[:TOP_K]
        return [d for d, _ in top10]
