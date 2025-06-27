import os
import re
import yaml
import spacy
from config import PROMPT_PATH, CSV_LOG_PATH, MAPPING_PATH
from dict import (
    COMPARISON_KEYWORDS, QUESTIONNAIRE_KEYWORDS, DISEASE_KEYWORDS, QUESTIONNAIRES, QUESTIONNAIRE_ALIASES
)

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

def log_to_csv(question, rag_sources, mistral_response):
    """
    Save the question, associated references, and model output to a CSV log file.
    Extracts document names or ICD-11 headers from the RAG results.

    Args:
        question (str): The user input.
        rag_sources (list): Documents retrieved and used for RAG.
        mistral_response (str): The model's textual output.
    """
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
        str: One of "definition", "comparison", or "questionnaire".
    """
    lowered = user_input.lower()

    # Detects comparison requests based on comparison-related keywords
    if any(kw in lowered for kw in COMPARISON_KEYWORDS):
        return "comparison"
    
    # Detects diagnostic requests based on questionnaire-related keywords
    if any(kw in lowered for kw in QUESTIONNAIRE_KEYWORDS):
        return "questionnaire"
    
    # Default: Definition
    return "definition"

def extract_scores(user_input):
    """
    Identify questionnaire results in the user text.
    Supports aliases and flexible formatting with colons, equals, or spaces.

    Args:
        user_input (str): Natural language text with embedded scores.

    Returns:
        tuple: A list of unique normalized test names, and a dict of raw scores.
    """
    results = {}
    questionnaires = set()

    for test in QUESTIONNAIRES:
        escaped_test = re.escape(test).replace(r"\\ ", r"\\s+")  # Convert test name to regex-friendly patter
        pattern = rf"{escaped_test}\\s*[:=]?\\s*(\d{{1,3}})"  # Allow ':' or '=' or whitespace before number
        match = re.search(pattern, user_input)
        if match:
            try:
                score = int(match.group(1))
                results[test] = score
                normalized = QUESTIONNAIRE_ALIASES.get(test, test)
                questionnaires.add(normalized)
            except ValueError:
                continue

    return list(questionnaires), results

def score_to_disorders(score_dict, filepath=MAPPING_PATH):
    """
    Evaluate questionnaire scores using a YAML-defined mapping schema.
    For each item, checks thresholds and returns associated ICD-11 disorders.

    Args:
        score_dict (dict): Questionnaire-to-score dictionary.
        filepath (str): Path to the YAML mapping file.

    Returns:
        list: All matching disorder labels without duplicates.
    """
    # Load YAML mapping at the start
    with open(filepath, "r") as f:
        mapping = yaml.safe_load(f)

    all_disorders = []
    for test, val in score_dict.items():
        config = mapping.get(test)
        if not config:
            continue

        direction = config.get("direction")
        threshold = config.get("threshold")
        thresholds = config.get("thresholds")

        # Evaluate score according to direction logic
        if direction == "gte" and val >= threshold:
            all_disorders.extend(config["disorders"])
        elif direction == "gt" and val > threshold:
            all_disorders.extend(config["disorders"])
        elif direction == "lt" and val < threshold:
            all_disorders.extend(config["disorders"])
        elif direction == "outside" and thresholds:
            low, high = thresholds
            if val <= low or val >= high:
                all_disorders.extend(config["disorders"])

    seen = set()
    unique_disorders = []
    for d in all_disorders:
        if d not in seen:
            seen.add(d)
            unique_disorders.append(d)
        
    return unique_disorders