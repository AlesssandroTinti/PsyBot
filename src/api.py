import re                     # For regular expressions and pattern matching
import logging                # For logging events and debug/error messages
import requests               # For making HTTP requests to the ICD-11 API
import urllib.parse           # For safely encoding query parameters in URLs
from typing import TypedDict  # For defining structured dictionaries with type hints
from config import SIMILARITY, ICD_CLIENT_ID, ICD_CLIENT_SECRET, AUTH_URL, SEARCH_URL

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Type Definitions 
class ICDEntry(TypedDict):
    title: str
    code: str
    definition: str
    score: float

class RAGResult(TypedDict):
    score: float
    text: str
    source: str

def normalize_icd_entity(ent: dict) -> ICDEntry:
    """
    Normalize and clean fields from an ICD-11 API search result.

    Strips HTML from title and definition, extracts MMS code and relevance score.

    Args:
        ent (dict): Raw ICD-11 entity data.

    Returns:
        dict: Dictionary with cleaned 'title', 'code', 'definition', and 'score'.
    """

    raw_title = ent.get("title", {})
    title = raw_title.get("value") if isinstance(raw_title, dict) else str(raw_title)
    code = ent.get("code") or ent.get("theCode") or ""
    definition = ent.get("definition", {}).get("value", "")
    score = float(ent.get("score", 0))

    # Inline HTML cleaner: strips tags and decodes HTML entities
    title = re.sub(r"<[^>]+>", "", title or "").strip()
    definition = re.sub(r"<[^>]+>", "", definition or "").strip()

    return {
        "title": title,
        "code": code.strip(),
        "definition": definition,
        "score": score
    }

def get_icd_token() -> str | None:
    """
    Requests a bearer token from the WHO ICD-11 API using client credentials.

    Returns:
        str/None: Access token if successful, otherwise None.
    """
    data = {
        'client_id': ICD_CLIENT_ID,
        'client_secret': ICD_CLIENT_SECRET,
        'scope': 'icdapi_access',
        'grant_type': 'client_credentials'
    }
    try:
        response = requests.post(AUTH_URL, data=data, timeout=10)
        response.raise_for_status()
        return response.json().get('access_token')
    except requests.RequestException as e:
        logger.error(f"Error retrieving ICD token: {e}")  # Error retrieving token
        return None

def icd_search(term: str, token: str) -> list[dict]:
    """
    Sends a search request to the ICD-11 API using a term and returns valid entities.

    Filters results by SIMILARITY threshold.

    Args:
        term (str): Search keyword or term.
        token (str): Valid access token for the API.

    Returns:
        list: List of valid ICD entity results above the similarity threshold.
    """
    search_url = f"{SEARCH_URL}{urllib.parse.quote_plus(term)}"
    headers = {
        'Authorization': f'Bearer {token}',
        'API-Version': 'v2',
        'Accept-Language': 'en'
    }

    logger.info(f"Final ICD search query: {search_url}")

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error during ICD API request: {e}")  # Error during request
        return []

    try:
        entities = response.json().get('destinationEntities', [])
        if not entities:
            logger.info(f"No entities returned for '{term}'")
            return []

        # Ordina per similarità discendente
        sorted_entities = sorted(entities, key=lambda x: float(x.get('score', 0)), reverse=True)
        
        # Prendi il primo con score >= SIMILARITY
        best_match = sorted_entities[0]
        if float(best_match.get('score', 0)) >= SIMILARITY:
            logger.info(f"Top match for '{term}': {best_match.get('score')}")
            return [best_match]
        else:
            logger.info(f"No match over threshold for '{term}' — top score was {best_match.get('score')}")
            return []

    except Exception as e:
        logger.error(f"Error parsing JSON response: {e}")
        return []

def add_icd_results_to_context(terms: list[str]) -> tuple[list[str], list[RAGResult]]:

    """
    For each search term, performs an ICD-11 API search and formats results for context injection.

    Args:
        terms (list of str): List of terms to search in ICD-11.

    Returns:
        tuple[list,list]: A List of formatted strings for prompt context and a list of dicts with score, formatted text, and source title.
    """
    icd_token = get_icd_token()
    context_chunks = []
    rag_results = []

    for raw_name in terms:
        name = raw_name.strip()
        if not name:
            continue
        
        logger.info(f"Searching ICD for: '{name}'")
        icd_results = icd_search(name, icd_token)

        if not icd_results:
            logger.info(f"No result ICD for: {name}")
            continue

        seen_codes = set()
        for ent in icd_results:
            norm = normalize_icd_entity(ent)  # Normalization 

            if norm["code"] in seen_codes:
                continue
            seen_codes.add(norm["code"])

            # Build structured chunk text
            chunk_text = (
                f"### {norm['title']} (ICD-11 Code: {norm['code']})\n\n"
                f"**Definition:** {norm['definition']}\n\n"
            )
            
            context_chunks.append(chunk_text)
            rag_results.append({
                "score": norm["score"],
                "text": chunk_text,
                "source": norm["title"]
            })

    return context_chunks, rag_results
