import os                                     # For environment variables and system operations
import threading                              # To run graceful shutdown in background
import time                                   # To delay before shutdown
import signal                                 # To send termination signal
import gradio as gr                           # For creating the user interface
import logging                                # For logging events and debug/error messages
from llama_cpp import Llama                   # Local LLM backend
from rag import YAMLRAGLoader, SimpleRAG      # For RAG functionality
from api import add_icd_results_to_context    # Helper functions
from typing import Generator, Any, TypedDict  # For defining structured dictionaries with type hints
from config import (
    N_CTX, N_BATCH, N_THREADS, TOP_K, MAX_TOKENS, EMBEDDING_MODEL, TEMPERATURE,
    DOCS_PATH, INDEX_PATH, CHUNKS_PATH, LLAMA_MODEL_PATH
)
from utils import (
    load_prompt_template, log_to_csv, extract_scores, score_to_disorders,
    detect_question_type, detect_disorders
)

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM")

# Type Definitions 
class ChatMessage(TypedDict):
    role: str
    content: str

class ChatState(TypedDict):
    messages: list[ChatMessage]


# ---------------------------------------------------
# Build or load RAG index from questionnaire YAMLs
# ---------------------------------------------------
rag_loader = YAMLRAGLoader(DOCS_PATH)    # Initialize loader with docs folder
rag_loader.load_documents()              # Parse and load all YAML questionnaire files
chunks = rag_loader.get_text_chunks()    # Convert to text chunks
rag_engine = SimpleRAG(EMBEDDING_MODEL)  # Initialize RAG engine with embedding model

# Check for cached index files
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    logger.info("Loading RAG index from cache...")  # Log cache loading
    rag_engine.load_index()  # Load cached embeddings and chunks
else:
    logger.info("No cached index found — building a new...")  # Log new build
    rag_engine.build_index(chunks, rag_loader.entries)  # Build embeddings index
    rag_engine.save_index()  # Persist index for future use

# ---------------------------------------------------
# Initialize and warm up the LLM
# ---------------------------------------------------
llm = Llama(
    model_path=LLAMA_MODEL_PATH,  # Path to local llama model .gguf
    n_ctx=N_CTX,                  # Context window size
    n_batch=N_BATCH,              # Batch size for token generation
    n_threads=N_THREADS,          # CPU threads to use
    use_mmap=True,                # Use memory-map file loading
    use_mlock=True,               # Lock model in RAM (avoid swap)
    preload=True,                 # Preload entire model into memory
    verbose=True,                 # Enable detailed inference logging
    n_gpu_layers=0                # Do not allocate GPU layers
)

# Perform a brief warmup run to load model
if not os.environ.get("SKIP_WARMUP"):
    llm("Example short response.", max_tokens=5)

# ---------------------------------------------------
# Streamed chat function using RAG + LLM
# ---------------------------------------------------
def chat_with_llama(user_input: str)  -> Generator[str, None, None]:
    """
    Processes the user's input by:
      1. Detecting intent (diagnosis vs. general query).
      2. Retrieving relevant context using RAG or ICD-11.
      3. Building the prompt and streaming the LLM's response token by token.

    Args:
        user_input (str): The text query provided by the user.

    Yields:
        str: Progressive output chunks from the language model until completion.
    """
    question_type = detect_question_type(user_input)  # Detect intent
    logger.info(f"User intent detected: {question_type.upper()}")  # Log intent

    context_chunks, rag_results = [], []  # Lists for aggregated context
    context = ""  # Full assembled context text

    if question_type == "diagnosis" or question_type == "clinical":
        questionnaires, scores = extract_scores(user_input)  # Parse provided questionnaire found and scores
        disorders = score_to_disorders(scores)  # Map scores to possible disorders
        logger.info(f"Used questionnaires: {questionnaires}")
        logger.info(f"Suspected disorders: {disorders}")

        # Add questionnaire-specific YAML context
        for q in questionnaires:
            match = next(
                ((i, name) for i, name in rag_engine.questionnaires
                 if q.lower() in name.lower()),
                None
            )

            if match:
                idx, _ = match
                src = os.path.basename(rag_engine.corpus_chunks[idx]["source"])
                context_chunks.append(rag_engine.corpus_chunks[idx]["text"])
                
                rag_results.append({
                    "score": 1.0,
                    "text": rag_engine.corpus_chunks[idx]["text"],
                    "source": src
                })

            else:
                logger.info(f"No chunk matched questionnaire '{q}'")

        # Add ICD context based on suspected disorders
        icd_chunks, icd_rags = add_icd_results_to_context(disorders)
        context_chunks += icd_chunks
        rag_results += icd_rags

        context = "\n\n".join(context_chunks)

    else:
        # Fallback for other question types (e.g., definitions/diagnoses/interpretations)
        terms = detect_disorders(user_input)
        logger.info(f"Extracted disorder terms: {terms}")
        
        icd_chunks, icd_rags = add_icd_results_to_context(terms)
        context_chunks = icd_chunks
        rag_results = icd_rags

        context = "\n\n".join(context_chunks)

        # If no ICD info found, fallback to YAML-based RAG
        if not context_chunks:
            logger.info("No ICD context found — using fallback RAG YAML search")
            rag_results = rag_engine.query(user_input, top_k=TOP_K)
            combined = "\n\n".join(r["text"] for r in rag_results)
            tokens = llm.tokenize(combined.encode("utf-8"))
            max_ct = N_CTX - MAX_TOKENS - 128

            if len(tokens) > max_ct:
                tokens = tokens[:max_ct]
                context = llm.detokenize(tokens).decode("utf-8") + "\n[...]"
            else:
                context = combined

    # Load prompt template 
    prompt_tmpl = load_prompt_template(question_type)
    prompt = prompt_tmpl.format(context=context, user_input=user_input)
    
    # Calculate available tokens
    tok_prompt = llm.tokenize(prompt.encode("utf-8"))
    available = max(16, min(MAX_TOKENS, N_CTX - len(tok_prompt) - 16))

    # Request answer with streaming from Llama
    stream = llm(
        prompt,
        max_tokens=available,
        stop=["</s>", "Domanda:", "\n\n"],  # Multiple stop patterns
        temperature=TEMPERATURE,
        stream=True
    )

    output = ""  # Accumulate streaming output
    for chunk in stream:
        tok = chunk.get("choices", [{}])[0].get("text", "")
        output += tok
        yield output  # Yield partial answer as it's generated

    log_to_csv(user_input, rag_results, output)  # Log interaction

# ---------------------------------------------------
# Gradio interface setup
# ---------------------------------------------------
css = """
#inputbox:disabled {
    border: 3px solid #3498db;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { border-color: #3498db; }
    50% { border-color: #85c1e9; }
    100% { border-color: #3498db; }
}
#title {
    text-align: center;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    # Chat header
    gr.Markdown("## PsyBot", elem_id="title")  

    # Chat widget
    chatbot = gr.Chatbot( 
        elem_id="chatbox", 
        height=1100, 
        show_label=False, 
        show_copy_button=True, 
        type='messages'
    )  

    # User input
    input_box = gr.Textbox(
        placeholder="Ask your question...", 
        show_label=False, 
        lines=1, 
        elem_id="inputbox"
    )  

    with gr.Row():
        clear_btn = gr.Button("Reset")  # Resets conversation
        exit_btn = gr.Button("Quit", variant="stop")  # Triggers shutdown
    
    state = gr.State({"messages": []})  # Persistent chat state

    def process(user_input: str, history: ChatState) -> Generator[tuple[Any, list[ChatMessage], ChatState], None, None]:
        """
        Manages full chat lifecycle within the UI:
        - Disables input box and updates conversation history.
        - Streams LLM-generated response through `chat_with_llama`.
        - Re-enables input after completion.

        Args:
            user_input (str): Message text from the user.
            history (dict): Existing chat state, with messages list.

        Yields:
            Tuple: Streaming updates for input, chat display, and state.
        """
        if not isinstance(history, dict) or "messages" not in history:
            history = {"messages": []}

        yield gr.update(value="", interactive=False), history["messages"], history  # Disable input
        history["messages"].append({"role": "user", "content": user_input})  # Add user message
        yield gr.update(), history["messages"], history  # Update UI message

        response = ""
        started = False

        for part in chat_with_llama(user_input):
            response = part

            if not started:
                history["messages"].append({"role": "assistant", "content": ""})
                started = True
            
            history["messages"][-1]["content"] = response
            yield gr.update(), history["messages"], history  # Stream partial answer

        yield gr.update(interactive=True), history["messages"], history  # Re-enable input

    def clear_chat():
        """
        Clears the entire chat session.

        Resets the input box, empties the chat display, and resets the state.

        Returns:
            Tuple: Updated UI elements and initial state.
        """
        return gr.update(value="", interactive=True), [], {"messages": []}

    def graceful_exit():
        """
        Initiates a graceful shutdown by spawning a background thread
        that waits briefly then sends SIGINT to terminate the process.

        This avoids blocking the main UI thread during exit.
        """
        def _exit():
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGINT)
        threading.Thread(target=_exit).start()

    # Bind UI actions
    input_box.submit(fn=process, inputs=[input_box, state], outputs=[input_box, chatbot, state], postprocess=False)
    clear_btn.click(fn=clear_chat, outputs=[input_box, chatbot, state])
    exit_btn.click(fn=graceful_exit, outputs=[chatbot])

# Launch the interface in browser
demo.launch(server_port=5000, inbrowser=True)