import platform
import argparse
import time
import json
import re
import os
import shutil
import logging
import random # Added for exponential backoff jitter
import autogen

from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# RAG related imports
import chromadb # Vector Database
from sentence_transformers import CrossEncoder # Reranker
from sentence_transformers import SentenceTransformer # Embedding model

from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY
from openai import OpenAI
from utils import get_web_element_rect, encode_image, extract_information, print_message,\
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only

from typing import Dict, Any, Tuple, List, Optional, Union

# RAG components initialisation
manual_vector_db_collection = None
reranker_model = None
embedding_model_st = None # For SentenceTransformer

def initialize_rag_components():
    """Initializes RAG components like vector DB client and reranker model."""
    global manual_vector_db_collection, reranker_model, embedding_model_st
    logging.info("RAG: Initializing RAG components...")
    try:
        # Initialize ChromaDB client and get collection
        # Ensure the path './chroma_db/' exists or ChromaDB can create it.
        # You need to populate this DB separately with your manual's embeddings.
        client = chromadb.PersistentClient(path="./chroma_db") # Or chromadb.Client() for in-memory
        manual_vector_db_collection = client.get_or_create_collection(name="booking_manual")
        logging.info(f"RAG: ChromaDB collection 'booking_manual' loaded/created. Item count: {manual_vector_db_collection.count()}")

        # Initialize reranker model
        # This model will be downloaded on first use if not cached.
        reranker_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        reranker_model = CrossEncoder(reranker_model_name)
        logging.info(f"RAG: Reranker model '{reranker_model_name}' initialized.")

        # Initialize SentenceTransformer embedding model
        embedding_model_name_st = 'all-MiniLM-L6-v2'
        embedding_model_st = SentenceTransformer(embedding_model_name_st)
        logging.info(f"RAG: SentenceTransformer embedding model '{embedding_model_name_st}' initialized.")

    except Exception as e:
        logging.error(f"RAG: Error initializing RAG components: {e}")
        logging.warning("RAG: Proceeding without RAG capabilities. Manual retrieval will be skipped or use dummy data.")


def retrieve_and_rerank_manual_snippets(task_query: str, current_observation_text: str, task_id_for_manual_lookup: str = None) -> str:
    """ 
    Retrieves and reranks manual snippets based on the task query and current observation.
    Args:
        task_query: The overall task query or goal.
        current_observation_text: Textual description of the current webpage observation.
        task_id_for_manual_lookup: The ID of the current task, can be used to determine if a manual is relevant.
    Returns:
        A formatted string of relevant manual snippets, or an empty string.
    """
    global manual_vector_db_collection, reranker_model, embedding_model_st
    
    if not manual_vector_db_collection or not reranker_model or not embedding_model_st:
        logging.warning("RAG: RAG components not fully initialized. Skipping actual retrieval.")
        return ""

    # General trigger: if DB has content and query hints at needing a manual or task ID suggests RAG.
    # Simple check for now: if DB has items and task ID contains RAG or query hints at manual.
    trigger_rag = False
    if manual_vector_db_collection.count() > 0:
        if (task_id_for_manual_lookup and "rag" in task_id_for_manual_lookup.lower()) or \
           any(keyword in task_query.lower() for keyword in ["manual", "handbook", "refer to", "guide"]):
            trigger_rag = True
    
    if not trigger_rag:
        logging.info("RAG: Retrieval not triggered for this task/query based on current conditions.")
        return ""

    logging.info(f"RAG: Attempting to retrieve manual snippets for query: {task_query[:100]}...")
        
    query_text_for_retrieval = f"{task_query}\n{current_observation_text}" 
    retrieved_docs_with_metadata = []
    
    try:
        # 1. Generate query embedding
        query_embedding = embedding_model_st.encode(query_text_for_retrieval).tolist()
        
        # 2. Initial retrieval from Vector DB (ChromaDB) using query_embeddings
        # We also want to retrieve metadata.
        n_results_retrieval = 10
        results = manual_vector_db_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results_retrieval,
            include=["documents", "metadatas"] # Request documents and their metadata
        )
        
        retrieved_docs_texts = results.get('documents', [[]])[0]
        retrieved_metadatas = results.get('metadatas', [[]])[0]

        if not retrieved_docs_texts:
            logging.info("RAG: No documents retrieved from ChromaDB for the query.")
            return ""
        
        # Combine documents with their metadata for reranking and final output
        for i in range(len(retrieved_docs_texts)):
            doc_text = retrieved_docs_texts[i]
            metadata = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
            retrieved_docs_with_metadata.append({
                "text": doc_text,
                "source": metadata.get("source", "Unknown Source"),
                "header": metadata.get("header", "Unknown Header")
            })
        logging.info(f"RAG: ChromaDB retrieved {len(retrieved_docs_with_metadata)} docs with metadata.")

    except Exception as e:
        logging.error(f"RAG: ChromaDB query or embedding generation failed: {e}")
        return ""

    # 3. Reranking using CrossEncoder
    try:
        # Prepare pairs for reranker: [query, document_text]
        sentence_pairs = [[query_text_for_retrieval, doc["text"]] for doc in retrieved_docs_with_metadata]
        if not sentence_pairs:
            logging.info("RAG: No sentence pairs to rerank.")
            return ""
            
        scores = reranker_model.predict(sentence_pairs)
        
        # Combine scores with the original document dictionary (including metadata)
        scored_docs_with_metadata = sorted(zip(scores, retrieved_docs_with_metadata), key=lambda x: x[0], reverse=True)
        
        # Select top N reranked snippets (e.g., top 3)
        top_n_reranked = 3 
        reranked_top_docs_with_metadata = [doc_meta for score, doc_meta in scored_docs_with_metadata[:top_n_reranked]]
        logging.info(f"RAG: Reranked {len(retrieved_docs_with_metadata)} docs. Selected top {len(reranked_top_docs_with_metadata)}.")
    
    except Exception as e:
        logging.error(f"RAG: Reranking failed: {e}")
        # Fallback to top N from initial retrieval if reranking fails, maintaining metadata structure
        top_n_fallback = 1
        reranked_top_docs_with_metadata = retrieved_docs_with_metadata[:top_n_fallback]
        logging.warning(f"RAG: Using top {len(reranked_top_docs_with_metadata)} from initial retrieval (with metadata) due to reranking error.")

    if reranked_top_docs_with_metadata:
        formatted_snippets = "\n\n[Relevant Manual Snippets]:\n"
        for doc_meta in reranked_top_docs_with_metadata:
            # Format with text, source, and header
            snippet_text = doc_meta['text']
            source_info = doc_meta.get('source', 'N/A')
            header_info = doc_meta.get('header', 'N/A')
            formatted_snippets += f"- Snippet: \"{snippet_text}\" (Source: {source_info}, Header: {header_info})\n"
        
        logging.info(f"RAG: Providing {len(reranked_top_docs_with_metadata)} snippet(s) after reranking.")
        return formatted_snippets.strip() # Remove trailing newline
    else:
        logging.info("RAG: No snippets to provide after retrieval/reranking.")
        return ""

def setup_logger(folder_path):
    """Setup logger configuration."""
    # Get a full path for the output directory
    log_file = os.path.join(folder_path, "agent.log")

    # Reset any existing handlers to avoid duplicated logs
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Create formatter
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.ERROR)  # Set console level to ERROR
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    logging.info(f"Logging to {log_file}")


def clean_text(text):
    """
    Clean text of problematic characters that might cause encoding issues.
    Specifically handles zero-width spaces and other invisible formatting characters.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return text
        
    # List of problematic Unicode characters to remove
    problematic_chars = [
        '\u200b',  # zero-width space
        '\u200c',  # zero-width non-joiner
        '\u200d',  # zero-width joiner
        '\u200e',  # left-to-right mark
        '\u200f',  # right-to-left mark
        '\u2028',  # line separator
        '\u2029',  # paragraph separator
        '\ufeff',  # zero-width no-break space
    ]
    
    # Replace all problematic characters
    for char in problematic_chars:
        text = text.replace(char, '')
        
    return text


def clean_json_data(data):
    """
    Recursively clean text in JSON structures (dicts, lists, strings).
    
    Args:
        data: The data structure to clean (can be dict, list, str, or other)
        
    Returns:
        The cleaned data structure
    """
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, str):
        return clean_text(data)
    else:
        return data


def safe_log(level, message):
    """
    Safely log messages that might contain problematic Unicode characters.
    
    Args:
        level: The logging level (e.g., logging.INFO, logging.ERROR)
        message: The message to log
    """
    try:
        # Try to log the message normally
        logging.log(level, message)
    except UnicodeEncodeError:
        # If that fails, try to clean the message first
        try:
            clean_message = clean_text(message)
            logging.log(level, clean_message)
        except UnicodeEncodeError: # If clean_text still fails to encode for the handler
            # Force ASCII replacement as a final fallback
            try:
                ascii_message = str(message).encode('ascii', 'replace').decode('ascii')
                logging.log(level, f"[ENCODING ISSUE DETECTED] {ascii_message}")
            except Exception as final_e: # Catch errors during final fallback logging
                print(f"CRITICAL LOGGING FAILURE: Could not log message even as ASCII. Error: {final_e}") # Use print as last resort
        except Exception as e: # Catch other potential errors during cleaning/logging
            try:
                ascii_message = str(message).encode('ascii', 'replace').decode('ascii')
                logging.log(level, f"[OTHER LOGGING ERROR: {type(e).__name__}] {ascii_message}")
            except Exception as final_e2:
                print(f"CRITICAL LOGGING FAILURE: Could not log message after other error. Error: {final_e2}") # Use print as last resort


def driver_config(args):
    options = webdriver.ChromeOptions()

    if args.save_accessibility_tree:
        args.force_device_scale = True

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if args.headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": args.download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )
    options.add_argument("--disable-blink-features=AutomationControlled")
    # Attempt to reduce console noise from ChromeDriver
    options.add_argument('--log-level=3') # Only log fatal errors
    options.add_argument('--disable-logging') 
    options.add_argument('--silent')
    options.add_experimental_option('excludeSwitches', ['enable-logging']) # Another way to attempt to disable logging

    return options


def format_msg(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text):
    if it == 1:
        init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        init_msg_format = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': init_msg},
            ]
        }
        init_msg_format['content'].append({"type": "image_url",
                                           "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}})
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        return curr_msg


def format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree):
    if it == 1:
        init_msg_format = {
            'role': 'user',
            'content': init_msg + '\n' + ac_tree
        }
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': f"Observation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}"
            }
        return curr_msg


def call_gpt4v_api(args, openai_client, messages):
    retry_times = 0
    while True:
        try:
            if not args.text_only:
                logging.info('Calling gpt4v API...')
                openai_response = openai_client.chat.completions.create(
                    model=args.api_model, messages=messages, max_tokens=1000, seed=args.seed
                )
            else:
                logging.info('Calling gpt4 API...')
                openai_response = openai_client.chat.completions.create(
                    model=args.api_model, messages=messages, max_tokens=1000, seed=args.seed, timeout=30
                )

            prompt_tokens = openai_response.usage.prompt_tokens
            completion_tokens = openai_response.usage.completion_tokens

            logging.info(f'Prompt Tokens: {prompt_tokens}; Completion Tokens: {completion_tokens}')

            gpt_call_error = False
            return prompt_tokens, completion_tokens, gpt_call_error, openai_response

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')

            if type(e).__name__ == 'RateLimitError':
                # Exponential backoff with jitter
                # wait_time = (base_wait_time * (2 ** retry_times)) + random.uniform(0, 1)
                # wait_time = min(wait_time, max_wait_time)
                # A simpler exponential backoff: (2^retry_count) + random_jitter
                # The retry_times starts at 0, so first actual retry (retry_times becomes 1) will wait (2**0 to 2**1) seconds
                # Let's adjust so that initial retry_times = 0 leads to a small wait.
                # Effective retry_count for wait calculation: retry_times + 1
                
                # Calculate wait time: (2^retry_attempt_number * base_interval) + jitter
                # Let's use retry_times directly as it increments before the next loop iteration.
                # So, for retry_times = 0 (first attempt fails, leading to retry_times = 1 for next actual call)
                # we want a small base wait. Let base be 1s.
                # Wait for (2^retry_times) seconds + random jitter
                # if retry_times = 0 (first failure), wait = 1 + jitter.
                # if retry_times = 1 (second failure), wait = 2 + jitter.
                # if retry_times = 2 (third failure), wait = 4 + jitter.
                wait_seconds = (2 ** retry_times) + random.uniform(0, 1) 
                wait_seconds = min(wait_seconds, 60) # Cap wait time at 60s
                logging.info(f"RateLimitError. Waiting for {wait_seconds:.2f} seconds before retrying... (Attempt {retry_times + 1})")
                time.sleep(wait_seconds)

            elif type(e).__name__ == 'APIError':
                time.sleep(15)

            elif type(e).__name__ == 'InvalidRequestError':
                gpt_call_error = True
                return None, None, gpt_call_error, None

            else:
                gpt_call_error = True
                return None, None, gpt_call_error, None

        retry_times += 1
        if retry_times == 10:
            logging.info('Retrying too many times')
            return None, None, True, None


def exec_action_click(info, web_ele, driver_task):
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(3)


def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    # outer_html = web_ele.get_attribute("outerHTML")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    try:
        # Not always work to delete
        web_ele.clear()
        # Another way to delete
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)

    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(10)
    return warn_obs


def exec_action_scroll(info, web_eles, driver_task, args, obs_info):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {args.window_height*2//3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-args.window_height*2//3});")
    else:
        if not args.text_only:
            scroll_ele_number = int(scroll_ele_number)
            web_ele = web_eles[scroll_ele_number]
        else:
            element_box = obs_info[scroll_ele_number]['union_bound']
            element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
            web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(3)


def get_current_observation(driver: webdriver.Chrome, args: argparse.Namespace, task_dir: str, iter_num: int) -> Tuple[str, str, Dict[int, WebElement]]:
    """
    Captures the current state of the browser including screenshot and element information.

    Returns:
        Tuple[str, str, Dict[int, WebElement]]: observation_text, base64_image, web_elements_cache
    """
    logging.info(f"Capturing observation for iteration {iter_num}...")
    web_elements_cache = {} # Use a local cache for this observation step
    observation_text = ""
    base64_image = ""
    
    try:
        # Get rectangles and elements, adding numerical labels
        rects, web_eles, web_eles_text = get_web_element_rect(driver, fix_color=args.fix_box_color)
        
        # Cache elements by their numerical label for action functions
        logging.info(f"Caching {len(web_eles)} web elements")
        for idx, ele in enumerate(web_eles):
            web_elements_cache[idx] = ele # Maps the numerical label (index) to the WebElement
        
        # Debug log what's in the cache
        element_ids = list(web_elements_cache.keys())
        logging.info(f"Cached element IDs: {element_ids}")

        # Save screenshot with labels
        img_path = os.path.join(task_dir, f'screenshot{iter_num}_observation.png')
        driver.save_screenshot(img_path)
        
        # Remove the labels from the page after screenshot
        if rects:
            logging.info(f"Num of interactive elements labeled: {len(rects)}")
            for rect_ele in rects:
                try:
                    driver.execute_script("arguments[0].remove()", rect_ele)
                except Exception as e:
                    logging.warning(f"Could not remove label rect: {e}")
            rects = [] 

        # Encode the captured image
        base64_image = encode_image(img_path)

        # Format observation text
        observation_text = (
            "Current page screenshot with interactive elements labeled numerically.\n"
            f"Element information:\n{web_eles_text}"
        )
        
        # Clean the observation text to remove problematic characters
        observation_text = clean_text(observation_text)
        
        # Optional: Save accessibility tree if needed
        if args.save_accessibility_tree:
             # accessibility_tree_path = os.path.join(task_dir, f'accessibility_tree{iter_num}.json')
             # _, _ = get_webarena_accessibility_tree(driver, accessibility_tree_path)
             # observation_text += f"\nAccessibility tree saved to: {accessibility_tree_path}"
             pass # Keep it simple for now

    except Exception as e:
        logging.error(f"Error during observation capture: {e}")
        observation_text = f"Error capturing observation: {e}"
        # Optionally take a basic screenshot even if labeling fails
        try:
            img_path = os.path.join(task_dir, f'screenshot{iter_num}_error.png')
            driver.save_screenshot(img_path)
            base64_image = encode_image(img_path)
            observation_text += "\nAttached basic screenshot."
        except Exception as se:
            logging.error(f"Could not even take basic screenshot: {se}")

    return observation_text, base64_image, web_elements_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/tasks_test.jsonl')
    parser.add_argument('--max_iter', type=int, default=15)
    parser.add_argument("--api_model", default="gpt-4.1", type=str, help="api model name for Planner")
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action='store_true')
    # for web browser
    parser.add_argument("--headless", action='store_true', help='The window of selenium')
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=768)  # for headless mode, there is no address bar
    parser.add_argument("--fix_box_color", action='store_true')
    parser.add_argument("--vision_model", default="gpt-4.1", type=str, help="Vision model name for Error Grounder")
    args = parser.parse_args()

    # Create output and download directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.download_dir, exist_ok=True)

    # Setup logger
    current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    result_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(result_dir, exist_ok=True)
    setup_logger(result_dir)

    # --- Autogen Agent Definitions ---

    # Load config from OAI_CONFIG_LIST file
    try:
        config_list = autogen.config_list_from_json(
            env_or_file="OAI_CONFIG_LIST",
            filter_dict={
                "model": [args.api_model, args.vision_model] # Ensure models used are in the list
            }
        )
        if not config_list:
            raise ValueError("No valid models found in OAI_CONFIG_LIST for specified models.")
        logging.info(f"Loaded LLM config from OAI_CONFIG_LIST: {[c.get('model') for c in config_list]}")
    except FileNotFoundError:
        logging.error("ERROR: OAI_CONFIG_LIST file not found. Please create it in the WebVoyager directory.")
        exit()
    except ValueError as e:
        logging.error(f"ERROR in OAI_CONFIG_LIST: {e}")
        exit()
    except Exception as e:
        logging.error(f"Error loading OAI_CONFIG_LIST: {e}")
        exit()

    # LLM Configuration using the loaded list
    llm_config = {
        "config_list": config_list, # Use the loaded config list
        "temperature": args.temperature if hasattr(args, 'temperature') else 0.7, # Add temp if needed
        "timeout": 120, # Increase timeout slightly
    }
    # Vision LLM config will implicitly use the same config_list
    # If vision_model is different, ensure it's in OAI_CONFIG_LIST
    vision_llm_config = llm_config 

    # System Prompts
    PLANNER_SYSTEM_PROMPT = (
        "You are a helpful AI assistant serving as the Planner in a web navigation task. "
        "Your goal is to achieve the user's objective by interacting with a web browser via specific function calls. "
        "You will receive multimodal input: a textual description of the current browser state (including interactive element IDs), a screenshot, and potentially relevant [Manual Snippets].\n"
        "You will also receive feedback from an Error Grounding agent after each action.\n\n"
        "**Workflow:**\n"
        "1. Analyze the user's overall objective.\n"
        "2. Analyze the current Observation (text and screenshot), any Error Feedback provided, and critically review any provided [Manual Snippets]. Prioritize guidance from [Manual Snippets] if relevant.\n"
        "3. **Critically review the conversation history (trajectory) provided. Pay close attention to previous actions, observations, and reported errors to avoid repeating mistakes or getting stuck in loops.**\n"
        "4. Based on your analysis, identify the *single best function call* to make for the current turn. You can only choose ONE function from the available list: `click_element`, `type_element`, `scroll_page`, `go_back`, `google_search`, `answer`. Any multi-step plan you formulate in your reasoning is for your understanding only; your output action MUST be the single, next, immediately executable step based on the *current* observation.\n"
        "5. Provide concise reasoning for your chosen action based on your analysis. If your reasoning is guided by a manual snippet, you MUST include a line in your reasoning formatted as: `Instruction Cue: Manual Ref: [Specific reference to the manual snippet, e.g., BookingComAdvancedFeatures.md, Section 2.1 - 'How to apply X filter']`\n"
        "6. After your reasoning and any `Instruction Cue`, output ONLY the JSON for the *single function call* you decided on in step 4. This JSON object MUST be enclosed in ```json ... ``` tags. You MUST NOT output more than one JSON object or any other actions in this turn.\n\n"
        "**Important Guidelines:**\n"
        "- Remember, you operate on a turn-by-turn basis. You propose ONE action, it gets executed, and then you will receive a new observation and feedback. Decide your next single action based on that new information.\n"
        "- **CRITICAL: The webpage is dynamic. Element IDs and page layout WILL change after actions. ALWAYS make your decisions and choose element IDs based SOLELY on the MOST RECENT observation provided. Do NOT rely on element IDs or assumptions from previous turns; they are likely outdated.**\n"
        "- **To scroll specific parts of a page (e.g., a sidebar, a list within a modal, or a scrollable `div`): If the target scrollable area (like a filter sidebar) is not immediately fully visible or its specific scrollable element ID isn't clear, first try one or two general window scrolls using `scroll_page(direction='down')` to ensure the area loads. Then, in a subsequent turn, if you can identify a numerical element ID from the new observation that corresponds to the scrollable container itself, use `scroll_page(direction='down', element_id=RELEVANT_ID)` to scroll it. If you still cannot identify a specific ID for the scrollable area after attempting general scrolls, or if you intend to scroll the entire page viewport from the start, continue using `scroll_page(direction='down')` (omitting `element_id`).**\n"
        "- Focus on the screenshot for layout and context; use the text for element IDs and details. However, if [Manual Snippets] provide conflicting or more specific instructions for the current step, prioritize the manual's guidance.\n"
        "- Interact with elements using their numerical IDs (e.g., `click_element(element_id=15)`).\n"
        "- If the Error Grounder reports an error (`errors: Yes`), carefully analyze its explanation and suggestion. Also, re-check [Manual Snippets] for alternative approaches if the error relates to a manually-guided step.\n"
        "- If you seem stuck (e.g., repeated actions yield no progress or errors), consider using `go_back` to return to a previous state, `google_search` to find information, or re-consulting [Manual Snippets] for missed details.\n"
        "- When the task is fully completed, call the `answer` function with the final answer.\n"
        "- Your final response for a completed task MUST conclude with the `answer` function call JSON, for example: ```json {\\\"function\\\": \\\"answer\\\", \\\"args\\\": {\\\"final_answer\\\": \\\"Summary of found information...\\\"}} ```"
    )

    # Executor doesn't need a complex prompt if UserProxyAgent executes functions
    # EXECUTOR_SYSTEM_PROMPT = "You execute web browser actions by calling registered functions based on Planner's instructions."

    ERROR_GROUNDER_SYSTEM_PROMPT = (\
        "You are an error-grounding robot analyzing web navigation steps.\\n"
        "You will receive the Planner's intended action (function call) and the result (status, message, and observation including a screenshot). The observation has a base64 encoded image.\\n"
        "Use your vision capabilities to analyze the screenshot provided in the observation message.\\n"
        "Compare the intended action with the observation and execution status/message. Did the action execute successfully and lead to an expected outcome based on the Planner's likely intent? \\n"
        "Common errors include: clicking the wrong element (page state didn't change as expected), typing into a non-editable element, form validation errors appearing, navigating to an unexpected URL, or the action having no effect (page unchanged after non-Wait action), or the action itself failing (status: Error).\\n"
        "IMPORTANT: You MUST respond EXACTLY in the following JSON format with NO additional text or explanation outside the JSON object:\\n"
        "{\\n"
        "  \\\"errors\\\": \\\"Yes\\\" or \\\"No\\\",\\n"
        "  \\\"explanation\\\": \\\"If Yes, describe the error, probable cause, and suggest a specific correction for the Planner to consider (e.g., 'Try clicking element 25 instead', 'Verify the input format', 'Scroll down to find the button'). If No, state 'No errors detected.'\\\"\n"
        "}\\n"
        "Always ensure your response is parseable JSON. Do not include backticks, code blocks, or any other text outside the JSON object."
    )

    # Agent Instantiation
    planner = autogen.AssistantAgent(
        name="Planner",
        llm_config=llm_config, # Pass the config dict containing the list
        system_message=PLANNER_SYSTEM_PROMPT
    )

    error_grounder = autogen.AssistantAgent(
        name="Error_Grounder",
        llm_config=vision_llm_config, # Pass the config dict containing the list
        system_message=ERROR_GROUNDER_SYSTEM_PROMPT
    )

    # UserProxyAgent - will execute functions and manage reflection call
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=args.max_iter, # Set a limit
        # Termination condition: Check if the last message contains the final answer call result
        is_termination_msg=lambda msg: isinstance(msg, dict) and msg.get("role") == "function" and msg.get("name") == "answer",
        code_execution_config=False, # Functions executed via registration, not code blocks
        # Default llm_config for potential internal processing (though not strictly needed here)
        # llm_config=llm_config, 
    )

    # --- End Autogen Agent Definitions ---

    # Call to initialize RAG components
    initialize_rag_components()

    # Load test file
    tasks = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))

    # Initialize global variables for tracking state
    global web_elements_cache_for_task, current_iter, last_action_details
    web_elements_cache_for_task = {}
    current_iter = 0 
    # Store the planner's last requested action
    last_action_details = None

    all_results = []

    for i, task in enumerate(tasks):
        task_id = task["id"]
        start_url = task['web']
        task_dir = os.path.join(result_dir, 'task{}'.format(task_id))
        os.makedirs(task_dir, exist_ok=True)
        logging.info(f'########## Starting TASK {task_id} ##########')
        logging.info(f"Instruction: {task['ques']}")
        logging.info(f"URL: {start_url}")

        options = driver_config(args)
        driver_task = webdriver.Chrome(options=options)
        driver_task.set_window_size(args.window_width, args.window_height)
        driver_task.get(start_url)
        time.sleep(3) # Allow initial page load
        try:
            driver_task.find_element(By.TAG_NAME, 'body').click()
        except Exception as e:
            logging.warning(f"Could not click body on initial load: {e}")
        time.sleep(2)
        
        # Reset iteration counter for this task
        current_iter = 0 
        # Store the planner's last requested action
        last_action_details = None

        # --- Define Browser Interaction Functions --- 
        # These functions need access to driver_task, args, task_dir, current_iter, web_elements_cache
        # They should now primarily return the result dict, observation handled separately or within
        
        def click_element(element_id: int) -> Dict[str, Any]:
            """Clicks a web element identified by its numerical label from the last observation."""
            global current_iter, web_elements_cache_for_task
            logging.info(f"Action: Clicking element {element_id}")
            status = "Error"
            message = f"Element ID {element_id} not found in the last observation's cache."
            new_observation_text = ""
            new_base64_image = ""

            if element_id in web_elements_cache_for_task:
                web_ele = web_elements_cache_for_task[element_id]
                try:
                    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
                    web_ele.click()
                    time.sleep(3) # Wait for page load
                    status = "Success"
                    message = f"Clicked element {element_id}."
                    
                    # TODO: Integrate PDF checking logic properly if needed from original exec_action_click

                except Exception as e:
                    logging.error(f"Error clicking element {element_id}: {e}")
                    message = f"Error clicking element {element_id}: {e}"
            else:
                logging.warning(message)

            current_iter += 1
            new_observation_text, new_base64_image, updated_cache = get_current_observation(driver_task, args, task_dir, current_iter)
            web_elements_cache_for_task = updated_cache
            return {
                "action_status": status, 
                "action_message": message, 
                "observation_text": new_observation_text, 
                "base64_image": new_base64_image
            }
        
        def type_element(element_id: int, text: str) -> Dict[str, Any]:
            """Clears and types content into a web element identified by its numerical label, then presses Enter."""
            global current_iter, web_elements_cache_for_task
            logging.info(f"Action: Typing '{text}' into element {element_id}")
            
            # Debug the cache state
            cache_keys = list(web_elements_cache_for_task.keys())
            logging.info(f"Current cache contains element IDs: {cache_keys}")
            
            status = "Error"
            message = f"Element ID {element_id} not found in the last observation's cache."
            new_observation_text = ""
            new_base64_image = ""
            
            if element_id in web_elements_cache_for_task:
                web_ele = web_elements_cache_for_task[element_id]
                logging.info(f"Found element {element_id} in cache, element tag: {web_ele.tag_name}")
                try:
                    # Original type logic from exec_action_type (combined and adapted)
                    warn_obs = ""
                    ele_tag_name = web_ele.tag_name.lower()
                    ele_type_attr = web_ele.get_attribute("type")
                    # Check if it seems like a valid input element
                    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or \
                       (ele_tag_name == 'input' and ele_type_attr not in ['text', 'search', 'password', 'email', 'tel', None]): # Allow type=None for some inputs
                        warn_obs = f"Warning: Element <{ele_tag_name}> type='{ele_type_attr}' might not be a standard textbox."
                        logging.warning(warn_obs)
                        
                    # Clear element (best effort)
                    try:
                        web_ele.clear()
                        # Add Ctrl+A Backspace for robustness
                        if platform.system() == 'Darwin': web_ele.send_keys(Keys.COMMAND + "a")
                        else: web_ele.send_keys(Keys.CONTROL + "a")
                        web_ele.send_keys(Keys.BACKSPACE)
                    except Exception as clear_err:
                         logging.warning(f"Could not reliably clear element {element_id}: {clear_err}")
                         # Try clicking first if clear fails, might help focus
                         web_ele.click() 
                         time.sleep(0.5)

                    # Type content using ActionChains
                    actions = ActionChains(driver_task)
                    # Use click_and_hold -> release to ensure focus before typing, then send keys
                    actions.click_and_hold(web_ele).release().pause(0.5).send_keys(text).pause(1).send_keys(Keys.ENTER).perform()
                    
                    time.sleep(5) # Wait for potential page change after Enter
                    status = "Success"
                    message = f"Typed '{text}' into element {element_id}. {warn_obs}".strip()

                except Exception as e:
                    logging.error(f"Error typing into element {element_id}: {e}")
                    message = f"Error typing into element {element_id}: {e}"
            else:
                 logging.warning(message)
                 # Try converting to int if it's a string
                 if isinstance(element_id, str) and element_id.isdigit():
                     int_element_id = int(element_id)
                     logging.info(f"Trying to convert string element_id '{element_id}' to int: {int_element_id}")
                     if int_element_id in web_elements_cache_for_task:
                         logging.info(f"Found element {int_element_id} after string-to-int conversion")
                         return type_element(int_element_id, text)

            current_iter += 1
            new_observation_text, new_base64_image, updated_cache = get_current_observation(driver_task, args, task_dir, current_iter)
            web_elements_cache_for_task = updated_cache
            return {
                "action_status": status, 
                "action_message": message, 
                "observation_text": new_observation_text, 
                "base64_image": new_base64_image
            }

        def scroll_page(direction: str, element_id: Optional[int] = None) -> Dict[str, Any]:
            """Scrolls the page window or a specific element ('up' or 'down')."""
            global current_iter, web_elements_cache_for_task
            logging.info(f"Action: Scrolling {direction}" + (f" on element {element_id}" if element_id is not None else " on window"))
            status = "Success" # Assume success unless error occurs
            message = f"Scrolled {direction}"
            new_observation_text = ""
            new_base64_image = ""

            try:
                if element_id is None: # Scroll window
                    scroll_amount = args.window_height * 2 // 3
                    scroll_command = f"window.scrollBy(0, {scroll_amount if direction == 'down' else -scroll_amount});"
                    driver_task.execute_script(scroll_command)
                    message += " window."
                elif element_id in web_elements_cache_for_task: # Scroll specific element
                    web_ele = web_elements_cache_for_task[element_id]
                    scroll_amount_js = "arguments[0].clientHeight * 0.7" # Scroll by 70% of element's visible height
                    if direction == 'down':
                        driver_task.execute_script(f"arguments[0].scrollTop += {scroll_amount_js};", web_ele)
                    else: # 'up'
                        driver_task.execute_script(f"arguments[0].scrollTop -= {scroll_amount_js};", web_ele)
                    message += f" element {element_id} using JavaScript."
                else: 
                    status = "Error"
                    message = f"Scroll failed: Element ID {element_id} not found in cache."
                    logging.warning(message)
                
                time.sleep(2) # Wait for scroll effect

            except Exception as e: 
                logging.error(f"Error scrolling {direction}: {e}")
                status = "Error"
                message = f"Error scrolling {direction}: {e}"

            current_iter += 1
            new_observation_text, new_base64_image, updated_cache = get_current_observation(driver_task, args, task_dir, current_iter)
            web_elements_cache_for_task = updated_cache
            return {
                "action_status": status, 
                "action_message": message, 
                "observation_text": new_observation_text, 
                "base64_image": new_base64_image
            }

        def wait_seconds(duration: int = 5) -> Dict[str, Any]:
            """Waits for a specified duration."""
            global current_iter, web_elements_cache_for_task
            logging.info(f"Action: Waiting for {duration} seconds")
            time.sleep(duration)
            status = "Success"
            message = f"Waited for {duration} seconds."
            
            current_iter += 1 # Count wait as an iteration step
            new_observation_text, new_base64_image, updated_cache = get_current_observation(driver_task, args, task_dir, current_iter)
            web_elements_cache_for_task = updated_cache
            # Return the state *after* waiting
            return {
                "action_status": status,
                "action_message": message,
                "observation_text": new_observation_text, 
                "base64_image": new_base64_image
            }
            
        def go_back() -> Dict[str, Any]:
            """Navigates back to the previous page."""
            global current_iter, web_elements_cache_for_task
            logging.info("Action: Going back")
            status = "Success"
            message = "Navigated back."
            try:
                driver_task.back()
                time.sleep(3) # Wait for page load after going back
            except Exception as e:
                 logging.error(f"Error going back: {e}")
                 status = "Error"
                 message = f"Error going back: {e}"

            current_iter += 1
            new_observation_text, new_base64_image, updated_cache = get_current_observation(driver_task, args, task_dir, current_iter)
            web_elements_cache_for_task = updated_cache
            return {
                "action_status": status, 
                "action_message": message, 
                "observation_text": new_observation_text, 
                "base64_image": new_base64_image
            }

        def google_search(query: str) -> Dict[str, Any]:
            """Navigates to Google and searches for the given query."""
            global current_iter, web_elements_cache_for_task
            logging.info(f"Action: Searching Google for '{query}'")
            status = "Success"
            message = f"Navigated to Google and initiated search for '{query}'."
            try:
                driver_task.get('https://www.google.com/')
                time.sleep(2)
                # Find search box - common name is 'q'
                search_box = driver_task.find_element(By.NAME, 'q') 
                actions = ActionChains(driver_task)
                actions.click(search_box).pause(0.5).send_keys(query).pause(1).send_keys(Keys.ENTER).perform()
                time.sleep(5) # Wait for results page to load
            except Exception as e:
                 logging.error(f"Error performing Google search for '{query}': {e}")
                 status = "Error"
                 message = f"Error performing Google search for '{query}': {e}"

            current_iter += 1
            new_observation_text, new_base64_image, updated_cache = get_current_observation(driver_task, args, task_dir, current_iter)
            web_elements_cache_for_task = updated_cache
            return {
                "action_status": status, 
                "action_message": message, 
                "observation_text": new_observation_text, 
                "base64_image": new_base64_image
            }
           
        def answer(final_answer: str) -> Dict[str, Any]:
            """Provides the final answer and terminates the task."""
            global current_iter
            
            # Clean the final answer text to remove problematic characters
            cleaned_answer = clean_text(final_answer)
            
            logging.info(f"Action: Providing final answer: {cleaned_answer}")
            answer_file = os.path.join(task_dir, "final_answer.txt")
            try:
                with open(answer_file, "w", encoding="utf-8") as f:
                    f.write(cleaned_answer)
                logging.info(f"Final answer saved to {answer_file}")
            except Exception as e:
                logging.error(f"Error saving final answer: {e}")
                # Try again with ASCII encoding if UTF-8 fails
                try:
                    with open(answer_file, "w", encoding="utf-8") as f:
                        # Replace problematic characters with their ASCII equivalents or spaces
                        ascii_answer = cleaned_answer.encode('ascii', 'replace').decode('ascii')
                        f.write(ascii_answer)
                    logging.info(f"Final answer saved with ASCII encoding to {answer_file}")
                except Exception as e2:
                    logging.error(f"Failed to save answer even with ASCII encoding: {e2}")
            
            return {
                "action_status": "Success",
                "action_message": "Final answer provided. Terminating.",
                "final_answer": cleaned_answer
            }
           
        # --- End Define Browser Interaction Functions ---

        # --- Custom Reply Function for UserProxyAgent ---
        def execute_web_action_and_reflect(recipient, messages, sender, config):
            # --- DEBUG: Log function entry and sender --- 
            logging.info(f"DEBUG: execute_web_action_and_reflect entered. Sender: {sender.name if hasattr(sender, 'name') else sender}")
            # --------------------------------------------
            global last_action_details, current_iter # Allow modification
            # Get the last message from the list
            last_message = messages[-1] # Make sure this line is here
            # Check if the last message content is a string (required for regex)
            if not isinstance(last_message.get("content"), str):
                 # Not a string content - cannot parse function call
                 logging.warning(f"DEBUG: Message content from {sender.name} is not a string. Last msg: {last_message}")
                 return (False, None) # Indicate no reply is generated
            
            try:
                # Planner should respond with ONLY the JSON for the function call
                # Optional: Add reasoning before the JSON, parse only the JSON part.
                # Updated Regex to be more robust against surrounding text
                json_match = re.search(r'```json\s*({.*?})\s*```', last_message["content"], re.DOTALL)
                if not json_match:
                    # Fallback: Check if the entire message is just the JSON object (less likely with reasoning)
                    json_match = re.search(r'^\s*({.*?})\s*$', last_message["content"], re.DOTALL)
                if not json_match:
                    logging.warning(f"Planner response did not contain expected function call JSON format in ```json ... ``` block. Content: {last_message['content']}")
                    # Maybe ask planner to retry or use a specific format?
                    return (True, "Planner Error: Please respond ONLY with the function call JSON enclosed in ```json ... ```.")
                
                # Extract group 1, which contains only the JSON part
                action_json_str = json_match.group(1)
                action_details = json.loads(action_json_str)
                function_name = action_details.get("function")
                function_args = action_details.get("args", {})
                
                # Handle different ways the args might be formatted
                if function_name == "google_search":
                    if isinstance(function_args, list):
                        # Handle case where args is a list for google_search
                        function_args = {"query": function_args[0]} if function_args else {}
                    elif isinstance(function_args, str):
                        # Handle case where args is a string
                        function_args = {"query": function_args}
                    elif isinstance(function_args, dict) and "content" in function_args:
                        # Handle case where args has "content" instead of "query"
                        function_args = {"query": function_args["content"]}
                    elif not isinstance(function_args, dict) or "query" not in function_args:
                        # If args doesn't have query parameter, extract from function name
                        function_args = {"query": str(function_args) if function_args else ""}
                        logging.warning(f"Reformatted google_search args to: {function_args}")
                
                # Handle element_id conversion for functions that use it
                if function_name in ["click_element", "type_element"] and isinstance(function_args, dict) and "element_id" in function_args:
                    # Convert element_id to int if it's a string
                    element_id = function_args["element_id"]
                    if isinstance(element_id, str) and element_id.isdigit():
                        function_args["element_id"] = int(element_id)
                        logging.info(f"Converted element_id from string '{element_id}' to int {function_args['element_id']}")
                
                last_action_details = action_details # Store for grounding agent
                logging.info(f"DEBUG: UserProxy received function call request: {function_name}({function_args})")

                # Execute the function using the registered map
                func = config["function_map"].get(function_name)
                if func is None:
                    logging.error(f"Function {function_name} not registered!")
                    return (True, f"Error: Function '{function_name}' is not registered.")
                
                # Execute the browser action function
                action_result = func(**function_args)
                logging.info(f"DEBUG: Action Result Received: {action_result}")

                # Check for termination via 'answer' function
                if function_name == "answer":
                    logging.info("DEBUG: 'answer' function called, returning termination message.")
                    return (True, {"role": "function", "name": function_name, "content": json.dumps(action_result)})

                # --- Call Error Grounder for Reflection --- (Happens after action execution)
                error_feedback_json = {"errors": "N/A", "explanation": "Grounder call skipped/failed"}
                try: 
                    grounding_message_content = [
                        {
                            "type": "text",
                            "text": f"Action Attempted: {json.dumps(last_action_details)}\nAction Result Status: {action_result.get('action_status')}\nAction Result Message: {action_result.get('action_message')}\n\nPlease analyze the observation (screenshot included) based on the attempted action and result. Provide feedback in the specified JSON format."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{action_result.get('base64_image', '')}"}
                        }
                    ]
                    
                    logging.info("DEBUG: Sending action result to Error_Grounder for reflection...")
                    error_feedback_chat = user_proxy.initiate_chat(
                        error_grounder,
                        message={"role": "user", "content": grounding_message_content},
                        max_turns=1,
                        clear_history=True,
                        silent=True,
                    )
                    # Check if chat was successful and history exists
                    if error_feedback_chat.chat_history:
                        error_feedback_msg = error_feedback_chat.chat_history[-1]
                        error_feedback_content = error_feedback_msg.get("content", "{}")
                        logging.info(f"DEBUG: Error_Grounder Raw Feedback: {error_feedback_content}")
                        
                        # Try to find and extract any JSON in the response
                        json_match = re.search(r'({.*})', error_feedback_content, re.DOTALL)
                        if json_match:
                            error_feedback_content = json_match.group(1)
                            
                        # Try parsing the JSON
                        try:
                            error_feedback_json = json.loads(error_feedback_content)
                            logging.info(f"DEBUG: Error_Grounder Parsed JSON: {error_feedback_json}")
                        except json.JSONDecodeError as json_err:
                            logging.error(f"Error Grounder response not valid JSON: {json_err}")
                            # Fall back to a default response
                            error_feedback_json = {
                                "errors": "No", 
                                "explanation": "Could not parse error grounder response as JSON, but action appears to have executed."
                            }
                    else:
                        logging.warning("Error Grounder chat failed or produced no history.")
            
                except Exception as grounder_ex:
                    logging.error(f"Error during Error Grounder call: {grounder_ex}")
                    # Use default error feedback if grounder call fails

                # --- Retrieve Manual Snippets AFTER action_result is available ---
                current_task_description = "Unknown task - RAG context might be limited"
                current_task_id_for_rag = None
                if 'task' in globals() and isinstance(globals()['task'], dict):
                    current_task_description = globals()['task'].get('ques', current_task_description)
                    current_task_id_for_rag = globals()['task'].get('id', None)

                manual_snippets_str = retrieve_and_rerank_manual_snippets(
                    task_query=current_task_description, 
                    current_observation_text=action_result.get('observation_text', ''), # Now uses observation from current action_result
                    task_id_for_manual_lookup=current_task_id_for_rag
                )

                # --- Combine Result and Feedback for Planner --- 
                base_text_for_planner = (
                    f"Action Status: {action_result.get('action_status')}\n"
                    f"Action Message: {action_result.get('action_message')}\n"
                    f"Error Grounding Feedback: {json.dumps(error_feedback_json)}\n"
                )
                
                # Prepend manual snippets if any
                if manual_snippets_str:
                    base_text_for_planner += f"{manual_snippets_str}\n"
                
                base_text_for_planner += f"New Observation Text:\n{action_result.get('observation_text', 'No text observation available.')}"

                combined_response_content = [
                    {
                        "type": "text",
                        "text": base_text_for_planner
                    },
                     {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{action_result.get('base64_image', '')}"}
                    }
                ]

                logging.info("DEBUG: Preparing combined response for Planner.")
                return (True, {"role": "user", "content": combined_response_content})

            except Exception as e:
                logging.error(f"Error in execute_web_action_and_reflect: {e}")
                import traceback
                traceback.print_exc()
                return (True, f"Internal Error in UserProxy: {e}")

        # --- End Custom Reply Function ---

        try: # Keep try/finally for driver cleanup

            # --- Register Functions & Custom Reply ---
            function_map = {
                "click_element": click_element,
                "type_element": type_element,
                "scroll_page": scroll_page,
                "wait_seconds": wait_seconds,
                "go_back": go_back,
                "google_search": google_search,
                "answer": answer,
            }
            user_proxy.register_function(function_map=function_map)
            # Register the custom reply function to intercept messages from the Planner
            user_proxy.register_reply(\
                planner, # Trigger specifically for the planner instance\
                execute_web_action_and_reflect,\
                config={"function_map": function_map} # Pass function map to reply func\
            ) # Closing parenthesis ends the call\n            logging.info(\"Functions registered and custom reply handler set for UserProxyAgent.\") # Corrected logging string\n            # -------------------------------------------
            
            # --- Initiate Autogen Chat ---
            # 1. Get initial observation
            logging.info("Getting initial observation...")
            initial_observation_text, initial_base64_image, initial_web_elements_cache = get_current_observation(driver_task, args, task_dir, current_iter)
            
            # Set the global cache with the initial elements
            web_elements_cache_for_task = initial_web_elements_cache
            logging.info(f"Initial web elements cache populated with {len(web_elements_cache_for_task)} elements")
            logging.info(f"Initial element IDs: {list(web_elements_cache_for_task.keys())}")
            
            logging.info("Initial Observation Captured.")

            # 2. Prepare initial message (multimodal)
            # Retrieve manual snippets for the initial task query
            initial_manual_snippets_str = retrieve_and_rerank_manual_snippets(
                task_query=task['ques'], 
                current_observation_text=initial_observation_text, # Initial observation
                task_id_for_manual_lookup=task['id']
            )

            initial_text_for_planner = (
                f"Task: {task['ques']}\n"
            )
            if initial_manual_snippets_str:
                initial_text_for_planner += f"{initial_manual_snippets_str}\n"
            initial_text_for_planner += f"Initial Observation Text:\n{initial_observation_text}"

            initial_message_content = [
                 {
                    "type": "text",
                    "text": initial_text_for_planner
                },
                 {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{initial_base64_image}"}
                }
            ]
            initial_message = {"role": "user", "content": initial_message_content}

            # 3. Start the chat
            logging.info("--- Starting Autogen Chat --- ")
            chat_res = user_proxy.initiate_chat(
                planner,
                message=initial_message,
                max_turns=args.max_iter * 2 # Allow more turns for planner/grounder interaction
            )
            logging.info("--- Autogen Chat Finished ---")

            # --- DEBUG: Log the chat result object ---
            logging.info(f"DEBUG: Raw chat_res object: {chat_res}")
            # ----------------------------------------

            # Optional: Log the chat history
            history_file = os.path.join(task_dir, "autogen_chat_history.json")
            try:
                # Clean the chat history data before saving
                cleaned_history = clean_json_data(chat_res.chat_history)
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(cleaned_history, f, indent=2, ensure_ascii=False)
                logging.info(f"Chat history saved to {history_file}")
            except Exception as e:
                logging.error(f"Error saving chat history: {e}")
                # Try with ASCII encoding as a fallback
                try:
                    with open(history_file, "w", encoding="utf-8") as f:
                        json.dump(chat_res.chat_history, f, indent=2, ensure_ascii=True)
                    logging.info(f"Chat history saved with ASCII encoding to {history_file}")
                except Exception as e2:
                    logging.error(f"Failed to save chat history even with ASCII encoding: {e2}")

            # Optional: Log summary or final answer if available
            summary = chat_res.summary if hasattr(chat_res, 'summary') else "No summary available."
            cost = chat_res.cost if hasattr(chat_res, 'cost') else "Cost info not available."
            logging.info(f"Chat Summary: {summary}")
            logging.info(f"Chat Cost: {cost}")

            # Extract final result/status for all_results list
            final_answer_details = None
            for msg in reversed(chat_res.chat_history):
                 if msg.get("role") == "function" and msg.get("name") == "answer":
                     try:
                         final_answer_content = msg.get("content", "{}")
                         # Clean any potential problematic characters
                         final_answer_content = clean_text(final_answer_content)
                         final_answer_details = json.loads(final_answer_content)
                     except Exception as e:
                         logging.error(f"Error parsing final answer: {e}")
                         # Provide a fallback answer details
                         final_answer_details = {"action_status": "Error", "error": "Failed to parse answer"}
                     break
                     
            # Make sure we have a clean task_id for file operations
            safe_task_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)
            
            all_results.append({
                "task_id": task_id,
                "safe_task_id": safe_task_id,
                "final_answer_details": final_answer_details,
                "chat_history_file": os.path.basename(history_file),
                "summary": clean_text(summary),
                "cost": cost
            })

        except Exception as task_error:
             # Use safe_log to handle potential encoding issues in the error message
             safe_log(logging.ERROR, f"Error in task {task_id}: {task_error}")
             # Clean the error message to remove problematic characters
             cleaned_error = clean_text(str(task_error))
             all_results.append({
                "task_id": task_id,
                "status": "Error",
                "error_message": cleaned_error
             })
        finally:
             # Always quit the driver (this is inside the task try/finally)
             logging.info(f"TASK {task_id} driver quit.")
             try: # Correct indentation for try
                driver_task.quit()
             except Exception as quit_e: # Correct indentation for except
                 logging.warning(f"Ignoring error during driver quit: {quit_e}")
                 pass # Ignore errors when quitting the driver

    # Write out all task results for analysis
    all_tasks_summary_file = os.path.join(result_dir, "all_tasks_summary.json")
    try:
        # Clean the results data before saving
        cleaned_results = clean_json_data(all_results)
        with open(all_tasks_summary_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_results, f, indent=2, ensure_ascii=False)
        logging.info(f"All tasks summary saved to {all_tasks_summary_file}")
    except Exception as e:
        logging.error(f"Failed to save all_tasks_summary.json: {e}")
        # Try with a plain ASCII version if UTF-8 encoding fails
        try:
            with open(all_tasks_summary_file, "w", encoding="utf-8") as f:
                # Use ensure_ascii=True to force ASCII encoding
                json.dump(all_results, f, indent=2, ensure_ascii=True)
            logging.info(f"All tasks summary saved with ASCII encoding to {all_tasks_summary_file}")
        except Exception as e2:
            logging.error(f"Failed even with ASCII encoding: {e2}")

    # Same for any task summary files
    try:
        with open(os.path.join(result_dir, "test_tasks.log"), "w", encoding="utf-8") as f:
            for result in all_results:
                # Clean each result before saving
                cleaned_result = clean_json_data(result)
                f.write(json.dumps(cleaned_result, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"Failed to save test_tasks.log: {e}")
        try:
            with open(os.path.join(result_dir, "test_tasks.log"), "w", encoding="utf-8") as f:
                for result in all_results:
                    # Use ensure_ascii=True to force ASCII encoding
                    f.write(json.dumps(result, ensure_ascii=True) + "\n")
            logging.info("test_tasks.log saved with ASCII encoding")
        except Exception as e2:
            logging.error(f"Failed even with ASCII encoding for test_tasks.log: {e2}")

if __name__ == '__main__':
    main()
