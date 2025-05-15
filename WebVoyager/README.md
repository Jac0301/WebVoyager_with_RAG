[View on GitHub](https://github.com/Jac0301/WebVoyager_with_RAG)

# WebVoyager: AI-Powered Web Navigation Agent

## 1. Overview

WebVoyager is an AI-powered agent designed to navigate websites and perform tasks based on natural language instructions. It uses a combination of vision-language models (VLMs) like GPT-4V, browser automation via Selenium, and Retrieval Augmented Generation (RAG) to understand web pages, plan actions, and interact with web elements. The system employs an agentic framework built with Autogen, featuring a Planner agent for decision-making, an Error Grounder agent for analyzing action outcomes, and a User Proxy agent for executing browser actions.

## 2. Key Features

-   **Multimodal Understanding:** Leverages VLMs to process both textual content and screenshots of web pages.
-   **Browser Automation:** Uses Selenium to interact with web elements (click, type, scroll).
-   **Retrieval Augmented Generation (RAG):** Enhances decision-making by retrieving relevant information from a knowledge base (e.g., product manuals) stored in a ChromaDB vector database.
-   **Agentic Framework (Autogen):**
    -   **Planner:** Decides the next best action based on the current observation, task, and retrieved manual snippets.
    -   **Error Grounder:** Analyzes the outcome of actions, identifies errors, and provides feedback to the Planner.
    -   **User Proxy:** Executes browser commands and communicates results back to the Planner.
-   **Task Execution:** Can perform tasks like navigating to URLs, filling forms, clicking buttons, and extracting information.
-   **Configurable:** Supports various command-line arguments for headless browsing, logging, API model selection, etc.
-   **Logging and Output:** Generates detailed logs and saves screenshots for each step, aiding in debugging and analysis.

## 3. Prerequisites

-   Python 3.8+
-   Google Chrome browser
-   ChromeDriver (compatible with your Chrome version)
-   OpenAI API Key

## 4. Setup

### 4.1. Clone the Repository (if applicable)

```bash
git clone <your-repository-url>
cd WebVoyager # Or your project directory
```

### 4.2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4.3. Install Python Dependencies

Create a `requirements.txt` file with the following content:

```text
# Core
openai
selenium
 Pillow # Image processing, often a dependency for encode_image

# Autogen
pyautogen

# RAG
chromadb
sentence-transformers # For embedding model and reranker
torch # Often a dependency for sentence-transformers
torchvision # Often a dependency for sentence-transformers
torchaudio # Often a dependency for sentence-transformers

# Other utilities (based on imports)
# argparse, time, json, re, os, shutil, logging, random are standard libraries
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 4.4. Install ChromeDriver

1.  Check your Google Chrome browser version (Help -> About Google Chrome).
2.  Download the corresponding ChromeDriver version from [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads).
3.  Place the `chromedriver` executable in a directory included in your system's PATH (e.g., `/usr/local/bin` on macOS/Linux, or add its directory to PATH on Windows), or place it directly in the project's root directory.

### 4.5. Configure OpenAI API

The script expects an `OAI_CONFIG_LIST` file in the `WebVoyager` directory (or the directory from which `run.py` is executed). This JSON file configures the OpenAI API access.

Create `OAI_CONFIG_LIST` with the following structure:

```json
[
    {
        "model": "gpt-4.1",
        "api_key": "YOUR_OPENAI_API_KEY"
    },
    {
        "model": "gpt-4-vision-preview", // Or the specific vision model you intend to use, ensure it matches args.vision_model if different
        "api_key": "YOUR_OPENAI_API_KEY"
    }
    // Add other models if needed, ensure they match --api_model and --vision_model arguments
]
```

-   Replace `"YOUR_OPENAI_API_KEY"` with your actual OpenAI API key.
-   The script defaults to `gpt-4.1` (as per `args.api_model` and `args.vision_model`). Ensure the models specified in `OAI_CONFIG_LIST` match those you intend to use or are passed via command-line arguments. The `run.py` script specifically filters for models defined by `args.api_model` and `args.vision_model`.

### 4.6. Setup RAG Components (ChromaDB & Models)

The script initializes RAG components (`initialize_rag_components` function):
-   **ChromaDB:** A persistent client is created at `./chroma_db`. This directory will be created if it doesn't exist. The collection name is `booking_manual`.
    ```python
    client = chromadb.PersistentClient(path="./chroma_db")
    manual_vector_db_collection = client.get_or_create_collection(name="booking_manual")
    ```
-   **Embedding and Reranker Models:**
    -   Sentence Transformer Embedding Model: `all-MiniLM-L6-v2`
    -   CrossEncoder Reranker Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
    These models will be downloaded automatically by the `sentence-transformers` library on their first use if not already cached.

**Important for RAG:**
The script's RAG functionality relies on the `booking_manual` collection in ChromaDB being populated with embeddings of your manual documents. The script itself *does not* populate this database. **You need to run a separate process to ingest and embed your documents into this ChromaDB collection.** Each document should ideally have `text`, `source`, and `header` metadata.

If the `booking_manual` collection is empty or RAG components fail to initialize, the agent will proceed without RAG capabilities for those tasks.

## 5. Running the Agent

The agent is run using the `run.py` script with various command-line arguments.

### 5.1. Command-Line Arguments

Here are some of the key arguments:

-   `--test_file TEST_FILE`: Path to the JSONL file containing tasks (default: `data/tasks_test.jsonl`). Each line should be a JSON object with task details (e.g., `id`, `web` (start URL), `ques` (task instruction)).
-   `--max_iter MAX_ITER`: Maximum number of iterations (turns) per task (default: 15).
-   `--api_model API_MODEL`: API model name for the Planner (default: "gpt-4.1"). Must be in `OAI_CONFIG_LIST`.
-   `--vision_model VISION_MODEL`: Vision model name for the Error Grounder (default: "gpt-4.1"). Must be in `OAI_CONFIG_LIST`.
-   `--output_dir OUTPUT_DIR`: Directory to save results (default: 'results'). A timestamped subdirectory will be created here.
-   `--seed SEED`: Random seed for reproducibility.
-   `--download_dir DOWNLOAD_DIR`: Directory for downloads (default: "downloads").
-   `--headless`: Run browser in headless mode.
-   `--text_only`: Use text-only mode (accessibility tree) instead of screenshots.
-   `--window_width WIDTH`, `--window_height HEIGHT`: Browser window dimensions.
-   `--save_accessibility_tree`: Save accessibility tree (used with `--text_only` or for debugging).
-   `--force_device_scale`: Force device scale factor to 1.
-   `--fix_box_color`: Fix color of bounding boxes drawn on screenshots.
-   `--temperature TEMP`: Temperature for LLM sampling (default: 1.0).

### 5.2. Example Usage

```bash
python run.py --test_file path/to/your/tasks.jsonl --api_model gpt-4-vision-preview --vision_model gpt-4-vision-preview --headless
```

This command runs the agent on tasks defined in `tasks.jsonl`, using `gpt-4-vision-preview` for both planner and error grounder, in headless mode.

### 5.3. Task File Format (`tasks.jsonl`)

Each line in the task file should be a JSON object, for example:

```json
{"id": "task_001", "web": "https://www.example.com", "ques": "Find the contact email address on the website."}
{"id": "task_002", "web": "https://www.wikipedia.org", "ques": "Search for 'Artificial Intelligence' and summarize the first paragraph."}
```

## 6. Key Components and Workflow

1.  **Initialization:**
    -   Sets up logging, RAG components (ChromaDB, embedding/reranker models), and Autogen agents (Planner, Error Grounder, User Proxy).
    -   Loads tasks from the specified test file.

2.  **Task Loop:** For each task:
    -   Initializes a Selenium WebDriver.
    -   Navigates to the starting URL.
    -   Captures the initial observation (screenshot, web element information).
    -   Retrieves relevant manual snippets using RAG based on the task query and initial observation.
    -   Constructs an initial multimodal message for the Planner, including the task, observation, and manual snippets.

3.. **Autogen Interaction Loop (per task):**
    -   The `User_Proxy` sends the current state (observation, feedback, manual snippets) to the `Planner`.
    -   **Planner:**
        -   Analyzes the input.
        -   Decides on the next browser action (e.g., `click_element`, `type_element`, `scroll_page`, `go_back`, `google_search`, `answer`).
        -   Outputs its reasoning and the chosen action as a JSON object.
    -   **User_Proxy:**
        -   Parses the Planner's action.
        -   Executes the action using registered Selenium-based functions (e.g., `click_element(element_id=10)`). These functions update the browser state and capture a new observation.
        -   The result of the action (status, message, new observation with screenshot) is prepared.
    -   **Error Grounder (via User_Proxy):**
        -   The `User_Proxy` sends the attempted action, action result, and new observation (including screenshot) to the `Error_Grounder`.
        -   The `Error_Grounder` analyzes the screenshot and action outcome to determine if an error occurred.
        -   It responds with a JSON object indicating `{"errors": "Yes/No", "explanation": "..."}`.
    -   **RAG (via User_Proxy):**
        -   The `User_Proxy` again calls `retrieve_and_rerank_manual_snippets` with the current task query and the *new* observation text to get updated relevant manual snippets.
    -   The `User_Proxy` then sends a combined message back to the `Planner`:
        -   Action status and message.
        -   Error Grounder's feedback.
        -   Newly retrieved manual snippets.
        -   The new observation (text and screenshot).
    -   This loop continues until the Planner calls the `answer` function or `max_iter` is reached.

4.  **Termination:**
    -   When the Planner calls `answer(final_answer="...")`, the task is considered complete. The final answer is saved.
    -   The WebDriver is closed.

5.  **Results:**
    -   Logs, screenshots, and chat history are saved in the `output_dir` within a timestamped subdirectory for each run, and further organized by task ID.
    -   A summary of all tasks is saved to `all_tasks_summary.json`.

## 7. Configuration Details

-   **`OAI_CONFIG_LIST`:** Crucial for API access. Ensure model names here match `args.api_model` and `args.vision_model`.
-   **RAG Triggering:** RAG is triggered if the ChromaDB collection has items AND (the task ID contains "rag" OR the task query contains keywords like "manual", "handbook", "guide").
-   **Element Interaction:** Elements are identified by numerical labels overlaid on screenshots. The Planner uses these IDs (e.g., `element_id=5`) in its function calls.
-   **Logging:** Detailed logs are saved to `agent.log` within each task's output directory. Console output is set to `ERROR` level by default.

## 8. Directory Structure (Simplified)

```
WebVoyager/
├── run.py                  # Main script
├── prompts.py              # Contains system prompts (imported but not shown in snippet)
├── utils.py                # Utility functions (imported)
├── OAI_CONFIG_LIST         # (You need to create this for OpenAI API keys)
├── requirements.txt        # (You need to create this)
├── data/
│   └── tasks_test.jsonl    # Example task file
├── chroma_db/              # Will be created by ChromaDB for persistent storage
│   └── ...                 # ChromaDB data
├── results/                # Default output directory
│   └── <timestamp>_run/
│       ├── task<ID>/
│       │   ├── screenshot<N>_observation.png
│       │   ├── agent.log
│       │   ├── autogen_chat_history.json
│       │   └── final_answer.txt (if task completes)
│       ├── all_tasks_summary.json
│       └── test_tasks.log
└── downloads/              # Default directory for file downloads by the browser
```

## 9. Troubleshooting/Notes

-   **ChromeDriver Version:** Ensure ChromeDriver version matches your Chrome browser version. Mismatches are a common source of errors.
-   **API Keys:** Double-check that `OAI_CONFIG_LIST` is correctly formatted and contains valid API keys with access to the specified models.
-   **RAG Data:** For RAG to be effective, the `booking_manual` collection in `chroma_db` must be populated with relevant, embedded documents.
-   **Element Not Found/Interactable:** Web pages are dynamic. If the agent fails to interact with an element, review the screenshots and logs. The Error Grounder should provide feedback. The Planner is designed to use the *most recent* observation.
-   **Rate Limits:** If you encounter OpenAI API rate limits, the script has a basic exponential backoff mechanism.
-   **Headless Mode:** For long runs or CI/CD, use the `--headless` flag.

## 10. Contributing (Placeholder)

Information on how to contribute to the project. (e.g., coding standards, pull request process).

## 11. License (Placeholder)

Specify the project's license (e.g., MIT, Apache 2.0).
