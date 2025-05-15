import os
import chromadb
from sentence_transformers import SentenceTransformer
from markdown_it import MarkdownIt

# --- Configuration ---
MANUAL_FILE_PATH = os.path.join("data", "BookingComAdvancedFeatures.md")
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "booking_manual"
# Using a smaller, efficient model for local embedding. 
# Other options: 'all-mpnet-base-v2' (larger, potentially better), or OpenAI embeddings via API.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
CHUNK_SIZE = 500 # Approximate characters per chunk. Tune as needed.
CHUNK_OVERLAP = 50 # Approximate characters overlap between chunks. Tune as needed.

# --- Helper Functions ---

def load_and_parse_markdown(file_path):
    """Loads and parses a markdown file into a list of text blocks (paragraphs, list items, etc.)."""
    print(f"Loading markdown from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"ERROR: Manual file not found at {file_path}")
        return []
    
    md = MarkdownIt()
    tokens = md.parse(content)
    
    chunks = []
    current_chunk = ""
    current_header = "General"

    for token in tokens:
        if token.type == 'heading_open':
            # When a new heading starts, save the previous chunk if it has content
            if current_chunk.strip():
                chunks.append({"content": current_chunk.strip(), "header": current_header, "source": os.path.basename(file_path)})
                current_chunk = "" # Reset for the new section
            # Update current_header based on heading level (h1, h2, etc.)
            # For simplicity, we'll just take the text of the next inline token as header
        elif token.type == 'inline' and tokens[tokens.index(token)-1].type == 'heading_open':
             current_header = token.content.strip()
             current_chunk += f"{token.content}\n" # Add header to the chunk itself
        elif token.type == 'text' or (token.type == 'inline' and token.content.strip()):
            current_chunk += token.content
        elif token.type == 'paragraph_open' or token.type == 'list_item_open' or token.type == 'bullet_list_open':
            # Add a separator for structure if current_chunk is not empty and the new content is significant
            if current_chunk.strip() and not current_chunk.endswith('\n'):
                 current_chunk += "\n"
        # Add other token type handling if needed (e.g., for tables, code blocks)

    # Add the last accumulated chunk
    if current_chunk.strip():
        chunks.append({"content": current_chunk.strip(), "header": current_header, "source": os.path.basename(file_path)})
    
    # Further refine chunking: simple text splitting if chunks are too large
    # This is a basic approach. More sophisticated text splitters (e.g., from LangChain) could be used.
    refined_chunks = []
    for chunk_dict in chunks:
        text = chunk_dict["content"]
        if len(text) > CHUNK_SIZE:
            # Split the large chunk
            start = 0
            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                # Try to find a natural break point (e.g., end of sentence) if possible, not implemented here for simplicity
                refined_chunks.append({"content": text[start:end], "header": chunk_dict["header"], "source": chunk_dict["source"]})
                start += CHUNK_SIZE - CHUNK_OVERLAP
        else:
            refined_chunks.append(chunk_dict)
            
    print(f"Parsed and refined into {len(refined_chunks)} chunks.")
    # for i, c in enumerate(refined_chunks):
    #     print(f"Chunk {i} (Header: {c['header']}): {c['content'][:100]}...")
    return refined_chunks


# --- Main Processing Logic ---
def main():
    print("Starting manual processing...")

    # 1. Load and Chunk Manual
    chunks_with_metadata = load_and_parse_markdown(MANUAL_FILE_PATH)
    if not chunks_with_metadata:
        print("No chunks to process. Exiting.")
        return

    documents_to_embed = [chunk["content"] for chunk in chunks_with_metadata]
    metadatas_to_store = [{key: value for key, value in chunk.items() if key != 'content'} for chunk in chunks_with_metadata]
    ids_to_store = [f"chunk_{i}_{os.path.basename(MANUAL_FILE_PATH).split('.')[0]}" for i in range(len(documents_to_embed))]

    # 2. Initialize Embedding Model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"ERROR: Could not load sentence transformer model '{EMBEDDING_MODEL_NAME}'. Error: {e}")
        print("Please ensure it's installed or a valid model name is provided.")
        return
    print("Embedding model loaded.")

    # 3. Generate Embeddings
    print(f"Generating embeddings for {len(documents_to_embed)} document(s)...")
    try:
        embeddings = model.encode(documents_to_embed, show_progress_bar=True)
    except Exception as e:
        print(f"ERROR: Failed to generate embeddings. Error: {e}")
        return
    print("Embeddings generated.")

    # 4. Initialize and Populate ChromaDB
    print(f"Initializing ChromaDB client at: {CHROMA_DB_PATH}")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    except Exception as e:
        print(f"ERROR: Could not initialize ChromaDB client. Error: {e}")
        print("Ensure ChromaDB is installed and the path is accessible.")
        return
    
    print(f"Checking for existing collection: {COLLECTION_NAME}")
    try:
        # Check if collection exists
        existing_collections = [col.name for col in client.list_collections()]
        if COLLECTION_NAME in existing_collections:
            print(f"Collection '{COLLECTION_NAME}' found. Deleting it to ensure fresh data ingestion.")
            client.delete_collection(name=COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
        else:
            print(f"Collection '{COLLECTION_NAME}' not found. A new one will be created.")
    except Exception as e:
        print(f"ERROR: Could not check for or delete existing collection '{COLLECTION_NAME}'. Error: {e}")
        # Depending on the error, you might want to decide if it's safe to proceed or not.
        # For now, we'll print the error and let it try to get_or_create_collection.
        # If get_or_create_collection fails, the existing error handling for that will catch it.

    print(f"Getting or creating collection: {COLLECTION_NAME}")
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            # embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME) # Alternative
            # If you specify embedding_function here, ChromaDB can generate embeddings for queries if you don't provide query_embeddings.
            # For add(), you must provide embeddings explicitly if not using a default ef on the collection that matches your data.
        )
    except Exception as e:
        print(f"ERROR: Could not get or create ChromaDB collection '{COLLECTION_NAME}'. Error: {e}")
        return

    print(f"Adding {len(documents_to_embed)} documents to collection '{COLLECTION_NAME}'...")
    try:
        # Check if IDs already exist to avoid duplicates, or clear collection first if re-populating
        # For simplicity, we'll try to add. ChromaDB usually handles duplicate IDs by updating or erroring.
        # Consider adding a step to check existing_ids = collection.get(ids=ids_to_store)['ids'] and filter.
        collection.add(
            embeddings=embeddings.tolist(), # Ensure embeddings are a list of lists
            documents=documents_to_embed,
            metadatas=metadatas_to_store,
            ids=ids_to_store
        )
        print("Documents added to ChromaDB successfully.")
        print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} items.")
    except Exception as e:
        print(f"ERROR: Could not add documents to ChromaDB. Error: {e}")
        print("This might happen if IDs are duplicated and the collection is not set to update, or other DB issues.")

    print("Manual processing finished.")

if __name__ == "__main__":
    main() 