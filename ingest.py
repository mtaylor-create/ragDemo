import pdfplumber
import ollama
import chromadb

# --- Configuration ---
PDF_PATH = "the-word-2025-2026.pdf"
COLLECTION_NAME = "cmu_handbook"
CHUNK_SIZE = 500       # approximate number of characters per chunk
CHUNK_OVERLAP = 50     # overlap between consecutive chunks
EMBEDDING_MODEL = "nomic-embed-text"

def extract_text_from_pdf(pdf_path):
    """Read every page of the PDF and return a list of (page_number, text) tuples."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append((i + 1, text))  # 1-indexed page numbers
    print(f"Extracted text from {len(pages)} pages (out of {i + 1} total).")
    return pages

def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split page texts into overlapping chunks. Each chunk remembers its source page."""
    chunks = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                "text": chunk,
                "page": page_num,
                "start_char": start,
            })
            start += chunk_size - overlap
    print(f"Created {len(chunks)} chunks.")
    return chunks

def embed_and_store(chunks):
    """Embed each chunk using Ollama and store it in ChromaDB."""
    # Create (or open) a persistent ChromaDB database in ./chroma_db
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete the collection if it already exists so we start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    # Process in batches of 50 to avoid overwhelming Ollama
    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Get embeddings from Ollama
        response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
        embeddings = response.embeddings

        # Prepare data for ChromaDB
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        metadatas = [{"page": c["page"], "start_char": c["start_char"]} for c in batch]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  Stored batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    print(f"\nDone! {len(chunks)} chunks stored in ./chroma_db")

if __name__ == "__main__":
    print("Step 1: Extracting text from PDF...")
    pages = extract_text_from_pdf(PDF_PATH)

    print("\nStep 2: Splitting text into chunks...")
    chunks = chunk_text(pages)

    print("\nStep 3: Embedding chunks and storing in ChromaDB...")
    embed_and_store(chunks)
