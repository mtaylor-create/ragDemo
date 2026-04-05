import ollama
import chromadb

# --- Configuration ---
COLLECTION_NAME = "cmu_handbook"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "mistral"
N_RESULTS = 5  # number of chunks to retrieve per question

def get_collection():
    """Open the existing ChromaDB collection."""
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(name=COLLECTION_NAME)

def retrieve(collection, query, n_results=N_RESULTS):
    """Embed the query and find the most relevant chunks."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=query)
    query_embedding = response.embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results["documents"][0], results["metadatas"][0]

def build_prompt(query, documents, metadatas):
    """Construct a prompt that includes the retrieved context."""
    context_parts = []
    for doc, meta in zip(documents, metadatas):
        context_parts.append(f"[Page {meta['page']}] {doc}")
    context = "\n\n".join(context_parts)

    return f"""You are a helpful assistant that answers questions about the CMU student handbook ("The Word").
Use ONLY the following excerpts from the handbook to answer the question. If the answer is not
contained in the excerpts, say "I don't see that information in the handbook."

--- HANDBOOK EXCERPTS ---
{context}
--- END EXCERPTS ---

Question: {query}
Answer:"""

def chat(collection):
    """Main chat loop."""
    print("CMU Handbook Assistant (type 'quit' to exit)")
    print("=" * 50)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        # Step 1: Retrieve relevant chunks
        documents, metadatas = retrieve(collection, query)

        # Step 2: Build the prompt with context
        prompt = build_prompt(query, documents, metadatas)

        # Step 3: Generate a response using Mistral
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.message.content

        print(f"\nAssistant: {answer}")

        # Show sources so students can verify
        pages = sorted(set(m["page"] for m in metadatas))
        print(f"\n  [Sources: pages {', '.join(str(p) for p in pages)}]")

if __name__ == "__main__":
    collection = get_collection()
    chat(collection)
