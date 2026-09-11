from src.loader import load_documents
from src.chunker import chunk_documents
from src.embeddings import EmbeddingService
from src.search import searchService
from src.vector_index import vectorIndex

def main():

    # ------------------------
    # Load documents
    # ------------------------

    documents = load_documents("data")

    print(f"Loaded {len(documents)} documents")


    # ------------------------
    # Chunk documents
    # ------------------------

    chunks = chunk_documents(documents, chunk_size=50, overlap=10)

    print(f"Created {len(chunks)} chunks")

    # ------------------------
    # Generate embeddings
    # ------------------------

    embedding_service = EmbeddingService()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = (
        embedding_service.embed_texts(
            texts
        )
    )

    print("Embedding shape:", embeddings.shape)


    # ------------------------
    # Create search engine and vector index
    # ------------------------
    vector_index = vectorIndex()
    for chunk, embedding in zip(chunks, embeddings):
        vector_index.add(embedding=embedding, metadata=chunk )

    # -------------------------
    # QUERY PHASE
    # -------------------------

    search_service = searchService(embedding_service, vector_index)
    
    query = (
        "How can I deploy an ML model "
        "using an API?"
    )
    results = search_service.search(query=query, top_k=3)

    # ------------------------
    # Display results
    # ------------------------

    print(f"\nQuery: {query}")

    for rank, result in enumerate(results, start=1):

        print("\n" + "=" * 60)

        print(f"Rank: {rank}" )

        print(f"Score: {result['score']:.4f}")

        print(f"Source: {result['source']}")

        print(f"Chunk ID: {result['chunk_id']}")

        print("\nText:")

        print(
            result["text"]
        )


if __name__ == "__main__":
    main()