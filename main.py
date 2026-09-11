from src.loader import load_documents
from src.chunker import chunk_documents
from src.embeddings import EmbeddingService
from src.search import SemanticSearch


def main():

    # ------------------------
    # Load documents
    # ------------------------

    documents = load_documents(
        "data"
    )

    print(
        f"Loaded {len(documents)} documents"
    )


    # ------------------------
    # Chunk documents
    # ------------------------

    chunks = chunk_documents(
        documents,
        chunk_size=50,
        overlap=10
    )

    print(
        f"Created {len(chunks)} chunks"
    )


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

    print(
        "Embedding shape:",
        embeddings.shape
    )


    # ------------------------
    # Create search engine
    # ------------------------

    search_engine = SemanticSearch(
        chunks=chunks,
        embeddings=embeddings,
        embedding_service=embedding_service
    )


    # ------------------------
    # Search
    # ------------------------

    query = (
        "How can I build APIs using Python?"
    )

    results = search_engine.search(
        query=query,
        top_k=3
    )


    # ------------------------
    # Display results
    # ------------------------

    print(
        f"\nQuery: {query}"
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "=" * 60)

        print(
            f"Rank: {rank}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print("\nText:")

        print(
            result["text"]
        )


if __name__ == "__main__":
    main()