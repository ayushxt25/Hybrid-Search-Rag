from pathlib import Path

from app.core.config import get_settings
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.vectorstore.qdrant import QdrantVectorStore


def main() -> None:
    """Index a sample document and query the real Qdrant service."""
    settings = get_settings()

    document_path = Path("datasets/sample-documents/remote_policy.txt")

    if not document_path.exists():
        raise FileNotFoundError(f"Sample document was not found: {document_path}")

    pipeline = DocumentIngestionPipeline(
        chunk_size=25,
        chunk_overlap=5,
    )

    ingested_document = pipeline.ingest(document_path)

    embedding_provider = SentenceTransformerEmbeddingProvider()

    if embedding_provider.dimensions != settings.dense_embedding_dimensions:
        raise RuntimeError("Embedding dimensions do not match application settings.")

    embeddings = embedding_provider.embed_chunks(ingested_document.chunks)

    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        vector_dimensions=settings.dense_embedding_dimensions,
    )

    vector_store.ensure_collection()

    written_count = vector_store.upsert_document(
        ingested_document=ingested_document,
        embeddings=embeddings,
    )

    query_embedding = embedding_provider.embed_query(
        "How many days can employees work from home?"
    )

    results = vector_store.search_dense(
        query_vector=query_embedding.vector,
        limit=3,
        document_id=ingested_document.document.document_id,
    )

    print(f"Document ID: {ingested_document.document.document_id}")
    print(f"Generated chunks: {ingested_document.chunk_count}")
    print(f"Indexed points: {written_count}")
    print("\nDense search results:")

    if not results:
        raise RuntimeError("Qdrant returned no dense-search results.")

    for position, result in enumerate(results, start=1):
        print(f"\n{position}. Score: {result.score:.4f}")
        print(f"   Point ID: {result.point_id}")
        print(f"   Chunk ID: {result.chunk_id}")
        print(f"   File: {result.file_name}")
        print(f"   Page: {result.page_number}")
        print(f"   Heading: {result.heading}")
        print(f"   Text: {result.text}")

    top_result = results[0].text.lower()

    if "three days per week" not in top_result:
        raise RuntimeError("The expected remote-work chunk was not ranked first.")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
