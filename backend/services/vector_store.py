import chromadb
import numpy as np
import logging
from typing import List, Dict, Any
from backend.config import settings
from backend.services.chunking import DocumentChunk

logger = logging.getLogger(__name__)

# Global lazy-loaded SentenceTransformer embedder to speed up module imports
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {str(e)}. Using Mock Embedder.")
            # Fallback mock embedder for offline/local development robustness
            class MockEmbedder:
                def encode(self, sentences):
                    # Return deterministic dummy vectors of dimension 384 (MiniLM-L6 dimension)
                    if isinstance(sentences, str):
                        sentences = [sentences]
                    res = []
                    for s in sentences:
                        val = sum(ord(c) for c in s) / 1000.0
                        vec = [float(val + i * 0.01) % 1.0 for i in range(384)]
                        # Normalize
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = (vec / norm).tolist()
                        res.append(vec)
                    return np.array(res)
            _embedder = MockEmbedder()
    return _embedder

class VectorStoreService:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    def _get_collection_name(self, report_id: str) -> str:
        # ChromaDB collections must be between 3 and 63 chars, start and end with alphanumeric, 
        # only contain alphanumeric, underscores, hyphens, and dots.
        # Clean the collection name accordingly.
        clean_id = report_id.replace("-", "_")
        return f"report_{clean_id}"

    def index_report(self, report_id: str, chunks: List[DocumentChunk]):
        """Index a list of DocumentChunks into ChromaDB for a specific report."""
        if not chunks:
            logger.warning(f"No chunks to index for report {report_id}")
            return

        collection_name = self._get_collection_name(report_id)
        
        # Get or create collection using cosine similarity
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"chk_{c.metadata['chunk_index']}" for c in chunks]

        # Generate embeddings
        embedder = get_embedder()
        embeddings = embedder.encode(texts).tolist()

        # Add to Chroma
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Indexed {len(chunks)} chunks into collection {collection_name}")

    def query_report(self, report_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks from ChromaDB for a query."""
        collection_name = self._get_collection_name(report_id)
        
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Collection {collection_name} not found: {str(e)}")
            return []

        # Generate query embedding
        embedder = get_embedder()
        query_embedding = embedder.encode([query]).tolist()[0]

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_chunks = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0]

            for i in range(len(docs)):
                # Convert distance (cosine distance) to similarity score
                # cosine similarity = 1 - cosine distance
                similarity = 1.0 - distances[i]
                retrieved_chunks.append({
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "score": float(similarity)
                })
        
        return retrieved_chunks

    def delete_report_collection(self, report_id: str):
        """Delete vector collection associated with a report."""
        collection_name = self._get_collection_name(report_id)
        try:
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted vector collection {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection {collection_name}: {str(e)}")
