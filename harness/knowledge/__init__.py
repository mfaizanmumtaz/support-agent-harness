"""The knowledge base: chunk markdown documents, embed them, search by meaning."""

from harness.knowledge.chunking import Chunk, chunk_document, chunk_folder
from harness.knowledge.embeddings import Embedder, HashEmbedder, OpenAIEmbedder
from harness.knowledge.index import KnowledgeIndex

__all__ = [
    "Chunk",
    "Embedder",
    "HashEmbedder",
    "KnowledgeIndex",
    "OpenAIEmbedder",
    "chunk_document",
    "chunk_folder",
]
