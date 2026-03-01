"""RAG (Retrieval-Augmented Generation) for library documentation."""

from __future__ import annotations

import re
from pathlib import Path

from pmca.models.config import RAGConfig
from pmca.utils.logger import get_logger

log = get_logger("utils.rag")

# Maximum characters per chunk (roughly 512 tokens at 4 chars/token)
_MAX_CHUNK_CHARS = 2048


def _chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks by paragraph boundaries, respecting max size."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)
        if current_len + para_len + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        # If a single paragraph exceeds max, split by lines
        if para_len > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for i in range(0, para_len, max_chars):
                chunks.append(para[i : i + max_chars])
        else:
            current.append(para)
            current_len += para_len + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _sanitize_collection_name(name: str) -> str:
    """Sanitize a string into a valid ChromaDB collection name.

    ChromaDB requires: 3-63 chars, alphanumeric start/end, only [a-zA-Z0-9._-].
    """
    # Strip non-alphanumeric characters (except . _ -)
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name.lower())
    # Collapse consecutive underscores/dots
    name = re.sub(r"[_.]{2,}", "_", name)
    # Strip leading/trailing non-alphanumeric
    name = name.strip("._-")
    # Enforce length bounds
    name = name[:63]
    if len(name) < 3:
        name = name + "_col"
    return name


class RAGManager:
    """Manages a ChromaDB vector store for library documentation retrieval."""

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._collection = None
        self._client = None
        self._embedding_fn = None

        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = Path(config.persist_dir).expanduser()
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            log.info(f"ChromaDB initialized at {persist_dir}")
        except ImportError:
            log.warning(
                "chromadb is not installed. Install with: pip install pmca[rag]"
            )
            return  # Skip embedding init if chromadb is missing

        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=config.embedding_model,
            )
            log.info(f"Using embedding model: {config.embedding_model}")
        except ImportError:
            log.warning(
                "sentence-transformers is not installed. "
                "Install with: pip install pmca[rag]"
            )

    @property
    def available(self) -> bool:
        """Whether the RAG system is fully functional."""
        return self._client is not None and self._embedding_fn is not None

    def _get_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        if not self.available:
            return None
        name = _sanitize_collection_name(name)
        kwargs = {"name": name, "embedding_function": self._embedding_fn}
        self._collection = self._client.get_or_create_collection(**kwargs)
        return self._collection

    def index_directory(self, docs_path: Path) -> int:
        """Chunk and embed all markdown/text files in a directory.

        Returns the number of chunks indexed.
        """
        if not self.available:
            log.warning("RAG not available — skipping indexing")
            return 0

        docs_path = Path(docs_path).resolve()
        if not docs_path.is_dir():
            log.error(f"Docs path does not exist: {docs_path}")
            return 0

        collection_name = docs_path.name
        collection = self._get_collection(collection_name)
        if collection is None:
            return 0

        doc_files = list(docs_path.glob("**/*.md")) + list(docs_path.glob("**/*.txt"))
        if not doc_files:
            log.warning(f"No .md or .txt files found in {docs_path}")
            return 0

        all_chunks: list[str] = []
        all_ids: list[str] = []
        all_metadata: list[dict] = []

        for doc_file in doc_files:
            try:
                text = doc_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                log.warning(f"Could not read {doc_file}: {e}")
                continue

            chunks = _chunk_text(text)
            rel_path = str(doc_file.relative_to(docs_path))
            for i, chunk in enumerate(chunks):
                chunk_id = f"{rel_path}::{i}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadata.append({"source": rel_path, "chunk_index": i})

        if not all_chunks:
            return 0

        # Upsert in batches (ChromaDB handles deduplication by ID)
        batch_size = 100
        for start in range(0, len(all_chunks), batch_size):
            end = start + batch_size
            collection.upsert(
                documents=all_chunks[start:end],
                ids=all_ids[start:end],
                metadatas=all_metadata[start:end],
            )

        log.info(f"Indexed {len(all_chunks)} chunks from {len(doc_files)} files")
        return len(all_chunks)

    def query(self, text: str, n_results: int | None = None) -> list[str]:
        """Retrieve relevant document chunks for a query.

        Returns a list of document chunk strings, most relevant first.
        """
        if not self.available or self._collection is None:
            return []

        if n_results is None:
            n_results = self._config.n_results

        try:
            results = self._collection.query(
                query_texts=[text],
                n_results=n_results,
            )
        except Exception as e:
            log.warning(f"RAG query failed: {e}")
            return []

        documents = results.get("documents", [[]])
        return documents[0] if documents else []

    def close(self) -> None:
        """Clean up resources."""
        self._collection = None
        self._client = None
