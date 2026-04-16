"""ExpeRepair dual-memory: episodic + semantic failure pattern storage."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pmca.utils.logger import get_logger

log = get_logger("utils.failure_memory")


@dataclass
class FailureEpisode:
    """A single failure episode from a task attempt."""
    task_spec_summary: str
    error_signature: str
    error_types: list[str]
    fix_strategy: str
    fix_description: str
    outcome: str  # "resolved" or "unresolved"
    timestamp: str = ""
    task_title: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_document(self) -> str:
        """Convert to text for ChromaDB embedding."""
        return (
            f"Task: {self.task_title}\n"
            f"Error: {self.error_signature}\n"
            f"Types: {', '.join(self.error_types)}\n"
            f"Strategy: {self.fix_strategy}\n"
            f"Fix: {self.fix_description}\n"
            f"Outcome: {self.outcome}"
        )

    def to_metadata(self) -> dict:
        """Convert to ChromaDB metadata dict."""
        return {
            "task_title": self.task_title,
            "error_signature": self.error_signature[:200],
            "error_types": ",".join(self.error_types),
            "fix_strategy": self.fix_strategy,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
        }


@dataclass
class RepairPattern:
    """A distilled repair pattern from episode clusters."""
    pattern_name: str
    description: str
    recommended_fix: str
    confidence: float
    episode_count: int

    def format_for_prompt(self) -> str:
        """Brief injection text for coder prompts."""
        return (
            f"- Pattern '{self.pattern_name}' ({self.confidence:.0%} confidence, "
            f"{self.episode_count} episodes): {self.recommended_fix}"
        )


class FailureMemoryManager:
    """Dual-memory system: episodic failures + semantic repair patterns.

    Uses two ChromaDB collections following the same patterns as RAGManager.
    Gracefully degrades if chromadb is not installed.
    """

    def __init__(
        self,
        persist_dir: str = ".pmca/failure_memory",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._persist_dir = persist_dir
        self._episodic = None
        self._semantic = None
        self._client = None

        try:
            import chromadb
            from chromadb.config import Settings

            persist_path = Path(persist_dir).expanduser()
            persist_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(anonymized_telemetry=False),
            )

            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )
                embedding_fn = SentenceTransformerEmbeddingFunction(
                    model_name=embedding_model,
                )
            except ImportError:
                embedding_fn = None
                log.warning("sentence-transformers not installed, failure memory disabled")
                return

            self._episodic = self._client.get_or_create_collection(
                name="episodic_failures",
                embedding_function=embedding_fn,
            )
            self._semantic = self._client.get_or_create_collection(
                name="semantic_patterns",
                embedding_function=embedding_fn,
            )
            log.info(f"Failure memory initialized at {persist_path}")
        except ImportError:
            log.warning("chromadb not installed, failure memory disabled")

    @property
    def available(self) -> bool:
        return self._episodic is not None and self._semantic is not None

    def store_episode(self, episode: FailureEpisode) -> None:
        """Store a failure episode in the episodic collection."""
        if not self.available:
            return
        doc_id = f"ep_{episode.timestamp}_{hash(episode.error_signature) & 0xFFFFFFFF:08x}"
        self._episodic.upsert(
            documents=[episode.to_document()],
            ids=[doc_id],
            metadatas=[episode.to_metadata()],
        )

    def query_similar(self, error_text: str, n_results: int = 3) -> list[str]:
        """Query similar past failures, returning brief summaries.

        Each summary is truncated to 150 chars to avoid contextual drag.
        """
        if not self.available:
            return []
        try:
            results = self._episodic.query(
                query_texts=[error_text],
                n_results=n_results,
            )
            docs = results.get("documents", [[]])
            summaries = []
            for doc in (docs[0] if docs and docs[0] else []):
                summary = doc[:150].replace("\n", " ")
                if len(doc) > 150:
                    summary += "..."
                summaries.append(summary)
            return summaries
        except Exception as exc:
            log.warning(f"Failure memory query failed: {exc}")
            return []

    def query_patterns(self, error_text: str, n_results: int = 2) -> list[RepairPattern]:
        """Query semantic patterns relevant to the current error."""
        if not self.available:
            return []
        try:
            results = self._semantic.query(
                query_texts=[error_text],
                n_results=n_results,
            )
            patterns = []
            metadatas = results.get("metadatas", [[]])
            for meta in (metadatas[0] if metadatas and metadatas[0] else []):
                patterns.append(RepairPattern(
                    pattern_name=meta.get("pattern_name", "unknown"),
                    description=meta.get("description", ""),
                    recommended_fix=meta.get("recommended_fix", ""),
                    confidence=float(meta.get("confidence", 0.0)),
                    episode_count=int(meta.get("episode_count", 0)),
                ))
            return patterns
        except Exception as exc:
            log.warning(f"Pattern query failed: {exc}")
            return []

    def distill_patterns(self) -> int:
        """Distill abstract patterns from episode clusters.

        Groups episodes by error_type, finds dominant fix strategy per group,
        and stores as semantic patterns. Fully deterministic — no LLM.

        Returns number of patterns created/updated.
        """
        if not self.available:
            return 0

        try:
            # Fetch all episodes
            all_data = self._episodic.get(include=["metadatas"])
            if not all_data or not all_data.get("metadatas"):
                return 0

            # Group by error type
            by_type: dict[str, list[dict]] = {}
            for meta in all_data["metadatas"]:
                for etype in meta.get("error_types", "").split(","):
                    etype = etype.strip()
                    if etype:
                        by_type.setdefault(etype, []).append(meta)

            count = 0
            for error_type, episodes in by_type.items():
                if len(episodes) < 2:
                    continue  # Need at least 2 episodes to form a pattern

                # Find dominant strategy among resolved episodes
                resolved = [e for e in episodes if e.get("outcome") == "resolved"]
                if not resolved:
                    continue

                strategy_counts = Counter(e.get("fix_strategy", "") for e in resolved)
                dominant_strategy, dominant_count = strategy_counts.most_common(1)[0]
                confidence = dominant_count / len(resolved) if resolved else 0.0

                pattern = RepairPattern(
                    pattern_name=f"{error_type}_fix",
                    description=f"Common fix pattern for {error_type} errors",
                    recommended_fix=f"Use '{dominant_strategy}' strategy",
                    confidence=confidence,
                    episode_count=len(episodes),
                )

                doc_id = f"pat_{error_type}"
                self._semantic.upsert(
                    documents=[pattern.format_for_prompt()],
                    ids=[doc_id],
                    metadatas=[{
                        "pattern_name": pattern.pattern_name,
                        "description": pattern.description,
                        "recommended_fix": pattern.recommended_fix,
                        "confidence": str(pattern.confidence),
                        "episode_count": str(pattern.episode_count),
                    }],
                )
                count += 1

            if count > 0:
                log.info(f"Distilled {count} repair pattern(s) from episodes")
            return count

        except Exception as exc:
            log.warning(f"Pattern distillation failed: {exc}")
            return 0
