"""Memory system for persistent agent memory."""

import json
import logging
from pathlib import Path

from nanobot.agent.memory_types import ActiveLearningState, FractalNode
from nanobot.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Upgraded memory system with Fractal Memory and Active Learning State.
    
    Maintains backward compatibility with legacy MEMORY.md and HISTORY.md,
    while adding new capabilities:
    - Fractal nodes stored as lesson_X.json
    - Lightweight fractal_index.json for fast retrieval
    - Active Learning State in ALS.json
    - Token-efficient top-K retrieval
    """

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        
        # Legacy files (preserved for backward compatibility)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        
        # --- NEW: Fractal Architecture ---
        self.archives_dir = ensure_dir(self.memory_dir / "archives")
        self.index_file = self.memory_dir / "fractal_index.json"
        self.als_file = self.memory_dir / "ALS.json"
        
        self._init_fractal_structure()
    
    def _init_fractal_structure(self) -> None:
        """Initialize index and ALS if they don't exist."""
        if not self.index_file.exists():
            self.index_file.write_text(json.dumps([], indent=2))
        if not self.als_file.exists():
            default_als = ActiveLearningState().model_dump_json(indent=2)
            self.als_file.write_text(default_als)

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""
    
    # --- NEW METHODS for Fractal Memory ---
    
    def save_fractal_node(self, content: str, tags: list[str], summary: str) -> FractalNode:
        """
        Creates a lesson archive and updates the index.
        
        Args:
            content: The core lesson/fact content
            tags: List of tags for categorization and retrieval
            summary: Brief summary for the index
        
        Returns:
            The created FractalNode
        """
        node = FractalNode(content=content, tags=tags, context_summary=summary)
        
        # 1. Save full archive (Lesson)
        archive_path = self.archives_dir / f"lesson_{node.id}.json"
        archive_path.write_text(node.model_dump_json(indent=2), encoding="utf-8")
        
        # 2. Update Index (Lightweight metadata only)
        try:
            index_data = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception:
            index_data = []
        
        # Only store metadata in the index, not the full content
        index_entry = {
            "id": node.id,
            "tags": node.tags,
            "summary": node.context_summary,
            "timestamp": node.timestamp.isoformat()
        }
        index_data.append(index_entry)
        self.index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
        
        logger.info(f"Saved Fractal Node: {node.id} with tags: {tags}")
        return node
    
    def retrieve_relevant_nodes(self, query: str, k: int = 5) -> str:
        """
        Token-efficient retrieval using simple keyword matching.
        
        This is a placeholder implementation. Future versions can integrate
        vector search with mem0 or OpenAI embeddings.
        
        Args:
            query: Search query
            k: Number of top results to return
        
        Returns:
            Formatted string with relevant memory nodes
        """
        try:
            index_data = json.loads(self.index_file.read_text(encoding="utf-8"))
            
            # Simple scoring: count overlapping words between query and tags/summary
            # TODO: Replace with vector similarity search
            query_words = set(query.lower().split())
            
            scored_nodes = []
            for entry in index_data:
                score = 0
                # Check tags (higher weight)
                for tag in entry.get("tags", []):
                    if tag.lower() in query_words:
                        score += 5
                # Check summary (lower weight)
                summary_words = set(entry.get("summary", "").lower().split())
                score += len(query_words.intersection(summary_words))
                
                scored_nodes.append((score, entry))
            
            # Sort by score descending and take top K
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            top_nodes = scored_nodes[:k]
            
            # Load full content for top K nodes
            context_str = "## Relevant Resources (Fractal Memory)\n"
            for score, entry in top_nodes:
                if score == 0:
                    continue  # Skip irrelevant nodes
                
                archive_path = self.archives_dir / f"lesson_{entry['id']}.json"
                if archive_path.exists():
                    try:
                        node = FractalNode.model_validate_json(
                            archive_path.read_text(encoding="utf-8")
                        )
                        timestamp_str = node.timestamp.strftime('%Y-%m-%d')
                        tags_str = ', '.join(node.tags)
                        context_str += f"- **[{timestamp_str}] {tags_str}**: {node.content}\n"
                    except Exception as e:
                        logger.warning(f"Could not load node {entry['id']}: {e}")
            
            return context_str if "**[" in context_str else ""
            
        except Exception as e:
            logger.error(f"Error retrieving nodes: {e}")
            return ""
    
    def get_als_context(self) -> str:
        """
        Returns the Active Learning State summary for context.
        
        Returns:
            Formatted ALS summary
        """
        if self.als_file.exists():
            try:
                als = ActiveLearningState.model_validate_json(
                    self.als_file.read_text(encoding="utf-8")
                )
                return (
                    f"## Active Learning State\n"
                    f"Current Focus: {als.current_focus}\n"
                    f"Evolution Stage: {als.evolution_stage}\n"
                )
            except Exception as e:
                logger.warning(f"Could not load ALS: {e}")
        return ""
    
    def update_als(
        self,
        focus: str | None = None,
        reflection: str | None = None,
        evolution_stage: int | None = None
    ) -> None:
        """
        Update the Active Learning State.
        
        Args:
            focus: New focus area (optional)
            reflection: New reflection to add (optional)
            evolution_stage: New evolution stage (optional)
        """
        try:
            # Load existing ALS
            if self.als_file.exists():
                als = ActiveLearningState.model_validate_json(
                    self.als_file.read_text(encoding="utf-8")
                )
            else:
                als = ActiveLearningState()
            
            # Update fields
            if focus:
                als.current_focus = focus
            if reflection:
                als.recent_reflections.append(reflection)
                # Keep only last 10 reflections
                als.recent_reflections = als.recent_reflections[-10:]
            if evolution_stage is not None:
                als.evolution_stage = evolution_stage
            
            from datetime import datetime
            als.last_updated = datetime.now()
            
            # Save
            self.als_file.write_text(als.model_dump_json(indent=2), encoding="utf-8")
            logger.info("Updated Active Learning State")
            
        except Exception as e:
            logger.error(f"Error updating ALS: {e}")
