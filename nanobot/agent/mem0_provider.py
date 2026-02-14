"""mem0 integration for advanced memory management with vector embeddings."""

import json
import logging
from pathlib import Path
from typing import Any

from nanobot.agent.memory_types import FractalNode, ContentType

logger = logging.getLogger(__name__)


class Mem0Provider:
    """
    Provider for mem0 integration with vector embeddings and semantic search.
    
    This provider wraps mem0's Memory class to provide:
    - Vector embeddings for all stored content
    - Semantic search using cosine similarity
    - Multi-modal memory support
    - Hierarchical relationships
    """
    
    def __init__(
        self,
        config: dict[str, Any],
        workspace: Path,
    ):
        """
        Initialize mem0 provider.
        
        Args:
            config: Memory configuration with mem0 settings
            workspace: Workspace directory path
        """
        self.config = config
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.archives_dir = self.memory_dir / "archives"
        self.archives_dir.mkdir(parents=True, exist_ok=True)
        
        self._mem0_client = None
        self._init_mem0()
    
    def _init_mem0(self) -> None:
        """Initialize mem0 client."""
        try:
            from mem0 import Memory
            
            # Configure mem0 based on settings
            # Note: Using mem0 v1.1 API. Update this version string when upgrading mem0.
            mem0_config = {
                "version": self.config.get("mem0_version", "v1.1"),
            }
            
            # Add API key if provided (for cloud)
            api_key = self.config.get("mem0_api_key", "")
            if api_key:
                mem0_config["api_key"] = api_key
            
            # Add organization/project IDs if provided
            if self.config.get("mem0_org_id"):
                mem0_config["org_id"] = self.config["mem0_org_id"]
            if self.config.get("mem0_project_id"):
                mem0_config["project_id"] = self.config["mem0_project_id"]
            
            self._mem0_client = Memory.from_config(mem0_config)
            logger.info("mem0 provider initialized successfully")
            
        except ImportError:
            logger.warning("mem0 package not available, falling back to local storage")
            self._mem0_client = None
        except Exception as e:
            logger.error(f"Failed to initialize mem0: {e}")
            self._mem0_client = None
    
    def is_available(self) -> bool:
        """Check if mem0 is available."""
        return self._mem0_client is not None
    
    def add_memory(
        self,
        content: str,
        tags: list[str],
        summary: str,
        content_type: ContentType = ContentType.TEXT,
        metadata: dict[str, Any] | None = None,
    ) -> FractalNode:
        """
        Add a memory with vector embeddings.
        
        Args:
            content: The memory content
            tags: Tags for categorization
            summary: Brief summary
            content_type: Type of content
            metadata: Additional metadata
            
        Returns:
            The created FractalNode
        """
        # Create the fractal node
        node = FractalNode(
            content=content,
            tags=tags,
            context_summary=summary,
            content_type=content_type,
        )
        
        # Add to mem0 if available
        if self._mem0_client:
            try:
                mem0_metadata = {
                    "node_id": node.id,
                    "tags": tags,
                    "summary": summary,
                    "content_type": content_type.value,
                    "timestamp": node.timestamp.isoformat(),
                }
                if metadata:
                    mem0_metadata.update(metadata)
                
                # Add memory to mem0
                user_id = self.config.get("mem0_user_id", "nanobot_user")
                result = self._mem0_client.add(
                    messages=content,
                    user_id=user_id,
                    metadata=mem0_metadata,
                )
                
                # Store the mem0 memory ID
                if result and "id" in result:
                    node.embedding = result.get("embedding", [])
                
                logger.info(f"Added memory to mem0: {node.id}")
                
            except Exception as e:
                logger.error(f"Failed to add memory to mem0: {e}")
        
        # Save locally as backup
        archive_path = self.archives_dir / f"lesson_{node.id}.json"
        archive_path.write_text(node.model_dump_json(indent=2), encoding="utf-8")
        
        return node
    
    def search_memories(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[FractalNode]:
        """
        Search memories using vector similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Optional filters (e.g., tags, content_type)
            
        Returns:
            List of relevant FractalNodes
        """
        if not self._mem0_client:
            # Fallback to local keyword search
            return self._local_keyword_search(query, k, filters)
        
        try:
            user_id = self.config.get("mem0_user_id", "nanobot_user")
            
            # Search using mem0's semantic search
            results = self._mem0_client.search(
                query=query,
                user_id=user_id,
                limit=k,
            )
            
            # Convert mem0 results to FractalNodes
            nodes = []
            for result in results:
                metadata = result.get("metadata", {})
                node_id = metadata.get("node_id")
                
                # Try to load the full node from local storage
                if node_id:
                    archive_path = self.archives_dir / f"lesson_{node_id}.json"
                    if archive_path.exists():
                        try:
                            node = FractalNode.model_validate_json(
                                archive_path.read_text(encoding="utf-8")
                            )
                            nodes.append(node)
                        except Exception as e:
                            logger.warning(f"Failed to load node {node_id}: {e}")
            
            return nodes
            
        except Exception as e:
            logger.error(f"mem0 search failed: {e}")
            return self._local_keyword_search(query, k, filters)
    
    def _local_keyword_search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[FractalNode]:
        """
        Fallback keyword-based search when mem0 is not available.
        
        Args:
            query: Search query
            k: Number of results
            filters: Optional filters
            
        Returns:
            List of FractalNodes
        """
        query_words = set(query.lower().split())
        scored_nodes = []
        
        # Scan all archive files
        for archive_path in self.archives_dir.glob("lesson_*.json"):
            try:
                node = FractalNode.model_validate_json(
                    archive_path.read_text(encoding="utf-8")
                )
                
                # Apply filters
                if filters:
                    if "content_type" in filters:
                        if node.content_type != filters["content_type"]:
                            continue
                    if "tags" in filters:
                        if not any(tag in node.tags for tag in filters["tags"]):
                            continue
                
                # Score based on keyword matches
                score = 0
                for tag in node.tags:
                    if tag.lower() in query_words:
                        score += 5
                
                summary_words = set(node.context_summary.lower().split())
                score += len(query_words.intersection(summary_words))
                
                content_words = set(node.content.lower().split())
                score += len(query_words.intersection(content_words)) * 0.5
                
                if score > 0:
                    scored_nodes.append((score, node))
                    
            except Exception as e:
                logger.warning(f"Failed to load node from {archive_path}: {e}")
        
        # Sort by score and return top k
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored_nodes[:k]]
    
    def get_memory_by_id(self, node_id: str) -> FractalNode | None:
        """
        Retrieve a specific memory by ID.
        
        Args:
            node_id: The node ID
            
        Returns:
            The FractalNode or None if not found
        """
        archive_path = self.archives_dir / f"lesson_{node_id}.json"
        if archive_path.exists():
            try:
                return FractalNode.model_validate_json(
                    archive_path.read_text(encoding="utf-8")
                )
            except Exception as e:
                logger.error(f"Failed to load node {node_id}: {e}")
        return None
    
    def update_memory(self, node: FractalNode) -> bool:
        """
        Update an existing memory.
        
        Args:
            node: The updated FractalNode
            
        Returns:
            True if successful
        """
        try:
            # Update local storage
            archive_path = self.archives_dir / f"lesson_{node.id}.json"
            archive_path.write_text(node.model_dump_json(indent=2), encoding="utf-8")
            
            # Update mem0 if available
            if self._mem0_client:
                try:
                    user_id = self.config.get("mem0_user_id", "nanobot_user")
                    # mem0's update method
                    self._mem0_client.update(
                        memory_id=node.id,
                        data=node.content,
                        user_id=user_id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update mem0: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            return False
    
    def delete_memory(self, node_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            node_id: The node ID to delete
            
        Returns:
            True if successful
        """
        try:
            # Delete from local storage
            archive_path = self.archives_dir / f"lesson_{node_id}.json"
            if archive_path.exists():
                archive_path.unlink()
            
            # Delete from mem0 if available
            if self._mem0_client:
                try:
                    self._mem0_client.delete(memory_id=node_id)
                except Exception as e:
                    logger.warning(f"Failed to delete from mem0: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False
    
    def get_all_memories(self, filters: dict[str, Any] | None = None) -> list[FractalNode]:
        """
        Get all memories with optional filters.
        
        Args:
            filters: Optional filters
            
        Returns:
            List of FractalNodes
        """
        nodes = []
        
        for archive_path in self.archives_dir.glob("lesson_*.json"):
            try:
                node = FractalNode.model_validate_json(
                    archive_path.read_text(encoding="utf-8")
                )
                
                # Apply filters
                if filters:
                    if "content_type" in filters:
                        if node.content_type != filters["content_type"]:
                            continue
                    if "tags" in filters:
                        if not any(tag in node.tags for tag in filters["tags"]):
                            continue
                
                nodes.append(node)
                
            except Exception as e:
                logger.warning(f"Failed to load node from {archive_path}: {e}")
        
        return nodes
