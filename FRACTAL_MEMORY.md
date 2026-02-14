# Fractal Memory and Active Learning State (ALS)

This document describes the upgraded memory system for Nanobot, which implements Fractal Memory architecture with Active Learning State and context-engineered workflows.

## Overview

The new memory system extends Nanobot's capabilities with:

1. **Fractal Memory Nodes**: Hierarchical knowledge storage with lesson archives
2. **Active Learning State (ALS)**: Tracks the agent's learning evolution and focus
3. **Token-Efficient Retrieval**: Top-K search for relevant memories
4. **6-Block Context Structure**: Structured prompt engineering for better LLM performance
5. **Backward Compatibility**: Maintains support for legacy MEMORY.md and HISTORY.md

## Architecture

### Directory Structure

```
workspace/
└── memory/
    ├── MEMORY.md              # Legacy long-term memory (still supported)
    ├── HISTORY.md             # Legacy history log (still supported)
    ├── ALS.json               # Active Learning State
    ├── fractal_index.json     # Lightweight index for fast retrieval
    └── archives/              # Fractal node storage
        ├── lesson_<uuid1>.json
        ├── lesson_<uuid2>.json
        └── ...
```

### Components

#### 1. FractalNode (memory_types.py)

A Pydantic model representing a knowledge node:

```python
class FractalNode(BaseModel):
    id: str                    # Unique identifier
    timestamp: datetime        # Creation time
    tags: list[str]           # Keywords for retrieval
    content: str              # The lesson/fact content
    context_summary: str      # Brief summary
    embedding: list[float]    # Future: vector embeddings
```

#### 2. ActiveLearningState (memory_types.py)

Tracks the agent's evolution:

```python
class ActiveLearningState(BaseModel):
    current_focus: str              # Current learning focus
    sparring_partners: list[str]    # User IDs or personas
    evolution_stage: int            # Learning stage
    recent_reflections: list[str]   # Recent insights
    last_updated: datetime          # Last update time
```

#### 3. MemoryStore (memory.py)

Enhanced memory manager with new methods:

- `save_fractal_node(content, tags, summary)`: Creates and stores a lesson
- `retrieve_relevant_nodes(query, k)`: Retrieves top-K relevant nodes
- `get_als_context()`: Returns formatted ALS for prompts
- `update_als(focus, reflection, stage)`: Updates the learning state

#### 4. ContextBuilder (context.py)

Implements the 6-block context structure:

1. **System & Persona**: Core identity and capabilities
2. **Resources & ALS**: Dynamic memory retrieval (ALS + Fractal nodes + Core memory)
3. **Tools/Skills**: Available capabilities
4. **Assistant Messages**: Conversation history
5. **User Message**: Current input
6. **Tool Calls**: Execution results (handled by LLM)

#### 5. AgentLoop (loop.py)

Adds reflection hooks:

- `_trigger_reflection()`: Automatically captures important interactions
- Triggers based on keywords ("remember", "important") or tool usage
- Creates fractal nodes and updates ALS

## Configuration

Add to `config.yaml`:

```yaml
memory:
  enabled: true
  provider: local           # or 'mem0', 'openai' (future)
  top_k: 5                 # Number of nodes to retrieve
  archive_dir: archives
  als_enabled: true
```

## Usage

### Creating Fractal Nodes

Nodes are created automatically when:
- User says keywords like "remember", "important", "note"
- Significant tools are used (write_file, exec, spawn)
- You can also create manually:

```python
from nanobot.agent.memory import MemoryStore
from pathlib import Path

memory = MemoryStore(Path("workspace"))
node = memory.save_fractal_node(
    content="Python uses 4 spaces for indentation",
    tags=["python", "syntax", "indentation"],
    summary="Python indentation rule"
)
```

### Retrieving Nodes

Nodes are automatically retrieved when building context based on the user's query:

```python
# Automatic in context building
results = memory.retrieve_relevant_nodes("python syntax", k=5)
```

### Updating ALS

The Active Learning State is updated automatically during reflection:

```python
memory.update_als(
    focus="Learning Python programming",
    reflection="User completed first Python script",
    evolution_stage=2
)
```

### Accessing in Prompts

The 6-block context automatically includes:

```
# RESOURCES & MEMORY

## Active Learning State
Current Focus: Learning Python programming
Evolution Stage: 2

## Long-term Memory
(Content from MEMORY.md)

## Relevant Resources (Fractal Memory)
- **[2024-02-14] python, syntax**: Python uses 4 spaces...
- **[2024-02-13] python, functions**: Functions are defined with def...
```

## API Reference

### MemoryStore

#### `save_fractal_node(content: str, tags: list[str], summary: str) -> FractalNode`
Creates and stores a new lesson node.

**Args:**
- `content`: The lesson/fact content
- `tags`: List of keywords for retrieval
- `summary`: Brief summary (shown in index)

**Returns:** The created FractalNode

#### `retrieve_relevant_nodes(query: str, k: int = 5) -> str`
Retrieves top-K relevant nodes for a query.

**Args:**
- `query`: Search query
- `k`: Number of results

**Returns:** Formatted string with relevant nodes

#### `get_als_context() -> str`
Returns formatted Active Learning State.

**Returns:** ALS summary for prompts

#### `update_als(focus, reflection, evolution_stage)`
Updates the Active Learning State.

**Args:**
- `focus`: New focus area (optional)
- `reflection`: New reflection (optional)
- `evolution_stage`: New stage (optional)

### ContextBuilder

#### `build_messages(history, current_message, ...) -> list[dict]`
Builds message list with 6-block context structure.

**Returns:** List of messages including system prompt with context

## Migration from Legacy System

The new system is **fully backward compatible**:

1. Existing MEMORY.md continues to work
2. Existing HISTORY.md continues to work
3. All legacy methods (`read_long_term()`, `write_long_term()`, `append_history()`) still function
4. New features are additive, not replacing

## Testing

Run the test suite:

```bash
# Unit tests
python tests/test_fractal_memory.py

# Integration tests
python tests/test_context_integration.py

# Demonstration
python demo_fractal_memory.py
```

## Future Enhancements

1. **Vector Embeddings**: Replace keyword matching with semantic search
2. **mem0 Integration**: Use mem0 for advanced memory management
3. **Hierarchical Nodes**: Implement parent-child relationships
4. **Memory Consolidation**: Automatic merging of related nodes
5. **Importance Decay**: Time-based importance scoring

## Implementation Checklist

- [x] Create memory_types.py with Pydantic models
- [x] Update config/schema.py with MemoryConfig
- [x] Implement fractal node storage and retrieval
- [x] Implement Active Learning State
- [x] Update ContextBuilder with 6-block structure
- [x] Add reflection hooks in AgentLoop
- [x] Create comprehensive tests
- [x] Maintain backward compatibility
- [x] Documentation

## References

- **Fractal Memory**: Hierarchical knowledge representation
- **Active Learning**: Continuous learning from interactions
- **Context Engineering**: Structured prompt design for better LLM performance
