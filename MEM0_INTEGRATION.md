# mem0 Integration Guide

This document explains how to use mem0 integration in nanobot for advanced memory management with vector embeddings, multi-modal content, and hierarchical relationships.

## Overview

The mem0 integration provides:
- **Vector embeddings** for semantic search using mem0's AI-powered memory engine
- **Multi-modal memory** supporting text, code snippets, and images
- **Hierarchical relationships** for organizing knowledge in tree structures
- **Hybrid search** combining keyword and semantic similarity
- **Backward compatibility** with the existing fractal memory system

## Installation

mem0 is already included as a dependency. If you need to install it separately:

```bash
pip install mem0ai
```

## Configuration

Add mem0 configuration to your `~/.nanobot/config.json`:

```json
{
  "memory": {
    "enabled": true,
    "provider": "mem0",
    "top_k": 5,
    "mem0_api_key": "",
    "mem0_user_id": "my_user",
    "mem0_org_id": "",
    "mem0_project_id": "",
    "embedding_model": "text-embedding-3-small",
    "embedding_dim": 1536,
    "use_hybrid_search": true
  }
}
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `provider` | `"local"` | Use `"mem0"` to enable mem0 integration |
| `top_k` | `5` | Number of most relevant memories to retrieve |
| `mem0_api_key` | `""` | API key for mem0 cloud (optional for local use) |
| `mem0_user_id` | `"nanobot_user"` | User ID for memory isolation |
| `mem0_org_id` | `""` | Organization ID (optional) |
| `mem0_project_id` | `""` | Project ID (optional) |
| `embedding_model` | `"text-embedding-3-small"` | Embedding model to use |
| `embedding_dim` | `1536` | Dimension of embeddings |
| `use_hybrid_search` | `true` | Combine keyword and vector search |

## Basic Usage

### Automatic Memory Management

The easiest way to use mem0 is to let nanobot manage memories automatically:

```bash
nanobot agent -m "Remember: I prefer Python over JavaScript"
```

Nanobot will automatically:
1. Create a memory node with the content
2. Generate vector embeddings using mem0
3. Store it for future retrieval

Later:

```bash
nanobot agent -m "What programming language do I prefer?"
```

Nanobot will:
1. Use mem0 to search for relevant memories
2. Find the Python preference memory
3. Use it to answer your question

### Manual Memory Management

You can also create and manage memories programmatically:

```python
from pathlib import Path
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import ContentType

# Initialize memory store with your workspace path
# Note: Update this path to match your actual nanobot workspace
# Default is typically ~/.nanobot/workspace
workspace = Path.home() / ".nanobot" / "workspace"
config = {
    "provider": "mem0",
    "mem0_user_id": "my_user",
    "top_k": 5,
}
memory = MemoryStore(workspace, config=config)

# Save a text memory
node = memory.save_fractal_node(
    content="Python uses duck typing",
    tags=["python", "typing", "dynamic"],
    summary="Python typing system",
    content_type=ContentType.TEXT,
)

# Save a code snippet
code_node = memory.save_code_snippet(
    code="def greet(name):\n    return f'Hello, {name}!'",
    language="python",
    tags=["python", "function"],
    summary="Greeting function",
)

# Retrieve relevant memories
results = memory.retrieve_relevant_nodes("python typing", k=5)
print(results)
```

## Multi-Modal Memory

### Text Memories

Standard text memories are the default:

```python
memory.save_fractal_node(
    content="Machine learning is a subset of AI",
    tags=["ml", "ai", "concepts"],
    summary="ML definition",
    content_type=ContentType.TEXT,
)
```

### Code Snippets

Save code with syntax highlighting metadata:

```python
memory.save_code_snippet(
    code="""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
    language="python",
    tags=["python", "recursion", "math"],
    summary="Factorial function",
)
```

The code will be stored with:
- Language metadata (for syntax highlighting)
- Code-specific tags
- Full code content searchable via vector embeddings

### Images

Save images with descriptions:

```python
from pathlib import Path

memory.save_image(
    image_path=Path("screenshot.png"),
    tags=["ui", "design", "screenshot"],
    summary="Dashboard UI",
    description="User dashboard with metrics",
)
```

Images are:
- Base64-encoded for storage
- Linked with text descriptions for searchability
- Retrievable with full binary data

Retrieve image data:

```python
image_data, mime_type = memory.get_image_data(node_id)
# image_data is bytes, mime_type is like "image/png"

# Save to file
with open("retrieved.png", "wb") as f:
    f.write(image_data)
```

## Hierarchical Relationships

Organize memories in tree structures:

```python
# Create root node
root = memory.save_fractal_node(
    content="Web Development",
    tags=["web", "development"],
    summary="Web dev root",
)

# Create child nodes
frontend = memory.save_fractal_node(
    content="Frontend development with React",
    tags=["web", "frontend", "react"],
    summary="Frontend",
    parent_id=root.id,
)

backend = memory.save_fractal_node(
    content="Backend development with FastAPI",
    tags=["web", "backend", "fastapi"],
    summary="Backend",
    parent_id=root.id,
)

# Add grandchild
react_hooks = memory.save_fractal_node(
    content="React Hooks API",
    tags=["react", "hooks", "api"],
    summary="React Hooks",
    parent_id=frontend.id,
)
```

### Navigating Hierarchies

```python
# Get all children of a node
children = memory.get_children(root.id)

# Get parent of a node
parent = memory.get_parent(react_hooks.id)

# Get full hierarchy tree
tree = memory.get_hierarchy_tree(root.id, max_depth=3)
# Returns:
# {
#   "id": "root-id",
#   "summary": "Web dev root",
#   "tags": ["web", "development"],
#   "children": [
#     {
#       "id": "frontend-id",
#       "summary": "Frontend",
#       "children": [
#         {"id": "hooks-id", "summary": "React Hooks", "children": []}
#       ]
#     },
#     {"id": "backend-id", "summary": "Backend", "children": []}
#   ]
# }
```

## Vector Search

mem0 automatically generates vector embeddings for all content, enabling semantic search:

```python
# These queries will find relevant content even without exact keyword matches:

# Semantic similarity
results = memory.retrieve_relevant_nodes("programming language features")
# Finds: "Python uses duck typing", "Python functions", etc.

results = memory.retrieve_relevant_nodes("photo of user interface")
# Finds: Image memories with "ui", "dashboard" descriptions

results = memory.retrieve_relevant_nodes("recursive algorithms")
# Finds: Code snippets with recursion
```

### Hybrid Search

When `use_hybrid_search: true` is set, nanobot combines:
1. **Keyword matching** (fast, exact matches)
2. **Vector similarity** (semantic, finds related concepts)

This provides the best of both worlds:
- Fast retrieval of exact matches
- Discovery of semantically related content

## Filtering and Search

### Search by Content Type

```python
# Find all code snippets
code_nodes = memory.search_by_type(ContentType.CODE)

# Find Python code
python_code = memory.search_by_type(
    ContentType.CODE,
    tags=["python"],
    limit=10,
)

# Find all images
images = memory.search_by_type(ContentType.IMAGE)
```

### Retrieve Specific Node

```python
node = memory.get_node_by_id("node-uuid-here")
if node:
    print(f"Content: {node.content}")
    print(f"Type: {node.content_type}")
    print(f"Tags: {node.tags}")
```

## Migration from Local Storage

The system automatically falls back to local storage if mem0 is not available. To migrate:

1. **Update configuration** to use mem0 provider
2. **Existing local nodes** will continue to work
3. **New nodes** will use mem0 with embeddings
4. **Searches** will use mem0's semantic search

No manual data migration is needed - the system maintains backward compatibility.

## Performance Considerations

### Local vs. Cloud

- **Local mem0**: Runs embeddings locally, slower but private
- **Cloud mem0**: Uses cloud API, faster but requires API key

### Memory Limits

- mem0 can handle millions of memories efficiently
- Local storage is recommended for small datasets (<10k memories)
- Cloud mem0 is recommended for large datasets (>10k memories)

### Embedding Dimensions

- Smaller dimensions (384, 768) = faster, less accurate
- Larger dimensions (1536, 3072) = slower, more accurate
- Default (1536) provides good balance

## Troubleshooting

### mem0 Not Available

If you see "mem0 provider not available" warnings:

```bash
pip install mem0ai
```

Then restart nanobot.

### Embeddings Taking Long

If embedding generation is slow:

1. Use a smaller embedding model
2. Reduce `embedding_dim` in config
3. Consider using cloud mem0 API

### No Results Found

If searches return no results:

1. Check that memories have relevant tags
2. Try both keyword and semantic queries
3. Increase `top_k` value
4. Verify memories were saved successfully

## API Reference

See the main API reference in `FRACTAL_MEMORY.md` for:
- `MemoryStore` class methods
- `FractalNode` data structure
- Configuration options
- Full API documentation

## Examples

See `tests/test_mem0_integration.py` for comprehensive examples of:
- Multi-modal memory usage
- Hierarchical relationships
- Vector search
- Hybrid search
- Error handling

## Next Steps

- Explore the demo: `python demo_fractal_memory.py`
- Read the tests: `tests/test_mem0_integration.py`
- Check the main docs: `FRACTAL_MEMORY.md`
