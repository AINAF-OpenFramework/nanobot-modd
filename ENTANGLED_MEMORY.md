# Entangled Memory

## 1. Concept
- Memory retrieval combines semantic matching with graph entanglement links between nodes.

## 2. Hybrid Scoring
- Final score uses weighted components:
  - semantic relevance
  - entanglement strength
  - (optional) decayed importance

## 3. Cycle Handling
- Retrieval expands through entanglement links using a visited set.
- Nodes already seen are skipped to prevent loops and duplicates.
- Expansion uses a bounded BFS (`MAX_ENTANGLEMENT_HOPS`) to prevent deep-chain blowups.

## 4. FractalNode shape
- `id`, `content`, `tags`, `embedding`
- `entangled_ids: {node_id: strength}`
- `importance` for optional long-term weighting
