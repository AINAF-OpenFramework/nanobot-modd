#!/usr/bin/env python3
"""
Demonstration of Fractal Memory and ALS system.

This script shows how the new memory architecture works:
1. Creates fractal nodes (lessons)
2. Updates Active Learning State
3. Retrieves relevant nodes based on queries
4. Shows the 6-block context structure
"""

import tempfile
from pathlib import Path

from nanobot.agent.memory import MemoryStore
from nanobot.agent.context import ContextBuilder


def demo_fractal_memory():
    """Demonstrate fractal memory creation and retrieval."""
    print("=" * 60)
    print("FRACTAL MEMORY DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        memory = MemoryStore(workspace)
        
        print("\n1. Creating Fractal Nodes (Lessons)...")
        print("-" * 60)
        
        # Create several lesson nodes
        node1 = memory.save_fractal_node(
            content="Python uses indentation (4 spaces) to define code blocks instead of curly braces.",
            tags=["python", "syntax", "indentation", "coding"],
            summary="Python indentation syntax"
        )
        print(f"✓ Created node: {node1.id[:8]}... [tags: {', '.join(node1.tags)}]")
        
        node2 = memory.save_fractal_node(
            content="Git workflow: Always pull before push to avoid conflicts. Use 'git pull --rebase' for cleaner history.",
            tags=["git", "workflow", "version-control", "best-practices"],
            summary="Git pull-before-push workflow"
        )
        print(f"✓ Created node: {node2.id[:8]}... [tags: {', '.join(node2.tags)}]")
        
        node3 = memory.save_fractal_node(
            content="Docker containers are lightweight, portable environments. Use docker-compose for multi-container apps.",
            tags=["docker", "containers", "devops", "deployment"],
            summary="Docker container basics"
        )
        print(f"✓ Created node: {node3.id[:8]}... [tags: {', '.join(node3.tags)}]")
        
        # Check archive directory
        archive_files = list(memory.archives_dir.glob("lesson_*.json"))
        print(f"\n📁 Archive directory: {len(archive_files)} lesson files created")
        
        # Check index
        import json
        index_data = json.loads(memory.index_file.read_text())
        print(f"📊 Index file: {len(index_data)} entries")
        
        print("\n2. Retrieving Relevant Nodes...")
        print("-" * 60)
        
        # Query 1: Python-related
        print("\nQuery: 'How does Python handle code structure?'")
        results = memory.retrieve_relevant_nodes("python code structure", k=3)
        print(results if results else "  (No relevant nodes found)")
        
        # Query 2: Git-related
        print("\nQuery: 'What's the best Git workflow?'")
        results = memory.retrieve_relevant_nodes("git workflow best", k=3)
        print(results if results else "  (No relevant nodes found)")
        
        # Query 3: Unrelated
        print("\nQuery: 'Tell me about machine learning'")
        results = memory.retrieve_relevant_nodes("machine learning neural networks", k=3)
        print(results if results else "  (No relevant nodes found)")
        
        print("\n3. Active Learning State...")
        print("-" * 60)
        
        # Update ALS
        memory.update_als(
            focus="Learning DevOps and Development Best Practices",
            reflection="User is exploring Python, Git, and Docker workflows",
            evolution_stage=2
        )
        print("✓ Updated ALS")
        
        # Show ALS context
        als_context = memory.get_als_context()
        print("\nALS Context:")
        print(als_context)
        
        print("\n4. Backward Compatibility...")
        print("-" * 60)
        
        # Legacy memory still works
        memory.write_long_term("## User Preferences\n- Prefers Python over JavaScript\n- Uses VS Code")
        memory.append_history("[2024-02-14 10:30] Discussed Python syntax")
        
        print("✓ Legacy MEMORY.md updated")
        print("✓ Legacy HISTORY.md updated")
        print(f"  MEMORY.md: {len(memory.read_long_term())} chars")
        print(f"  HISTORY.md exists: {memory.history_file.exists()}")


def demo_six_block_context():
    """Demonstrate the 6-block context structure."""
    print("\n\n" + "=" * 60)
    print("6-BLOCK CONTEXT STRUCTURE DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        # Setup memory
        memory = MemoryStore(workspace)
        memory.save_fractal_node(
            content="Async/await in Python allows non-blocking I/O operations",
            tags=["python", "async", "concurrency"],
            summary="Python async/await"
        )
        memory.update_als(
            focus="Learning Python async programming",
            evolution_stage=1
        )
        
        # Build context
        context_builder = ContextBuilder(workspace)
        messages = context_builder.build_messages(
            history=[
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level programming language."}
            ],
            current_message="How does async/await work in Python?"
        )
        
        print("\nBuilt message structure:")
        print("-" * 60)
        for i, msg in enumerate(messages):
            role = msg["role"]
            content_preview = str(msg["content"])[:100] + "..." if len(str(msg["content"])) > 100 else str(msg["content"])
            print(f"\nMessage {i+1} [{role.upper()}]:")
            print(f"  {content_preview}")
        
        # Analyze system prompt
        system_content = messages[0]["content"]
        
        print("\n\nSystem Prompt Analysis:")
        print("-" * 60)
        
        blocks_found = []
        if "nanobot" in system_content:
            blocks_found.append("✓ Block 1: System & Persona")
        if "Active Learning State" in system_content or "RESOURCES" in system_content:
            blocks_found.append("✓ Block 2: Resources & ALS")
        if "Skills" in system_content:
            blocks_found.append("✓ Block 3: Tools/Skills")
        if len(messages) > 1:
            blocks_found.append("✓ Block 4: Assistant Messages/History")
        if messages[-1]["role"] == "user":
            blocks_found.append("✓ Block 5: User Message")
        
        for block in blocks_found:
            print(block)
        
        print(f"\nSystem prompt length: {len(system_content)} characters")
        print(f"Total messages: {len(messages)}")
        
        # Check if fractal nodes were retrieved
        if "async" in system_content.lower():
            print("\n✓ Relevant fractal nodes were retrieved based on query!")


if __name__ == "__main__":
    demo_fractal_memory()
    demo_six_block_context()
    
    print("\n\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("  • Fractal node creation and storage")
    print("  • Token-efficient top-K retrieval")
    print("  • Active Learning State (ALS) tracking")
    print("  • 6-block context structure")
    print("  • Backward compatibility with legacy system")
    print("\nNext Steps:")
    print("  • Run 'nanobot agent' to test with real interactions")
    print("  • Check workspace/memory/archives/ for lesson files")
    print("  • Check workspace/memory/ALS.json for learning state")
    print("  • Check workspace/memory/fractal_index.json for index")
