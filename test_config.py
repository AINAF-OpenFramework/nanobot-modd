#!/usr/bin/env python3
"""
Test script to validate Nanobot configuration files.

Usage:
    python test_config.py [config_file]

If no config file is provided, tests the example configs.
"""

import json
import sys
from pathlib import Path

try:
    from nanobot.config.loader import load_config
except ImportError:
    print("Error: nanobot package not installed. Run: pip install -e .")
    sys.exit(1)


def test_config(config_path: Path) -> bool:
    """Test if a config file is valid."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_path}")
    print('='*60)
    
    if not config_path.exists():
        print(f"✗ File not found: {config_path}")
        return False
    
    try:
        config = load_config(config_path)
        print("✓ Config is valid!")
        print(f"\nConfiguration Summary:")
        print(f"  Model: {config.agents.defaults.model}")
        print(f"  Workspace: {config.agents.defaults.workspace}")
        print(f"  Max Tokens: {config.agents.defaults.max_tokens}")
        print(f"  Temperature: {config.agents.defaults.temperature}")
        
        print(f"\nMemory Configuration:")
        print(f"  Enabled: {config.memory.enabled}")
        print(f"  Provider: {config.memory.provider}")
        print(f"  Top K: {config.memory.top_k}")
        print(f"  ALS Enabled: {config.memory.als_enabled}")
        
        print(f"\nConfigured Providers:")
        providers_with_keys = []
        for name in ['openai', 'gemini', 'anthropic', 'deepseek', 'groq', 
                     'openrouter', 'zhipu', 'dashscope', 'moonshot', 'minimax', 
                     'vllm', 'aihubmix', 'custom']:
            provider = getattr(config.providers, name, None)
            if provider and provider.api_key:
                providers_with_keys.append(name)
                # Mask the API key
                key = provider.api_key
                if len(key) > 8:
                    masked = key[:4] + '...' + key[-4:]
                else:
                    masked = '***'
                print(f"  ✓ {name}: {masked}")
        
        if not providers_with_keys:
            print("  ⚠ No providers configured with API keys")
        
        print(f"\nTools Configuration:")
        print(f"  Web Search: {'✓' if config.tools.web.search.api_key else '✗'}")
        print(f"  Exec Timeout: {config.tools.exec.timeout}s")
        print(f"  Restrict to Workspace: {config.tools.restrict_to_workspace}")
        
        print(f"\nEnabled Channels:")
        channels = []
        for name in ['telegram', 'whatsapp', 'discord', 'feishu', 'mochat', 
                     'dingtalk', 'email', 'slack', 'qq']:
            channel = getattr(config.channels, name, None)
            if channel and channel.enabled:
                channels.append(name)
        if channels:
            for ch in channels:
                print(f"  ✓ {ch}")
        else:
            print("  None (CLI mode only)")
        
        return True
        
    except Exception as e:
        print(f"✗ Config validation failed!")
        print(f"\nError: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    if len(sys.argv) > 1:
        # Test user-provided config
        config_path = Path(sys.argv[1])
        success = test_config(config_path)
        sys.exit(0 if success else 1)
    else:
        # Test example configs
        script_dir = Path(__file__).parent
        configs = [
            script_dir / "config.minimal.json",
            script_dir / "config.example.json",
        ]
        
        # Also test the default location if it exists
        default_config = Path.home() / ".nanobot" / "config.json"
        if default_config.exists():
            configs.append(default_config)
        
        results = []
        for config_path in configs:
            results.append(test_config(config_path))
        
        print(f"\n{'='*60}")
        print("Summary")
        print('='*60)
        for config_path, result in zip(configs, results):
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {config_path.name}")
        
        all_passed = all(results)
        print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
