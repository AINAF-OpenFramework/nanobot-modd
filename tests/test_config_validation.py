"""
Test configuration validation for the memory field issue.

This test demonstrates the problem described in the issue:
- Old configs with extra fields in 'memory' section cause Pydantic validation errors
- New example configs avoid these errors by only using valid schema fields
"""

import json
from pathlib import Path

from nanobot.config.loader import convert_keys
from nanobot.config.schema import Config, MemoryConfig


def test_memory_config_ignores_extra_fields():
    """Test that MemoryConfig ignores extra/unknown fields (Pydantic V2 default behavior)."""
    # By default, Pydantic V2 ignores extra fields rather than raising errors
    memory_data_with_extra = {
        "enabled": True,
        "provider": "local",
        "memory_type": "fractal",  # Extra field - will be ignored
        "vector_store": "chromadb",  # Extra field - will be ignored
    }

    # This should succeed - extra fields are simply ignored
    memory_config = MemoryConfig(**memory_data_with_extra)
    assert memory_config.enabled is True
    assert memory_config.provider == "local"

    # Extra fields should not be in the model
    assert not hasattr(memory_config, 'memory_type')
    assert not hasattr(memory_config, 'vector_store')


def test_memory_config_accepts_valid_fields():
    """Test that MemoryConfig accepts all valid fields from the schema."""
    valid_memory_data = {
        "enabled": True,
        "provider": "local",
        "top_k": 5,
        "archive_dir": "archives",
        "als_enabled": True,
        "mem0_api_key": "",
        "mem0_user_id": "test_user",
        "mem0_org_id": "",
        "mem0_project_id": "",
        "mem0_version": "v1.1",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
        "use_hybrid_search": True,
        "latent_retry_attempts": 3,
        "latent_retry_min_wait": 1.0,
        "latent_retry_max_wait": 5.0,
        "latent_retry_multiplier": 1.0,
    }

    # This should succeed
    memory_config = MemoryConfig(**valid_memory_data)
    assert memory_config.enabled is True
    assert memory_config.provider == "local"
    assert memory_config.top_k == 5
    assert memory_config.latent_retry_attempts == 3


def test_example_config_validates():
    """Test that config.example.json validates successfully."""
    example_path = Path(__file__).parent.parent / "config.example.json"

    # Load and validate
    with open(example_path) as f:
        data = json.load(f)

    # Convert camelCase to snake_case and validate
    config = Config.model_validate(convert_keys(data))

    # Verify it's configured as expected
    assert config.memory.enabled is True
    assert config.memory.provider == "local"
    assert config.memory.top_k == 5
    assert config.memory.als_enabled is True

    # Verify providers are set up
    assert config.providers.openai.api_key != ""
    assert config.providers.gemini.api_key != ""


def test_minimal_config_validates():
    """Test that config.minimal.json validates successfully."""
    minimal_path = Path(__file__).parent.parent / "config.minimal.json"

    # Load and validate
    with open(minimal_path) as f:
        data = json.load(f)

    # Convert camelCase to snake_case and validate
    config = Config.model_validate(convert_keys(data))

    # Verify defaults are applied
    assert config.memory.enabled is True  # Default value
    assert config.memory.provider == "local"  # Default value
    assert config.agents.defaults.workspace == "~/.nanobot/workspace"  # Default value


def test_old_config_with_extra_memory_fields_succeeds():
    """Test that a config with old/extra memory fields succeeds (extra fields ignored)."""
    old_config_data = {
        "agents": {
            "defaults": {
                "model": "gpt-4"
            }
        },
        "providers": {
            "openai": {
                "apiKey": "test-key"
            }
        },
        "memory": {
            "enabled": True,
            "provider": "local",
            "memory_type": "fractal",  # Extra field - will be ignored
            "vector_store": "chromadb",  # Extra field - will be ignored
            "als_enabled": True
        }
    }

    # This should succeed - extra fields are simply ignored
    config = Config.model_validate(convert_keys(old_config_data))
    assert config.memory.enabled is True
    assert config.memory.provider == "local"
    assert config.memory.als_enabled is True

    # Extra fields should not be in the model
    assert not hasattr(config.memory, 'memory_type')
    assert not hasattr(config.memory, 'vector_store')


def test_config_with_all_memory_fields():
    """Test that a config with all valid memory fields succeeds."""
    full_config_data = {
        "agents": {
            "defaults": {
                "model": "gpt-4"
            }
        },
        "providers": {
            "openai": {
                "apiKey": "test-key"
            }
        },
        "memory": {
            "enabled": True,
            "provider": "mem0",
            "topK": 10,
            "archiveDir": "my_archives",
            "alsEnabled": False,
            "mem0ApiKey": "mem0-key",
            "mem0UserId": "user123",
            "mem0OrgId": "org456",
            "mem0ProjectId": "proj789",
            "mem0Version": "v2.0",
            "embeddingModel": "text-embedding-3-large",
            "embeddingDim": 3072,
            "useHybridSearch": False,
            "latentRetryAttempts": 4,
            "latentRetryMinWait": 0.5,
            "latentRetryMaxWait": 10.0,
            "latentRetryMultiplier": 2.0,
        }
    }

    config = Config.model_validate(convert_keys(full_config_data))

    # Verify all fields are set correctly
    assert config.memory.enabled is True
    assert config.memory.provider == "mem0"
    assert config.memory.top_k == 10
    assert config.memory.archive_dir == "my_archives"
    assert config.memory.als_enabled is False
    assert config.memory.mem0_api_key == "mem0-key"
    assert config.memory.mem0_user_id == "user123"
    assert config.memory.mem0_org_id == "org456"
    assert config.memory.mem0_project_id == "proj789"
    assert config.memory.mem0_version == "v2.0"
    assert config.memory.embedding_model == "text-embedding-3-large"
    assert config.memory.embedding_dim == 3072
    assert config.memory.use_hybrid_search is False
    assert config.memory.latent_retry_attempts == 4
    assert config.memory.latent_retry_min_wait == 0.5
    assert config.memory.latent_retry_max_wait == 10.0
    assert config.memory.latent_retry_multiplier == 2.0


def test_multi_provider_config():
    """Test that configs can have multiple providers configured."""
    multi_provider_config = {
        "agents": {
            "defaults": {
                "model": "gemini/gemini-2.0-flash-exp"
            }
        },
        "providers": {
            "openai": {
                "apiKey": "sk-openai-key"
            },
            "gemini": {
                "apiKey": "gemini-key"
            },
            "anthropic": {
                "apiKey": "sk-ant-key"
            },
            "deepseek": {
                "apiKey": "deepseek-key"
            }
        }
    }

    config = Config.model_validate(convert_keys(multi_provider_config))

    # Verify all providers are configured
    assert config.providers.openai.api_key == "sk-openai-key"
    assert config.providers.gemini.api_key == "gemini-key"
    assert config.providers.anthropic.api_key == "sk-ant-key"
    assert config.providers.deepseek.api_key == "deepseek-key"

    # Verify model selection
    assert config.agents.defaults.model == "gemini/gemini-2.0-flash-exp"
