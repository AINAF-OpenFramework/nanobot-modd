"""Tests for Triune Memory System Phase 2.

Comprehensive test suite for checksums, verifier, loaders, and CLI commands.
"""


import yaml

from nanobot.game.loader import GameLoader
from nanobot.governance.loader import GovernanceLoader
from nanobot.latent.loader import LatentLoader
from nanobot.memory.loader import MemoryLoader
from nanobot.triune.checksums import ChecksumManager
from nanobot.triune.verifier import TriuneVerifier


class TestChecksumManager:
    """Tests for ChecksumManager."""

    def test_compute_checksum_md5(self, tmp_path):
        """Test MD5 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        manager = ChecksumManager(checksums_file)

        checksum = manager.compute_checksum(test_file, algorithm="md5")
        assert checksum is not None
        assert len(checksum) == 32  # MD5 is 32 hex chars

    def test_compute_checksum_sha256(self, tmp_path):
        """Test SHA256 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        manager = ChecksumManager(checksums_file)

        checksum = manager.compute_checksum(test_file, algorithm="sha256")
        assert checksum is not None
        assert len(checksum) == 64  # SHA256 is 64 hex chars

    def test_update_checksums(self, tmp_path):
        """Test updating checksums for MD-YAML pair."""
        md_file = tmp_path / "test.md"
        yaml_file = tmp_path / "test.yaml"
        md_file.write_text("# Test", encoding="utf-8")
        yaml_file.write_text("title: Test", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        manager = ChecksumManager(checksums_file)

        success = manager.update_checksums(md_file, yaml_file)
        assert success

        checksums = manager.get_all_checksums()
        assert str(md_file) in checksums
        assert checksums[str(md_file)].md_path == str(md_file)
        assert checksums[str(md_file)].yaml_path == str(yaml_file)

    def test_verify_pair_valid(self, tmp_path):
        """Test verification of valid MD-YAML pair."""
        md_file = tmp_path / "test.md"
        yaml_file = tmp_path / "test.yaml"
        md_file.write_text("# Test", encoding="utf-8")
        yaml_file.write_text("title: Test", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        manager = ChecksumManager(checksums_file)

        manager.update_checksums(md_file, yaml_file)
        is_valid, reason = manager.verify_pair(md_file, yaml_file)

        assert is_valid
        assert reason == "valid"

    def test_verify_pair_md_modified(self, tmp_path):
        """Test detection of modified MD file."""
        md_file = tmp_path / "test.md"
        yaml_file = tmp_path / "test.yaml"
        md_file.write_text("# Test", encoding="utf-8")
        yaml_file.write_text("title: Test", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        manager = ChecksumManager(checksums_file)

        manager.update_checksums(md_file, yaml_file)

        # Modify MD file
        md_file.write_text("# Modified", encoding="utf-8")

        is_valid, reason = manager.verify_pair(md_file, yaml_file)
        assert not is_valid
        assert reason == "md_modified"

    def test_save_and_load_checksums(self, tmp_path):
        """Test saving and loading checksums from file."""
        md_file = tmp_path / "test.md"
        yaml_file = tmp_path / "test.yaml"
        md_file.write_text("# Test", encoding="utf-8")
        yaml_file.write_text("title: Test", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        manager = ChecksumManager(checksums_file)

        manager.update_checksums(md_file, yaml_file)
        manager.save_checksums()

        # Create new manager to load from file
        manager2 = ChecksumManager(checksums_file)
        checksums = manager2.get_all_checksums()

        assert str(md_file) in checksums
        assert checksums[str(md_file)].md_checksum == manager.get_all_checksums()[str(md_file)].md_checksum

    def test_remove_checksum(self, tmp_path):
        """Test removing checksum record."""
        md_file = tmp_path / "test.md"
        yaml_file = tmp_path / "test.yaml"
        md_file.write_text("# Test", encoding="utf-8")
        yaml_file.write_text("title: Test", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        manager = ChecksumManager(checksums_file)

        manager.update_checksums(md_file, yaml_file)
        assert str(md_file) in manager.get_all_checksums()

        removed = manager.remove_checksum(md_file)
        assert removed
        assert str(md_file) not in manager.get_all_checksums()


class TestTriuneVerifier:
    """Tests for TriuneVerifier."""

    def test_verify_all_no_files(self, tmp_path):
        """Test verification with no MD files."""
        checksums_file = tmp_path / "checksums.json"
        verifier = TriuneVerifier(tmp_path, checksums_file)

        result = verifier.verify_all()

        assert result.total_md_files == 0
        assert result.sync_status == "no_files"

    def test_verify_all_missing_yaml(self, tmp_path):
        """Test detection of missing YAML files."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        verifier = TriuneVerifier(tmp_path, checksums_file)

        result = verifier.verify_all()

        assert result.total_md_files == 1
        assert len(result.missing_yaml) == 1
        assert result.sync_status == "no_sync"

    def test_verify_all_with_fix(self, tmp_path):
        """Test auto-fixing missing YAML files."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\nContent here.", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        verifier = TriuneVerifier(tmp_path, checksums_file)

        result = verifier.verify_all(fix=True)

        yaml_file = tmp_path / "test.yaml"
        assert yaml_file.exists()
        assert result.synced_yaml_files == 1

    def test_verify_all_orphaned_yaml(self, tmp_path):
        """Test detection of orphaned YAML files."""
        yaml_file = tmp_path / "orphan.yaml"
        yaml_file.write_text("title: Orphan", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        verifier = TriuneVerifier(tmp_path, checksums_file)

        result = verifier.verify_all()

        assert len(result.orphaned_yaml) == 1
        assert "orphan.yaml" in result.orphaned_yaml[0]

    def test_verify_all_invalid_yaml(self, tmp_path):
        """Test detection of invalid YAML files."""
        md_file = tmp_path / "test.md"
        yaml_file = tmp_path / "test.yaml"
        md_file.write_text("# Test", encoding="utf-8")
        yaml_file.write_text("invalid: yaml: syntax:", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        verifier = TriuneVerifier(tmp_path, checksums_file)

        result = verifier.verify_all()

        assert len(result.invalid_yaml) == 1

    def test_get_detailed_report(self, tmp_path):
        """Test detailed report generation."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test", encoding="utf-8")

        checksums_file = tmp_path / "checksums.json"
        verifier = TriuneVerifier(tmp_path, checksums_file)

        result = verifier.verify_all()
        report = verifier.get_detailed_report(result)

        assert "Triune Sync Status" in report
        assert "Missing YAML" in report


class TestGovernanceLoader:
    """Tests for GovernanceLoader."""

    def test_loader_singleton(self, tmp_path):
        """Test singleton pattern."""
        config_path = tmp_path / "governance.yaml"
        loader1 = GovernanceLoader.get_instance(config_path)
        loader2 = GovernanceLoader.get_instance(config_path)

        assert loader1 is loader2

    def test_load_missing_file(self, tmp_path):
        """Test loading when file doesn't exist."""
        config_path = tmp_path / "governance.yaml"
        loader = GovernanceLoader.get_instance(config_path)

        config = loader.load()

        assert isinstance(config, dict)
        assert len(config) == 0

    def test_load_valid_config(self, tmp_path):
        """Test loading valid governance config."""
        config_path = tmp_path / "governance.yaml"
        config_data = {
            "policies": [
                {"name": "policy1", "rule": "test"},
            ],
            "rules": [
                {"name": "rule1", "condition": "test"},
            ],
            "constraints": {
                "max_tokens": 1000,
            },
        }
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = GovernanceLoader.get_instance(config_path)
        config = loader.load()

        assert len(config["policies"]) == 1
        assert config["policies"][0]["name"] == "policy1"

    def test_get_policies(self, tmp_path):
        """Test getting policies."""
        config_path = tmp_path / "governance.yaml"
        config_data = {
            "policies": [
                {"name": "policy1"},
                {"name": "policy2"},
            ],
        }
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = GovernanceLoader.get_instance(config_path)
        GovernanceLoader.reset_instance()  # Reset for test isolation
        loader = GovernanceLoader.get_instance(config_path)
        policies = loader.get_policies()

        assert len(policies) == 2
        assert policies[0]["name"] == "policy1"


class TestMemoryLoader:
    """Tests for MemoryLoader."""

    def test_load_valid_config(self, tmp_path):
        """Test loading valid memory config."""
        config_path = tmp_path / "memory.yaml"
        config_data = {
            "schemas": [
                {"name": "schema1", "type": "test"},
            ],
            "templates": [
                {"name": "template1"},
            ],
            "configuration": {
                "cache_size": 100,
            },
        }
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = MemoryLoader.get_instance(config_path)
        config = loader.load()

        assert len(config["schemas"]) == 1
        assert config["configuration"]["cache_size"] == 100

    def test_get_schemas(self, tmp_path):
        """Test getting schemas."""
        config_path = tmp_path / "memory.yaml"
        config_data = {
            "schemas": [
                {"name": "schema1"},
            ],
        }
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = MemoryLoader.get_instance(config_path)
        MemoryLoader.reset_instance()
        loader = MemoryLoader.get_instance(config_path)
        schemas = loader.get_schemas()

        assert len(schemas) == 1


class TestLatentLoader:
    """Tests for LatentLoader."""

    def test_load_valid_config(self, tmp_path):
        """Test loading valid latent config."""
        config_path = tmp_path / "latent.yaml"
        config_data = {
            "patterns": [
                {"name": "pattern1", "type": "test"},
            ],
            "heuristics": [
                {"name": "heuristic1"},
            ],
        }
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = LatentLoader.get_instance(config_path)
        config = loader.load()

        assert len(config["patterns"]) == 1
        assert len(config["heuristics"]) == 1

    def test_get_patterns(self, tmp_path):
        """Test getting patterns."""
        config_path = tmp_path / "latent.yaml"
        config_data = {
            "patterns": [
                {"name": "pattern1"},
            ],
        }
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = LatentLoader.get_instance(config_path)
        LatentLoader.reset_instance()
        loader = LatentLoader.get_instance(config_path)
        patterns = loader.get_patterns()

        assert len(patterns) == 1


class TestGameLoader:
    """Tests for GameLoader."""

    def test_load_valid_config(self, tmp_path):
        """Test loading valid game config."""
        config_path = tmp_path / "game.yaml"
        config_data = {
            "configurations": [
                {"name": "config1"},
            ],
            "reward_models": [
                {"name": "model1"},
            ],
            "learning": {
                "rate": 0.01,
            },
        }
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = GameLoader.get_instance(config_path)
        config = loader.load()

        assert len(config["configurations"]) == 1
        assert config["learning"]["rate"] == 0.01

    def test_get_reward_models(self, tmp_path):
        """Test getting reward models."""
        config_path = tmp_path / "game.yaml"
        config_data = {
            "reward_models": [
                {"name": "model1"},
            ],
        }
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = GameLoader.get_instance(config_path)
        GameLoader.reset_instance()
        loader = GameLoader.get_instance(config_path)
        models = loader.get_reward_models()

        assert len(models) == 1
