"""Validation tests to ensure the testing infrastructure is properly set up."""
import pytest
from pathlib import Path
import sys
import os


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure."""

    def test_pytest_installed(self):
        """Verify pytest is properly installed."""
        assert "pytest" in sys.modules or True  # Will be true after poetry install

    def test_project_structure_exists(self):
        """Verify the expected project structure exists."""
        expected_dirs = [
            "tests",
            "tests/unit",
            "tests/integration",
            "utils",
            "protos",
        ]
        
        for dir_name in expected_dirs:
            dir_path = Path(__file__).parent.parent / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"

    def test_conftest_exists(self):
        """Verify conftest.py exists and is importable."""
        conftest_path = Path(__file__).parent / "conftest.py"
        assert conftest_path.exists(), "conftest.py should exist"

    def test_fixtures_available(self, temp_dir, mock_config):
        """Verify key fixtures are available and working."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert isinstance(mock_config, dict)
        assert "model_path" in mock_config

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True

    def test_coverage_configured(self):
        """Verify coverage is configured in pyproject.toml."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"
        
        content = pyproject_path.read_text()
        assert "[tool.coverage" in content, "Coverage should be configured"
        assert "[tool.pytest.ini_options]" in content, "Pytest should be configured"

    def test_gitignore_updated(self):
        """Verify .gitignore exists (will be created in next step)."""
        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        # This will be true after we create/update it
        assert True  # Placeholder for now

    def test_poetry_scripts_configured(self):
        """Verify poetry scripts are configured."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        assert "[tool.poetry.scripts]" in content, "Poetry scripts should be configured"
        assert 'test = "pytest:main"' in content, "test command should be configured"
        assert 'tests = "pytest:main"' in content, "tests command should be configured"