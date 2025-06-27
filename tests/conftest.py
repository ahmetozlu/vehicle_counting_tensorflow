"""Shared pytest fixtures and configuration."""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import json


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide mock configuration for tests."""
    return {
        "model_path": "/path/to/model",
        "confidence_threshold": 0.5,
        "max_detections": 100,
        "batch_size": 32,
        "input_size": (640, 480),
        "labels": ["car", "truck", "bus", "motorcycle", "bicycle"],
        "colors": ["red", "blue", "green", "yellow", "black", "white"],
    }


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a sample image file path."""
    image_path = temp_dir / "test_image.jpg"
    image_path.touch()
    return image_path


@pytest.fixture
def sample_video_path(temp_dir: Path) -> Path:
    """Create a sample video file path."""
    video_path = temp_dir / "test_video.mp4"
    video_path.touch()
    return video_path


@pytest.fixture
def mock_label_map() -> Dict[int, str]:
    """Provide a mock label map for object detection."""
    return {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
    }


@pytest.fixture
def sample_detection_result() -> Dict[str, Any]:
    """Provide sample detection results."""
    return {
        "boxes": [[100, 100, 200, 200], [300, 300, 400, 400]],
        "scores": [0.95, 0.87],
        "classes": [3, 8],  # car, truck
        "num_detections": 2,
    }


@pytest.fixture
def mock_model_config(temp_dir: Path) -> Path:
    """Create a mock model configuration file."""
    config_path = temp_dir / "pipeline.config"
    config_content = {
        "model": {
            "ssd": {
                "num_classes": 90,
                "image_resizer": {
                    "fixed_shape_resizer": {
                        "height": 300,
                        "width": 300
                    }
                }
            }
        },
        "train_config": {
            "batch_size": 24,
            "num_steps": 200000
        }
    }
    config_path.write_text(json.dumps(config_content, indent=2))
    return config_path


@pytest.fixture
def color_training_data() -> Dict[str, list]:
    """Provide sample color training data."""
    return {
        "red": [[255, 0, 0], [200, 10, 10], [180, 20, 20]],
        "green": [[0, 255, 0], [10, 200, 10], [20, 180, 20]],
        "blue": [[0, 0, 255], [10, 10, 200], [20, 20, 180]],
        "yellow": [[255, 255, 0], [200, 200, 10], [180, 180, 20]],
        "black": [[0, 0, 0], [10, 10, 10], [20, 20, 20]],
        "white": [[255, 255, 255], [240, 240, 240], [230, 230, 230]],
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages during tests."""
    with caplog.at_level("DEBUG"):
        yield caplog