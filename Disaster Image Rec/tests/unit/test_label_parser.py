"""Unit tests for xBD label parser."""

import json
import tempfile
from pathlib import Path
import pytest
from shapely.geometry import Polygon

from xbd_damage_assessment.data.label_parser import xBDLabelParser, DAMAGE_CLASSES


@pytest.fixture
def sample_json_data():
    """Create sample xBD JSON data."""
    return {
        "features": {
            "xy": [
                {
                    "wkt": "POLYGON ((0 0, 100 0, 100 100, 0 100, 0 0))",
                    "properties": {
                        "feature_type": "building",
                        "subtype": "no-damage",
                        "uid": "building_001",
                    },
                },
                {
                    "wkt": "POLYGON ((200 200, 300 200, 300 300, 200 300, 200 200))",
                    "properties": {
                        "feature_type": "building",
                        "subtype": "destroyed",
                        "uid": "building_002",
                    },
                },
            ]
        },
        "metadata": {"width": 1024, "height": 1024},
    }


@pytest.fixture
def sample_json_file(sample_json_data):
    """Create temporary JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_json_data, f)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    Path(temp_path).unlink()


def test_parser_initialization():
    """Test parser initialization."""
    parser = xBDLabelParser()
    assert parser.damage_class_map == DAMAGE_CLASSES

    custom_map = {"no-damage": 0, "destroyed": 1}
    parser = xBDLabelParser(damage_class_map=custom_map)
    assert parser.damage_class_map == custom_map


def test_parse_valid_json(sample_json_file):
    """Test parsing valid JSON file."""
    parser = xBDLabelParser()
    polygons, damage_classes, building_uids = parser.parse(sample_json_file)

    assert len(polygons) == 2
    assert len(damage_classes) == 2
    assert len(building_uids) == 2

    # Check first building
    assert isinstance(polygons[0], Polygon)
    assert polygons[0].is_valid
    assert damage_classes[0] == 0  # no-damage
    assert building_uids[0] == "building_001"

    # Check second building
    assert damage_classes[1] == 3  # destroyed


def test_parse_nonexistent_file():
    """Test parsing nonexistent file."""
    parser = xBDLabelParser()

    with pytest.raises(FileNotFoundError):
        parser.parse("nonexistent_file.json")


def test_parse_empty_features():
    """Test parsing JSON with no features."""
    data = {"features": {"xy": []}, "metadata": {}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        parser = xBDLabelParser()
        polygons, damage_classes, building_uids = parser.parse(temp_path)

        assert len(polygons) == 0
        assert len(damage_classes) == 0
        assert len(building_uids) == 0
    finally:
        Path(temp_path).unlink()


def test_get_image_dimensions(sample_json_file):
    """Test extracting image dimensions."""
    width, height = xBDLabelParser.get_image_dimensions(sample_json_file)
    assert width == 1024
    assert height == 1024


def test_damage_class_mapping():
    """Test damage class mapping."""
    parser = xBDLabelParser()

    assert parser.damage_class_map["no-damage"] == 0
    assert parser.damage_class_map["minor-damage"] == 1
    assert parser.damage_class_map["major-damage"] == 2
    assert parser.damage_class_map["destroyed"] == 3
    assert parser.damage_class_map["un-classified"] == 0
