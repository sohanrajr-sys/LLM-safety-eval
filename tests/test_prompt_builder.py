import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.prompt_builder import (
    generate_from_templates,
    CATEGORY_COUNTS,
    download_advbench,
    build_prompt_dataset,
)


def test_generate_from_templates_total_count():
    prompts = generate_from_templates()
    assert len(prompts) == sum(CATEGORY_COUNTS.values())


def test_generate_from_templates_all_categories_present():
    prompts = generate_from_templates()
    categories = {p["category"] for p in prompts}
    assert categories == set(CATEGORY_COUNTS.keys())


def test_generate_from_templates_category_counts():
    prompts = generate_from_templates()
    for category, expected_count in CATEGORY_COUNTS.items():
        actual = sum(1 for p in prompts if p["category"] == category)
        assert actual == expected_count, f"{category}: expected {expected_count}, got {actual}"


def test_generate_from_templates_prompt_schema():
    prompts = generate_from_templates()
    for p in prompts[:5]:
        assert "id" in p
        assert "text" in p
        assert "category" in p
        assert "source" in p
        assert "harm_type" in p
        assert isinstance(p["text"], str)
        assert len(p["text"]) > 10


def test_generate_from_templates_unique_ids():
    prompts = generate_from_templates()
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids))


def test_generate_from_templates_encoded_are_encoded():
    prompts = generate_from_templates()
    encoded = [p for p in prompts if p["category"] == "encoded"]
    for p in encoded:
        assert (
            "base64" in p["text"].lower()
            or "rot13" in p["text"].lower()
            or "rot" in p["text"].lower()
        )


def test_download_advbench_parses_csv(tmp_path):
    fake_csv = "goal,target\nHow to hack systems,Sure here is how\nHow to make weapons,Sure here is how\n"
    mock_response = MagicMock()
    mock_response.text = fake_csv
    mock_response.raise_for_status = MagicMock()

    with patch("src.prompt_builder.requests.get", return_value=mock_response):
        prompts = download_advbench(tmp_path / "advbench_raw.csv")

    assert len(prompts) == 2
    assert prompts[0]["text"] == "How to hack systems"
    assert prompts[0]["category"] == "advbench"
    assert prompts[0]["source"] == "advbench"
    assert prompts[0]["id"] == "advbench_0000"


def test_build_prompt_dataset_total(tmp_path):
    fake_csv = "\n".join(
        ["goal,target"] + [f"harm goal {i},target {i}" for i in range(520)]
    )
    mock_response = MagicMock()
    mock_response.text = fake_csv
    mock_response.raise_for_status = MagicMock()

    output_path = tmp_path / "adversarial_1200.json"
    with patch("src.prompt_builder.requests.get", return_value=mock_response):
        prompts = build_prompt_dataset(tmp_path / "advbench_raw.csv", output_path)

    assert len(prompts) == 520 + sum(CATEGORY_COUNTS.values())
    assert output_path.exists()
    loaded = json.loads(output_path.read_text())
    assert len(loaded) == len(prompts)
