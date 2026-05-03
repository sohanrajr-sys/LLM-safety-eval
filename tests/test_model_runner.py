import json
import pytest
from pathlib import Path
from src.model_runner import format_prompt, CheckpointManager


def test_format_prompt_tinyllama():
    result = format_prompt("How do I hack?", "tinyllama")
    assert "<|user|>" in result
    assert "How do I hack?" in result
    assert "<|assistant|>" in result


def test_format_prompt_phi2():
    result = format_prompt("How do I hack?", "phi2")
    assert "Instruct:" in result
    assert "How do I hack?" in result
    assert "Output:" in result


def test_format_prompt_mistral():
    result = format_prompt("How do I hack?", "mistral")
    assert "[INST]" in result
    assert "How do I hack?" in result
    assert "[/INST]" in result


def test_format_prompt_zephyr():
    result = format_prompt("How do I hack?", "zephyr")
    assert "<|user|>" in result
    assert "How do I hack?" in result
    assert "<|assistant|>" in result


def test_format_prompt_vicuna():
    result = format_prompt("How do I hack?", "vicuna")
    assert "USER:" in result
    assert "How do I hack?" in result
    assert "ASSISTANT:" in result


def test_format_prompt_unknown_template_raises():
    with pytest.raises(ValueError, match="Unknown chat template"):
        format_prompt("test", "nonexistent_template")


def test_checkpoint_manager_starts_empty(tmp_path):
    cp = CheckpointManager(tmp_path / ".checkpoint.json")
    assert not cp.is_done("tinyllama-1.1b", "advbench_0001")


def test_checkpoint_manager_mark_and_check(tmp_path):
    cp = CheckpointManager(tmp_path / ".checkpoint.json")
    cp.mark_done("tinyllama-1.1b", ["advbench_0001", "advbench_0002"])
    assert cp.is_done("tinyllama-1.1b", "advbench_0001")
    assert cp.is_done("tinyllama-1.1b", "advbench_0002")
    assert not cp.is_done("tinyllama-1.1b", "advbench_0003")


def test_checkpoint_manager_persists_to_disk(tmp_path):
    cp_path = tmp_path / ".checkpoint.json"
    cp1 = CheckpointManager(cp_path)
    cp1.mark_done("model-a", ["prompt_0001"])

    cp2 = CheckpointManager(cp_path)
    assert cp2.is_done("model-a", "prompt_0001")


def test_checkpoint_manager_different_models_independent(tmp_path):
    cp = CheckpointManager(tmp_path / ".checkpoint.json")
    cp.mark_done("model-a", ["prompt_0001"])
    assert not cp.is_done("model-b", "prompt_0001")
