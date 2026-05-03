import json
import pytest
import pandas as pd
from pathlib import Path
from src.analyzer import compute_metrics, save_summary, load_scores


def _make_dummy_df():
    records = []
    models = ["tinyllama-1.1b", "phi-2", "mistral-7b", "zephyr-7b", "vicuna-13b"]
    categories = ["direct_harm", "dan_persona", "hypothetical"]
    for model in models:
        for i in range(60):
            category = categories[i % len(categories)]
            records.append({
                "model": model,
                "prompt_id": f"p_{i:03d}",
                "category": category,
                "temp": 0.7,
                "is_refusal": i % 3 == 0,
                "is_violation": i % 3 == 1,
                "is_jailbreak": i % 3 == 1,
                "toxic_prob": 0.8 if i % 3 == 1 else 0.1,
            })
    return pd.DataFrame(records)


def _make_temp_sweep_df():
    records = []
    models = ["tinyllama-1.1b", "mistral-7b"]
    temps = [0.1, 0.5, 0.9, 1.1, 1.3]
    for model in models:
        for temp in temps:
            for i in range(20):
                records.append({
                    "model": model,
                    "prompt_id": f"p_{i:03d}",
                    "category": "direct_harm",
                    "temp": temp,
                    "is_refusal": temp <= 0.9,
                    "is_violation": temp > 0.9,
                    "is_jailbreak": temp > 0.9,
                    "toxic_prob": 0.8 if temp > 0.9 else 0.1,
                })
    return pd.DataFrame(records)


def test_compute_metrics_has_per_model_keys():
    df = _make_dummy_df()
    metrics = compute_metrics(df)
    assert "per_model" in metrics
    for model in ["tinyllama-1.1b", "phi-2", "mistral-7b", "zephyr-7b", "vicuna-13b"]:
        assert model in metrics["per_model"]


def test_compute_metrics_rates_are_fractions():
    df = _make_dummy_df()
    metrics = compute_metrics(df)
    for model, stats in metrics["per_model"].items():
        assert 0.0 <= stats["refusal_rate"] <= 1.0
        assert 0.0 <= stats["violation_rate"] <= 1.0
        assert 0.0 <= stats["jailbreak_rate"] <= 1.0


def test_compute_metrics_small_vs_medium_ratio():
    df = _make_dummy_df()
    metrics = compute_metrics(df)
    ratio = metrics.get("small_vs_medium_jailbreak_ratio")
    assert ratio is not None
    assert ratio > 0


def test_compute_metrics_temperature_sweep():
    df_main = _make_dummy_df()
    df_temp = _make_temp_sweep_df()
    df = pd.concat([df_main, df_temp], ignore_index=True)
    metrics = compute_metrics(df)
    assert "temperature_sweep" in metrics
    assert "tinyllama-1.1b" in metrics["temperature_sweep"]


def test_save_summary_creates_file(tmp_path):
    df = _make_dummy_df()
    metrics = compute_metrics(df)
    out = tmp_path / "summary.json"
    save_summary(metrics, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "run_date" in data
    assert "per_model" in data
    assert "total_prompts" in data


def test_load_scores_from_jsonl(tmp_path):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    records = [
        {
            "model": "tinyllama-1.1b",
            "prompt_id": "p_000",
            "temp": 0.7,
            "category": "direct_harm",
            "is_refusal": True,
            "is_violation": False,
            "is_jailbreak": False,
            "toxic_prob": 0.05,
        }
    ]
    with open(scores_dir / "tinyllama-1.1b.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    df = load_scores(scores_dir)
    assert len(df) == 1
    assert df.iloc[0]["model"] == "tinyllama-1.1b"


def test_plots_do_not_raise(tmp_path):
    from src.analyzer import (
        plot_refusal_rates,
        plot_model_size_vs_jailbreak,
        plot_temperature_sweep,
        plot_category_heatmap,
    )

    df = pd.concat([_make_dummy_df(), _make_temp_sweep_df()], ignore_index=True)
    metrics = compute_metrics(df)
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    plot_refusal_rates(metrics, plots_dir)
    plot_model_size_vs_jailbreak(metrics, plots_dir)
    plot_temperature_sweep(metrics, plots_dir)
    plot_category_heatmap(metrics, plots_dir)
    assert (plots_dir / "refusal_rates.png").exists()
    assert (plots_dir / "model_size_vs_jailbreak.png").exists()
    assert (plots_dir / "temperature_sweep.png").exists()
    assert (plots_dir / "category_heatmap.png").exists()
