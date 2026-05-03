import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL_SIZES = {
    "tinyllama-1.1b": 1.1,
    "phi-2": 2.7,
    "mistral-7b": 7.0,
    "zephyr-7b": 7.0,
    "vicuna-13b": 13.0,
}

SMALL_MODELS = ["tinyllama-1.1b", "phi-2"]
MEDIUM_MODELS = ["mistral-7b", "zephyr-7b"]


def load_scores(scores_dir: Path) -> pd.DataFrame:
    records = []
    for f in scores_dir.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return pd.DataFrame(records)


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"per_model": {}, "per_category": {}}

    main_df = df[df["temp"].isna() | (df["temp"] == 0.7)].copy()

    for model in main_df["model"].unique():
        mdf = main_df[main_df["model"] == model]
        n = len(mdf)
        metrics["per_model"][model] = {
            "refusal_rate": round(float(mdf["is_refusal"].sum()) / n, 4),
            "violation_rate": round(float(mdf["is_violation"].sum()) / n, 4),
            "jailbreak_rate": round(float(mdf["is_jailbreak"].sum()) / n, 4),
            "n": n,
        }

    for category in main_df["category"].unique():
        cdf = main_df[main_df["category"] == category]
        metrics["per_category"][category] = {
            model: round(float(cdf[cdf["model"] == model]["is_violation"].mean()), 4)
            for model in main_df["model"].unique()
        }

    present_small = [m for m in SMALL_MODELS if m in metrics["per_model"]]
    present_medium = [m for m in MEDIUM_MODELS if m in metrics["per_model"]]

    small_jb = (
        sum(metrics["per_model"][m]["jailbreak_rate"] for m in present_small) / len(present_small)
        if present_small else 0.0
    )
    medium_jb = (
        sum(metrics["per_model"][m]["jailbreak_rate"] for m in present_medium) / len(present_medium)
        if present_medium else 0.0
    )
    metrics["small_vs_medium_jailbreak_ratio"] = (
        round(small_jb / medium_jb, 2) if medium_jb > 0 else None
    )

    temp_df = df[df["temp"].notna() & (df["temp"] != 0.7)].copy()
    temp_sweep: Dict[str, Any] = {}
    if not temp_df.empty:
        for model in temp_df["model"].unique():
            temp_sweep[model] = {}
            for temp in sorted(temp_df["temp"].unique()):
                tdf = temp_df[(temp_df["model"] == model) & (temp_df["temp"] == temp)]
                temp_sweep[model][str(round(temp, 1))] = round(
                    float(tdf["is_violation"].mean()), 4
                )
        low = temp_df[temp_df["temp"] <= 0.9]["is_violation"].mean()
        high = temp_df[temp_df["temp"] > 0.9]["is_violation"].mean()
        metrics["violation_increase_above_0.9_temp"] = (
            round((high - low) / low, 4) if low > 0 else None
        )

    metrics["temperature_sweep"] = temp_sweep
    return metrics


def plot_refusal_rates(metrics: Dict, output_dir: Path) -> None:
    models = list(metrics["per_model"].keys())
    refusal = [metrics["per_model"][m]["refusal_rate"] for m in models]
    violation = [metrics["per_model"][m]["violation_rate"] for m in models]
    jailbreak = [metrics["per_model"][m]["jailbreak_rate"] for m in models]
    x = list(range(len(models)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width for i in x], refusal, width, label="Refusal Rate", color="steelblue")
    ax.bar(x, violation, width, label="Violation Rate", color="tomato")
    ax.bar([i + width for i in x], jailbreak, width, label="Jailbreak Rate", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Refusal, Violation, and Jailbreak Rates by Model")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "refusal_rates.png", dpi=150)
    plt.close()


def plot_model_size_vs_jailbreak(metrics: Dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, stats in metrics["per_model"].items():
        size = MODEL_SIZES.get(model, 0)
        jb = stats["jailbreak_rate"]
        ax.scatter(size, jb, s=120, zorder=5)
        ax.annotate(model, (size, jb), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Model Size (B parameters)")
    ax.set_ylabel("Jailbreak Success Rate")
    ax.set_title("Model Size vs Jailbreak Success Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "model_size_vs_jailbreak.png", dpi=150)
    plt.close()


def plot_temperature_sweep(metrics: Dict, output_dir: Path) -> None:
    if not metrics.get("temperature_sweep"):
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, temp_data in metrics["temperature_sweep"].items():
        temps = sorted(float(t) for t in temp_data)
        rates = [temp_data[str(round(t, 1))] for t in temps]
        ax.plot(temps, rates, marker="o", label=model)
    ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.6, label="temp=0.9 threshold")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Violation Rate")
    ax.set_title("Violation Rate vs Temperature by Model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_sweep.png", dpi=150)
    plt.close()


def plot_category_heatmap(metrics: Dict, output_dir: Path) -> None:
    per_cat = metrics.get("per_category", {})
    if not per_cat:
        return
    categories = list(per_cat.keys())
    models = list(next(iter(per_cat.values())).keys())
    data = [[per_cat[cat].get(model, 0.0) for model in models] for cat in categories]
    frame = pd.DataFrame(data, index=categories, columns=models)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(frame, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Violation Rate: Attack Category × Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Attack Category")
    plt.tight_layout()
    plt.savefig(output_dir / "category_heatmap.png", dpi=150)
    plt.close()


def plot_top350_yield(
    candidates_path: Path, top350_path: Path, output_dir: Path
) -> None:
    all_candidates = []
    with open(candidates_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_candidates.append(json.loads(line))
    with open(top350_path) as f:
        top350 = json.load(f)
    top_ids = {c["id"] for c in top350}
    top_scores = [c.get("yield_score", 0.0) for c in all_candidates if c["id"] in top_ids]
    rest_scores = [c.get("yield_score", 0.0) for c in all_candidates if c["id"] not in top_ids]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rest_scores, bins=50, alpha=0.6, label="Rest", color="steelblue")
    ax.hist(top_scores, bins=50, alpha=0.7, label="Top 350", color="tomato")
    ax.set_xlabel("Yield Score (toxicity probability)")
    ax.set_ylabel("Count")
    ax.set_title("Yield Score Distribution: Top 350 vs Rest")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "top350_yield.png", dpi=150)
    plt.close()


def save_summary(
    metrics: Dict,
    output_path: Path,
    total_prompts: int = 1200,
    candidates_generated: int = 4700,
    top_k: int = 350,
) -> None:
    summary = {
        "run_date": datetime.now().isoformat(),
        "total_prompts": total_prompts,
        "adversarial_candidates_generated": candidates_generated,
        "top_yield_prompts": top_k,
        "small_vs_medium_jailbreak_ratio": metrics.get("small_vs_medium_jailbreak_ratio"),
        "violation_increase_above_0.9_temp": metrics.get("violation_increase_above_0.9_temp"),
        "per_model": metrics["per_model"],
        "per_category": metrics["per_category"],
        "temperature_sweep": metrics.get("temperature_sweep", {}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
