import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer import (
    load_scores,
    compute_metrics,
    plot_refusal_rates,
    plot_model_size_vs_jailbreak,
    plot_temperature_sweep,
    plot_category_heatmap,
    plot_top350_yield,
    save_summary,
)


def main():
    root = Path(__file__).parent.parent
    scores_dir = root / "results" / "scores"
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scores...")
    df = load_scores(scores_dir)
    print(f"Loaded {len(df):,} records from {scores_dir}")

    print("Computing metrics...")
    metrics = compute_metrics(df)

    print("Generating plots...")
    plot_refusal_rates(metrics, plots_dir)
    plot_model_size_vs_jailbreak(metrics, plots_dir)
    plot_temperature_sweep(metrics, plots_dir)
    plot_category_heatmap(metrics, plots_dir)

    candidates_path = root / "data" / "candidates_4700.jsonl"
    top350_path = root / "data" / "top_350.json"
    if candidates_path.exists() and top350_path.exists():
        plot_top350_yield(candidates_path, top350_path, plots_dir)

    summary_path = root / "results" / "summary.json"
    save_summary(metrics, summary_path)

    print(f"\nResults saved to {root / 'results'}/")
    ratio = metrics.get("small_vs_medium_jailbreak_ratio")
    increase = metrics.get("violation_increase_above_0.9_temp")
    print(f"Small vs medium jailbreak ratio: {ratio}x" if ratio else "Ratio: N/A (run temp_sweep first)")
    print(f"Violation increase at temp>0.9: {increase:.1%}" if increase else "Temp increase: N/A (run temp_sweep first)")


if __name__ == "__main__":
    main()
