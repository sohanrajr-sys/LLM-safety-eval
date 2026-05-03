import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_builder import build_prompt_dataset


def main():
    root = Path(__file__).parent.parent
    advbench_path = root / "data" / "advbench_raw.csv"
    output_path = root / "data" / "adversarial_1200.json"
    prompts = build_prompt_dataset(advbench_path, output_path)
    advbench_count = sum(1 for p in prompts if p["source"] == "advbench")
    generated_count = sum(1 for p in prompts if p["source"] == "generated")
    print(f"Built {len(prompts)} prompts ({advbench_count} AdvBench + {generated_count} generated)")
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
