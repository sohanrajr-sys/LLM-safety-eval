import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adversarial_search import generate_candidates, score_candidates, filter_top_k
from src.model_runner import ModelRunner
from src.safety_scorer import score_text


def main():
    root = Path(__file__).parent.parent

    with open(root / "data" / "adversarial_1200.json") as f:
        prompts = json.load(f)

    print(f"Generating candidates from {len(prompts)} prompts...")
    candidates = generate_candidates(prompts)
    print(f"Generated {len(candidates)} unique candidates")

    with open(root / "config" / "models.yaml") as f:
        models_config = yaml.safe_load(f)

    tiny_cfg = next(m for m in models_config["models"] if "tinyllama" in m["slug"])
    runner = ModelRunner(
        model_id=tiny_cfg["id"],
        chat_template=tiny_cfg["chat_template"],
        batch_size=tiny_cfg["batch_size_fp16"],
        quantize=tiny_cfg.get("quantize", False),
    )
    print(f"Loading {tiny_cfg['slug']} for candidate scoring...")
    runner.load()

    scored = score_candidates(candidates, runner, score_text)
    runner.unload()

    candidates_path = root / "data" / "candidates_4700.jsonl"
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with open(candidates_path, "w") as f:
        for c in scored:
            f.write(json.dumps(c) + "\n")
    print(f"All {len(scored)} scored candidates → {candidates_path}")

    top350 = filter_top_k(scored, top_k=350)
    top350_path = root / "data" / "top_350.json"
    with open(top350_path, "w") as f:
        json.dump(top350, f, indent=2)
    print(f"Top 350 high-yield prompts → {top350_path}")
    if top350:
        print(f"Yield score range: {top350[-1]['yield_score']:.3f} – {top350[0]['yield_score']:.3f}")


if __name__ == "__main__":
    main()
