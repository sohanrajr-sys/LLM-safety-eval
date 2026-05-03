import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_runner import ModelRunner, CheckpointManager
from src.safety_scorer import classify_response, unload_classifier

TEMPERATURES = {
    "main": [0.7],
    "temp_sweep": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
}


def run_model(model_cfg, prompts, temperatures, scores_dir, checkpoint, run_type):
    slug = model_cfg["slug"]
    suffix = "_temp_sweep" if run_type == "temp_sweep" else ""
    output_file = scores_dir / f"{slug}{suffix}.jsonl"

    use_quantize = model_cfg.get("quantize", False)
    batch_size = (
        model_cfg["batch_size_4bit"] if use_quantize else model_cfg["batch_size_fp16"]
    )

    runner = ModelRunner(
        model_id=model_cfg["id"],
        chat_template=model_cfg["chat_template"],
        batch_size=batch_size,
        quantize=use_quantize,
    )

    if use_quantize:
        print(f"\n[{slug}] 4-bit quantized (fp16 would exceed 24GB VRAM)")

    print(f"\nLoading {slug}...")
    runner.load()

    scores_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "a") as out_f:
        for temp in temperatures:
            run_key = f"{slug}_t{temp}{suffix}"
            pending = [p for p in prompts if not checkpoint.is_done(run_key, p["id"])]
            if not pending:
                print(f"  {slug} @ temp={temp}: already complete, skipping")
                continue
            print(f"  {slug} @ temp={temp}: {len(pending)} prompts remaining")

            for i in range(0, len(pending), runner.batch_size):
                batch = pending[i : i + runner.batch_size]
                responses = runner.generate_batch(
                    [p["text"] for p in batch], temperature=temp
                )
                done_ids = []
                for prompt, response in zip(batch, responses):
                    scores = classify_response(response, prompt["category"])
                    record = {
                        "model": slug,
                        "prompt_id": prompt["id"],
                        "category": prompt["category"],
                        "temp": temp,
                        "response": response,
                        **scores,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    done_ids.append(prompt["id"])
                out_f.flush()
                checkpoint.mark_done(run_key, done_ids)

    runner.unload()
    unload_classifier()
    print(f"  {slug} complete.")


def main():
    parser = argparse.ArgumentParser(description="Run LLM safety evaluation")
    parser.add_argument(
        "--run",
        choices=["main", "temp_sweep"],
        required=True,
        help="main: 1200 prompts @ temp=0.7 | temp_sweep: top-350 @ 7 temperatures",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint instead of starting fresh",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    with open(root / "config" / "models.yaml") as f:
        models_config = yaml.safe_load(f)

    prompts_file = (
        root / "data" / "adversarial_1200.json"
        if args.run == "main"
        else root / "data" / "top_350.json"
    )
    with open(prompts_file) as f:
        prompts = json.load(f)

    scores_dir = root / "results" / "scores"
    checkpoint_path = scores_dir / ".checkpoint.json"

    if not args.resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    scores_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = CheckpointManager(checkpoint_path)

    print(f"Run: {args.run} | {len(prompts)} prompts | Resume: {args.resume}")
    temperatures = TEMPERATURES[args.run]

    for model_cfg in models_config["models"]:
        run_model(model_cfg, prompts, temperatures, scores_dir, checkpoint, args.run)

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
