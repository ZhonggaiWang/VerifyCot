"""Run a single-coordinate online counterfactual CoT intervention."""

import argparse
from pathlib import Path

from PIL import Image

from model.load_model import counterfactual_infer, load_model


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weights" / "Volcano-7b"
IMAGE_PATH = ROOT / "figs" / "sample_input.jpg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default='Is there a event "the cat is below the bed" in this image?')
    parser.add_argument("--perturb-index", type=int, default=None, help="1-based baseline coordinate index.")
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--perturb-seed", type=int, default=None)
    parser.add_argument("--iou-min", type=float, default=0.0)
    parser.add_argument("--iou-max", type=float, default=0.1)
    parser.add_argument("--perturb-box-mode", choices=("random", "same_shape"), default="random")
    parser.add_argument("--random-box-min-size", type=float, default=0.05)
    parser.add_argument("--random-box-max-size", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    model, preprocessor = load_model(str(MODEL_PATH), precision="fp16")
    image = Image.open(IMAGE_PATH).convert("RGB")
    result = counterfactual_infer(
        model,
        preprocessor,
        image,
        args.query,
        cot=True,
        max_new_tokens=args.max_new_tokens,
        perturb_index=args.perturb_index,
        selection_seed=args.selection_seed,
        perturb_seed=args.perturb_seed,
        perturb_iou_range=(args.iou_min, args.iou_max),
        perturb_box_mode=args.perturb_box_mode,
        random_box_min_size=args.random_box_min_size,
        random_box_max_size=args.random_box_max_size,
    )
    print("baseline response:", result["baseline"]["response"])
    print("baseline boxes:", result["baseline"]["boxes"])
    print("intervention:", result["intervention"])
    print("counterfactual response:", result["counterfactual"]["response"])
    print("counterfactual boxes:", result["counterfactual"]["boxes"])


if __name__ == "__main__":
    main()
