"""Run the Volcano Quick Start with dynamically randomized coordinate bindings."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from model.load_model import infer, load_model


MODEL_PATH = ROOT / "weights" / "Volcano-7b"
IMAGE_PATH = ROOT / "figs" / "sample_input.jpg"
QUERIES = [
    'Is there a event "the cat is below the bed" in this image?',
    "Why is the cat on the bed?",
    "Describe the image.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible random boxes.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--min-box-size", type=float, default=0.05)
    parser.add_argument(
        "--max-randomized-coors",
        type=int,
        default=1,
        help="Randomize at most the first K coordinates (default: 1).",
    )
    args = parser.parse_args()

    model, preprocessor = load_model(str(MODEL_PATH), precision="fp16")
    input_image = Image.open(IMAGE_PATH).convert("RGB")

    for index, query in enumerate(QUERIES, start=1):
        seed = None if args.seed is None else args.seed + index - 1
        responses, metadata = infer(
            model,
            preprocessor,
            input_image,
            query,
            cot=True,
            max_new_tokens=args.max_new_tokens,
            randomize_coor=True,
            random_coor_seed=seed,
            random_coor_min_size=args.min_box_size,
            max_randomized_coors=args.max_randomized_coors,
            return_metadata=True,
        )
        print(f"response {index}: {responses[0]}")
        print(f"forced boxes {index}: {metadata['forced_boxes']}")


if __name__ == "__main__":
    main()
