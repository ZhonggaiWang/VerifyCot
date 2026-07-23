"""Run the official Volcano-7b Quick Start example with local weights."""

from pathlib import Path

from PIL import Image

from model.load_model import infer, load_model


ROOT = Path(__file__).resolve().parent
print(ROOT)
MODEL_PATH = ROOT / "weights" / "Volcano-7b"
IMAGE_PATH = ROOT / "figs" / "sample_input.jpg"


def main() -> None:
    model, preprocessor = load_model(str(MODEL_PATH), precision="fp16")
    input_image = Image.open(IMAGE_PATH).convert("RGB")

    response_1 = infer(
        model,
        preprocessor,
        input_image,
        'Is there a event "the cat is below the bed" in this image?',
        cot=True,
    )
    response_2 = infer(
        model,
        preprocessor,
        input_image,
        "Why is the cat on the bed?",
        cot=True,
    )
    response_3 = infer(model, preprocessor, input_image, "Describe the image.", cot=True)

    print("response 1:", response_1[0])
    print("response 2:", response_2[0])
    print("response 3:", response_3[0])


if __name__ == "__main__":
    main()
