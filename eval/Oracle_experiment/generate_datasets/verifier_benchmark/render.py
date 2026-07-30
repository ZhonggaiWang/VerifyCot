"""Render a uniform candidate outline without exposing benchmark labels."""

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw


CANDIDATE_COLOR = (255, 0, 255)


def render_candidate(
        source_path: Path,
        candidate_box: Sequence[float],
        output_path: Path) -> None:
    with Image.open(source_path) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image)
    line_width = max(3, round(min(image.size) / 160))
    x1, y1, x2, y2 = (round(float(value)) for value in candidate_box)
    # Draw inward so border pixels never fall outside the image canvas.
    x1 = max(0, min(image.width - 1, x1))
    y1 = max(0, min(image.height - 1, y1))
    x2 = max(x1 + 1, min(image.width - 1, x2))
    y2 = max(y1 + 1, min(image.height - 1, y2))
    for offset in range(line_width):
        if x1 + offset >= x2 - offset or y1 + offset >= y2 - offset:
            break
        draw.rectangle(
            (x1 + offset, y1 + offset, x2 - offset, y2 - offset),
            outline=CANDIDATE_COLOR,
            width=1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format='PNG')
