import pytest

from utils.coordinate_intervention import normalized_box_to_square_padding


def test_portrait_box_moves_into_horizontal_padding():
    box = normalized_box_to_square_padding(
        [27 / 235, 1 / 500, 225 / 235, 482 / 500],
        235,
        500,
    )
    assert box == pytest.approx((159 / 500, 1 / 500, 357 / 500, 482 / 500))


def test_landscape_box_moves_into_vertical_padding():
    box = normalized_box_to_square_padding([0.2, 0.25, 0.8, 0.75], 500, 300)
    assert box == pytest.approx((0.2, 175 / 500, 0.8, 325 / 500))


def test_square_box_is_unchanged():
    source = (0.1, 0.2, 0.7, 0.9)
    assert normalized_box_to_square_padding(source, 448, 448) == source


@pytest.mark.parametrize(
    'box,width,height',
    [
        ([0.2, 0.2, 0.1, 0.4], 100, 100),
        ([0.0, 0.0, 1.1, 1.0], 100, 100),
        ([0.0, 0.0, 1.0], 100, 100),
        ([0.0, 0.0, 1.0, 1.0], 0, 100),
    ],
)
def test_rejects_invalid_input(box, width, height):
    with pytest.raises(ValueError):
        normalized_box_to_square_padding(box, width, height)
