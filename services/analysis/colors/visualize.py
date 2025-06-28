import argparse
import collections
import json

import numpy as np
from skimage import io

COLORS = {
    "black": [0.00, 0.00, 0.00],
    "blue": [0.00, 0.00, 1.00],
    "brown": [0.50, 0.40, 0.25],
    "grey": [0.50, 0.50, 0.50],
    "green": [0.00, 1.00, 0.00],
    "orange": [1.00, 0.80, 0.00],
    "pink": [1.00, 0.50, 1.00],
    "purple": [1.00, 0.00, 1.00],
    "red": [1.00, 0.00, 0.00],
    "white": [1.00, 1.00, 1.00],
    "yellow": [1.00, 1.00, 0.00],
}


def main(args: argparse.Namespace) -> None:
    with open(args.color_map, "r") as file:
        color_map = json.load(file)

    hwhw = np.tile((args.height, args.width), 2).reshape(1, 4)

    scores = np.array(color_map["scores"])
    labels = np.array(color_map["labels"])
    boxes = np.array(color_map["boxes"])
    boxes = np.round(boxes * hwhw).astype(int)

    colors_per_box = collections.defaultdict(list)
    output = np.zeros((args.height, args.width, 4))

    for box, label, score in zip(boxes, labels, scores):
        colors_per_box[tuple(box)].append((label, score))

    for box, colors in colors_per_box.items():
        y0, x0, y1, x1 = box
        n = 1.0 / len(colors)
        for i, (label, score) in enumerate(colors):
            yi0 = int(y0 + i * n * (y1 - y0))
            yi1 = int(y0 + (i + 1) * n * (y1 - y0))
            output[yi0:yi1, x0:x1, :3] = COLORS[label]
            output[yi0:yi1, x0:x1, 3] = score

    io.imsave(args.output, (output * 255).astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize color annotations.")
    parser.add_argument("color_map", help="json file of the color map")
    parser.add_argument("output", help="output color map image file")
    parser.add_argument(
        "--width", type=int, default=512, help="width of the output visualization"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="height of the output visualization"
    )
    args = parser.parse_args()
    main(args)
