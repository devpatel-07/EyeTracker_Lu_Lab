import json
import math
from pathlib import Path

import cv2
import numpy as np


ANNOTATIONS_PATH = Path("pupil_training_labelled_json.json")
IMAGE_DIR = Path("training_data")
OUTPUT_DIR = Path("pupil_masks")


def render_mask(image_shape, regions):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for region in regions:
        shape = region["shape_attributes"]
        shape_name = shape["name"]

        if shape_name == "ellipse":
            center = (round(shape["cx"]), round(shape["cy"]))
            axes = (round(shape["rx"]), round(shape["ry"]))
            angle_degrees = math.degrees(shape.get("theta", 0.0))
            cv2.ellipse(mask, center, axes, angle_degrees, 0, 360, 255, -1)
        elif shape_name == "polyline":
            points = np.column_stack(
                (shape["all_points_x"], shape["all_points_y"])
            ).astype(np.int32)
            cv2.fillPoly(mask, [points], 255)
        else:
            raise ValueError(f"Unsupported VIA shape: {shape_name}")

    return mask


def convert_annotations(json_path, image_dir, output_dir):
    json_path = Path(json_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    annotations = json.loads(json_path.read_text(encoding="utf-8"))
    entries = annotations.get("_via_img_metadata", annotations)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"total": 0, "pupil": 0, "blink": 0}

    for entry in entries.values():
        filename = entry["filename"]
        image_path = image_dir / filename
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read source image: {image_path}")

        regions = entry.get("regions", [])
        mask = render_mask(image.shape, regions)
        output_path = output_dir / f"{Path(filename).stem}.png"

        if not cv2.imwrite(str(output_path), mask):
            raise OSError(f"Could not write mask: {output_path}")

        summary["total"] += 1
        summary["pupil" if regions else "blink"] += 1

    return summary


def main():
    summary = convert_annotations(ANNOTATIONS_PATH, IMAGE_DIR, OUTPUT_DIR)
    print(f"Created {summary['total']} masks in: {OUTPUT_DIR}")
    print(f"Pupil masks: {summary['pupil']}")
    print(f"Blink masks: {summary['blink']}")


if __name__ == "__main__":
    main()
