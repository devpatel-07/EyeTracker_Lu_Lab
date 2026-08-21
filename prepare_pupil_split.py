import json
import re
from pathlib import Path


ANNOTATIONS_PATH = Path("pupil_training_labelled_json.json")
OUTPUT_PATH = Path("pupil_dataset_split.json")
VALIDATION_START_IMAGE = 236


def build_split(entries, validation_start=VALIDATION_START_IMAGE):
    samples = []

    for entry in entries:
        filename = entry["filename"]
        match = re.fullmatch(r"image_(\d+)\.[^.]+", filename)
        if match is None:
            raise ValueError(f"Could not parse image number from: {filename}")

        number = int(match.group(1))
        samples.append(
            {
                "filename": filename,
                "image": f"training_data/{filename}",
                "mask": f"pupil_masks/{Path(filename).stem}.png",
                "image_number": number,
                "blink": len(entry.get("regions", [])) == 0,
            }
        )

    samples.sort(key=lambda sample: sample["image_number"])
    train = [sample for sample in samples if sample["image_number"] < validation_start]
    validation = [
        sample for sample in samples if sample["image_number"] >= validation_start
    ]

    if not train or not validation:
        raise ValueError("Both training and validation splits must contain samples")

    def count_samples(split):
        blink_count = sum(sample["blink"] for sample in split)
        return {
            "total": len(split),
            "pupil": len(split) - blink_count,
            "blink": blink_count,
        }

    return {
        "validation_start_image": validation_start,
        "train": train,
        "validation": validation,
        "summary": {
            "train": count_samples(train),
            "validation": count_samples(validation),
        },
    }


def main():
    annotations = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    metadata = annotations.get("_via_img_metadata", annotations)
    manifest = build_split(metadata.values())

    for split_name in ("train", "validation"):
        for sample in manifest[split_name]:
            for path_key in ("image", "mask"):
                path = Path(sample[path_key])
                if not path.exists():
                    raise FileNotFoundError(f"Missing {path_key}: {path}")

    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train = manifest["summary"]["train"]
    validation = manifest["summary"]["validation"]
    print(
        f"Train: {train['total']} "
        f"({train['pupil']} pupil, {train['blink']} blink)"
    )
    print(
        f"Validation: {validation['total']} "
        f"({validation['pupil']} pupil, {validation['blink']} blink)"
    )
    print(f"Saved split to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
