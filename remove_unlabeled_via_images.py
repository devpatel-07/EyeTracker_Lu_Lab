import json
import re
from pathlib import Path


JSON_PATH = Path("pupil_training_labelled_json.json")
DELETED_LIST_PATH = Path("deleted_unlabeled_image_numbers.txt")


def image_number(filename):
    match = re.search(r"image_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def has_regions(item):
    return len(item.get("regions", [])) > 0


def main():
    with JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # VIA exports can either be a flat dict or store images under _via_img_metadata.
    if "_via_img_metadata" in data:
        metadata = data["_via_img_metadata"]
        is_nested_via = True
    else:
        metadata = data
        is_nested_via = False

    kept = {}
    deleted_numbers = []

    for key, item in metadata.items():
        if has_regions(item):
            kept[key] = item
        else:
            number = image_number(item.get("filename", key))
            if number is not None:
                deleted_numbers.append(number)

    deleted_numbers.sort()

    backup_path = JSON_PATH.with_suffix(".backup.json")
    backup_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if is_nested_via:
        data["_via_img_metadata"] = kept
        output_data = data
    else:
        output_data = kept

    JSON_PATH.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    DELETED_LIST_PATH.write_text(
        "\n".join(str(number) for number in deleted_numbers),
        encoding="utf-8",
    )

    print(f"Deleted {len(deleted_numbers)} unlabeled image entries.")
    print("Deleted image numbers:")
    print(deleted_numbers)
    print(f"Backup saved to: {backup_path}")
    print(f"Deleted list saved to: {DELETED_LIST_PATH}")


if __name__ == "__main__":
    main()
