"""
Class Mapping Diagnostic
--------------------------
Your data.yaml's `names:` list got corrupted (it contains README text
instead of real class names: crazing, inclusion, patches, pitted_surface,
rolled-in_scale, scratches).

This script checks whether your actual LABEL FILES (train/labels/*.txt)
used correct class IDs even though the yaml's names were wrong. It does
this by matching each image's filename prefix (e.g. "crazing_30_jpg...")
against the class ID that appears in its paired label .txt file.

If every image consistently maps to the same class ID for its filename
prefix, your annotations are trustworthy and we ONLY need to fix data.yaml
(no retraining needed) - just re-map training was already correct,
only the printed names were wrong.

If the mapping is inconsistent or filenames don't have recognizable
prefixes, your training data itself may be unreliable and switching to
a different, properly-labeled dataset is the safer path.

USAGE:
    Run this from your project root (same folder as data.yaml):
    python check_class_mapping.py
"""

import os
import glob
from collections import defaultdict

KNOWN_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
    "rolled", "pitted", "rs",  # a few common short-form variants seen in NEU exports
]

SPLITS = ["train", "valid", "test"]


def find_prefix(filename):
    """Try to detect which known class this filename belongs to."""
    lower = filename.lower()
    for cls in KNOWN_CLASSES:
        if lower.startswith(cls.lower()):
            return cls
    return None


def main():
    mapping_votes = defaultdict(lambda: defaultdict(int))  # {prefix: {class_id: count}}
    unmatched = 0
    total_checked = 0

    for split in SPLITS:
        img_dir = os.path.join(split, "images")
        lbl_dir = os.path.join(split, "labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        images = glob.glob(os.path.join(img_dir, "*"))
        for img_path in images:
            filename = os.path.basename(img_path)
            prefix = find_prefix(filename)

            base = os.path.splitext(filename)[0]
            label_path = os.path.join(lbl_dir, base + ".txt")

            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                continue

            total_checked += 1
            if prefix is None:
                unmatched += 1
                continue

            for line in lines:
                class_id = line.split()[0]
                mapping_votes[prefix][class_id] += 1

    print(f"Checked {total_checked} labeled images ({unmatched} had unrecognized filename prefixes)\n")

    if not mapping_votes:
        print("No matches found. Either your folder structure differs from")
        print("train/valid/test with images/ and labels/ subfolders, or")
        print("filenames don't start with a recognizable class name.")
        print("Tell me your actual folder layout and I'll adjust this script.")
        return

    print("Reconstructed mapping (filename prefix -> class ID used in labels):")
    print("-" * 65)
    final_mapping = {}
    for prefix, votes in sorted(mapping_votes.items()):
        total = sum(votes.values())
        best_id, best_count = max(votes.items(), key=lambda x: x[1])
        confidence = best_count / total * 100
        consistent = "CONSISTENT" if confidence > 95 else f"INCONSISTENT ({confidence:.0f}% agreement)"
        print(f"  {prefix:20s} -> class_id {best_id:3s}  [{consistent}]  (votes: {dict(votes)})")
        final_mapping[best_id] = prefix

    print("\n" + "-" * 65)
    if all("CONSISTENT" in "" for _ in [1]):  # placeholder, real check below
        pass

    inconsistent_found = any(
        max(votes.values()) / sum(votes.values()) <= 0.95
        for votes in mapping_votes.values()
    )

    if inconsistent_found:
        print("WARNING: Some prefixes had inconsistent class IDs across images.")
        print("This suggests the underlying annotations may be unreliable -")
        print("consider switching to a different, properly-labeled dataset.")
    else:
        print("All mappings were consistent. Your annotations are trustworthy.")
        print("\nSuggested corrected `names:` list for data.yaml, ordered by class ID:")
        try:
            ordered = [final_mapping[str(i)] for i in range(len(final_mapping))]
            print(f"names: {ordered}")
        except KeyError:
            print("Class IDs found don't run cleanly from 0..N-1 - print raw mapping above")
            print("and share it so I can help order it correctly.")


if __name__ == "__main__":
    main()
