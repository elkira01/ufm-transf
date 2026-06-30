"""Flatten nested face/ and fingerprint/ subdirectories into the subject root.

Transforms:
    ID000001/
        face/000.jpg        ->  ID000001/face_000.jpg
        fingerprint/123.BMP ->  ID000001/fingerprint_123.bmp

Run from project root: python scripts/flatten_dataset.py
"""

import shutil
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FACE_DIR = "face"
FINGERPRINT_DIR = "fingerprint"


def flatten_subject(subject_dir: Path) -> tuple[int, int]:
    face_count = 0
    fp_count = 0

    for subdir_name, prefix in [(FACE_DIR, "face"), (FINGERPRINT_DIR, "fingerprint")]:
        subdir = subject_dir / subdir_name
        if not subdir.is_dir():
            continue
        for fpath in subdir.iterdir():
            if not fpath.is_file():
                continue
            new_name = f"{prefix}_{fpath.name}"
            new_path = subject_dir / new_name.lower()
            shutil.move(str(fpath), str(new_path))
            if subdir_name == FACE_DIR:
                face_count += 1
            else:
                fp_count += 1
        subdir.rmdir()

    return face_count, fp_count


def main():
    total_face = 0
    total_fp = 0

    for subject_dir in sorted(DATA_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        name = subject_dir.name
        if not name.startswith("ID"):
            continue

        f, fp = flatten_subject(subject_dir)
        if f or fp:
            total_face += f
            total_fp += fp
            print(f"  {name}: {f} face images, {fp} fingerprint images")

    print(f"\nTotal flattened: {total_face} face, {total_fp} fingerprint")


if __name__ == "__main__":
    main()
