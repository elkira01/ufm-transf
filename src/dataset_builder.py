"""
dataset_builder.py — Build the multimodal dataset directory tree on disk.

For each ``Pair``:
1. Create ``Multimodal_Dataset/ID{NNNNNN}/``.
2. Copy all face images (``*.jpg``) from the CASIA identity directory
   into ``ID{NNNNNN}/face/``.
3. Copy all fingerprint images from *all* SOCOFing sub-folders
   (Real, Altered-Easy, Altered-Medium, Altered-Hard) that belong to
   the paired subject into ``ID{NNNNNN}/fingerprint/``.

Uses ``shutil.copy2`` to preserve file metadata and ``tqdm`` for
progress display.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from pairing_engine import Pair
from tqdm import tqdm


def build_dataset(
    pairs: List[Pair],
    face_source_root: Path,
    fp_source_root: Path,
    output_root: Path,
) -> Tuple[int, int]:
    """Build the full multimodal dataset directory tree.

    Args:
        pairs: List of multimodal pairings.
        face_source_root: Path to ``casia-webface-extracted/``.
        fp_source_root: Path to ``SOCOFing/`` (parent of ``Real/`` and
            ``Altered/``).
        output_root: Path where ``Multimodal_Dataset/`` will be created.

    Returns:
        Tuple of ``(total_face_copied, total_fp_copied)``.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    total_faces = 0
    total_fps = 0

    # SOCOFing sub-directories that hold fingerprints
    fp_subdirs = {
        "Real": fp_source_root / "Real",
        "Altered-Easy": fp_source_root / "Altered" / "Altered-Easy",
        "Altered-Medium": fp_source_root / "Altered" / "Altered-Medium",
        "Altered-Hard": fp_source_root / "Altered" / "Altered-Hard",
    }

    for pair in tqdm(pairs, desc="Building multimodal dataset", unit="identity"):
        identity_dir = output_root / pair.multimodal_id
        face_dir = identity_dir / "face"
        fp_dir = identity_dir / "fingerprint"

        face_dir.mkdir(parents=True, exist_ok=True)
        fp_dir.mkdir(parents=True, exist_ok=True)

        # ---- Copy face images ----
        casia_src = face_source_root / pair.casia_identity.name
        face_copied = 0
        if casia_src.is_dir():
            for jpg_file in casia_src.glob("*.jpg"):
                shutil.copy2(jpg_file, face_dir / jpg_file.name)
                face_copied += 1
        total_faces += face_copied

        # ---- Copy fingerprint images ----
        fp_copied = 0
        sid = pair.socofing_identity.name

        for _label, sub_path in fp_subdirs.items():
            if not sub_path.is_dir():
                continue
            for bmp_file in sub_path.iterdir():
                if not bmp_file.suffix.lower() == ".bmp":
                    continue
                # Match: filename starts with the subject ID followed by "__"
                name = bmp_file.name
                if name.startswith(f"{sid}__"):
                    shutil.copy2(bmp_file, fp_dir / name)
                    fp_copied += 1

        total_fps += fp_copied

    return total_faces, total_fps
