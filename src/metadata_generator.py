"""
metadata_generator.py — Generate metadata and mapping files for the multimodal dataset.

Produces three files in the dataset root:

1. ``metadata.csv`` — Full per-identity metadata.
2. ``mapping.csv`` — Lightweight mapping table.
3. ``pairing_report.json`` — Machine-readable statistics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from dataset_analyzer import DatasetReport
from pairing_engine import Pair


def generate_metadata(
    pairs: List[Pair],
    output_root: Path,
    total_faces: int,
    total_fps: int,
    face_report: DatasetReport,
    fp_report: DatasetReport,
) -> None:
    """Write ``metadata.csv`` and ``mapping.csv`` and ``pairing_report.json``.

    Args:
        pairs: The list of multimodal pairings.
        output_root: Root of the multimodal dataset.
        total_faces: Total number of face images copied.
        total_fps: Total number of fingerprint images copied.
        face_report: Original CASIA analysis report.
        fp_report: Original SOCOFing analysis report.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # metadata.csv
    # ------------------------------------------------------------------
    metadata_path = output_root / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "multimodal_id",
            "casia_identity",
            "socofing_identity",
            "nb_face_images",
            "nb_fingerprint_images",
        ])
        for pair in pairs:
            writer.writerow([
                pair.multimodal_id,
                pair.casia_identity.name,
                pair.socofing_identity.name,
                pair.casia_identity.nb_images,
                pair.socofing_identity.nb_images,
            ])

    # ------------------------------------------------------------------
    # mapping.csv
    # ------------------------------------------------------------------
    mapping_path = output_root / "mapping.csv"
    with open(mapping_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["multimodal_id", "casia_id", "socofing_id"])
        for pair in pairs:
            writer.writerow([
                pair.multimodal_id,
                pair.casia_identity.name,
                pair.socofing_identity.name,
            ])

    # ------------------------------------------------------------------
    # pairing_report.json
    # ------------------------------------------------------------------
    face_counts = [p.casia_identity.nb_images for p in pairs]
    fp_counts = [p.socofing_identity.nb_images for p in pairs]

    report = {
        "dataset_name": "Multimodal Biometric Dataset (Synthetic)",
        "total_identities": len(pairs),
        "face": {
            "source_dataset": "CASIA-WebFace",
            "original_identities": face_report.total_identities,
            "original_images": face_report.total_images,
            "selected_identities": len(pairs),
            "copied_images": total_faces,
            "mean_images_per_identity": sum(face_counts) / len(face_counts)
            if face_counts
            else 0,
            "min_images_per_identity": min(face_counts) if face_counts else 0,
            "max_images_per_identity": max(face_counts) if face_counts else 0,
        },
        "fingerprint": {
            "source_dataset": "SOCOFing",
            "original_identities": fp_report.total_identities,
            "original_images": fp_report.total_images,
            "selected_identities": len(pairs),
            "copied_images": total_fps,
            "mean_images_per_identity": sum(fp_counts) / len(fp_counts)
            if fp_counts
            else 0,
            "min_images_per_identity": min(fp_counts) if fp_counts else 0,
            "max_images_per_identity": max(fp_counts) if fp_counts else 0,
        },
        "pairing_strategy": {
            "method": "shuffle_then_zip_1_to_1",
            "seed": 42,
            "casia_selection": "top N by image count, excluding corrupted",
            "socofing_selection": "all subjects, no filtering",
        },
    }

    report_path = output_root / "pairing_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"  metadata.csv       → {metadata_path}")
    print(f"  mapping.csv        → {mapping_path}")
    print(f"  pairing_report.json → {report_path}")


def print_final_summary(
    pairs: List[Pair],
    total_faces: int,
    total_fps: int,
) -> None:
    """Display the final construction summary."""

    face_counts = [p.casia_identity.nb_images for p in pairs]
    fp_counts = [p.socofing_identity.nb_images for p in pairs]

    print("\n" + "=" * 60)
    print("  DATASET MULTIMODAL CREATE AVEC SUCCES")
    print("=" * 60)

    print(f"\n  Identites creees      : {len(pairs):>8,}")
    print(f"  Images visage         : {total_faces:>8,}")
    print(f"  Images empreintes     : {total_fps:>8,}")
    print(f"  ──────────────────────────────────")
    print(f"  Visages / identite")
    print(f"    Moyenne              : {sum(face_counts)/len(face_counts):>10.1f}")
    print(f"    Min                  : {min(face_counts):>10,}")
    print(f"    Max                  : {max(face_counts):>10,}")
    print(f"  ──────────────────────────────────")
    print(f"  Empreintes / identite")
    print(f"    Moyenne              : {sum(fp_counts)/len(fp_counts):>10.1f}")
    print(f"    Min                  : {min(fp_counts):>10,}")
    print(f"    Max                  : {max(fp_counts):>10,}")

    # Distribution of face counts
    print(f"\n  Distribution visages/identite (deciles):")
    sorted_faces = sorted(face_counts)
    n = len(sorted_faces)
    for pct in [10, 25, 50, 75, 90]:
        idx = min(int(n * pct / 100), n - 1)
        print(f"    {pct:>2}%  : {sorted_faces[idx]:>8,}")

    print(f"\n  Distribution empreintes/identite (deciles):")
    sorted_fps = sorted(fp_counts)
    n = len(sorted_fps)
    for pct in [10, 25, 50, 75, 90]:
        idx = min(int(n * pct / 100), n - 1)
        print(f"    {pct:>2}%  : {sorted_fps[idx]:>8,}")

    print()
