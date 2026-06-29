#!/usr/bin/env python3
"""
create_multimodal_dataset.py — Build a synthetic multimodal biometric dataset.

Pairs CASIA-WebFace (face) identities with SOCOFing (fingerprint) subjects
in a strict 1:1 mapping.  The output dataset is ready for training a
multimodal biometric verification system.

Usage::

    python create_multimodal_dataset.py \\
        --face_path /path/to/casia-webface-extracted \\
        --fp_path /path/to/SOCOFing \\
        --output_dir /path/to/Multimodal_Dataset \\
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Local modules (same directory)
from dataset_analyzer import run_analysis
from identity_selector import select_identities, print_selection_summary
from pairing_engine import create_pairs
from dataset_builder import build_dataset
from metadata_generator import generate_metadata, print_final_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a synthetic multimodal biometric dataset "
        "(CASIA-WebFace + SOCOFing)."
    )
    parser.add_argument(
        "--face_path",
        type=Path,
        required=True,
        help="Path to casia-webface-extracted/ directory.",
    )
    parser.add_argument(
        "--fp_path",
        type=Path,
        required=True,
        help="Path to SOCOFing/ directory (parent of Real/ and Altered/).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("Multimodal_Dataset"),
        help="Where to create the multimodal dataset. Default: ./Multimodal_Dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # -------------------------------------------------------------------
    # 1. Analyse automatique
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ETAPE 1 — Analyse automatique des datasets")
    print("=" * 60)

    face_report, fp_report = run_analysis(args.face_path, args.fp_path)

    # -------------------------------------------------------------------
    # 2. Selection des identites
    # -------------------------------------------------------------------
    print("=" * 60)
    print("  ETAPE 2 — Selection des identites")
    print("=" * 60)

    selected_casia, selected_soco, N = select_identities(face_report, fp_report)
    print_selection_summary(
        selected_casia, selected_soco, face_report, fp_report, N
    )

    # -------------------------------------------------------------------
    # 3. Appariement 1:1
    # -------------------------------------------------------------------
    print("=" * 60)
    print("  ETAPE 3 — Appariement 1:1 (seed={})".format(args.seed))
    print("=" * 60)

    pairs = create_pairs(selected_casia, selected_soco, seed=args.seed)
    print(f"  Paires creees : {len(pairs):,}")
    print(f"  Exemple : {pairs[0].multimodal_id} → "
          f"CASIA={pairs[0].casia_identity.name} / "
          f"SOCO={pairs[0].socofing_identity.name}")

    # -------------------------------------------------------------------
    # 4. Construction du dataset
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ETAPE 4 — Construction du dataset sur disque")
    print("=" * 60)

    total_faces, total_fps = build_dataset(
        pairs=pairs,
        face_source_root=args.face_path,
        fp_source_root=args.fp_path,
        output_root=args.output_dir,
    )

    # -------------------------------------------------------------------
    # 5. Metadonnees
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ETAPE 5 — Generation des metadonnees")
    print("=" * 60)

    generate_metadata(
        pairs=pairs,
        output_root=args.output_dir,
        total_faces=total_faces,
        total_fps=total_fps,
        face_report=face_report,
        fp_report=fp_report,
    )

    # -------------------------------------------------------------------
    # 6. Rapport final
    # -------------------------------------------------------------------
    print_final_summary(pairs, total_faces, total_fps)


if __name__ == "__main__":
    main()
