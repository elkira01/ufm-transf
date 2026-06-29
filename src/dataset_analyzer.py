"""
dataset_analyzer.py — Automatic analysis of CASIA-WebFace and SOCOFing datasets.

Scans both datasets to determine:
- Total number of identities / subjects
- Number of images per identity
- Total number of images
- Corrupted or incomplete identities
- Exact directory structure

Provides a summary report before any dataset mutation occurs.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IdentityInfo:
    """Metadata about a single identity / subject."""

    name: str
    nb_images: int
    corrupted: bool = False
    corrupted_files: List[str] = field(default_factory=list)
    nb_images_real: int = 0
    nb_images_altered_easy: int = 0
    nb_images_altered_medium: int = 0
    nb_images_altered_hard: int = 0


@dataclass
class DatasetReport:
    """Full analysis report for one dataset."""

    name: str
    total_identities: int
    total_images: int
    identities: List[IdentityInfo] = field(default_factory=list)
    corrupted_identity_count: int = 0
    min_images: int = 0
    max_images: int = 0
    mean_images: float = 0.0


# ---------------------------------------------------------------------------
# JPEG / BMP integrity checks
# ---------------------------------------------------------------------------


def _is_valid_jpeg(filepath: Path) -> bool:
    """Check whether *filepath* contains a valid JPEG image."""
    if not filepath.is_file() or filepath.stat().st_size < 4:
        return False
    with open(filepath, "rb") as fh:
        magic = fh.read(2)
    return magic == b"\xff\xd8"


def _is_valid_bmp(filepath: Path) -> bool:
    """Check whether *filepath* contains a valid BMP image."""
    if not filepath.is_file() or filepath.stat().st_size < 2:
        return False
    with open(filepath, "rb") as fh:
        magic = fh.read(2)
    return magic == b"\x42\x4d"


# ---------------------------------------------------------------------------
# SOCOFing filename parser
# ---------------------------------------------------------------------------

# Example:  100__M_Left_index_finger.BMP
#           100__M_Left_index_finger_CR.BMP
_SOCO_PATTERN = re.compile(
    r"^(?P<subject_id>\d+)"
    r"__(?P<gender>[MF])"
    r"_(?P<hand>Left|Right)"
    r"_(?P<finger>index_finger|little_finger|middle_finger|ring_finger|thumb_finger)"
    r"(?:_(?P<alteration>CR|Obl|Zcut))?"
    r"\.BMP$",
    re.IGNORECASE,
)


def parse_socofing_filename(filename: str) -> Optional[Dict[str, str]]:
    """Extract fields from a SOCOFing filename. Returns None if unparseable."""
    match = _SOCO_PATTERN.match(filename)
    if not match:
        return None
    return match.groupdict()


# ---------------------------------------------------------------------------
# CASIA-WebFace scanner
# ---------------------------------------------------------------------------


def analyze_casia(face_path: Path) -> DatasetReport:
    """Analyze the CASIA-WebFace dataset structure."""
    report = DatasetReport(name="CASIA-WebFace", total_identities=0, total_images=0)

    if not face_path.is_dir():
        report.identities = []
        return report

    identities: List[IdentityInfo] = []
    total_images = 0

    for subj_dir in sorted(face_path.iterdir()):
        if not subj_dir.is_dir():
            continue

        jpg_files = sorted(subj_dir.glob("*.jpg"))
        nb = len(jpg_files)
        corrupted = False
        corrupted_list: List[str] = []

        for jpg in jpg_files:
            if not _is_valid_jpeg(jpg):
                corrupted = True
                corrupted_list.append(str(jpg))

        total_images += nb
        identities.append(
            IdentityInfo(
                name=subj_dir.name,
                nb_images=nb,
                corrupted=corrupted,
                corrupted_files=corrupted_list,
            )
        )

    report.total_identities = len(identities)
    report.total_images = total_images
    report.identities = identities
    report.corrupted_identity_count = sum(
        1 for ident in identities if ident.corrupted
    )

    if identities:
        counts = [ident.nb_images for ident in identities]
        report.min_images = min(counts)
        report.max_images = max(counts)
        report.mean_images = total_images / len(identities)

    return report


# ---------------------------------------------------------------------------
# SOCOFing scanner
# ---------------------------------------------------------------------------


def analyze_socofing(fp_path: Path) -> DatasetReport:
    """Analyze the SOCOFing dataset structure.

    Expects *fp_path* to point to ``SOCOFing/``, which contains
    ``Real/``, ``Altered/Altered-Easy/``, etc.
    """
    report = DatasetReport(name="SOCOFing", total_identities=0, total_images=0)

    if not fp_path.is_dir():
        return report

    # Map: subject_id (str) -> IdentityInfo
    subject_map: Dict[str, IdentityInfo] = {}

    # Sub-directories to scan
    sub_dirs = [
        ("real", fp_path / "Real"),
        ("altered_easy", fp_path / "Altered" / "Altered-Easy"),
        ("altered_medium", fp_path / "Altered" / "Altered-Medium"),
        ("altered_hard", fp_path / "Altered" / "Altered-Hard"),
    ]

    for category, cat_path in sub_dirs:
        if not cat_path.is_dir():
            continue
        for bmp_file in sorted(cat_path.glob("*.BMP")):
            parsed = parse_socofing_filename(bmp_file.name)
            if parsed is None:
                continue
            sid = parsed["subject_id"]
            if sid not in subject_map:
                subject_map[sid] = IdentityInfo(name=sid, nb_images=0)

            info = subject_map[sid]
            info.nb_images += 1

            # Per-category count
            if category == "real":
                info.nb_images_real += 1
            elif category == "altered_easy":
                info.nb_images_altered_easy += 1
            elif category == "altered_medium":
                info.nb_images_altered_medium += 1
            elif category == "altered_hard":
                info.nb_images_altered_hard += 1

            # Integrity check
            if not _is_valid_bmp(bmp_file):
                info.corrupted = True
                info.corrupted_files.append(str(bmp_file))

    identities = sorted(subject_map.values(), key=lambda x: int(x.name))
    total_images = sum(ident.nb_images for ident in identities)

    report.total_identities = len(identities)
    report.total_images = total_images
    report.identities = identities
    report.corrupted_identity_count = sum(
        1 for ident in identities if ident.corrupted
    )

    if identities:
        counts = [ident.nb_images for ident in identities]
        report.min_images = min(counts)
        report.max_images = max(counts)
        report.mean_images = total_images / len(identities)

    return report


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------


def print_report(face_report: DatasetReport, fp_report: DatasetReport) -> None:
    """Display a human-readable summary of both datasets."""

    def _print_one(report: DatasetReport) -> None:
        print(f"\n{'─' * 60}")
        print(f"  Dataset : {report.name}")
        print(f"{'─' * 60}")
        print(f"  Total identities          : {report.total_identities:>8,}")
        print(f"  Total images              : {report.total_images:>8,}")
        print(f"  Corrupted identities      : {report.corrupted_identity_count:>8,}")
        print(f"  Images/identity (min)     : {report.min_images:>8,}")
        print(f"  Images/identity (max)     : {report.max_images:>8,}")
        print(f"  Images/identity (mean)    : {report.mean_images:>12.2f}")

        # Distribution deciles (if enough identities)
        if report.identities:
            counts = sorted(ident.nb_images for ident in report.identities)
            n = len(counts)
            print(f"  Distribution (deciles)    :")
            for pct in [10, 25, 50, 75, 90]:
                idx = min(int(n * pct / 100), n - 1)
                print(f"    {pct:>2}%                     : {counts[idx]:>8,}")

            # Corrupted details
            corrupted_ids = [
                ident for ident in report.identities if ident.corrupted
            ]
            if corrupted_ids:
                print(f"\n  ⚠  Corrupted identities ({len(corrupted_ids)}):")
                for ident in corrupted_ids[:10]:
                    print(
                        f"    - {ident.name}: "
                        f"{len(ident.corrupted_files)} bad file(s)"
                    )
                if len(corrupted_ids) > 10:
                    print(f"    ... and {len(corrupted_ids) - 10} more.")

    _print_one(face_report)
    _print_one(fp_report)

    N = min(face_report.total_identities, fp_report.total_identities)
    print(f"\n{'─' * 60}")
    print(f"  N = min(|CASIA|, |SOCOFing|) = {N:,} identities")
    print(f"{'─' * 60}\n")


def run_analysis(
    face_path: Path, fp_path: Path
) -> Tuple[DatasetReport, DatasetReport]:
    """Entry point: run full analysis and print the report."""
    face_report = analyze_casia(face_path)
    fp_report = analyze_socofing(fp_path)
    print_report(face_report, fp_report)
    return face_report, fp_report
