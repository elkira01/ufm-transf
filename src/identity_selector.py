"""
identity_selector.py — Intelligent selection of identities for the multimodal dataset.

Given the analysis reports from ``dataset_analyzer.py``, this module:

1. Determines ``N = min(|CASIA|, |SOCOFing|)``.
2. Keeps **all** SOCOFing subjects (no filtering).
3. Reduces CASIA-WebFace to N identities by selecting the richest ones:
   - Exclude identities with corrupted files.
   - Sort by ``(nb_images DESC, identity_name ASC)``.
   - Take the top N.
4. Returns the two lists of selected identities.
"""

from __future__ import annotations

from typing import List

from dataset_analyzer import DatasetReport, IdentityInfo


def select_identities(
    face_report: DatasetReport,
    fp_report: DatasetReport,
) -> tuple[List[IdentityInfo], List[IdentityInfo], int]:
    """Select the best N identities from each dataset.

    Args:
        face_report: CASIA-WebFace analysis report.
        fp_report: SOCOFing analysis report.

    Returns:
        Tuple of ``(selected_casia, selected_soco, N)``.

        *selected_casia* contains the N best CASIA identities ordered by
        ascending name (the pairing engine will shuffle them).
        *selected_soco* contains all 600 SOCOFing subjects (unfiltered).
    """
    N = min(face_report.total_identities, fp_report.total_identities)

    # -----------------------------------------------------------------------
    # SOCOFing: keep all subjects (the user explicitly requested no filtering)
    # -----------------------------------------------------------------------
    selected_soco = sorted(fp_report.identities, key=lambda ident: int(ident.name))

    # -----------------------------------------------------------------------
    # CASIA-WebFace: exclude corrupted, then pick the top N by image count
    # -----------------------------------------------------------------------
    valid_casia = [
        ident for ident in face_report.identities if not ident.corrupted
    ]

    # Sort: most images first; on tie, lexicographic name (deterministic)
    valid_casia.sort(key=lambda ident: (-ident.nb_images, ident.name))

    selected_casia = valid_casia[:N]

    # Sort by name ascending for a stable join with shuffled SOCO list
    selected_casia.sort(key=lambda ident: ident.name)

    return selected_casia, selected_soco, N


def print_selection_summary(
    selected_casia: List[IdentityInfo],
    selected_soco: List[IdentityInfo],
    face_report: DatasetReport,
    fp_report: DatasetReport,
    N: int,
) -> None:
    """Print a summary of which identities were selected and why."""

    excluded_casia = face_report.total_identities - len(selected_casia)
    corrupted_excluded = face_report.corrupted_identity_count

    print("\n" + "=" * 60)
    print("  SELECTION SUMMARY")
    print("=" * 60)

    print(f"\n  Target N              : {N:,}")
    print(f"  ──────────────────────────────────")
    print(f"  CASIA-WebFace")
    print(f"    Total               : {face_report.total_identities:,}")
    print(f"    Corrupted (excluded): {corrupted_excluded:,}")
    print(f"    Dropped (below top N): {excluded_casia - corrupted_excluded:,}")
    print(f"    Selected            : {len(selected_casia):,}")
    print(f"  ──────────────────────────────────")
    print(f"  SOCOFing")
    print(f"    Total               : {fp_report.total_identities:,}")
    print(f"    Selected (all)      : {len(selected_soco):,}")

    # Statistics on selected CASIA identities
    if selected_casia:
        counts = [ident.nb_images for ident in selected_casia]
        print(f"\n  Selected CASIA identities:")
        print(f"    Min images/identity : {min(counts):,}")
        print(f"    Max images/identity : {max(counts):,}")
        print(f"    Mean images/identity: {sum(counts)/len(counts):.1f}")
        print(f"    Total face images   : {sum(counts):,}")

        # Top and bottom of the selection
        top3_idx = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)[:3]
        bot3_idx = sorted(range(len(counts)), key=lambda i: counts[i])[:3]
        print(f"\n  Top 3 richest identities:")
        for idx in top3_idx:
            ident = selected_casia[idx]
            print(f"    {ident.name}: {ident.nb_images:,} images")
        print(f"  Bottom 3 (threshold at N={N}):")
        for idx in bot3_idx:
            ident = selected_casia[idx]
            print(f"    {ident.name}: {ident.nb_images:,} images")

    # Statistics on selected SOCOFing identities
    if selected_soco:
        counts = [ident.nb_images for ident in selected_soco]
        print(f"\n  Selected SOCOFing subjects:")
        print(f"    Min images/subject : {min(counts):,}")
        print(f"    Max images/subject : {max(counts):,}")
        print(f"    Mean images/subject: {sum(counts)/len(counts):.1f}")
        print(f"    Total fp images    : {sum(counts):,}")

    print()
