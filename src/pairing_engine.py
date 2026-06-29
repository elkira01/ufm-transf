"""
pairing_engine.py — Reproducible 1:1 pairing between two identity lists.

Algorithm:
1. Shuffle both lists independently with a fixed random seed.
2. Zip them together: identity i from list A ↔ identity i from list B.
3. Assign a sequential multimodal ID (``ID000001``, ``ID000002``, ...).

The shuffle-before-zip approach guarantees:
- No systematic bias (rich CASIA identity always paired with the same
  SOCOFing subject index).
- Full reproducibility when the same seed is used.
- Each identity is used exactly once (strict 1:1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dataset_analyzer import IdentityInfo


@dataclass
class Pair:
    """A single multimodal pairing."""

    multimodal_id: str
    casia_identity: IdentityInfo
    socofing_identity: IdentityInfo


def create_pairs(
    casia_list: List[IdentityInfo],
    soco_list: List[IdentityInfo],
    seed: int = 42,
) -> List[Pair]:
    """Create 1:1 multimodal pairings between two identity lists.

    The lists must be of equal length (the caller must have reduced the
    larger dataset beforehand).

    Args:
        casia_list: Selected CASIA-WebFace identities.
        soco_list: Selected SOCOFing identities.
        seed: Random seed for reproducibility.

    Returns:
        List of ``Pair`` objects, one per multimodal identity.

    Raises:
        ValueError: If the input lists have different lengths.
    """
    if len(casia_list) != len(soco_list):
        raise ValueError(
            f"Identity lists must have equal length. "
            f"Got {len(casia_list)} CASIA vs {len(soco_list)} SOCOFing."
        )

    N = len(casia_list)

    # ---- Step 1: shuffle independently with seed ----
    random.seed(seed)
    shuffled_casia = casia_list.copy()
    random.shuffle(shuffled_casia)

    random.seed(seed + 1)
    shuffled_soco = soco_list.copy()
    random.shuffle(shuffled_soco)

    # ---- Step 2: 1:1 zip ----
    pairs: List[Pair] = []
    for idx, (casia, soco) in enumerate(zip(shuffled_casia, shuffled_soco), start=1):
        multimodal_id = f"ID{idx:06d}"
        pairs.append(
            Pair(
                multimodal_id=multimodal_id,
                casia_identity=casia,
                socofing_identity=soco,
            )
        )

    return pairs


def generate_mapping_dict(pairs: List[Pair]) -> Dict[str, Tuple[str, str]]:
    """Build a mapping dict: ``{multimodal_id: (casia_name, socofing_name)}``."""
    return {
        pair.multimodal_id: (pair.casia_identity.name, pair.socofing_identity.name)
        for pair in pairs
    }
