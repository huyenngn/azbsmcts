from __future__ import annotations

import hashlib

_UINT32_MASK = 0xFFFFFFFF


def _mix_to_u32(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") & _UINT32_MASK


def derive_seed(
    base_seed: int,
    *,
    purpose: str,
    run_id: str = "",
    step: int | None = None,
    game_idx: int | None = None,
    player_id: int | None = None,
    extra: str = "",
) -> int:
    """
    Deterministic seed derivation with namespacing.
    Same inputs -> same seed; different purpose/ids -> different seed.
    """
    key = (
        f"{base_seed}|{purpose}|{run_id}|{step}|{game_idx}|{player_id}|{extra}"
    )
    return _mix_to_u32(key)
