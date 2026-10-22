"""Select measured KV ranks without changing the modeled TP/PP topology."""

import json
from typing import Any


def parse_active_ranks(value: object) -> list[int] | None:
    """Accept a list (YAML/Python), JSON/CSV (env/CLI), or 'all'.

    Null means all modeled ranks. Empty lists, duplicates and coercions such as
    booleans/floats are errors, not requests to silently start extra engines.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "all":
            return None
        if text.startswith("["):
            value = json.loads(text)
        else:
            parts = text.split(",")
            if any(not part.strip() or not part.strip().isascii() or not part.strip().isdecimal() for part in parts):
                raise ValueError("active_ranks must be a nonempty list of nonnegative integer rank IDs, or 'all'")
            value = [int(part.strip()) for part in parts]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("active_ranks must be a nonempty list of rank IDs, or null for all ranks")
    if any(type(rank) is not int or rank < 0 for rank in value):
        raise ValueError("active_ranks must contain nonnegative integer rank IDs")
    if len(set(value)) != len(value):
        raise ValueError("active_ranks must not contain duplicate rank IDs")
    return sorted(value)


def resolve_active_ranks(rank_count: int, selection: list[int] | None) -> list[int]:
    if type(rank_count) is not int or rank_count < 1:
        raise ValueError("Modeled rank count must be a positive integer")
    ranks = parse_active_ranks(selection)
    if ranks is None:
        return list(range(rank_count))
    if any(rank >= rank_count for rank in ranks):
        raise ValueError(f"active_ranks must be within the modeled topology: 0..{rank_count - 1}")
    return ranks


def rank_selection_info(layout: list[dict[str, int]], selection: list[int] | None) -> dict[str, Any]:
    active = resolve_active_ranks(len(layout), selection)
    ids = set(active)
    return {
        "modeled_rank_count": len(layout),
        "active_rank_count": len(active),
        "active_rank_ids": active,
        "inactive_rank_ids": [rank["rank"] for rank in layout if rank["rank"] not in ids],
        "is_subset": len(active) != len(layout),
        "cache_result_scope": "active_ranks",
        "active_stored_bytes_per_token": sum(r["bytes_per_token"] for r in layout if r["rank"] in ids),
        "modeled_stored_bytes_per_token": sum(r["bytes_per_token"] for r in layout),
    }
