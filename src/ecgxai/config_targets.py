from __future__ import annotations

from typing import Dict, List, TypedDict


class TargetSpec(TypedDict):
    """Metadata for a single diagnosis target."""
    name: str
    aliases: List[str]


TARGET_META: Dict[str, TargetSpec] = {
    "164889003": {
        "name": "atrial fibrillation",
        "aliases": ["164889003"],
    },
    "426783006": {
        "name": "sinus rhythm",
        "aliases": ["426783006"],
    },
}

__all__ = ["TARGET_META", "TargetSpec", "validate_target_meta", "canonical_code_from_alias"]


def validate_target_meta(meta: Dict[str, TargetSpec] = TARGET_META) -> None:
    """Validate that TARGET_META is internally consistent."""
    alias_to_canonical: Dict[str, str] = {}

    for canonical, spec in meta.items():
        aliases = spec.get("aliases", [])

        if not isinstance(canonical, str) or not canonical:
            raise ValueError(f"Canonical code must be a non-empty string. Got: {canonical!r}")

        if canonical not in aliases:
            raise ValueError(f"Canonical code {canonical} missing from its aliases: {aliases}")

        seen = set()
        for a in aliases:
            if not isinstance(a, str) or not a:
                raise ValueError(f"Alias must be a non-empty string. Got: {a!r} in {canonical}")
            if a in seen:
                raise ValueError(f"Duplicate alias {a} in canonical {canonical}")
            seen.add(a)

            if a in alias_to_canonical and alias_to_canonical[a] != canonical:
                raise ValueError(
                    f"Alias {a} maps to multiple canonicals: "
                    f"{alias_to_canonical[a]} and {canonical}"
                )
            alias_to_canonical[a] = canonical


def canonical_code_from_alias(code: str, meta: Dict[str, TargetSpec] = TARGET_META) -> str:
    """Map any alias code to its canonical code.

    Args:
        code: SNOMED code string (canonical or alias).
        meta: Registry.

    Returns:
        Canonical code.

    Raises:
        KeyError: If code is not found in any alias list.
    """
    code = str(code)
    for canonical, spec in meta.items():
        if code in spec.get("aliases", []):
            return canonical
    raise KeyError(f"Code {code} not found in TARGET_META aliases.")
