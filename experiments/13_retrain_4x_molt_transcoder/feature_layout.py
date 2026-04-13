"""MOLT feature-id flattening.

Single source of truth for converting between:
    - flat feature_id (int)  — what delphi sees
    - (group_idx, transform_idx) — MOLT's native addressing

Canonical order follows MOLT rank groups from largest rank to smallest,
then transform index within each group.

For rank_multiplier N:
    group 0: 1*N  transforms at rank 512    → feature_ids [0,         1*N)
    group 1: 2*N  transforms at rank 256    → feature_ids [1*N,       3*N)
    group 2: 4*N  transforms at rank 128    → feature_ids [3*N,       7*N)
    group 3: 8*N  transforms at rank 64     → feature_ids [7*N,      15*N)
    group 4: 16*N transforms at rank 32     → feature_ids [15*N,     31*N)

Total features = 31 * N.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from molt.config import MOLTConfig


@dataclass(frozen=True)
class FeatureLayout:
    """Flat-id ↔ (group, transform) mapping for a given MOLT config."""

    rank_multiplier: int
    group_sizes: tuple[int, ...]   # transforms per group, in group order
    group_ranks: tuple[int, ...]   # rank per group, in group order
    group_offsets: tuple[int, ...] # cumulative start index per group

    @property
    def n_features(self) -> int:
        return sum(self.group_sizes)

    def flat_to_gt(self, feature_id: int) -> tuple[int, int]:
        """Map flat feature_id → (group_idx, transform_idx_within_group)."""
        if not 0 <= feature_id < self.n_features:
            raise IndexError(f"feature_id {feature_id} out of range [0, {self.n_features})")
        for g, (off, size) in enumerate(zip(self.group_offsets, self.group_sizes)):
            if feature_id < off + size:
                return g, feature_id - off
        raise RuntimeError("unreachable")

    def gt_to_flat(self, group_idx: int, transform_idx: int) -> int:
        """Map (group_idx, transform_idx) → flat feature_id."""
        if not 0 <= group_idx < len(self.group_sizes):
            raise IndexError(f"group_idx {group_idx} out of range")
        if not 0 <= transform_idx < self.group_sizes[group_idx]:
            raise IndexError(
                f"transform_idx {transform_idx} out of range for group {group_idx} "
                f"(size={self.group_sizes[group_idx]})"
            )
        return self.group_offsets[group_idx] + transform_idx

    def rank_of(self, feature_id: int) -> int:
        """Return the rank of the transform at this feature_id."""
        g, _ = self.flat_to_gt(feature_id)
        return self.group_ranks[g]

    def to_dict(self) -> dict:
        return asdict(self)


def layout_for_config(config: MOLTConfig) -> FeatureLayout:
    """Derive the canonical layout from a MOLTConfig's rank_distribution."""
    dist = config.rank_distribution  # [(count, rank), ...] in canonical order
    sizes = tuple(count for count, _ in dist)
    ranks = tuple(rank for _, rank in dist)
    offsets = []
    running = 0
    for s in sizes:
        offsets.append(running)
        running += s
    return FeatureLayout(
        rank_multiplier=config.rank_multiplier,
        group_sizes=sizes,
        group_ranks=ranks,
        group_offsets=tuple(offsets),
    )


if __name__ == "__main__":
    # Smoke test
    from molt.config import MOLTConfig

    for N in (1, 2, 4):
        cfg = MOLTConfig.from_preset("openai-community/gpt2", rank_multiplier=N)
        layout = layout_for_config(cfg)
        assert layout.n_features == 31 * N, f"N={N}: expected {31*N}, got {layout.n_features}"
        # Round-trip every feature
        for fid in range(layout.n_features):
            g, t = layout.flat_to_gt(fid)
            assert layout.gt_to_flat(g, t) == fid
        print(f"N={N}: {layout.n_features} features, groups {layout.group_sizes}, "
              f"ranks {layout.group_ranks}, offsets {layout.group_offsets} — OK")
