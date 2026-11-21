# Sokoban solver

Solves make.fun Sokoban levels push-optimally: the level is split into its
hallway rooms, each room is solved with push-optimal A* over pushes
(min-cost box-to-goal matching heuristic, static dead-square pruning), and
the rooms' walks are stitched together. A whole-level A* is the fallback.
No dependencies, no network, deterministic.
