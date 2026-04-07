"""Function-preserving transform primitives for ROTQ.

Three composable transforms:
  - equalization: cross-layer magnitude balancing
  - permutation: channel reordering for better g128 grouping
  - rotation: structured orthogonal basis change within foldable subspaces
"""

from .equalization import equalize_pair, equalize_model_weights
from .permutation import find_best_permutation, apply_input_permutation, apply_output_permutation
from .rotation import search_rotation, apply_rotation_to_weight
