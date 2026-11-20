"""Load paired scene features from exports or a trusted aligned cache."""
from pathlib import Path

import numpy as np


def load_pair(cache, x_dir, y_dir, max_scenes=None):
    if cache is not None:
        with np.load(cache, allow_pickle=True) as data:
            x, y = data["X"], data["Y"]
            ids = data["common_ids"].tolist()
    else:
        from analyze_pca_whitening_cca import align_features, load_features_from_dir

        x, y, ids = align_features(load_features_from_dir(x_dir), load_features_from_dir(y_dir))
    if len(x) != len(y) or len(x) != len(ids):
        raise ValueError("Feature matrices and scene tokens must have equal lengths.")
    if max_scenes is not None and max_scenes < 2:
        raise ValueError("--max-scenes must be at least 2.")
    x, y, ids = x[:max_scenes], y[:max_scenes], ids[:max_scenes]
    if len(ids) < 2:
        raise ValueError("At least two matched scenes are required.")
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False), ids


def add_pair_arguments(parser):
    parser.add_argument("--cache", type=Path, help="Existing aligned NPZ containing X, Y and common_ids.")
    parser.add_argument("--x-dir", type=Path, help="Scene feature directory for model X.")
    parser.add_argument("--y-dir", type=Path, help="Scene feature directory for model Y.")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--output", type=Path, required=True)


def parse_pair_arguments(parser):
    args = parser.parse_args()
    cache_only = args.cache is not None and args.x_dir is None and args.y_dir is None
    directories_only = args.cache is None and args.x_dir is not None and args.y_dir is not None
    if not (cache_only or directories_only):
        parser.error("Supply either --cache or both --x-dir and --y-dir.")
    return args
