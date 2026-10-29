"""Centralized filesystem paths for proj-2pt-surf-2026.

Import this instead of hardcoding 'data/...', 'figures', or any absolute
path. All paths are anchored to the repository root (the directory
containing this file), so they resolve correctly regardless of the
process's current working directory.

If your raw TNG100 downloads live somewhere else on disk, either symlink
your download location to the data/... subdirectories below, or edit the
RAW_*_DIR constants to point at them directly -- don't reintroduce
hardcoded paths at call sites.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache_files"
FIGURES_DIR = REPO_ROOT / "figures"

# --- raw TNG100 download subdirectories under data/ ---
HYDRO_RAW_DIR = DATA_DIR / "hydro_data"
DMO_RAW_DIR = DATA_DIR / "dmo_data"
COLOR_RAW_DIR = DATA_DIR / "color_data"
TREE_RAW_DIR = DATA_DIR / "tree_data"
MATCH_RAW_DIR = DATA_DIR / "match_data"
FAVOLE_RAW_DIR = DATA_DIR / "favole_data"

# --- specific cache/data files referenced by more than one script/notebook ---
DMO_TREE_CACHE = CACHE_DIR / "dmo_tree_data.csv"
HYDRO_COLOR_TREE_CACHE = CACHE_DIR / "hydro_color_tree_subhaloflag_check_data.csv"
HYDRO_COLOR_TREE_NO_FLAG_CACHE = CACHE_DIR / "hydro_color_tree_no_subhaloflag_check_data.csv"
MATCH_TREES_NPZ = MATCH_RAW_DIR / "matches-trees-tng100.npz"
