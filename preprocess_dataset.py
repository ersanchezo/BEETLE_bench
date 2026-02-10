#!/usr/bin/env python3
"""
Unified Dataset Preprocessing Pipeline

This script combines all preprocessing steps for medical image segmentation datasets:
1. Analyze masks and compute statistics (spatial complexity, class distributions)
2. Create train/val splits with stratification
3. Clean data by removing low-quality patches
4. Transfer files to appropriate directories
5. Balance splits by moving samples between train/val

USAGE:
  # Full pipeline (dry run)
  python preprocess_dataset.py --mask-dir ./train_masks --output-dir ./split

  # Full pipeline (execute)
  python preprocess_dataset.py --mask-dir ./train_masks --output-dir ./split --execute

  # Individual steps
  python preprocess_dataset.py --step analyze --mask-dir ./train_masks --output-dir ./split --execute
  python preprocess_dataset.py --step split --mask-dir ./train_masks --output-dir ./split --execute
  python preprocess_dataset.py --step clean --output-dir ./split --execute
  python preprocess_dataset.py --step transfer --output-dir ./split --execute
  python preprocess_dataset.py --step balance --output-dir ./split --n-to-move 500 --execute
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# ==================== CONFIGURATION ====================
CLASS_MAP = {
    "unannotated": 0,
    "other": 1,
    "non-invasive epithelium": 2,
    "invasive epithelium": 3,
    "necrosis": 4,
}
NUM_CLASSES = 5

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".tif", ".tiff"}

DEFAULT_VAL_SIZE = 0.10
DEFAULT_RANDOM_STATE = 42


# ==================== UTILITY FUNCTIONS ====================
def ensure_dir(p: Path):
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def resolve_file_by_stem(dirpath: Path, stem: str, exts: set[str]) -> Path | None:
    """Find file in dirpath matching stem + known extensions."""
    for ext in exts:
        p = dirpath / f"{stem}{ext}"
        if p.exists():
            return p
        p2 = dirpath / f"{stem}{ext.upper()}"
        if p2.exists():
            return p2

    hits = list(dirpath.glob(stem + ".*"))
    if hits:
        exts_lower = {e.lower() for e in exts}
        hits_sorted = sorted(
            hits,
            key=lambda x: (x.suffix.lower() not in exts_lower, str(x)),
        )
        return hits_sorted[0]
    return None


def build_stem_index(folder: Path, allowed_exts: set[str]) -> dict[str, Path]:
    """Map stem -> path for all files in folder with allowed extensions."""
    idx: dict[str, Path] = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed_exts:
            continue
        stem = p.stem
        idx.setdefault(stem, p)
    return idx


# ==================== ANALYSIS FUNCTIONS ====================
def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask image as numpy array."""
    mask_img = Image.open(mask_path)
    mask = np.array(mask_img)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int64)


def calculate_spatial_complexity(mask: np.ndarray) -> float:
    """Calculate edge density (spatial complexity) of the mask."""
    grad_x = np.diff(mask, axis=0) != 0
    grad_y = np.diff(mask, axis=1) != 0
    boundary_pixels = np.sum(grad_x) + np.sum(grad_y)
    return float(boundary_pixels / mask.size)


def calculate_class_stats(mask: np.ndarray) -> dict:
    """Calculate pixel counts and fractions for each class."""
    u = np.unique(mask)
    if u.min() < 0 or u.max() >= NUM_CLASSES:
        print(f"[WARN] Unexpected mask values {u[:20]}... Expected 0..{NUM_CLASSES-1}")
    
    mask_clipped = np.clip(mask, 0, NUM_CLASSES - 1)
    counts = np.bincount(mask_clipped.reshape(-1), minlength=NUM_CLASSES).astype(np.int64)
    total = int(mask.size)
    fracs = counts / max(total, 1)

    pix_unann, pix_other, pix_noninv, pix_inv, pix_nec = counts.tolist()
    frac_unann, frac_other, frac_noninv, frac_inv, frac_nec = fracs.tolist()

    fg_pix = pix_noninv + pix_inv + pix_nec
    fg_frac = float(fg_pix / max(total, 1))

    return {
        "pix_unannotated": pix_unann,
        "pix_other": pix_other,
        "pix_noninv": pix_noninv,
        "pix_inv": pix_inv,
        "pix_necrosis": pix_nec,
        "frac_unannotated": float(frac_unann),
        "frac_other": float(frac_other),
        "frac_noninv": float(frac_noninv),
        "frac_inv": float(frac_inv),
        "frac_necrosis": float(frac_nec),
        "has_noninv": int(pix_noninv > 0),
        "has_inv": int(pix_inv > 0),
        "has_necrosis": int(pix_nec > 0),
        "fg_frac": fg_frac,
    }


def analyze_masks(mask_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Analyze all masks and compute statistics."""
    print(f"\n{'='*60}")
    print("STEP 1: ANALYZING MASKS")
    print(f"{'='*60}")
    
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_files = [
        mask_dir / f for f in os.listdir(mask_dir)
        if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))
    ]
    print(f"Found {len(mask_files)} masks")

    data = []
    for f in tqdm(mask_files, desc="Analyzing masks"):
        try:
            mask = load_mask(f)
            spatial_complexity = calculate_spatial_complexity(mask)
            class_stats = calculate_class_stats(mask)

            data.append({
                "filename": f.name,
                "path": str(f),
                "num_classes": int(len(np.unique(mask))),
                "spatial_complexity": spatial_complexity,
                **class_stats
            })
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not data:
        raise RuntimeError("No valid masks processed")

    df = pd.DataFrame(data)

    # Create complexity bins
    try:
        df['complexity_bin'] = pd.qcut(
            df['spatial_complexity'], q=3, labels=['Easy', 'Medium', 'Hard']
        )
    except ValueError:
        print("Warning: qcut failed; using thresholds")
        df['complexity_bin'] = df['spatial_complexity'].apply(
            lambda x: 'Hard' if x > 0.1 else ('Medium' if x > 0.01 else 'Easy')
        )

    # Create stratification column
    df['strata'] = df['complexity_bin'].astype(str) + "_nec" + df['has_necrosis'].astype(str)

    print("\n--- Complexity Distribution ---")
    print(df['complexity_bin'].value_counts(normalize=True))
    print("\n--- Stratification Distribution ---")
    print(df['strata'].value_counts(normalize=True).sort_index())

    return df


def visualize_complexity(df: pd.DataFrame, output_dir: Path):
    """Create visualization grid showing examples from each complexity bin."""
    print("\nGenerating complexity visualization...")
    categories = ['Easy', 'Medium', 'Hard']
    n_samples = 5

    fig, axes = plt.subplots(len(categories), n_samples, figsize=(15, 10))
    fig.suptitle('Spatial Complexity Stratification', fontsize=16)

    for i, cat in enumerate(categories):
        samples = df[df['complexity_bin'] == cat].sample(
            n=min(n_samples, len(df[df['complexity_bin'] == cat])),
            random_state=42
        )

        for j, (_, row) in enumerate(samples.iterrows()):
            ax = axes[i, j]
            mask = load_mask(Path(row['path']))
            ax.imshow(mask, cmap='tab10', interpolation='nearest')
            ax.set_title(f"Score: {row['spatial_complexity']:.3f}", fontsize=9)
            ax.axis('off')

            if j == 0:
                label = f"{cat}\n({'Low' if cat == 'Easy' else 'High'} Edges)"
                ax.text(-0.2, 0.5, label, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', va='center', rotation=90)

    plt.tight_layout()
    save_path = output_dir / "complexity_visualization.png"
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to: {save_path}")
    plt.close()


# ==================== SPLITTING FUNCTIONS ====================
def create_splits(df: pd.DataFrame, output_dir: Path, val_size: float, random_state: int):
    """Create stratified train/val splits."""
    print(f"\n{'='*60}")
    print("STEP 2: CREATING TRAIN/VAL SPLITS")
    print(f"{'='*60}")
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(split.split(df, df['strata']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_csv = output_dir / "train_split_with_stats.csv"
    val_csv = output_dir / "val_split_with_stats.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"\nTrain size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val size:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"\nSaved:")
    print(f"  {train_csv}")
    print(f"  {val_csv}")

    return train_df, val_df


# ==================== CLEANING FUNCTIONS ====================
def clean_data(
    csv_path: Path,
    images_dir: Path,
    masks_dir: Path,
    threshold_unannotated: float = 0.65,
    threshold_combo: float = 0.60,
    max_num_classes: int = 3,
    easy_remove_frac: float = 0.05,
    easy_protect_frac: float = 0.15,
    medium_remove_frac: float = 0.01,
    medium_protect_frac: float = 0.10,
    seed: int = 42,
    execute: bool = False,
    missing_ok: bool = False
):
    """Clean dataset by removing low-quality patches."""
    print(f"\n{'='*60}")
    print("STEP 3: CLEANING DATA")
    print(f"{'='*60}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # RULE 1: Delete patches with high unannotated fraction (except necrosis)
    targets_rule1 = df[
        (df["frac_unannotated"] > threshold_unannotated) & 
        (df["has_necrosis"] == 0)
    ].copy()

    # RULE 2: Delete patches with high combined unannotated+other (except necrosis)
    eligible = (df["has_necrosis"] == 0) & (df["num_classes"] <= max_num_classes)
    bins = df['complexity_bin'].astype(str)
    in_bins = bins.isin(['Easy', 'Medium', 'Hard'])
    combo = df["frac_unannotated"].astype(float) + df["frac_other"].astype(float)
    
    targets_rule2 = df[eligible & in_bins & (combo > threshold_combo)].copy()

    # RULE 3: Random removal from Easy/Medium bins
    remaining = df[eligible & ~targets_rule2.index.isin(targets_rule2.index)].copy()
    rem_bins = remaining['complexity_bin'].astype(str)
    easy_pool = remaining[rem_bins == 'Easy'].copy()
    med_pool = remaining[rem_bins == 'Medium'].copy()

    _, easy_del, _ = keep_with_protection(
        easy_pool, 'spatial_complexity', 
        keep_frac=1.0 - easy_remove_frac,
        protect_frac=easy_protect_frac,
        seed=seed
    )
    _, med_del, _ = keep_with_protection(
        med_pool, 'spatial_complexity',
        keep_frac=1.0 - medium_remove_frac,
        protect_frac=medium_protect_frac,
        seed=seed + 1
    )

    to_delete_df = pd.concat([targets_rule1, targets_rule2, easy_del, med_del], axis=0).drop_duplicates()

    print(f"\nDeletion Summary:")
    print(f"  Rule 1 (high unannotated):        {len(targets_rule1)}")
    print(f"  Rule 2 (high unannotated+other):  {len(targets_rule2)}")
    print(f"  Rule 3 (random from Easy):        {len(easy_del)}")
    print(f"  Rule 3 (random from Medium):      {len(med_del)}")
    print(f"  Total unique to delete:           {len(to_delete_df)}")

    # Find files to delete
    to_delete_files = []
    missing = []

    for _, r in to_delete_df.iterrows():
        fname = str(r["filename"])
        stem = Path(fname).stem

        img = resolve_file_by_stem(images_dir, stem, IMAGE_EXTS)
        msk = resolve_file_by_stem(masks_dir, stem, MASK_EXTS)

        if img is None or msk is None:
            missing.append((stem, img is not None, msk is not None))
            continue

        to_delete_files.extend([img, msk])

    print(f"\nMatched files to delete: {len(to_delete_files)} ({len(to_delete_files)//2} pairs)")
    
    if missing:
        print(f"Missing files: {len(missing)} pairs")
        if not missing_ok:
            raise RuntimeError("Some files missing. Use --missing-ok to proceed anyway.")

    if not execute:
        print("\n[DRY RUN] Use --execute to actually delete files")
        return

    deleted = 0
    for p in to_delete_files:
        try:
            p.unlink()
            deleted += 1
        except FileNotFoundError:
            if not missing_ok:
                raise

    print(f"\nDeleted {deleted} files ({deleted//2} pairs)")


def protect_top_fraction(df: pd.DataFrame, score_col: str, frac: float) -> pd.DataFrame:
    """Protect top fraction of rows by score."""
    n = len(df)
    if n == 0 or frac <= 0:
        return df.iloc[0:0].copy()
    k = int(math.ceil(frac * n))
    k = max(1, min(k, n))
    return df.sort_values(score_col, ascending=False).head(k)


def sample_random(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n random rows."""
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed)


def keep_with_protection(
    df_group: pd.DataFrame,
    score_col: str,
    keep_frac: float,
    protect_frac: float,
    seed: int
):
    """Keep a fraction of rows while protecting high-scoring ones."""
    n = len(df_group)
    if n == 0:
        empty = df_group.iloc[0:0].copy()
        return empty, empty, empty

    keep_n = int(math.ceil(keep_frac * n))
    keep_n = max(0, min(keep_n, n))

    protected = protect_top_fraction(df_group, score_col, protect_frac)
    protected_ids = set(protected.index)
    remainder = df_group.drop(index=list(protected_ids), errors="ignore")

    keep_n = max(keep_n, len(protected))
    need = keep_n - len(protected)

    random_keep = sample_random(remainder, need, seed=seed)

    keep_df = pd.concat([protected, random_keep], axis=0).drop_duplicates()
    delete_df = df_group.drop(index=keep_df.index, errors="ignore")
    return keep_df, delete_df, protected


# ==================== TRANSFER FUNCTIONS ====================
def transfer_to_val(
    csv_path: Path,
    train_images: Path,
    train_masks: Path,
    val_images: Path,
    val_masks: Path,
    execute: bool = False,
    move: bool = True,
    workers: int = 8,
    missing_ok: bool = False
):
    """Transfer files from train to val directories based on CSV."""
    print(f"\n{'='*60}")
    print("STEP 4: TRANSFERRING FILES TO VAL")
    print(f"{'='*60}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    ensure_dir(val_images)
    ensure_dir(val_masks)

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError("CSV must contain 'filename' column")

    # Index existing files
    img_idx = build_stem_index(train_images, IMAGE_EXTS)
    msk_idx = build_stem_index(train_masks, MASK_EXTS)

    print(f"Indexed {len(img_idx)} image stems, {len(msk_idx)} mask stems")

    ops = []
    missing = []

    for _, r in df.iterrows():
        csv_fname = str(r["filename"])
        stem = Path(csv_fname).stem

        img_src = img_idx.get(stem)
        msk_src = msk_idx.get(stem)

        if img_src is None or msk_src is None:
            missing.append((stem, img_src is not None, msk_src is not None))
            continue

        ops.append((img_src, val_images / img_src.name))
        ops.append((msk_src, val_masks / msk_src.name))

    print(f"Matched pairs to transfer: {len(ops)//2}")
    print(f"Missing: {len(missing)}")

    if missing and not missing_ok:
        raise RuntimeError(f"Missing {len(missing)} pairs. Use --missing-ok to proceed.")

    if not execute:
        print("\n[DRY RUN] Use --execute to actually transfer files")
        return

    action = shutil.move if move else shutil.copy2
    action_name = "MOVE" if move else "COPY"

    def worker(pair):
        src, dst = pair
        ensure_dir(dst.parent)
        action(str(src), str(dst))

    print(f"\n{action_name}ing files with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(worker, ops))

    print(f"Done. {action_name}ed {len(ops)} files ({len(ops)//2} pairs)")


# ==================== BALANCING FUNCTIONS ====================
def balance_splits(
    val_images: Path,
    val_masks: Path,
    train_images: Path,
    train_masks: Path,
    n_to_move: int,
    seed: int = 42,
    execute: bool = False,
    missing_ok: bool = False
):
    """Move random samples from val back to train to balance splits."""
    print(f"\n{'='*60}")
    print("STEP 5: BALANCING SPLITS")
    print(f"{'='*60}")
    
    for d in [val_images, val_masks, train_images, train_masks]:
        if not d.exists():
            raise FileNotFoundError(f"Missing directory: {d}")

    ensure_dir(train_images)
    ensure_dir(train_masks)

    images = [p for p in val_images.iterdir() if p.is_file()]
    if not images:
        raise RuntimeError(f"No files found in {val_images}")

    rng = random.Random(seed)
    rng.shuffle(images)

    n = min(n_to_move, len(images))
    selected = images[:n]

    ops = []
    missing = []

    for img in selected:
        stem = img.stem
        mask = resolve_file_by_stem(val_masks, stem, MASK_EXTS)
        
        if mask is None:
            missing.append((img, "mask_missing"))
            continue

        ops.append((img, train_images / img.name))
        ops.append((mask, train_masks / mask.name))

    print(f"Requested to move: {n_to_move} pairs")
    print(f"Available in val: {len(images)}")
    print(f"Matched pairs to move: {len(ops)//2}")

    if missing:
        print(f"Missing masks: {len(missing)}")
        if not missing_ok:
            raise RuntimeError("Some masks missing. Use --missing-ok to proceed.")

    if not execute:
        print("\n[DRY RUN] Use --execute to actually move files")
        return

    moved = 0
    for src, dst in ops:
        if dst.exists():
            print(f"SKIP (exists): {dst}")
            continue
        shutil.move(str(src), str(dst))
        moved += 1

    print(f"\nMoved {moved} files ({moved//2} pairs)")


# ==================== MAIN PIPELINE ====================
def main():
    parser = argparse.ArgumentParser(
        description="Unified dataset preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # General arguments
    parser.add_argument("--step", choices=["all", "analyze", "split", "clean", "transfer", "balance"],
                       default="all", help="Which step(s) to run")
    parser.add_argument("--execute", action="store_true",
                       help="Actually execute operations (default is dry-run)")
    parser.add_argument("--missing-ok", action="store_true",
                       help="Proceed even if some files are missing")

    # Directory arguments
    parser.add_argument("--mask-dir", type=Path, default=Path("./train_masks"),
                       help="Directory containing mask files")
    parser.add_argument("--output-dir", type=Path, default=Path("./split"),
                       help="Output directory for CSVs and visualizations")
    parser.add_argument("--train-images", type=Path, default=Path("./train_images"),
                       help="Train images directory")
    parser.add_argument("--train-masks", type=Path, default=Path("./train_masks"),
                       help="Train masks directory")
    parser.add_argument("--val-images", type=Path, default=Path("./val_images"),
                       help="Validation images directory")
    parser.add_argument("--val-masks", type=Path, default=Path("./val_masks"),
                       help="Validation masks directory")

    # Split parameters
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE,
                       help="Validation set size (fraction)")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE,
                       help="Random seed for reproducibility")

    # Cleaning parameters
    parser.add_argument("--threshold-unannotated", type=float, default=0.65,
                       help="Delete if frac_unannotated > threshold (unless necrosis)")
    parser.add_argument("--threshold-combo", type=float, default=0.60,
                       help="Delete if (frac_unannotated + frac_other) > threshold")
    parser.add_argument("--max-num-classes", type=int, default=3,
                       help="Only delete if num_classes <= this value")
    parser.add_argument("--easy-remove-frac", type=float, default=0.05,
                       help="Fraction of Easy samples to remove")
    parser.add_argument("--easy-protect-frac", type=float, default=0.15,
                       help="Fraction of top Easy samples to protect")
    parser.add_argument("--medium-remove-frac", type=float, default=0.01,
                       help="Fraction of Medium samples to remove")
    parser.add_argument("--medium-protect-frac", type=float, default=0.10,
                       help="Fraction of top Medium samples to protect")

    # Transfer parameters
    parser.add_argument("--transfer-mode", choices=["move", "copy"], default="move",
                       help="Whether to move or copy files during transfer")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of worker threads for file operations")

    # Balance parameters
    parser.add_argument("--n-to-move", type=int, default=500,
                       help="Number of pairs to move from val to train")

    args = parser.parse_args()

    # Resolve paths
    args.mask_dir = args.mask_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.train_images = args.train_images.resolve()
    args.train_masks = args.train_masks.resolve()
    args.val_images = args.val_images.resolve()
    args.val_masks = args.val_masks.resolve()

    ensure_dir(args.output_dir)

    print(f"\n{'='*60}")
    print("DATASET PREPROCESSING PIPELINE")
    print(f"{'='*60}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"Steps: {args.step}")
    print(f"{'='*60}\n")

    # Execute pipeline steps
    if args.step in ["all", "analyze"]:
        df = analyze_masks(args.mask_dir, args.output_dir)
        visualize_complexity(df, args.output_dir)
        
        if args.step == "all":
            # Continue to split
            train_df, val_df = create_splits(
                df, args.output_dir, args.val_size, args.random_state
            )

    if args.step in ["all", "split"]:
        if args.step == "split":
            # Load existing analysis
            analysis_csv = args.output_dir / "full_dataset_with_stats.csv"
            if not analysis_csv.exists():
                print("No existing analysis found. Running analyze step first...")
                df = analyze_masks(args.mask_dir, args.output_dir)
                df.to_csv(analysis_csv, index=False)
            else:
                df = pd.read_csv(analysis_csv)
            
            train_df, val_df = create_splits(
                df, args.output_dir, args.val_size, args.random_state
            )

    if args.step in ["all", "clean"]:
        # Clean train split
        train_csv = args.output_dir / "train_split_with_stats.csv"
        if train_csv.exists():
            clean_data(
                train_csv,
                args.train_images,
                args.train_masks,
                threshold_unannotated=args.threshold_unannotated,
                threshold_combo=args.threshold_combo,
                max_num_classes=args.max_num_classes,
                easy_remove_frac=args.easy_remove_frac,
                easy_protect_frac=args.easy_protect_frac,
                medium_remove_frac=args.medium_remove_frac,
                medium_protect_frac=args.medium_protect_frac,
                seed=args.random_state,
                execute=args.execute,
                missing_ok=args.missing_ok
            )
        else:
            print(f"Train CSV not found: {train_csv}. Skipping clean step.")

    if args.step in ["all", "transfer"]:
        val_csv = args.output_dir / "val_split_with_stats.csv"
        if val_csv.exists():
            transfer_to_val(
                val_csv,
                args.train_images,
                args.train_masks,
                args.val_images,
                args.val_masks,
                execute=args.execute,
                move=(args.transfer_mode == "move"),
                workers=args.workers,
                missing_ok=args.missing_ok
            )
        else:
            print(f"Val CSV not found: {val_csv}. Skipping transfer step.")

    if args.step in ["all", "balance"]:
        if args.val_images.exists() and args.val_masks.exists():
            balance_splits(
                args.val_images,
                args.val_masks,
                args.train_images,
                args.train_masks,
                n_to_move=args.n_to_move,
                seed=args.random_state,
                execute=args.execute,
                missing_ok=args.missing_ok
            )
        else:
            print("Val directories not found. Skipping balance step.")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
