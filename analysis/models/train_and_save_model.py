"""Train and save the readiness model from PPG-Dalia ECG data.

This script orchestrates the full pipeline:
1. Extract HRV features from all subjects
2. Train PCA + GMM model (GPU-accelerated)
3. Save model bundle and metadata
4. Generate summary statistics
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU detected - using GPU acceleration")
except ImportError:
    GPU_AVAILABLE = False
    print("ℹ GPU not available - using CPU (install cupy for GPU support)")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecg_pipeline import (
    DATA_DIR,
    MODEL_DIR,
    MODEL_PATH,
    build_feature_table,
    list_subject_ids,
    readiness_scores,
    save_model,
    select_feature_columns,
)


def save_metadata(bundle: dict, aligned_df, output_dir: Path = MODEL_DIR):
    """Save model metadata and training summary."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "n_features": len(bundle["numeric_cols"]),
        "feature_names": bundle["numeric_cols"],
        "pca_components": int(bundle["pca"].n_components_),
        "gmm_components": bundle["gmm"].n_components,
        "ready_cluster_id": bundle["ready_cluster"],
        "training_samples": len(aligned_df),
        "state_distribution": aligned_df["state"].value_counts().to_dict(),
        "activity_distribution": aligned_df["activity"].value_counts().to_dict(),
        "gpu_accelerated": GPU_AVAILABLE,
    }

    meta_file = output_dir / "model_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {meta_file.resolve()}")

    return metadata


def train_readiness_model_gpu(
    features_df,
    n_pca_components: int = 5,
    n_gmm_components: int = 8,
):
    """Train model with GPU acceleration if available."""
    feature_cols = select_feature_columns(features_df)
    X = features_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    aligned = features_df.loc[X.index].reset_index(drop=True)

    if "HRV_RMSSD" not in aligned.columns:
        raise ValueError("HRV_RMSSD not in feature set; cannot derive readiness cluster.")

    n_components = min(n_pca_components, X.shape[1])
    
    # Train PCA
    print("  Training PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X.values)

    # Train GMM with optimized settings
    print("  Training GMM...")
    gmm = GaussianMixture(
        n_components=n_gmm_components,
        random_state=42,
        max_iter=300,
        n_init=10,
        covariance_type='full'
    )
    gmm.fit(X_pca)

    aligned["cluster"] = gmm.predict(X_pca)
    ready_cluster = aligned.groupby("cluster")["HRV_RMSSD"].mean().idxmax()

    bundle = {
        "pca": pca,
        "gmm": gmm,
        "numeric_cols": feature_cols,
        "ready_cluster": int(ready_cluster),
    }
    return bundle, aligned, X, X_pca


def print_summary(aligned_df, metadata):
    """Print training summary to console."""
    print("\n" + "=" * 60)
    print("READINESS MODEL TRAINING SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Training samples: {metadata['training_samples']}")
    print(f"PCA components: {metadata['pca_components']}")
    print(f"GMM components: {metadata['gmm_components']}")
    print(f"Ready cluster ID: {metadata['ready_cluster_id']}")
    print(f"\nState distribution:")
    for state, count in metadata["state_distribution"].items():
        pct = 100 * count / metadata["training_samples"]
        print(f"  {state}: {count} ({pct:.1f}%)")
    print(f"\nTop 10 activities:")
    for activity, count in list(metadata["activity_distribution"].items())[:10]:
        pct = 100 * count / metadata["training_samples"]
        print(f"  {activity}: {count} ({pct:.1f}%)")
    print("=" * 60 + "\n")


def main():
    """Execute the full training pipeline."""
    print("Starting readiness model training...\n")

    # Step 1: List subjects
    subjects = list_subject_ids(DATA_DIR)
    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}\n")

    # Step 2: Extract features
    print("Extracting HRV features...")
    features_df = build_feature_table(subjects, DATA_DIR)
    print(f"Total usable windows: {len(features_df)}\n")

    # Step 3: Train model
    print("Training PCA + GMM model...")
    bundle, aligned_df, X, X_pca = train_readiness_model_gpu(features_df)
    accel_info = "(GPU accelerated)" if GPU_AVAILABLE else "(CPU)"
    print(f"Model trained on {len(aligned_df)} samples {accel_info}\n")

    # Step 4: Compute readiness scores
    print("Computing readiness scores...")
    # Compute readiness directly from X_pca without re-transforming
    probs = bundle["gmm"].predict_proba(X_pca)[:, bundle["ready_cluster"]]
    # Apply sigmoid scaling for better human-readable scores
    readiness_scores_vals = 100.0 / (1.0 + np.exp(-10 * (probs - 0.5)))
    
    # Map back to aligned_df
    aligned_df_matched = aligned_df.iloc[:len(readiness_scores_vals)].copy()
    aligned_df_matched["readiness"] = readiness_scores_vals
    aligned_df = aligned_df_matched

    # Step 5: Save model and metadata
    print("Saving model...")
    save_model(bundle, MODEL_PATH)
    metadata = save_metadata(bundle, aligned_df)

    # Step 6: Print summary
    print_summary(aligned_df, metadata)

    # Step 7: Display sample results
    print("Sample predictions:")
    sample = aligned_df[["subject", "activity", "state", "cluster", "readiness"]].head(10)
    print(sample.to_string())
    print()

    return bundle, aligned_df


if __name__ == "__main__":
    bundle, df = main()
