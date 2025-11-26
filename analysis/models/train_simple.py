"""Debug version - simpler training script"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ecg_pipeline import (
    DATA_DIR,
    MODEL_DIR,
    MODEL_PATH,
    build_feature_table,
    list_subject_ids,
    save_model,
    select_feature_columns,
)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
import pickle
import json
from datetime import datetime

# Step 1: List subjects
print("Step 1: Finding subjects...")
subjects = list_subject_ids(DATA_DIR)
print(f"✓ Found {len(subjects)} subjects\n")

# Step 2: Extract features  
print("Step 2: Extracting HRV features (this may take 5-10 minutes)...")
features_df = build_feature_table(subjects, DATA_DIR)
print(f"✓ Total usable windows: {len(features_df)}\n")

# Step 3: Train model
print("Step 3: Training PCA + GMM model...")
feature_cols = select_feature_columns(features_df)
X = features_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
aligned_df = features_df.loc[X.index].reset_index(drop=True)

print(f"  Features shape: {X.shape}")
print(f"  Training samples: {len(aligned_df)}")

n_components = min(5, X.shape[1])
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X.values)
print(f"  PCA fitted: {X_pca.shape}")

gmm = GaussianMixture(n_components=8, random_state=42, max_iter=300, n_init=10)
gmm.fit(X_pca)
print(f"  GMM fitted")

# Step 4: Label clusters
aligned_df["cluster"] = gmm.predict(X_pca)
ready_cluster = aligned_df.groupby("cluster")["HRV_RMSSD"].mean().idxmax()
print(f"✓ Ready cluster: {ready_cluster}\n")

# Step 5: Save model
print("Step 4: Saving model...")
bundle = {
    "pca": pca,
    "gmm": gmm,
    "numeric_cols": feature_cols,
    "ready_cluster": int(ready_cluster),
}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)
print(f"✓ Model saved to {MODEL_PATH}\n")

# Step 6: Save metadata
metadata = {
    "timestamp": datetime.now().isoformat(),
    "n_features": len(feature_cols),
    "feature_names": feature_cols,
    "pca_components": int(pca.n_components_),
    "gmm_components": gmm.n_components,
    "ready_cluster_id": ready_cluster,
    "training_samples": len(aligned_df),
}

meta_file = MODEL_DIR / "model_metadata.json"
with open(meta_file, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved to {meta_file}\n")

print("=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Samples trained: {len(aligned_df)}")
print(f"Features: {len(feature_cols)}")
print(f"PCA components: {pca.n_components_}")
print(f"GMM clusters: {gmm.n_components}")
print(f"Ready cluster: {ready_cluster}")
print()
