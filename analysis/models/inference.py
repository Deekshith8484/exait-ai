"""Load and use the trained readiness model for inference.

This module provides utilities to:
1. Load the saved model bundle
2. Predict readiness scores for new ECG data
3. Batch inference on multiple windows
4. GPU-accelerated predictions (optional)
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecg_pipeline import (
    MODEL_PATH,
    compute_hrv_window,
    readiness_scores,
)


class ReadinessModel:
    """Wrapper for the trained readiness model."""

    def __init__(self, model_path: Union[str, Path] = MODEL_PATH):
        """Load the model bundle from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(model_path, "rb") as f:
            self.bundle = pickle.load(f)

        self.pca = self.bundle["pca"]
        self.gmm = self.bundle["gmm"]
        self.feature_cols = self.bundle["numeric_cols"]
        self.ready_cluster = self.bundle["ready_cluster"]
        self.gpu_available = GPU_AVAILABLE

        accel = "GPU" if GPU_AVAILABLE else "CPU"
        print(f"✓ Model loaded from {model_path.resolve()}")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  PCA components: {self.pca.n_components_}")
        print(f"  GMM clusters: {self.gmm.n_components}")
        print(f"  Inference device: {accel}")

    def predict_from_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Predict readiness score (0-100) and confidence from HRV feature dictionary.

        Args:
            features: Dictionary with keys matching the model's feature_cols

        Returns:
            Dictionary with 'score' (0-100) and 'confidence' (0-1)
        """
        try:
            x = np.array([features[col] for col in self.feature_cols]).reshape(1, -1)
            x_pca = self.pca.transform(x)
            
            # Get probabilities for both clusters
            probs = self.gmm.predict_proba(x_pca)[0]
            prob_ready = probs[self.ready_cluster]
            prob_other = 1.0 - prob_ready
            
            # Readiness score (0-100)
            score = 100.0 / (1.0 + np.exp(-10 * (prob_ready - 0.5)))
            
            # Confidence: how certain is the model (0-1)
            # Max of the two probabilities shows how certain
            confidence = max(prob_ready, prob_other)
            
            return {
                "score": float(score),
                "confidence": float(confidence),
                "prob_ready": float(prob_ready),
                "prob_other": float(prob_other),
            }
        except (KeyError, ValueError) as e:
            raise ValueError(f"Feature extraction failed: {e}")

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict readiness scores for multiple samples.

        Args:
            df: DataFrame with rows containing feature columns

        Returns:
            Series of readiness scores (0-100)
        """
        return readiness_scores(df, self.bundle) * 100

    def predict_from_ecg(
        self,
        ecg_segment: np.ndarray,
        fs: int = 700,
    ) -> Optional[Dict[str, float]]:
        """
        Predict readiness and confidence directly from raw ECG segment.

        Args:
            ecg_segment: 1D array of ECG samples
            fs: Sampling frequency (default 700 Hz)

        Returns:
            Dictionary with 'score', 'confidence', or None if HRV computation fails
        """
        try:
            feats = compute_hrv_window(ecg_segment, fs)
            if feats is None:
                return None
            return self.predict_from_features(feats)
        except Exception as e:
            print(f"ECG processing failed: {e}")
            return None

    def batch_predict_ecg(
        self,
        ecg_signal: np.ndarray,
        fs: int = 700,
        window_sec: int = 90,
        step_sec: int = 15,
    ) -> pd.DataFrame:
        """
        Slide over a long ECG signal and compute readiness at each window.
        Also captures HRV features (RMSSD, SDNN, LF/HF).
        GPU-accelerated when available.

        Args:
            ecg_signal: 1D array of ECG samples
            fs: Sampling frequency
            window_sec: Window duration in seconds
            step_sec: Slide step in seconds

        Returns:
            DataFrame with columns: center_time_sec, readiness, confidence, 
                                   prob_ready, prob_other, HRV_RMSSD, HRV_SDNN, 
                                   HRV_LF, HRV_HF, HRV_LF_HF, HR_est
        """
        window_samples = int(window_sec * fs)
        step_samples = int(step_sec * fs)

        results = []
        start = 0
        window_idx = 0

        while start + window_samples <= len(ecg_signal):
            end = start + window_samples
            center_time_sec = (start + end) / 2.0 / fs

            ecg_seg = ecg_signal[start:end]
            
            # Get HRV features directly for this segment
            try:
                feats = compute_hrv_window(ecg_seg, fs)
                if feats is None:
                    start += step_samples
                    continue
                    
                pred_dict = self.predict_from_features(feats)
                
                # Extract HRV values
                hrv_lf = feats.get("HRV_LF", np.nan)
                hrv_hf = feats.get("HRV_HF", np.nan)
                hrv_lf_hf = hrv_lf / max(hrv_hf, 0.001) if not np.isnan(hrv_lf) and not np.isnan(hrv_hf) else np.nan
                
                results.append(
                    {
                        "window_idx": window_idx,
                        "center_time_sec": center_time_sec,
                        "readiness": pred_dict["score"],
                        "confidence": pred_dict["confidence"],
                        "prob_ready": pred_dict["prob_ready"],
                        "prob_other": pred_dict["prob_other"],
                        # HRV features
                        "HRV_RMSSD": feats.get("HRV_RMSSD", np.nan),
                        "HRV_SDNN": feats.get("HRV_SDNN", np.nan),
                        "HRV_LF": hrv_lf,
                        "HRV_HF": hrv_hf,
                        "HRV_LF_HF": hrv_lf_hf,
                        "HR_est": feats.get("HR_est", np.nan),
                    }
                )
                window_idx += 1
            except Exception as e:
                print(f"Error processing window at {center_time_sec:.1f}s: {e}")
                
            start += step_samples

        return pd.DataFrame(results)


def load_model(model_path: Union[str, Path] = MODEL_PATH) -> ReadinessModel:
    """Convenience function to load the model."""
    return ReadinessModel(model_path)


if __name__ == "__main__":
    # Example: load model and show info
    try:
        model = load_model()
        print(f"\nModel is ready for inference.")
        print(f"Ready cluster: {model.ready_cluster}")
        print(f"Feature columns: {model.feature_cols[:5]}... ({len(model.feature_cols)} total)")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Train the model first using train_and_save_model.py")
