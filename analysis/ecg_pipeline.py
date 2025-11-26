"""Train a readiness model from the PPG-Dalia ECG data using PCA + GMM.

The pipeline mirrors the notebook workflow:
1) load ECG + activity labels for each subject,
2) compute HRV features on sliding windows,
3) fit PCA (dimensionality reduction) and a GMM,
4) save the fitted objects so they can be reused for inference.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List, Optional

import neurokit2 as nk
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Root/directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ppg+dalia" / "data" / "PPG_FieldStudy"
MODEL_DIR = PROJECT_ROOT / "analysis" / "models"
MODEL_PATH = MODEL_DIR / "readiness_model.pkl"

# Sampling and windowing
FS_ECG = 700
WINDOW_SEC = 90
STEP_SEC = 15

# Label mapping
READY_ACTIVITIES = {"NO_ACTIVITY", "BASELINE", "CLEAN_BASELINE", "DRIVING", "LUNCH", "WORKING"}
FATIGUE_ACTIVITIES = {"STAIRS", "SOCCER", "CYCLING", "WALKING"}


def list_subject_ids(data_dir: Path) -> List[str]:
    """Return subject directory names (S1, S2, ...)."""
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir() and p.name.upper().startswith("S"))


def load_ecg(subject_dir: Path) -> np.ndarray:
    """Load ECG signal for a subject."""
    pkl_path = subject_dir / f"{subject_dir.name}.pkl"
    with pkl_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    ecg = np.asarray(data["signal"]["chest"]["ECG"]).flatten()
    return ecg


def load_activity(subject_dir: Path) -> pd.DataFrame:
    """Load and clean activity annotations."""
    act_path = subject_dir / f"{subject_dir.name}_activity.csv"
    df = pd.read_csv(act_path, header=None, names=["raw_label", "time_sec"])
    df["activity"] = df["raw_label"].astype(str).str.replace("#", "", regex=False).str.strip()
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df.dropna(subset=["time_sec"]).sort_values("time_sec").reset_index(drop=True)
    return df[["activity", "time_sec"]]


def activity_to_state(activity: str) -> str:
    if activity in READY_ACTIVITIES:
        return "ready"
    if activity in FATIGUE_ACTIVITIES:
        return "fatigued"
    return "unknown"


def activity_at_time(t_sec: float, activity_df: pd.DataFrame) -> str:
    """Return the last known activity at or before t_sec."""
    rows = activity_df[activity_df["time_sec"] <= t_sec]
    if rows.empty:
        return "unknown"
    return rows.iloc[-1]["activity"]


def compute_hrv_window(ecg_seg: np.ndarray, fs: int) -> Optional[dict]:
    """Compute HRV + simple SQI for one ECG window."""
    ecg_clean = nk.ecg_clean(ecg_seg, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
    r_locs = rpeaks["ECG_R_Peaks"]

    duration_sec = len(ecg_seg) / fs
    if duration_sec < 30 or len(r_locs) < 3:
        return None

    hrv_time = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
    hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False)

    hr = len(r_locs) * 60.0 / duration_sec
    sqi_hr = 1.0 if 35 <= hr <= 200 else 0.0
    min_expected_beats = 35 * duration_sec / 60.0
    sqi_beats = min(1.0, len(r_locs) / max(1.0, min_expected_beats))

    feats: dict[str, float] = {}
    feats.update(hrv_time.iloc[0].to_dict())
    feats.update(hrv_freq.iloc[0].to_dict())
    feats["HR_est"] = hr
    feats["n_beats"] = len(r_locs)
    feats["window_duration_sec"] = duration_sec
    feats["SQI"] = float(0.5 * sqi_hr + 0.5 * sqi_beats)
    return feats


def extract_features_for_subject(
    subject_id: str,
    base_dir: Path,
    fs: int = FS_ECG,
    window_sec: int = WINDOW_SEC,
    step_sec: int = STEP_SEC,
) -> pd.DataFrame:
    """Slide over a subject's ECG and compute HRV features."""
    subject_dir = base_dir / subject_id
    ecg = load_ecg(subject_dir)
    activity_df = load_activity(subject_dir)

    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    feature_rows = []
    start = 0
    while start + window_samples <= len(ecg):
        end = start + window_samples
        center_time_sec = (start + end) / 2.0 / fs

        activity = activity_at_time(center_time_sec, activity_df)
        state = activity_to_state(activity)
        if state == "unknown":
            start += step_samples
            continue

        ecg_seg = ecg[start:end]
        try:
            feats = compute_hrv_window(ecg_seg, fs)
        except Exception:
            start += step_samples
            continue

        if feats is None:
            start += step_samples
            continue

        feats.update(
            {
                "activity": activity,
                "state": state,
                "center_time": center_time_sec,
                "subject": subject_id,
            }
        )
        feature_rows.append(feats)
        start += step_samples

    return pd.DataFrame(feature_rows)


def build_feature_table(subject_ids: Iterable[str], base_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for sid in subject_ids:
        print(f"Processing {sid} ...")
        df = extract_features_for_subject(sid, base_dir)
        if df.empty:
            print(f"  No usable windows for {sid}")
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No features computed; check data paths or parameters.")

    return pd.concat(frames, ignore_index=True)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Pick numeric columns suitable for modeling."""
    numeric_cols = [c for c in df.columns if df[c].dtype != "object"]
    numeric_cols = [c for c in numeric_cols if df[c].notna().sum() > 0]
    numeric_cols = [c for c in numeric_cols if c not in {"center_time"}]
    numeric_cols = [c for c in numeric_cols if df[c].isna().mean() < 0.5]
    return numeric_cols


def train_readiness_model(
    features: pd.DataFrame,
    n_pca_components: int = 5,
    n_gmm_components: int = 8,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, np.ndarray]:
    feature_cols = select_feature_columns(features)
    X = features[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    aligned = features.loc[X.index].reset_index(drop=True)

    if "HRV_RMSSD" not in aligned.columns:
        raise ValueError("HRV_RMSSD not in feature set; cannot derive readiness cluster.")

    n_components = min(n_pca_components, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    gmm = GaussianMixture(n_components=n_gmm_components, random_state=42)
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


def save_model(bundle: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved readiness model to {path.resolve()}")


def readiness_scores(
    df: pd.DataFrame,
    bundle: dict,
    X_numeric: Optional[pd.DataFrame] = None,
    X_pca: Optional[np.ndarray] = None,
) -> pd.Series:
    """Compute readiness probability (0-100) for each row of df."""
    if X_numeric is None:
        X_numeric = df[bundle["numeric_cols"]].replace([np.inf, -np.inf], np.nan).dropna()
    if X_pca is None:
        X_pca = bundle["pca"].transform(X_numeric)
    probs = bundle["gmm"].predict_proba(X_pca)[:, bundle["ready_cluster"]]
    scores = pd.Series(probs * 100, index=X_numeric.index, name="readiness")
    return scores


def main():
    subjects = list_subject_ids(DATA_DIR)
    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    features_df = build_feature_table(subjects, DATA_DIR)
    print(f"Total usable windows: {len(features_df)}")

    bundle, aligned_df, X_numeric, X_pca = train_readiness_model(features_df)
    aligned_df["readiness"] = readiness_scores(aligned_df, bundle, X_numeric, X_pca)

    save_model(bundle, MODEL_PATH)
    print(aligned_df[["subject", "activity", "state", "cluster", "readiness"]].head())


if __name__ == "__main__":
    main()
