"""Subject-wise activity recognition on the PAMAP2 dataset.

PAMAP2 (Reiss & Stricker, 2012): 9 participants wore three inertial
measurement units (hand, chest, ankle; 100 Hz) and a heart-rate monitor
(about 9 Hz) while performing up to 12 protocol activities.

This module:
  1. reads the Protocol/*.dat files,
  2. cuts the signals into 5.12 s windows (512 samples, 50% overlap),
  3. computes time- and frequency-domain features per window,
  4. evaluates a random forest in two ways:
       - a random split of windows (optimistic: neighbouring windows from
         the same person end up in both training and test data), and
       - leave-one-subject-out (LOSO): train on all participants but one,
         test on the unseen participant. LOSO is the honest estimate of how
         the model would work on a new person.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

FS = 100  # Hz, IMU sampling rate
WIN = 512  # samples per window (5.12 s)
STEP = 256  # 50% overlap
SEED = 42

ACTIVITIES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "Nordic walking", 12: "ascending stairs",
    13: "descending stairs", 16: "vacuum cleaning", 17: "ironing",
    24: "rope jumping",
}

# Column layout of a PAMAP2 .dat row (54 columns).
IMU_OFFSET = {"hand": 3, "chest": 20, "ankle": 37}
# Within each 17-column IMU block: 0 temperature, 1-3 acc (+-16 g),
# 4-6 acc (+-6 g), 7-9 gyroscope, 10-12 magnetometer, 13-16 orientation (invalid).
ACC16 = (1, 2, 3)
GYRO = (7, 8, 9)


def read_subject(path: str) -> pd.DataFrame:
    """Read one subjectXXX.dat file into a tidy data frame."""
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    out = pd.DataFrame({
        "t": raw[0].values,
        "activity": raw[1].astype(int).values,
        "hr": raw[2].values,
    })
    for loc, off in IMU_OFFSET.items():
        for name, idx in (("acc", ACC16), ("gyro", GYRO)):
            for axis, j in zip("xyz", idx):
                out[f"{loc}_{name}_{axis}"] = raw[off + j].values
    # Heart rate is sampled at ~9 Hz: carry the last value forward.
    out["hr"] = out["hr"].ffill().bfill()
    # Short sensor drop-outs: interpolate within the file.
    out = out.interpolate(limit_direction="both")
    out["subject"] = int(os.path.basename(path)[7:10])
    return out


def load_protocol(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "subject1*.dat")))
    if not files:
        raise FileNotFoundError(f"No subject*.dat files found in {folder}")
    df = pd.concat([read_subject(f) for f in files], ignore_index=True)
    return df[df["activity"].isin(ACTIVITIES)].reset_index(drop=True)


def _spectral(sig: np.ndarray) -> tuple[float, float, float]:
    """Dominant frequency, spectral entropy and total spectral energy."""
    f, p = welch(sig - sig.mean(), fs=FS, nperseg=min(256, len(sig)))
    total = p.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    pn = p / total
    entropy = float(-(pn * np.log2(pn + 1e-12)).sum() / np.log2(len(pn)))
    return float(f[np.argmax(p)]), entropy, float(total)


def window_features(win: pd.DataFrame) -> dict:
    feats = {}
    channels = [c for c in win.columns if "_acc_" in c or "_gyro_" in c]
    for loc in IMU_OFFSET:
        mag = np.sqrt(sum(win[f"{loc}_acc_{a}"].values ** 2 for a in "xyz"))
        win = win.assign(**{f"{loc}_accmag": mag})
        channels.append(f"{loc}_accmag")
    for c in channels:
        x = win[c].values
        q75, q25 = np.percentile(x, [75, 25])
        feats[f"{c}_mean"] = x.mean()
        feats[f"{c}_std"] = x.std()
        feats[f"{c}_min"] = x.min()
        feats[f"{c}_max"] = x.max()
        feats[f"{c}_iqr"] = q75 - q25
        feats[f"{c}_domfreq"], feats[f"{c}_specent"], feats[f"{c}_energy"] = _spectral(x)
    feats["hr_mean"] = win["hr"].mean()
    feats["hr_std"] = win["hr"].std()
    return feats


def make_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Cut each continuous (subject, activity) bout into overlapping windows."""
    rows = []
    bout = (df["activity"] != df["activity"].shift()) | (df["subject"] != df["subject"].shift())
    df = df.assign(bout=bout.cumsum())
    for _, seg in df.groupby("bout", sort=False):
        n = len(seg)
        for start in range(0, n - WIN + 1, STEP):
            w = seg.iloc[start:start + WIN]
            f = window_features(w)
            f["subject"] = int(w["subject"].iloc[0])
            f["activity"] = int(w["activity"].iloc[0])
            rows.append(f)
    return pd.DataFrame(rows)


def _model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED,
                                  class_weight="balanced")


def evaluate(windows: pd.DataFrame) -> dict:
    X = windows.drop(columns=["subject", "activity"]).values
    y = windows["activity"].values
    groups = windows["subject"].values

    # 1) Random window split (optimistic, shown for comparison only).
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    m = _model().fit(Xtr, ytr)
    pred = m.predict(Xte)
    random_split = {"accuracy": accuracy_score(yte, pred),
                    "macro_f1": f1_score(yte, pred, average="macro")}

    # 2) Leave-one-subject-out.
    per_subject = []
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        m = _model().fit(X[tr], y[tr])
        pred = m.predict(X[te])
        per_subject.append({
            "subject": int(s), "n_windows": int(te.sum()),
            "n_activities": int(len(np.unique(y[te]))),
            "accuracy": accuracy_score(y[te], pred),
            "macro_f1": f1_score(y[te], pred, average="macro"),
        })
    loso = pd.DataFrame(per_subject)
    return {"random_split": random_split, "loso": loso}


def summarise(res: dict) -> str:
    loso = res["loso"]
    lines = [
        f"Random window split   accuracy {res['random_split']['accuracy']:.3f}   "
        f"macro-F1 {res['random_split']['macro_f1']:.3f}   (optimistic)",
        f"Leave-one-subject-out accuracy {loso['accuracy'].mean():.3f} +/- {loso['accuracy'].std():.3f}   "
        f"macro-F1 {loso['macro_f1'].mean():.3f} +/- {loso['macro_f1'].std():.3f}   "
        f"(mean +/- SD over {len(loso)} held-out participants)",
    ]
    return "\n".join(lines)
