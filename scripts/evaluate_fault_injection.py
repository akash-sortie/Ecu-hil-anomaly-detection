"""
Complete Offline Fault-Injection Evaluation Script for ecu_hil_anomaly
-----------------------------------------------------------------------
1. Splits healthy telemetry contiguously (by drive cycle if present, else chronological)
   into train / validation / test with zero cross-split window overlap or shuffling.
2. Generates reproducible synthetic faults on copies of held-out healthy test episodes:
   - sudden_spike
   - gradual_drift
   - stuck_at_flatline
   - noise_increase
   - cross_signal_idle_overheat (low/idle RPM + elevated temperature)
3. Computes thresholds ONLY on healthy validation windows (P95, P99, mean + 3*std)
   and compares against existing_threshold (0.2598).
4. Computes classification metrics, confusion matrix, ROC-AUC, PR-AUC, healthy false-alarm rate,
   per-fault-type recall, per-channel recall, missed fault types, root-cause top-1/top-2
   channel accuracy, and detection latency (windows + seconds).
5. Plots threshold sensitivity curve and precision-recall curve.
6. Compares against static min/max rule bounds, PCA reconstruction error, and Isolation Forest.
7. Saves CSV, JSON, PNG curves, and Markdown summary report.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import yaml
except ImportError:
    yaml = None


# ==============================================================================
# 1. PROJECT CONFIGURATION (AUTO-DETECTS CONFIGS/FEATURES IF SET TO AUTO/NONE)
# ==============================================================================
CONFIG = {
    "seed": 42,
    "signals_yaml": "configs/signals.yaml",
    "model_config_yaml": "configs/model_config.yaml",
    # Candidate healthy CSVs in priority order if "data_path" is None
    "data_path": None,  # e.g. "data/features/baseline_features.csv" or "data/raw/baseline_drive_cycle.csv"
    "candidate_data_paths": [
        "data/features/baseline_features.csv",
        "data/raw/baseline_drive_cycle.csv",
    ],
    "cycle_col": "auto",        # "auto" checks common cycle columns; or specify column name; or None
    "time_col": "auto",         # "auto" checks common time columns; or specify column name; or None
    "sampling_hz": "auto",      # "auto" reads from configs/*.yaml if present, else defaults to 10.0
    "feature_cols": "auto",     # "auto" reads from configs/signals.yaml if present, else numeric CSV cols
    "window_size": "auto",      # "auto" reads seq_len/window_size from configs/model_config.yaml, else 50
    "stride": 1,
    "split_ratios": (0.60, 0.20, 0.20),  # (train, val, test) contiguous split
    # Model & scaler paths from your repository structure
    "onnx_model_path": "mlflow/models/lstm_ae.onnx",
    "iso_model_path": "mlflow/models/isolation_forest.pkl",
    "scaler_path": "data/processed/scaler.pkl",
    "fit_fresh_train_scaler": True,     # True = fit StandardScaler strictly on TRAIN split (recommended to avoid leakage)
    "train_fallback_ae_if_needed": True, # If ONNX input shape mismatches feature count, train a local sequence AE on TRAIN
    "existing_threshold": 0.2598,
    "episode_length": 150,
    "n_fault_episodes_per_type": 20,
    "output_dir": "eval_outputs",
}


# ==============================================================================
# 2. CONFIG & DATA LOADING + STRICT CONTIGUOUS SPLIT (NO LEAKAGE)
# ==============================================================================
def _load_yaml_safe(path_str: Optional[str]) -> dict:
    if not path_str or yaml is None:
        return {}
    p = Path(path_str)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def resolve_runtime_config(cfg: dict) -> dict:
    resolved = dict(cfg)
    sig_cfg = _load_yaml_safe(cfg.get("signals_yaml"))
    mod_cfg = _load_yaml_safe(cfg.get("model_config_yaml"))

    # Resolve dataset path
    if not resolved.get("data_path"):
        for cand in resolved.get("candidate_data_paths", []):
            if Path(cand).exists():
                resolved["data_path"] = cand
                break
    if not resolved.get("data_path") or not Path(resolved["data_path"]).exists():
        raise FileNotFoundError(
            "No telemetry CSV found. Set CONFIG['data_path'] to your CSV file."
        )

    df_peek = pd.read_csv(resolved["data_path"], nrows=5)

    # Resolve sampling_hz
    if resolved.get("sampling_hz") == "auto":
        hz = None
        for src in (sig_cfg, mod_cfg):
            for key in ("sampling_hz", "sample_rate_hz", "hz", "fs"):
                if key in src and isinstance(src[key], (int, float)):
                    hz = float(src[key])
                    break
            if hz is not None:
                break
        resolved["sampling_hz"] = hz if hz is not None else 10.0

    # Resolve window_size
    if resolved.get("window_size") == "auto":
        w = None
        for src in (mod_cfg, sig_cfg):
            for key in ("seq_len", "sequence_length", "window_size", "lookback"):
                if key in src and isinstance(src[key], int):
                    w = int(src[key])
                    break
            if w is not None:
                break
        resolved["window_size"] = w if w is not None else 50

    # Resolve cycle_col
    if resolved.get("cycle_col") == "auto":
        candidates = ["cycle_id", "drive_cycle", "cycle", "episode_id", "run_id", "segment_id"]
        resolved["cycle_col"] = next((c for c in candidates if c in df_peek.columns), None)

    # Resolve time_col
    if resolved.get("time_col") == "auto":
        candidates = ["timestamp", "time", "datetime", "date", "t_sec", "elapsed_time"]
        resolved["time_col"] = next((c for c in candidates if c in df_peek.columns), None)

    # Resolve feature_cols
    if resolved.get("feature_cols") == "auto":
        extracted: List[str] = []
        for key in ("features", "feature_cols", "signals", "channels", "sensors"):
            val = sig_cfg.get(key)
            if isinstance(val, list) and len(val) > 0:
                if all(isinstance(x, str) for x in val):
                    extracted = val
                elif all(isinstance(x, dict) and "name" in x for x in val):
                    extracted = [x["name"] for x in val]
                break

        available_in_df = [c for c in extracted if c in df_peek.columns]
        if available_in_df:
            resolved["feature_cols"] = available_in_df
        else:
            exclude = {
                resolved["cycle_col"],
                resolved["time_col"],
                "label",
                "anomaly",
                "is_anomaly",
                "fault",
                "is_fault",
                "ai_score",
                "anomaly_score",
                "reconstruction_error",
            }
            numeric_cols = [
                c
                for c in df_peek.select_dtypes(include=[np.number]).columns
                if c not in exclude and not c.startswith("Unnamed")
            ]
            if not numeric_cols:
                raise ValueError(f"Could not determine numeric feature columns from {resolved['data_path']}.")
            resolved["feature_cols"] = numeric_cols

    return resolved


def load_and_split_contiguous(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    df = pd.read_csv(cfg["data_path"])
    time_col = cfg.get("time_col")
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    feat_cols = cfg["feature_cols"]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {cfg['data_path']}. Available: {list(df.columns)}")

    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    train_r, val_r, test_r = cfg["split_ratios"]
    assert abs((train_r + val_r + test_r) - 1.0) < 1e-6, "split_ratios must sum to 1.0"

    cycle_col = cfg.get("cycle_col")
    if cycle_col and cycle_col in df.columns and df[cycle_col].nunique() >= 3:
        cycles = df[cycle_col].drop_duplicates().tolist()
        n_c = len(cycles)
        n_tr = max(1, int(round(n_c * train_r)))
        n_va = max(1, int(round(n_c * val_r)))
        tr_c = set(cycles[:n_tr])
        va_c = set(cycles[n_tr : n_tr + n_va])
        te_c = set(cycles[n_tr + n_va :])

        train_df = df[df[cycle_col].isin(tr_c)].reset_index(drop=True)
        val_df = df[df[cycle_col].isin(va_c)].reset_index(drop=True)
        test_df = df[df[cycle_col].isin(te_c)].reset_index(drop=True)
        desc = (
            f"Drive-cycle split on '{cycle_col}' (no shuffling): "
            f"Train={len(tr_c)} cycles ({len(train_df)} rows, {train_r*100:.0f}% target), "
            f"Val={len(va_c)} cycles ({len(val_df)} rows, {val_r*100:.0f}% target), "
            f"Test={len(te_c)} cycles ({len(test_df)} rows, {test_r*100:.0f}% target)."
        )
    else:
        n = len(df)
        i_tr = int(n * train_r)
        i_va = int(n * (train_r + val_r))
        train_df = df.iloc[:i_tr].reset_index(drop=True)
        val_df = df.iloc[i_tr:i_va].reset_index(drop=True)
        test_df = df.iloc[i_va:].reset_index(drop=True)
        desc = (
            f"Chronological contiguous split (no shuffling): "
            f"Train rows [0:{i_tr}] ({len(train_df)} rows, {train_r*100:.0f}%), "
            f"Val rows [{i_tr}:{i_va}] ({len(val_df)} rows, {val_r*100:.0f}%), "
            f"Test rows [{i_va}:{n}] ({len(test_df)} rows, {test_r*100:.0f}%)."
        )
    return train_df, val_df, test_df, desc


def make_sliding_windows(arr_2d: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    if len(arr_2d) < window_size:
        return np.empty((0, window_size, arr_2d.shape[1]), dtype=np.float32)
    starts = range(0, len(arr_2d) - window_size + 1, stride)
    return np.stack([arr_2d[s : s + window_size] for s in starts], axis=0).astype(np.float32)


# ==============================================================================
# 3. REPRODUCIBLE SYNTHETIC FAULT INJECTION ON HELD-OUT TEST EPISODES
# ==============================================================================
@dataclass
class FaultManifestRow:
    episode_id: int
    is_fault: int
    fault_type: str
    fault_channel: str
    secondary_channel: str
    onset_step: int
    severity_info: str


def build_fault_test_episodes(
    test_raw: np.ndarray,
    train_raw: np.ndarray,
    feature_cols: List[str],
    cfg: dict,
) -> Tuple[List[np.ndarray], pd.DataFrame]:
    rng = np.random.default_rng(cfg["seed"])
    ep_len = int(cfg["episode_length"])
    win_size = int(cfg["window_size"])
    n_per_type = int(cfg["n_fault_episodes_per_type"])
    n_feat = len(feature_cols)

    if len(test_raw) < ep_len:
        ep_len = max(win_size + 10, len(test_raw) // 4)

    n_base = len(test_raw) // ep_len
    if n_base < 2:
        raise ValueError(f"Not enough test rows ({len(test_raw)}) for episode_length={ep_len}.")

    base_episodes = [test_raw[i * ep_len : (i + 1) * ep_len].copy() for i in range(n_base)]
    ch_std = np.std(train_raw, axis=0) + 1e-6

    episodes: List[np.ndarray] = []
    manifest: List[FaultManifestRow] = []
    ep_id = 0

    # Healthy held-out episodes (label = 0)
    for ep in base_episodes:
        episodes.append(ep.copy())
        manifest.append(
            FaultManifestRow(
                episode_id=ep_id,
                is_fault=0,
                fault_type="healthy",
                fault_channel="none",
                secondary_channel="none",
                onset_step=-1,
                severity_info="unmodified_held_out_test",
            )
        )
        ep_id += 1

    fault_types = [
        "sudden_spike",
        "gradual_drift",
        "stuck_at_flatline",
        "noise_increase",
        "cross_signal_idle_overheat",
    ]

    rpm_idx = next((i for i, c in enumerate(feature_cols) if "rpm" in c.lower()), 0)
    temp_idx = next((i for i, c in enumerate(feature_cols) if "temp" in c.lower()), min(1, n_feat - 1))

    min_onset = max(1, win_size // 2)
    max_onset = max(min_onset + 1, ep_len - win_size)

    for ftype in fault_types:
        for _ in range(n_per_type):
            src = base_episodes[int(rng.integers(0, len(base_episodes)))].copy()
            onset = int(rng.integers(min_onset, max_onset))
            ch = int(rng.integers(0, n_feat))
            sec_ch = "none"

            if ftype == "sudden_spike":
                sign = float(rng.choice([-1.0, 1.0]))
                mag = float(rng.uniform(3.5, 5.5)) * ch_std[ch]
                dur = int(rng.integers(6, min(20, ep_len - onset)))
                end = min(ep_len, onset + dur)
                src[onset:end, ch] += sign * mag
                sev = f"delta={sign*mag:+.3f} ({mag/ch_std[ch]:.2f}x train_std) over {end-onset} steps"

            elif ftype == "gradual_drift":
                sign = float(rng.choice([-1.0, 1.0]))
                target_shift = float(rng.uniform(3.0, 5.0)) * ch_std[ch]
                ramp = np.linspace(0.0, sign * target_shift, ep_len - onset)
                src[onset:, ch] += ramp
                sev = f"ramp_to={sign*target_shift:+.3f} ({target_shift/ch_std[ch]:.2f}x train_std)"

            elif ftype == "stuck_at_flatline":
                frozen_val = float(src[onset, ch])
                src[onset:, ch] = frozen_val
                sev = f"held_constant={frozen_val:.3f}"

            elif ftype == "noise_increase":
                sigma = float(rng.uniform(2.5, 4.0)) * ch_std[ch]
                src[onset:, ch] += rng.normal(0.0, sigma, size=ep_len - onset)
                sev = f"gaussian_sigma={sigma:.3f} ({sigma/ch_std[ch]:.2f}x train_std)"

            elif ftype == "cross_signal_idle_overheat":
                ch = temp_idx
                sec_ch = feature_cols[rpm_idx]
                idle_rpm = float(np.percentile(train_raw[:, rpm_idx], 10))
                hot_temp = float(np.percentile(train_raw[:, temp_idx], 95)) + 0.5 * ch_std[temp_idx]
                src[onset:, rpm_idx] = idle_rpm + rng.normal(0.0, 0.05 * ch_std[rpm_idx], size=ep_len - onset)
                src[onset:, temp_idx] = hot_temp + rng.normal(0.0, 0.05 * ch_std[temp_idx], size=ep_len - onset)
                sev = f"{feature_cols[rpm_idx]}~{idle_rpm:.2f}, {feature_cols[temp_idx]}~{hot_temp:.2f}"

            episodes.append(src)
            manifest.append(
                FaultManifestRow(
                    episode_id=ep_id,
                    is_fault=1,
                    fault_type=ftype,
                    fault_channel=feature_cols[ch],
                    secondary_channel=sec_ch,
                    onset_step=onset,
                    severity_info=sev,
                )
            )
            ep_id += 1

    return episodes, pd.DataFrame([asdict(m) for m in manifest])


# ==============================================================================
# 4. AUTOENCODER & BASELINE MODELS
# ==============================================================================
class SequenceAutoencoderEvaluator:
    def __init__(self, cfg: dict, train_windows: np.ndarray):
        self.mode = "onnx"
        self.session = None
        self.fallback_mlp = None
        self.shape_wf = train_windows.shape[1:]

        onnx_path = Path(cfg["onnx_model_path"])
        if onnx_path.exists():
            try:
                import onnxruntime as ort

                sess = ort.InferenceSession(str(onnx_path))
                inp = sess.get_inputs()[0]
                expected_feat = inp.shape[-1] if len(inp.shape) >= 3 else None
                expected_win = inp.shape[1] if len(inp.shape) >= 3 else None

                feat_ok = (expected_feat is None) or (not isinstance(expected_feat, int)) or (expected_feat == train_windows.shape[2])
                win_ok = (expected_win is None) or (not isinstance(expected_win, int)) or (expected_win == train_windows.shape[1])

                if feat_ok and win_ok:
                    self.session = sess
                    self.input_name = inp.name
                    self.mode = "onnx"
                    print(f"Using ONNX model: {onnx_path} (input shape={inp.shape})")
                else:
                    raise ValueError(
                        f"ONNX input shape {inp.shape} does not match windows {train_windows.shape}."
                    )
            except Exception as exc:
                print(f"Warning: Could not run ONNX model directly ({exc}).")
                if not cfg.get("train_fallback_ae_if_needed", True):
                    raise
                self.mode = "fallback_mlp"
        else:
            if not cfg.get("train_fallback_ae_if_needed", True):
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
            self.mode = "fallback_mlp"

        if self.mode == "fallback_mlp":
            from sklearn.neural_network import MLPRegressor

            n, w, f = train_windows.shape
            flat = train_windows.reshape(n, w * f)
            self.fallback_mlp = MLPRegressor(
                hidden_layer_sizes=(64, 16, 64),
                max_iter=60,
                random_state=cfg["seed"],
                early_stopping=True,
            )
            self.fallback_mlp.fit(flat, flat)
            print("Fitted local fallback reconstruction autoencoder strictly on TRAIN windows.")

    def reconstruct_and_errors(self, windows_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            window_mse: (N,) mean squared reconstruction error across time and features
            feature_mse: (N, n_features) mean squared reconstruction error per feature
        """
        if len(windows_scaled) == 0:
            return np.array([]), np.empty((0, windows_scaled.shape[2]))

        x = windows_scaled.astype(np.float32)
        if self.mode == "onnx":
            recon = self.session.run(None, {self.input_name: x})[0]
        else:
            flat = x.reshape(len(x), -1)
            recon = self.fallback_mlp.predict(flat).reshape(len(x), *self.shape_wf)

        sq_diff = (x - recon) ** 2
        feature_mse = np.mean(sq_diff, axis=1)
        window_mse = np.mean(feature_mse, axis=1)
        return window_mse, feature_mse


class StaticBoundsBaseline:
    def __init__(self, std_margin: float = 0.5):
        self.std_margin = std_margin

    def fit(self, train_raw: np.ndarray):
        std = np.std(train_raw, axis=0)
        self.low = np.min(train_raw, axis=0) - self.std_margin * std
        self.high = np.max(train_raw, axis=0) + self.std_margin * std
        return self

    def score_windows(self, windows_raw: np.ndarray) -> np.ndarray:
        below = np.maximum(0.0, self.low[None, None, :] - windows_raw)
        above = np.maximum(0.0, windows_raw - self.high[None, None, :])
        span = (self.high - self.low) + 1e-6
        return np.max((below + above) / span[None, None, :], axis=(1, 2))


class PCABaseline:
    def __init__(self, variance_ratio: float = 0.90, seed: int = 42):
        self.pca = PCA(n_components=variance_ratio, random_state=seed)

    def fit(self, windows_scaled: np.ndarray):
        flat = windows_scaled.reshape(len(windows_scaled), -1)
        self.pca.fit(flat)
        return self

    def score_windows(self, windows_scaled: np.ndarray) -> np.ndarray:
        flat = windows_scaled.reshape(len(windows_scaled), -1)
        recon = self.pca.inverse_transform(self.pca.transform(flat))
        return np.mean((flat - recon) ** 2, axis=1)


class IsolationForestEvaluator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.use_saved = False

    @staticmethod
    def _window_stats(windows_scaled: np.ndarray) -> np.ndarray:
        mean = np.mean(windows_scaled, axis=1)
        std = np.std(windows_scaled, axis=1)
        wmin = np.min(windows_scaled, axis=1)
        wmax = np.max(windows_scaled, axis=1)
        return np.hstack([mean, std, wmin, wmax])

    def fit(self, windows_scaled: np.ndarray):
        p = Path(self.cfg.get("iso_model_path", ""))
        if p.exists():
            try:
                loaded = joblib.load(p)
                stats = self._window_stats(windows_scaled)
                if hasattr(loaded, "n_features_in_") and loaded.n_features_in_ == stats.shape[1]:
                    self.model = loaded
                    self.use_saved = True
                    return self
            except Exception:
                pass

        self.model = IsolationForest(
            n_estimators=150, contamination="auto", random_state=self.cfg["seed"]
        )
        self.model.fit(self._window_stats(windows_scaled))
        self.use_saved = False
        return self

    def score_windows(self, windows_scaled: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(self._window_stats(windows_scaled))


# ==============================================================================
# 5. EVALUATION, ROOT CAUSE, LATENCY & CURVES
# ==============================================================================
def binary_report(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    healthy_far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "healthy_false_alarm_rate": float(healthy_far),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def run_episode_evaluation(
    episodes_raw: List[np.ndarray],
    manifest_df: pd.DataFrame,
    scaler: StandardScaler,
    ae: SequenceAutoencoderEvaluator,
    rules: StaticBoundsBaseline,
    pca_b: PCABaseline,
    iso_b: IsolationForestEvaluator,
    feature_cols: List[str],
    window_size: int,
    sampling_hz: float,
    ae_thresh: float,
    pca_thresh: float,
    iso_thresh: float,
) -> dict:
    y_true_parts, ae_parts, rule_parts, pca_parts, iso_parts = [], [], [], [], []
    win_records = []
    top1_list, top2_list = [], []
    lat_steps, lat_secs, ep_hit = [], [], []

    for idx, row in manifest_df.iterrows():
        ep_raw = episodes_raw[idx]
        ep_scaled = scaler.transform(ep_raw)

        win_raw = make_sliding_windows(ep_raw, window_size, stride=1)
        win_scaled = make_sliding_windows(ep_scaled, window_size, stride=1)
        n_w = len(win_scaled)
        end_steps = np.arange(window_size - 1, window_size - 1 + n_w)

        if int(row["is_fault"]) == 1:
            w_y = (end_steps >= int(row["onset_step"])).astype(int)
        else:
            w_y = np.zeros(n_w, dtype=int)

        ae_s, f_err = ae.reconstruct_and_errors(win_scaled)
        ru_s = rules.score_windows(win_raw)
        pc_s = pca_b.score_windows(win_scaled)
        is_s = iso_b.score_windows(win_scaled)

        y_true_parts.append(w_y)
        ae_parts.append(ae_s)
        rule_parts.append(ru_s)
        pca_parts.append(pc_s)
        iso_parts.append(is_s)

        for i in range(n_w):
            win_records.append(
                {
                    "episode_id": int(row["episode_id"]),
                    "y_true": int(w_y[i]),
                    "fault_type": row["fault_type"] if w_y[i] == 1 else "healthy",
                    "fault_channel": row["fault_channel"] if w_y[i] == 1 else "none",
                    "ae_score": float(ae_s[i]),
                    "ae_pred": int(ae_s[i] > ae_thresh),
                }
            )

        if int(row["is_fault"]) == 1:
            f_mask = w_y == 1
            f_ends = end_steps[f_mask]
            f_scores = ae_s[f_mask]
            f_errs = f_err[f_mask]

            alerts = np.where(f_scores > ae_thresh)[0]
            if len(alerts) > 0:
                first_i = alerts[0]
                lag_s = int(f_ends[first_i] - int(row["onset_step"]))
                lat_steps.append(lag_s)
                lat_secs.append(lag_s / float(sampling_hz))
                ep_hit.append(1)

                ch_name = row["fault_channel"]
                if ch_name in feature_cols:
                    ch_i = feature_cols.index(ch_name)
                    for err_vec in f_errs[alerts]:
                        order = np.argsort(err_vec)[::-1]
                        top1_list.append(int(order[0] == ch_i))
                        top2_list.append(int(ch_i in order[: min(2, len(feature_cols))]))
            else:
                ep_hit.append(0)

    win_df = pd.DataFrame(win_records)
    active_faults = win_df[win_df["y_true"] == 1]

    by_type = (
        active_faults.groupby("fault_type")
        .agg(
            fault_windows=("ae_pred", "count"),
            detected_windows=("ae_pred", "sum"),
            recall=("ae_pred", "mean"),
            mean_score=("ae_score", "mean"),
        )
        .reset_index()
        .sort_values("recall", ascending=False)
    )

    by_channel = (
        active_faults.groupby("fault_channel")
        .agg(
            fault_windows=("ae_pred", "count"),
            detected_windows=("ae_pred", "sum"),
            recall=("ae_pred", "mean"),
            mean_score=("ae_score", "mean"),
        )
        .reset_index()
        .sort_values("recall", ascending=False)
    )

    return {
        "y_true": np.concatenate(y_true_parts),
        "ae_scores": np.concatenate(ae_parts),
        "rule_scores": np.concatenate(rule_parts),
        "pca_scores": np.concatenate(pca_parts),
        "iso_scores": np.concatenate(iso_parts),
        "by_fault_type": by_type,
        "by_channel": by_channel,
        "root_cause": {
            "top1_accuracy": float(np.mean(top1_list)) if top1_list else float("nan"),
            "top2_accuracy": float(np.mean(top2_list)) if top2_list else float("nan"),
            "alert_windows_checked": int(len(top1_list)),
        },
        "latency": {
            "episode_recall": float(np.mean(ep_hit)) if ep_hit else 0.0,
            "mean_latency_windows": float(np.mean(lat_steps)) if lat_steps else float("nan"),
            "median_latency_windows": float(np.median(lat_steps)) if lat_steps else float("nan"),
            "mean_latency_seconds": float(np.mean(lat_secs)) if lat_secs else float("nan"),
            "median_latency_seconds": float(np.median(lat_secs)) if lat_secs else float("nan"),
        },
    }


def generate_plots(
    y_true: np.ndarray,
    ae_scores: np.ndarray,
    val_thresholds: Dict[str, float],
    existing_thresh: float,
    out_dir: Path,
):
    lo = float(np.percentile(ae_scores, 5))
    hi = float(np.percentile(ae_scores, 98))
    grid = np.linspace(lo, hi, 120)

    prec_vals, rec_vals, f1_vals = [], [], []
    for th in grid:
        yp = (ae_scores > th).astype(int)
        prec_vals.append(precision_score(y_true, yp, zero_division=0))
        rec_vals.append(recall_score(y_true, yp, zero_division=0))
        f1_vals.append(f1_score(y_true, yp, zero_division=0))

    plt.figure(figsize=(9, 5))
    plt.plot(grid, prec_vals, label="Precision", linewidth=2)
    plt.plot(grid, rec_vals, label="Recall", linewidth=2)
    plt.plot(grid, f1_vals, label="F1 Score", linewidth=2.5)
    for label, val in val_thresholds.items():
        if lo <= val <= hi:
            plt.axvline(val, linestyle="--", alpha=0.8, label=f"{label} ({val:.4f})")
    if lo <= existing_thresh <= hi:
        plt.axvline(existing_thresh, color="black", linestyle=":", linewidth=1.8, label=f"Existing ({existing_thresh:.4f})")
    plt.title("Threshold Sensitivity Curve (Held-Out Injected Faults)")
    plt.xlabel("Reconstruction Error Threshold")
    plt.ylabel("Metric Value")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sensitivity_curve.png", dpi=160)
    plt.close()

    p_curve, r_curve, _ = precision_recall_curve(y_true, ae_scores)
    pr_auc = average_precision_score(y_true, ae_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(r_curve, p_curve, linewidth=2.2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.title("Precision-Recall Curve (Held-Out Injected Faults)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.0, 1.02)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=160)
    plt.close()


# ==============================================================================
# 6. MAIN WORKFLOW
# ==============================================================================
def main():
    cfg = resolve_runtime_config(CONFIG)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, split_desc = load_and_split_contiguous(cfg)
    print("\n=== SPLIT SUMMARY ===")
    print(split_desc)
    print(f"Dataset used: {cfg['data_path']}")
    print(f"Features ({len(cfg['feature_cols'])}): {cfg['feature_cols']}")
    print(f"Window size: {cfg['window_size']} | Sampling rate: {cfg['sampling_hz']} Hz")

    train_raw = train_df[cfg["feature_cols"]].to_numpy(dtype=np.float32)
    val_raw = val_df[cfg["feature_cols"]].to_numpy(dtype=np.float32)
    test_raw = test_df[cfg["feature_cols"]].to_numpy(dtype=np.float32)

    scaler_path = Path(cfg["scaler_path"])
    if not cfg.get("fit_fresh_train_scaler", True) and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"Loaded existing scaler from {scaler_path}")
    else:
        scaler = StandardScaler().fit(train_raw)
        print("Fitted StandardScaler strictly on TRAIN split.")

    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)

    train_win = make_sliding_windows(train_scaled, cfg["window_size"], cfg["stride"])
    val_win = make_sliding_windows(val_scaled, cfg["window_size"], cfg["stride"])

    ae = SequenceAutoencoderEvaluator(cfg, train_win)
    rules = StaticBoundsBaseline(std_margin=0.5).fit(train_raw)
    pca_b = PCABaseline(variance_ratio=0.90, seed=cfg["seed"]).fit(train_win)
    iso_b = IsolationForestEvaluator(cfg).fit(train_win)

    # Validation-only thresholds
    val_ae_scores, _ = ae.reconstruct_and_errors(val_win)
    val_pca_scores = pca_b.score_windows(val_win)
    val_iso_scores = iso_b.score_windows(val_win)

    val_thresholds = {
        "val_p95": float(np.percentile(val_ae_scores, 95)),
        "val_p99": float(np.percentile(val_ae_scores, 99)),
        "val_mean_plus_3std": float(np.mean(val_ae_scores) + 3.0 * np.std(val_ae_scores)),
    }
    chosen_ae_thresh = val_thresholds["val_p99"]
    pca_thresh = float(np.percentile(val_pca_scores, 99))
    iso_thresh = float(np.percentile(val_iso_scores, 99))

    episodes_raw, manifest_df = build_fault_test_episodes(test_raw, train_raw, cfg["feature_cols"], cfg)
    manifest_df.to_csv(out_dir / "fault_injection_manifest.csv", index=False)

    res = run_episode_evaluation(
        episodes_raw=episodes_raw,
        manifest_df=manifest_df,
        scaler=scaler,
        ae=ae,
        rules=rules,
        pca_b=pca_b,
        iso_b=iso_b,
        feature_cols=cfg["feature_cols"],
        window_size=cfg["window_size"],
        sampling_hz=cfg["sampling_hz"],
        ae_thresh=chosen_ae_thresh,
        pca_thresh=pca_thresh,
        iso_thresh=iso_thresh,
    )

    y_true = res["y_true"]
    ae_scores = res["ae_scores"]

    rows = []
    for label, th in [
        ("LSTM AE (Val P99)", val_thresholds["val_p99"]),
        ("LSTM AE (Val P95)", val_thresholds["val_p95"]),
        ("LSTM AE (Val Mean+3Std)", val_thresholds["val_mean_plus_3std"]),
        ("LSTM AE (Existing 0.2598)", cfg["existing_threshold"]),
    ]:
        rows.append({"model": label, **binary_report(y_true, ae_scores, th)})

    rows.append({"model": "Baseline: Static Min/Max Rules", **binary_report(y_true, res["rule_scores"], 0.0)})
    rows.append({"model": "Baseline: PCA Reconstruction (Val P99)", **binary_report(y_true, res["pca_scores"], pca_thresh)})
    rows.append({"model": "Baseline: Isolation Forest (Val P99)", **binary_report(y_true, res["iso_scores"], iso_thresh)})

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_dir / "model_comparison.csv", index=False)
    res["by_fault_type"].to_csv(out_dir / "recall_by_fault_type.csv", index=False)
    res["by_channel"].to_csv(out_dir / "recall_by_channel.csv", index=False)

    missed_types = res["by_fault_type"][res["by_fault_type"]["recall"] < 0.50]["fault_type"].tolist()

    generate_plots(y_true, ae_scores, val_thresholds, cfg["existing_threshold"], out_dir)

    summary_payload = {
        "split_description": split_desc,
        "resolved_config": {k: v for k, v in cfg.items() if k != "candidate_data_paths"},
        "validation_thresholds": val_thresholds,
        "model_comparison": rows,
        "root_cause": res["root_cause"],
        "latency": res["latency"],
        "recall_by_fault_type": res["by_fault_type"].to_dict(orient="records"),
        "recall_by_channel": res["by_channel"].to_dict(orient="records"),
        "missed_fault_types_below_50pct_recall": missed_types,
    }
    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    md_summary = "\n".join(
        [
            "# ECU Telemetry Fault-Injection Evaluation Report",
            "",
            f"**Data Split:** {split_desc}",
            "",
            "## Model & Baseline Comparison",
            "",
            comp_df.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Recall by Fault Type (LSTM AE @ Val P99)",
            "",
            res["by_fault_type"].to_markdown(index=False, floatfmt=".4f"),
            "",
            f"**Missed / Low-Recall Fault Types (<0.50 recall):** {', '.join(missed_types) or 'None'}",
            "",
            "## Recall by Faulted Channel",
            "",
            res["by_channel"].to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Root-Cause Isolation & Detection Latency",
            "",
            f"- **Top-1 Channel Accuracy:** {res['root_cause']['top1_accuracy']:.4f}",
            f"- **Top-2 Channel Accuracy:** {res['root_cause']['top2_accuracy']:.4f}",
            f"- **Evaluated Alert Windows:** {res['root_cause']['alert_windows_checked']}",
            f"- **Episode Detection Rate:** {res['latency']['episode_recall']:.4f}",
            f"- **Mean Detection Latency:** {res['latency']['mean_latency_windows']:.2f} windows ({res['latency']['mean_latency_seconds']:.2f} s)",
            f"- **Median Detection Latency:** {res['latency']['median_latency_windows']:.2f} windows ({res['latency']['median_latency_seconds']:.2f} s)",
        ]
    )
    with open(out_dir / "evaluation_summary.md", "w", encoding="utf-8") as f:
        f.write(md_summary)

    print("\n" + md_summary)
    print(f"\nSaved all artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()