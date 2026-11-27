"""
Hybrid Evaluation: LSTM Autoencoder @ Val P95 + Stuck/Flatline Detector
------------------------------------------------------------------------
Fits flatline thresholds strictly on TRAIN healthy data, checks/reports
false alarms on VAL healthy data, and evaluates on the exact same
chronological 60/20/20 split and reproducible fault-injected test episodes.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Reuse the exact split, fault injection, and model loader from evaluate_fault_injection.py
from evaluate_fault_injection import (
    CONFIG,
    SequenceAutoencoderEvaluator,
    build_fault_test_episodes,
    load_and_split_contiguous,
    make_sliding_windows,
    resolve_runtime_config,
)


class FlatlineWindowDetector:
    """
    Detects windows where a normally dynamic channel freezes (near-zero first differences
    across a window while its healthy training dynamic range exceeds a minimum activity floor).
    Calibrated strictly on healthy TRAIN data—never on validation or test faults.
    """

    def __init__(self, min_train_diff_std: float = 0.25, flat_ratio_thresh: float = 0.05):
        self.min_train_diff_std = min_train_diff_std
        self.flat_ratio_thresh = flat_ratio_thresh
        self.active_ch_mask = None
        self.baseline_diff_std = None

    def fit(self, train_raw: np.ndarray, window_size: int):
        # Compute per-step first differences on raw training series
        diffs = np.abs(np.diff(train_raw, axis=0))
        self.baseline_diff_std = np.std(diffs, axis=0) + 1e-9
        # Only apply flatline check to channels that actually move in healthy training data
        self.active_ch_mask = self.baseline_diff_std > self.min_train_diff_std
        return self

    def window_flatness_score(self, windows_raw: np.ndarray) -> np.ndarray:
        """
        Returns a score > 1.0 when a normally active channel shows unusually small
        step-to-step changes across the window (0.0 if no active channel is frozen).
        """
        if len(windows_raw) == 0 or not np.any(self.active_ch_mask):
            return np.zeros(len(windows_raw), dtype=np.float64)

        win_diffs = np.abs(np.diff(windows_raw, axis=1))  # (N, W-1, F)
        win_diff_std = np.std(win_diffs, axis=1) + 1e-9   # (N, F)

        ratio = win_diff_std[:, self.active_ch_mask] / self.baseline_diff_std[self.active_ch_mask][None, :]
        # Score > 1.0 means ratio is well below flat_ratio_thresh
        flat_severity = np.maximum(0.0, (self.flat_ratio_thresh - ratio) / self.flat_ratio_thresh)
        return np.max(flat_severity, axis=1)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, continuous_scores: np.ndarray, thresh_label: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "variant": thresh_label,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, continuous_scores)),
        "pr_auc": float(average_precision_score(y_true, continuous_scores)),
        "healthy_false_alarm_rate": float(far),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def main():
    cfg = resolve_runtime_config(CONFIG)
    out_dir = Path(cfg["output_dir"]) / "hybrid_p95_flatline"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, split_desc = load_and_split_contiguous(cfg)
    train_raw = train_df[cfg["feature_cols"]].to_numpy(dtype=np.float32)
    val_raw = val_df[cfg["feature_cols"]].to_numpy(dtype=np.float32)
    test_raw = test_df[cfg["feature_cols"]].to_numpy(dtype=np.float32)

    scaler = StandardScaler().fit(train_raw)
    train_win = make_sliding_windows(scaler.transform(train_raw), cfg["window_size"], cfg["stride"])
    val_win = make_sliding_windows(scaler.transform(val_raw), cfg["window_size"], cfg["stride"])
    val_raw_win = make_sliding_windows(val_raw, cfg["window_size"], cfg["stride"])

    ae = SequenceAutoencoderEvaluator(cfg, train_win)
    flat_det = FlatlineWindowDetector(min_train_diff_std=0.25, flat_ratio_thresh=0.05).fit(
        train_raw, cfg["window_size"]
    )

    # Validation-only thresholds
    val_ae_scores, _ = ae.reconstruct_and_errors(val_win)
    p99_thresh = float(np.percentile(val_ae_scores, 99))
    p95_thresh = float(np.percentile(val_ae_scores, 95))

    # Check flatline false-alarm rate on healthy validation windows
    val_flat_scores = flat_det.window_flatness_score(val_raw_win)
    flat_thresh = 1.0
    val_flat_far = float(np.mean(val_flat_scores > flat_thresh))
    print(f"Val P99={p99_thresh:.6f} | Val P95={p95_thresh:.6f} | Val flatline FAR={val_flat_far:.4f}")

    episodes_raw, manifest_df = build_fault_test_episodes(test_raw, train_raw, cfg["feature_cols"], cfg)

    y_true_list, ae_score_list = [], []
    pred_p99_list, pred_p95_list, pred_hybrid_list = [], [], []
    win_rows = []

    for idx, row in manifest_df.iterrows():
        ep_raw = episodes_raw[idx]
        ep_scaled = scaler.transform(ep_raw)
        w_scaled = make_sliding_windows(ep_scaled, cfg["window_size"], stride=1)
        w_raw = make_sliding_windows(ep_raw, cfg["window_size"], stride=1)

        n_w = len(w_scaled)
        end_steps = np.arange(cfg["window_size"] - 1, cfg["window_size"] - 1 + n_w)
        y_w = (end_steps >= int(row["onset_step"])).astype(int) if int(row["is_fault"]) == 1 else np.zeros(n_w, dtype=int)

        ae_s, _ = ae.reconstruct_and_errors(w_scaled)
        fl_s = flat_det.window_flatness_score(w_raw)

        pred_p99 = (ae_s > p99_thresh).astype(int)
        pred_p95 = (ae_s > p95_thresh).astype(int)
        pred_hyb = ((ae_s > p95_thresh) | (fl_s > flat_thresh)).astype(int)

        y_true_list.append(y_w)
        ae_score_list.append(ae_s)
        pred_p99_list.append(pred_p99)
        pred_p95_list.append(pred_p95)
        pred_hybrid_list.append(pred_hyb)

        for i in range(n_w):
            win_rows.append(
                {
                    "episode_id": int(row["episode_id"]),
                    "y_true": int(y_w[i]),
                    "fault_type": row["fault_type"] if y_w[i] == 1 else "healthy",
                    "p99_pred": int(pred_p99[i]),
                    "p95_pred": int(pred_p95[i]),
                    "hybrid_pred": int(pred_hyb[i]),
                }
            )

    y_true = np.concatenate(y_true_list)
    ae_scores = np.concatenate(ae_score_list)
    pred_p99 = np.concatenate(pred_p99_list)
    pred_p95 = np.concatenate(pred_p95_list)
    pred_hyb = np.concatenate(pred_hybrid_list)
    win_df = pd.DataFrame(win_rows)

    comp_rows = [
        metrics_dict(y_true, pred_p99, ae_scores, f"LSTM AE @ Val P99 ({p99_thresh:.4f})"),
        metrics_dict(y_true, pred_p95, ae_scores, f"LSTM AE @ Val P95 ({p95_thresh:.4f})"),
        metrics_dict(y_true, pred_hyb, ae_scores, f"Hybrid: LSTM AE @ Val P95 + Flatline Rule"),
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_dir / "hybrid_comparison.csv", index=False)

    # Recall per fault type across all three configurations
    active = win_df[win_df["y_true"] == 1]
    by_type = (
        active.groupby("fault_type")
        .agg(
            fault_windows=("hybrid_pred", "count"),
            recall_p99=("p99_pred", "mean"),
            recall_p95=("p95_pred", "mean"),
            recall_hybrid=("hybrid_pred", "mean"),
        )
        .reset_index()
        .sort_values("recall_hybrid", ascending=False)
    )
    by_type.to_csv(out_dir / "hybrid_recall_by_fault_type.csv", index=False)

    summary = {
        "split_description": split_desc,
        "val_p99_threshold": p99_thresh,
        "val_p95_threshold": p95_thresh,
        "val_flatline_false_alarm_rate": val_flat_far,
        "comparison": comp_rows,
        "recall_by_fault_type": by_type.to_dict(orient="records"),
    }
    with open(out_dir / "hybrid_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== COMPARISON ===")
    print(comp_df.to_string(index=False))
    print("\n=== RECALL BY FAULT TYPE ===")
    print(by_type.to_string(index=False))
    print(f"\nSaved outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()