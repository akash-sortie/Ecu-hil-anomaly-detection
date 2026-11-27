"""
Synthetic Fault Injector for ECU HIL Signals
Injects labelled faults into normal signal traces for supervised evaluation.
"""
import numpy as np
import pandas as pd
from typing import Literal

FAULT_TYPES = ["voltage_spike", "signal_dropout", "stuck_at", "noise_burst", "timing_jitter"]


def inject_fault(signal: pd.Series, fault_type: str, start: int, duration: int) -> pd.Series:
    s = signal.copy()
    end = min(start + duration, len(s))
    window = s.iloc[start:end]

    if fault_type == "voltage_spike":
        s.iloc[start:end] = window * np.random.uniform(2.5, 4.0)
    elif fault_type == "signal_dropout":
        s.iloc[start:end] = 0.0
    elif fault_type == "stuck_at":
        s.iloc[start:end] = float(window.iloc[0])
    elif fault_type == "noise_burst":
        s.iloc[start:end] = window + np.random.normal(0, window.std() * 3, len(window))
    elif fault_type == "timing_jitter":
        s.iloc[start:end] = window.shift(np.random.randint(3, 8)).fillna(method="bfill")
    return s


def inject_random_faults(df: pd.DataFrame, fault_rate: float = 0.02):
    faulted = df.copy()
    labels = []
    for col in df.columns:
        n_faults = max(1, int(len(df) * fault_rate))
        for _ in range(n_faults):
            start = np.random.randint(0, len(df) - 50)
            duration = np.random.randint(5, 30)
            fault_type = np.random.choice(FAULT_TYPES)
            faulted[col] = inject_fault(faulted[col], fault_type, start, duration)
            labels.append({"signal": col, "fault_type": fault_type,
                           "start_idx": start, "end_idx": start + duration})
    return faulted, pd.DataFrame(labels)
