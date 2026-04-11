"""
Change detection: detects WHICH component just changed (delta between frames).

Enhancements vs original:
  - Loads per-label optimal thresholds from checkpoints/thresholds.json
    (produced by evaluate.py).  Falls back to symmetric defaults if file absent.
  - 3-frame rolling buffer with hysteresis to reduce false positives.
"""
import json
import torch
import numpy as np
from pathlib import Path
from architecture import CarComponentCNN, LABELS

_DEFAULT_OPEN_THR  = 0.6
_DEFAULT_CLOSE_THR = 0.4


def _load_thresholds() -> tuple[list[float], list[float]]:
    """
    Load per-label optimal thresholds from checkpoints/thresholds.json.

    Returns (open_thresholds, close_thresholds) as lists of length 5.
    If the file doesn't exist, symmetric defaults are used.
    """
    thr_path = Path(__file__).resolve().parents[1] / "checkpoints" / "thresholds.json"
    if not thr_path.exists():
        open_t  = [_DEFAULT_OPEN_THR]  * len(LABELS)
        close_t = [_DEFAULT_CLOSE_THR] * len(LABELS)
        return open_t, close_t

    with open(thr_path) as f:
        thr_map: dict = json.load(f)

    open_t, close_t = [], []
    for label in LABELS:
        opt = thr_map.get(label, 0.5)
        # Apply wider hysteresis band for more stable real-time detection
        hysteresis = 0.12
        open_t.append(min(opt + hysteresis, 0.95))
        close_t.append(max(opt - hysteresis, 0.05))

    print(f"Loaded per-label thresholds from {thr_path}")
    for label, ot, ct in zip(LABELS, open_t, close_t):
        print(f"  {label}: open>={ot:.2f}  close<={ct:.2f}")

    return open_t, close_t


class ChangeDetectorWrapper:
    """
    Wraps the CNN with temporal smoothing:
      1. Rolling buffer of N frames → averaged probabilities.
      2. Hysteresis thresholds (per-label, loaded from evaluate.py output).
      3. Returns current state dict + list of labels that just changed.
    """

    def __init__(self, model: CarComponentCNN, buffer_size: int = 5):
        self.model        = model
        self.buffer_size  = buffer_size
        self.open_thr, self.close_thr = _load_thresholds()
        self.prob_buffer  = []
        self.confirmed_state = {i: 0 for i in range(len(LABELS))}

    def update(self, frame_tensor: torch.Tensor):
        """
        Process one frame.

        Returns
        -------
        current_state : dict  {label_index: 0|1}
        changed       : list  of label names that just flipped
        avg_probs     : list  of averaged sigmoid probabilities (length 5)
        """
        with torch.no_grad():
            logits = self.model(frame_tensor)
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

        self.prob_buffer.append(probs)
        if len(self.prob_buffer) > self.buffer_size:
            self.prob_buffer.pop(0)

        avg_probs = np.mean(self.prob_buffer, axis=0)
        new_state = []
        for i, prob in enumerate(avg_probs):
            prev = self.confirmed_state[i]
            if prob >= self.open_thr[i]:
                new_state.append(1)
            elif prob <= self.close_thr[i]:
                new_state.append(0)
            else:
                new_state.append(prev)   # stay in current state (hysteresis zone)

        new_state = np.asarray(new_state, dtype=int)

        changed = [LABELS[i] for i in range(len(LABELS))
                   if new_state[i] != self.confirmed_state[i]]

        self.confirmed_state = {i: int(new_state[i]) for i in range(len(LABELS))}
        return self.confirmed_state, changed, avg_probs.tolist()

    def reset(self):
        self.prob_buffer     = []
        self.confirmed_state = {i: 0 for i in range(len(LABELS))}
        self.open_thr, self.close_thr = _load_thresholds()
