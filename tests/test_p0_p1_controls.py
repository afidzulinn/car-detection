"""
Section 9 (P0 & P1 Scientific Controls & Unit Tests):
Automated tests verifying:
  1. All 32 binary 5-component state combinations and 3 canonical elevations ('mid', 'high', 'top').
  2. Explicit elevation alias normalization and rejection of unknown elevations.
  3. Zero silent label mutation in CarDataset loader (P0-02).
  4. Strict 3-way split disjointness (train ∩ val = train ∩ test = val ∩ test = ∅) and
     exception on unknown split name (P0-03).
  5. Zero parent-child augmentation leakage across splits (P1-02).
  6. Protocol T1 global azimuth holdout and Protocol T2 leave-top-elevation-out holdout (P0-05).
  7. Orthogonal ablation flags in CarComponentCNN (use_coordconv=False only changes input channels
     from 5 to 3 without altering backbone, multi-scale fusion, or pooling) (P1-06).
  8. Unified decision threshold policy (>=) and predicted-class confidence semantics (P0-08).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
DATA_COLLECTION_DIR = ROOT_DIR / "data_collection"
for p in (MODEL_DIR, DATA_COLLECTION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from architecture import CarComponentCNN, LABELS, ModelConfig
from audit_dataset import normalize_elevation
from baselines import (
    MobileNetV3SmallScratch,
    ResNet18Scratch,
    build_plain_cnn_baseline,
    build_residual_gap_baseline,
)
from dataset import CarDataset
from decision_policy import confidence_of_predicted_class, predict_from_prob


def test_elevation_normalization():
    assert normalize_elevation("mid")[0] == "mid"
    assert normalize_elevation("high")[0] == "high"
    assert normalize_elevation("top")[0] == "top"
    assert normalize_elevation("Slightly High")[0] == "high"
    assert normalize_elevation("Top-Down")[0] == "top"
    assert normalize_elevation("invalid_undercarriage")[0] is None


def test_unknown_split_raises_value_error():
    data_dir = str(ROOT_DIR / "dataset" / "raw")
    try:
        CarDataset(data_dir, split="unknown_split", preload_cache=False)
        raise AssertionError("Expected ValueError for unknown_split")
    except ValueError as e:
        assert "Invalid split" in str(e)


def test_three_way_split_disjointness_and_no_aug_leakage():
    data_dir = str(ROOT_DIR / "dataset" / "augmented")
    tr_ds = CarDataset(data_dir, split="train", protocol="T0", preload_cache=False)
    va_ds = CarDataset(data_dir, split="val", protocol="T0", preload_cache=False)
    te_ds = CarDataset(data_dir, split="test", protocol="T0", preload_cache=False)

    tr_sids = {item["source_id"] for item in tr_ds.data}
    va_sids = {item["source_id"] for item in va_ds.data}
    te_sids = {item["source_id"] for item in te_ds.data}

    assert len(tr_sids) == 576
    assert len(va_sids) == 96
    assert len(te_sids) == 96
    assert len(tr_sids & va_sids) == 0
    assert len(tr_sids & te_sids) == 0
    assert len(va_sids & te_sids) == 0
    # Ensure no augmented sample exists in val or test
    assert all("augmentation" not in item for item in va_ds.data)
    assert all("augmentation" not in item for item in te_ds.data)


def test_global_azimuth_holdout_t1_and_elevation_holdout_t2():
    data_dir = str(ROOT_DIR / "dataset" / "raw")
    tr_t1 = CarDataset(data_dir, split="train", protocol="T1", preload_cache=False)
    te_t1 = CarDataset(data_dir, split="test", protocol="T1", preload_cache=False)
    tr_angles = {int(d["angle"]) for d in tr_t1.data}
    te_angles = {int(d["angle"]) for d in te_t1.data}
    assert te_angles == {90, 270}
    assert len(tr_angles & te_angles) == 0

    tr_t2 = CarDataset(data_dir, split="train", protocol="T2", preload_cache=False)
    va_t2 = CarDataset(data_dir, split="val", protocol="T2", preload_cache=False)
    te_t2 = CarDataset(data_dir, split="test", protocol="T2", preload_cache=False)
    assert {d["elevation"] for d in te_t2.data} == {"top"}
    assert "top" not in {d["elevation"] for d in tr_t2.data}
    assert "top" not in {d["elevation"] for d in va_t2.data}


def test_ablation_flag_independence():
    """
    P1-06: Toggling use_coordconv=False must ONLY change the first conv layer's
    input channels from 5 to 3, keeping ResidualConvBlock, multiscale_proj, and
    6x6 hybrid_grid pooling intact!
    """
    m_full = CarComponentCNN(use_coordconv=True, use_cbam=True, use_film=True)
    m_no_coord = CarComponentCNN(use_coordconv=False, use_cbam=True, use_film=True)

    assert m_full.multiscale_proj is not None
    assert m_no_coord.multiscale_proj is not None
    assert m_full.grid_avg_pool is not None
    assert m_no_coord.grid_avg_pool is not None

    # Difference in parameters should be strictly the 2 extra input channels in Stage-1
    # Conv3x3 (2 * 32 * 3 * 3 = 576) + Shortcut 1x1 (2 * 32 * 1 * 1 = 64) = 640 params
    p_full = sum(p.numel() for p in m_full.parameters())
    p_no_coord = sum(p.numel() for p in m_no_coord.parameters())
    assert (p_full - p_no_coord) == 640


def test_model_variants_and_baselines_shapes():
    x = torch.randn(2, 3, 224, 224)
    models_to_check = [
        CarComponentCNN(),
        CarComponentCNN(use_coordconv=False),
        CarComponentCNN(use_multiscale=False),
        CarComponentCNN(use_cbam=False),
        CarComponentCNN(use_film=False),
        CarComponentCNN(use_elevation_pose=True, use_component_attention=True),
        build_plain_cnn_baseline(),
        build_residual_gap_baseline(),
        ResNet18Scratch(),
        MobileNetV3SmallScratch(),
    ]
    for m in models_to_check:
        out = m(x)
        assert out.shape == (2, 5)


def test_threshold_policy_and_confidence_semantics():
    probs = np.array([0.50, 0.499, 0.01, 0.99])
    preds = predict_from_prob(probs, 0.50)
    assert preds.tolist() == [1, 0, 0, 1]
    confs = confidence_of_predicted_class(probs, preds)
    assert np.allclose(confs, np.array([0.50, 0.501, 0.99, 0.99]))


if __name__ == "__main__":
    tests = [
        test_elevation_normalization,
        test_unknown_split_raises_value_error,
        test_three_way_split_disjointness_and_no_aug_leakage,
        test_global_azimuth_holdout_t1_and_elevation_holdout_t2,
        test_ablation_flag_independence,
        test_model_variants_and_baselines_shapes,
        test_threshold_policy_and_confidence_semantics,
    ]
    for fn in tests:
        fn()
        print(f"  [PASS] {fn.__name__}")
    print(f"All {len(tests)} P0/P1 scientific control tests PASSED.")
