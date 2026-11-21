"""
P0-07, P1-04, P1-05, P1-06:
Systematic P1 Experiment Suite & Experiment Registry Builder for Paper A.

Runs and logs to results/experiment_registry.csv and results/metrics/p1_experiment_suite.json:
  Part A — P1-04 ROI Policy Comparison (PA-SCAN v1 on Protocol T0 3-way split):
    - ROI_A0_legacy_fixed: (0.17, 0.19, 0.83, 0.62) [88.11% mean fg retention, truncates top/high]
    - ROI_A1_full_frame:   (0.00, 0.00, 1.00, 1.00) [100% fg retention, 1.00x density]
    - ROI_A2_full_vehicle: (0.24, 0.08, 0.76, 0.92) [100% fg retention, 2.29x density]

  Part B — P1-05 Fair Comparison Baselines from Scratch (on T0 & full_vehicle ROI):
    - Base_PlainCNN:          5-stage Plain Conv + GAP (1.82M params)
    - Base_ResidualGAP:       5-stage Strided ResNet + GAP (5.00M params)
    - Base_MobileNetV3Small:  MobileNetV3-Small from scratch (1.52M params)
    - Base_ResNet18:          ResNet-18 from scratch (11.18M params)

  Part C — P1-06 Orthogonal Factor-by-Factor Ablation (on T0 & full_vehicle ROI):
    - Abl_1_ResNet_Grid6x6:   Residual + 6x6 Grid (CoordConv=OFF, MultiScale=OFF, CBAM=OFF, FiLM=OFF)
    - Abl_2_Plus_MultiScale:  + Stage-4/5 Multi-Scale Fusion (CoordConv=OFF, MultiScale=ON, CBAM=OFF, FiLM=OFF)
    - Abl_3_Plus_CoordConv:   + 5-Ch CoordConv (CoordConv=ON, MultiScale=ON, CBAM=OFF, FiLM=OFF)
    - Abl_4_Plus_CBAM:        + Unit-Centered Residual CBAM (CoordConv=ON, MultiScale=ON, CBAM=ON, FiLM=OFF)
    - Abl_5_Full_PASCAN_v1:   + Pose-Conditioned FiLM (CoordConv=ON, MultiScale=ON, CBAM=ON, FiLM=ON)

All thresholds are calibrated ONLY on Validation (N=96) and evaluated ONCE on Independent Test (N=96).
"""

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from architecture import CarComponentCNN, LABELS, ModelConfig
from baselines import (
    MobileNetV3SmallScratch,
    ResNet18Scratch,
    build_plain_cnn_baseline,
    build_residual_gap_baseline,
)
from dataset import CarDataset, get_dataloaders
from decision_policy import calibrate_thresholds_on_val
from evaluate import _collect_split_outputs, _compute_split_metrics
from train import compute_pos_weight, set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = str(ROOT_DIR / "dataset" / "augmented")


def train_and_evaluate_variant(
    exp_id: str,
    group: str,
    description: str,
    model_builder,
    roi_policy: str = "full_vehicle",
    protocol: str = "T0",
    epochs: int = 35,
    batch_size: int = 16,
    lr: float = 2.5e-3,
    seed: int = 42,
    use_pose_loss: bool = True,
) -> dict:
    set_seed(seed)
    use_amp = DEVICE == "cuda"
    model = model_builder().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_dl, val_dl, test_dl = get_dataloaders(
        DATA_DIR,
        batch_size=batch_size,
        return_angle_train=True,
        max_aug_per_sample=0,
        protocol=protocol,
        roi_policy=roi_policy,
        include_test=True,
    )

    pos_weight = compute_pos_weight(train_dl.dataset).to(DEVICE)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_dl),
        pct_start=0.2,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels, angle_idx in train_dl:
            imgs = imgs.to(DEVICE, non_blocking=use_amp)
            labels = labels.to(DEVICE, non_blocking=use_amp)
            angle_idx = angle_idx.to(DEVICE, non_blocking=use_amp)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, pose_logits = model(imgs, return_pose=True)
                smooth_y = labels * (1.0 - 0.03) + 0.5 * 0.03
                loss_bce = bce_criterion(logits, smooth_y)
                if use_pose_loss and getattr(model, "use_film", False):
                    loss = loss_bce + 0.35 * ce_criterion(pose_logits, angle_idx)
                else:
                    loss = loss_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0.0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                bs = imgs.size(0)
                imgs = imgs.to(DEVICE, non_blocking=use_amp)
                labels = labels.to(DEVICE, non_blocking=use_amp)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(imgs)
                    v_loss = bce_criterion(logits, labels).item()
                val_loss += v_loss * bs
                preds = (torch.sigmoid(logits) >= 0.50).float()
                correct += (preds == labels).float().sum().item()
                total += bs * len(LABELS)

        val_acc = correct / max(total, 1)
        val_loss /= max(total // len(LABELS), 1)
        if (val_acc > best_val_acc + 1e-6) or (
            abs(val_acc - best_val_acc) <= 1e-6 and val_loss < best_val_loss
        ):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_time_s = round(time.time() - t_start, 2)
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate on Validation (N=96) -> calibrate thresholds -> freeze -> evaluate on Independent Test (N=96)
    val_out = _collect_split_outputs(model, val_dl.dataset)
    test_out = _collect_split_outputs(model, test_dl.dataset)

    default_thr = {l: 0.50 for l in LABELS}
    frozen_val_thr = calibrate_thresholds_on_val(val_out["probs"], val_out["targets"], LABELS)

    val_m05 = _compute_split_metrics(
        val_out["probs"], val_out["targets"], default_thr,
        val_out["elevations"], val_out["pose_preds"], val_out["pose_targets"]
    )
    val_mopt = _compute_split_metrics(
        val_out["probs"], val_out["targets"], frozen_val_thr,
        val_out["elevations"], val_out["pose_preds"], val_out["pose_targets"]
    )
    test_m05 = _compute_split_metrics(
        test_out["probs"], test_out["targets"], default_thr,
        test_out["elevations"], test_out["pose_preds"], test_out["pose_targets"]
    )
    test_mopt = _compute_split_metrics(
        test_out["probs"], test_out["targets"], frozen_val_thr,
        test_out["elevations"], test_out["pose_preds"], test_out["pose_targets"]
    )

    print(
        f"[{exp_id:<24}] Params={n_params/1e6:5.2f}M | "
        f"Val Acc@0.5={val_m05['mean_accuracy']:.2%} | "
        f"Test Acc@0.5={test_m05['mean_accuracy']:.2%} (Exact={test_m05['exact_match']:.2%}) | "
        f"Test Acc@tau*={test_mopt['mean_accuracy']:.2%} (F1={test_mopt['macro_f1']:.4f}, Exact={test_mopt['exact_match']:.2%}, WorstElevF1={test_mopt['worst_elevation_macro_f1']:.4f})"
    )

    return {
        "exp_id": exp_id,
        "group": group,
        "description": description,
        "protocol": protocol,
        "roi_policy": roi_policy,
        "seed": seed,
        "epochs": epochs,
        "params": n_params,
        "params_m": round(n_params / 1e6, 3),
        "train_time_s": train_time_s,
        "frozen_val_thresholds": frozen_val_thr,
        "val_at_050": val_m05,
        "val_at_tuned": val_mopt,
        "test_at_050": test_m05,
        "test_at_frozen_val_thr": test_mopt,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_all_p1_experiments():
    experiments = [
        # Part A: P1-04 ROI Policy Comparison (PA-SCAN v1)
        (
            "E1_ROI_A0_legacy_fixed",
            "P1-04_ROI",
            "PA-SCAN v1 with legacy fixed ROI (0.17, 0.19, 0.83, 0.62)",
            lambda: CarComponentCNN(),
            "legacy_fixed",
            True,
        ),
        (
            "E1_ROI_A1_full_frame",
            "P1-04_ROI",
            "PA-SCAN v1 with uncropped full frame (0.00, 0.00, 1.00, 1.00)",
            lambda: CarComponentCNN(),
            "full_frame",
            True,
        ),
        (
            "E1_ROI_A2_full_vehicle",
            "P1-04_ROI",
            "PA-SCAN v1 with verified 100%-retention full-vehicle ROI (0.24, 0.08, 0.76, 0.92)",
            lambda: CarComponentCNN(),
            "full_vehicle",
            True,
        ),
        # Part B: P1-05 Fair Baselines from Scratch (on full_vehicle ROI)
        (
            "B1_PlainCNN_GAP",
            "P1-05_Baselines",
            "5-stage Plain Conv-BN-ReLU-MaxPool + GlobalAvgPool from scratch",
            build_plain_cnn_baseline,
            "full_vehicle",
            False,
        ),
        (
            "B2_ResidualCNN_GAP",
            "P1-05_Baselines",
            "5-stage Strided Residual Conv + GlobalAvgPool from scratch",
            build_residual_gap_baseline,
            "full_vehicle",
            False,
        ),
        (
            "B3_MobileNetV3Small_Scratch",
            "P1-05_Baselines",
            "MobileNetV3-Small trained from scratch (weights=None)",
            MobileNetV3SmallScratch,
            "full_vehicle",
            False,
        ),
        (
            "B4_ResNet18_Scratch",
            "P1-05_Baselines",
            "ResNet-18 trained from scratch (weights=None)",
            ResNet18Scratch,
            "full_vehicle",
            False,
        ),
        # Part C: P1-06 Orthogonal Factor-by-Factor Ablation (on full_vehicle ROI)
        (
            "Abl1_ResNet_Grid6x6",
            "P1-06_Isolated_Ablation",
            "Residual Backbone + 6x6 Hybrid Grid (CoordConv=OFF, MultiScale=OFF, CBAM=OFF, FiLM=OFF)",
            lambda: CarComponentCNN(
                backbone="residual",
                use_coordconv=False,
                use_multiscale=False,
                use_cbam=False,
                use_film=False,
                pooling="hybrid_grid",
            ),
            "full_vehicle",
            False,
        ),
        (
            "Abl2_Plus_MultiScale",
            "P1-06_Isolated_Ablation",
            "Residual + 6x6 Grid + Stage-4/5 Multi-Scale Fusion (CoordConv=OFF, CBAM=OFF, FiLM=OFF)",
            lambda: CarComponentCNN(
                backbone="residual",
                use_coordconv=False,
                use_multiscale=True,
                use_cbam=False,
                use_film=False,
                pooling="hybrid_grid",
            ),
            "full_vehicle",
            False,
        ),
        (
            "Abl3_Plus_CoordConv",
            "P1-06_Isolated_Ablation",
            "Residual + 6x6 Grid + Multi-Scale + 5-Ch CoordConv (CBAM=OFF, FiLM=OFF)",
            lambda: CarComponentCNN(
                backbone="residual",
                use_coordconv=True,
                use_multiscale=True,
                use_cbam=False,
                use_film=False,
                pooling="hybrid_grid",
            ),
            "full_vehicle",
            False,
        ),
        (
            "Abl4_Plus_CBAM",
            "P1-06_Isolated_Ablation",
            "Residual + 6x6 Grid + Multi-Scale + CoordConv + Unit-Centered CBAM (FiLM=OFF)",
            lambda: CarComponentCNN(
                backbone="residual",
                use_coordconv=True,
                use_multiscale=True,
                use_cbam=True,
                use_film=False,
                pooling="hybrid_grid",
            ),
            "full_vehicle",
            False,
        ),
    ]

    all_results = []
    for exp_id, grp, desc, builder, roi_pol, use_p in experiments:
        res = train_and_evaluate_variant(
            exp_id=exp_id,
            group=grp,
            description=desc,
            model_builder=builder,
            roi_policy=roi_pol,
            protocol="T0",
            epochs=35,
            batch_size=16,
            lr=2.5e-3,
            seed=42,
            use_pose_loss=use_p,
        )
        all_results.append(res)

    # Reuse E1_ROI_A2_full_vehicle as Abl5_Full_PASCAN_v1 in the ablation table
    full_v1 = dict(all_results[2])
    full_v1["exp_id"] = "Abl5_Full_PASCAN_v1"
    full_v1["group"] = "P1-06_Isolated_Ablation"
    full_v1["description"] = "Full PA-SCAN v1 (+ Pose-Conditioned FiLM on full_vehicle ROI)"
    all_results.append(full_v1)

    out_dir = ROOT_DIR / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "p1_experiment_suite.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Save P0-07 Experiment Registry CSV
    reg_path = ROOT_DIR / "results" / "experiment_registry.csv"
    with open(reg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id",
            "group",
            "protocol",
            "roi_policy",
            "seed",
            "epochs",
            "params_m",
            "val_acc_050",
            "val_f1_050",
            "val_exact_050",
            "test_acc_050",
            "test_f1_050",
            "test_exact_050",
            "test_acc_frozen_opt",
            "test_f1_frozen_opt",
            "test_exact_frozen_opt",
            "test_roc_auc",
            "test_worst_elev_f1",
            "test_ece",
            "test_brier",
            "timestamp_utc",
        ])
        for r in all_results:
            v05 = r["val_at_050"]
            t05 = r["test_at_050"]
            topt = r["test_at_frozen_val_thr"]
            writer.writerow([
                r["exp_id"],
                r["group"],
                r["protocol"],
                r["roi_policy"],
                r["seed"],
                r["epochs"],
                r["params_m"],
                v05["mean_accuracy"],
                v05["macro_f1"],
                v05["exact_match"],
                t05["mean_accuracy"],
                t05["macro_f1"],
                t05["exact_match"],
                topt["mean_accuracy"],
                topt["macro_f1"],
                topt["exact_match"],
                topt["macro_roc_auc"],
                topt["worst_elevation_macro_f1"],
                topt["calibration"]["macro_ece"],
                topt["calibration"]["macro_brier_score"],
                r["timestamp_utc"],
            ])

    print(f"\n[P0-07/P1] Saved full experiment suite to {json_path} and registry to {reg_path}")


if __name__ == "__main__":
    run_all_p1_experiments()
