"""
Automated Ablation Study for Paper 1:
Evaluates the progressive architectural contribution of each module in Multi-Scale PA-SCAN:
  1. Baseline Standard CNN           (RGB 3-ch, GlobalAvgPool 1x1, no CoordConv/CBAM/FiLM)
  2. CNN + CoordConv                 (+ 5-ch CoordConv & 6x6 Multi-Scale Spatial Grid)
  3. CNN + CoordConv + CBAM          (+ Channel SE & 2D Spatial Attention)
  4. Full PA-SCAN (+ FiLM Pose-Cond) (+ Auxiliary Pose Head & Pose-Conditioned FiLM Gating)

Outputs:
  - Per-variant checkpoints in checkpoints/ablation/
  - JSON metrics in docs/tables/ablation_results.json
  - LaTeX table in docs/tables/ablation_study.tex

Usage:
    uv run python -u model/ablation_study.py
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from architecture import CarComponentCNN, LABELS
from dataset import CarDataset, get_dataloaders
from evaluate import find_optimal_threshold

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_AUG = ROOT_DIR / "dataset" / "augmented"
DATA_RAW = ROOT_DIR / "dataset" / "raw"
DEFAULT_DATA = str(DATA_AUG if (DATA_AUG / "annotations.json").exists() else DATA_RAW)
ABLATION_CKPT_DIR = ROOT_DIR / "checkpoints" / "ablation"
ABLATION_CKPT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


ABLATION_CONFIGS = [
    {
        "id": "v1_baseline_cnn",
        "name": "Baseline Standard CNN",
        "use_coordconv": False,
        "use_cbam": False,
        "use_film": False,
    },
    {
        "id": "v2_cnn_coordconv",
        "name": "CNN + CoordConv",
        "use_coordconv": True,
        "use_cbam": False,
        "use_film": False,
    },
    {
        "id": "v3_cnn_coordconv_cbam",
        "name": "CNN + CoordConv + CBAM",
        "use_coordconv": True,
        "use_cbam": True,
        "use_film": False,
    },
    {
        "id": "v4_full_pascan",
        "name": "Full PA-SCAN (+ FiLM)",
        "use_coordconv": True,
        "use_cbam": True,
        "use_film": True,
    },
]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def compute_pos_weight(train_ds: CarDataset) -> torch.Tensor:
    pos = torch.zeros(len(LABELS))
    neg = torch.zeros(len(LABELS))
    for item in train_ds.data:
        for i, label in enumerate(LABELS):
            if item["labels"][label] == 1:
                pos[i] += 1
            else:
                neg[i] += 1
    return neg / pos.clamp(min=1)


def evaluate_model(model: nn.Module, val_dl, device: str) -> dict:
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs = imgs.to(device, non_blocking=(device == "cuda"))
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    # Metrics @ 0.50
    preds_05 = (all_probs >= 0.50).astype(int)
    per_label_acc_05 = (preds_05 == all_targets).mean(axis=0)
    mean_acc_05 = float((preds_05 == all_targets).mean())
    exact_05 = float(np.all(preds_05 == all_targets, axis=1).mean())

    # Metrics @ optimal threshold & ROC-AUC
    preds_opt = np.zeros_like(preds_05)
    f1s_opt = []
    aucs = []
    for i in range(len(LABELS)):
        thr = find_optimal_threshold(all_probs[:, i], all_targets[:, i])
        preds_opt[:, i] = (all_probs[:, i] >= thr).astype(int)
        f1s_opt.append(float(f1_score(all_targets[:, i], preds_opt[:, i], zero_division=0)))
        aucs.append(float(roc_auc_score(all_targets[:, i], all_probs[:, i])))

    per_label_acc_opt = (preds_opt == all_targets).mean(axis=0)
    mean_acc_opt = float((preds_opt == all_targets).mean())
    exact_opt = float(np.all(preds_opt == all_targets, axis=1).mean())

    return {
        "mean_acc_05": mean_acc_05,
        "exact_05": exact_05,
        "per_label_acc_05": {LABELS[i]: float(per_label_acc_05[i]) for i in range(len(LABELS))},
        "mean_acc_opt": mean_acc_opt,
        "exact_opt": exact_opt,
        "macro_f1_opt": float(np.mean(f1s_opt)),
        "macro_auc": float(np.mean(aucs)),
        "per_label_acc_opt": {LABELS[i]: float(per_label_acc_opt[i]) for i in range(len(LABELS))},
        "per_label_auc": {LABELS[i]: aucs[i] for i in range(len(LABELS))},
    }


def train_and_evaluate_variant(
    cfg: dict,
    train_dl,
    val_dl,
    pos_weight: torch.Tensor,
    device: str,
    epochs: int = 40,
    lr: float = 2.5e-3,
    pose_weight: float = 0.35,
    label_smoothing: float = 0.03,
    reuse_full_ckpt: bool = True,
) -> dict:
    set_seed(SEED)
    use_amp = device.startswith("cuda")

    model = CarComponentCNN(
        use_coordconv=cfg["use_coordconv"],
        use_cbam=cfg["use_cbam"],
        use_film=cfg["use_film"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_path = ABLATION_CKPT_DIR / f"{cfg['id']}.pt"
    main_ckpt = ROOT_DIR / "checkpoints" / "best_model.pt"

    print("\n" + "=" * 72)
    print(
        f"VARIANT: {cfg['name']} | Params: {n_params:,} ({n_params/1e6:.2f}M) | "
        f"CoordConv={cfg['use_coordconv']} | CBAM={cfg['use_cbam']} | FiLM={cfg['use_film']}"
    )
    print("=" * 72)

    # For Variant 4 (Full PA-SCAN), reuse main checkpoint if present so numbers match main evaluation
    if cfg["id"] == "v4_full_pascan" and reuse_full_ckpt and main_ckpt.exists():
        print(f"  Loading verified Full PA-SCAN weights from {main_ckpt}...")
        state = torch.load(main_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        torch.save(state, ckpt_path)
    else:
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        ce_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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

        best_val_loss = float("inf")
        best_mean_acc = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            train_bce = 0.0
            for imgs, labels, angle_idx in train_dl:
                imgs = imgs.to(device, non_blocking=use_amp)
                labels = labels.to(device, non_blocking=use_amp)
                angle_idx = angle_idx.to(device, non_blocking=use_amp)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, pose_logits = model(imgs, return_pose=True)
                    smooth_labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
                    loss_bce = bce_criterion(logits, smooth_labels)
                    if cfg["use_film"]:
                        loss_pose = ce_criterion(pose_logits, angle_idx)
                        loss = loss_bce + pose_weight * loss_pose
                    else:
                        loss = loss_bce

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_bce += loss_bce.item()

            # Validate
            model.eval()
            val_loss = 0.0
            per_label_acc = torch.zeros(len(LABELS))
            exact_match = 0.0
            n_val = 0
            with torch.no_grad():
                for imgs, labels in val_dl:
                    bs = imgs.size(0)
                    imgs = imgs.to(device, non_blocking=use_amp)
                    labels = labels.to(device, non_blocking=use_amp)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(imgs)
                        batch_loss = bce_criterion(logits, labels).item()
                    val_loss += batch_loss * bs
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    per_label_acc += (preds == labels).float().sum(dim=0).cpu()
                    exact_match += (preds == labels).all(dim=1).float().sum().item()
                    n_val += bs

            val_loss /= max(1, n_val)
            per_label_acc /= max(1, n_val)
            exact_match /= max(1, n_val)
            mean_acc = per_label_acc.mean().item()
            dt = time.time() - t0

            if (mean_acc > best_mean_acc) or (
                abs(mean_acc - best_mean_acc) < 1e-6 and val_loss < best_val_loss
            ):
                best_mean_acc = max(best_mean_acc, mean_acc)
                best_val_loss = min(best_val_loss, val_loss)
                torch.save(model.state_dict(), ckpt_path)

            if epoch % 10 == 0 or epoch == epochs:
                print(
                    f"  Epoch {epoch:02d}/{epochs:02d} ({dt:4.1f}s) | "
                    f"BCE: {train_bce/len(train_dl):.4f} | Val: {val_loss:.4f} | "
                    f"Mean@0.5: {mean_acc:.2%} | Best: {best_mean_acc:.2%}"
                )

        # Reload best checkpoint for final evaluation
        best_state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        if cfg["id"] == "v4_full_pascan":
            torch.save(best_state, main_ckpt)
            print(f"  >> Updated main checkpoint: {main_ckpt}")

    metrics = evaluate_model(model, val_dl, device)
    metrics["id"] = cfg["id"]
    metrics["name"] = cfg["name"]
    metrics["params"] = n_params
    metrics["use_coordconv"] = cfg["use_coordconv"]
    metrics["use_cbam"] = cfg["use_cbam"]
    metrics["use_film"] = cfg["use_film"]

    print(
        f"  >> RESULT [{cfg['name']}]: "
        f"Mean Acc (0.50)={metrics['mean_acc_05']:.2%} | "
        f"Mean Acc (Opt)={metrics['mean_acc_opt']:.2%} | "
        f"Macro F1={metrics['macro_f1_opt']:.4f} | "
        f"ROC-AUC={metrics['macro_auc']:.4f} | "
        f"Exact Match={metrics['exact_opt']:.2%}"
    )
    return metrics


def generate_latex_ablation_table(results: list[dict]) -> str:
    rows = []
    base_acc = results[0]["mean_acc_opt"]
    for r in results:
        c_mark = "\\checkmark" if r["use_coordconv"] else "--"
        cb_mark = "\\checkmark" if r["use_cbam"] else "--"
        f_mark = "\\checkmark" if r["use_film"] else "--"
        pl = r["per_label_acc_opt"]
        delta = (r["mean_acc_opt"] - base_acc) * 100.0
        delta_str = "---" if abs(delta) < 1e-6 else f"{delta:+.2f}\\%"

        row = (
            f"{r['name']} & {c_mark} & {cb_mark} & {f_mark} & "
            f"{r['params']/1e6:.2f}\\,M & "
            f"{pl['front_left_door']*100:.1f}\\% & "
            f"{pl['front_right_door']*100:.1f}\\% & "
            f"{pl['rear_left_door']*100:.1f}\\% & "
            f"{pl['rear_right_door']*100:.1f}\\% & "
            f"{pl['hood']*100:.1f}\\% & "
            f"\\textbf{{{r['mean_acc_opt']*100:.2f}\\%}} ({delta_str}) & "
            f"{r['macro_f1_opt']:.4f} & "
            f"{r['macro_auc']:.4f} & "
            f"{r['exact_opt']*100:.2f}\\% \\\\"
        )
        rows.append(row)

    rows_joined = "\n".join(rows)
    return f"""% Table: Architectural Ablation Study for Paper 1 (Objective 1: Multi-Scale PA-SCAN)
\\begin{{table*}}[htbp]
\\centering
\\caption{{Architectural ablation study of Multi-Scale PA-SCAN on the paired angle-stratified validation set ($N=128$).}}
\\label{{tab:ablation_study}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lccc c ccccc cccc}}
\\hline
\\textbf{{Architecture Variant}} & \\textbf{{CoordConv}} & \\textbf{{CBAM}} & \\textbf{{FiLM}} & \\textbf{{Params}} & \\textbf{{FL}} & \\textbf{{FR}} & \\textbf{{RL}} & \\textbf{{RR}} & \\textbf{{Hood}} & \\textbf{{Mean Acc ($\\Delta$)}} & \\textbf{{Macro F1}} & \\textbf{{ROC-AUC}} & \\textbf{{Exact Match}} \\\\
\\hline
{rows_joined}
\\hline
\\end{{tabular}}%
}}
\\end{{table*}}
"""


def main():
    parser = argparse.ArgumentParser(description="Run Automated Ablation Study for Paper 1")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--retrain-all", action="store_true", help="Force retraining Variant 4 from scratch")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Automated Ablation Study on device: {device}")

    set_seed(SEED)
    train_dl, val_dl = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        return_angle_train=True,
        max_aug_per_sample=0,
    )
    pos_weight = compute_pos_weight(train_dl.dataset).to(device)

    results = []
    for cfg in ABLATION_CONFIGS:
        res = train_and_evaluate_variant(
            cfg,
            train_dl,
            val_dl,
            pos_weight,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            reuse_full_ckpt=(not args.retrain_all),
        )
        results.append(res)

    out_dir = ROOT_DIR / "docs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    latex_table = generate_latex_ablation_table(results)
    tex_path = out_dir / "ablation_study.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("\n" + "=" * 72)
    print("ABLATION STUDY COMPLETE")
    print(f"JSON saved to  : {json_path}")
    print(f"LaTeX saved to : {tex_path}")
    print("=" * 72)
    print(latex_table)


if __name__ == "__main__":
    main()
