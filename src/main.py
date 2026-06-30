"""
Main entry point for UFM-Transformer training and evaluation.

This script orchestrates the full lifecycle of the Unified Face and
Fingerprint Multimodal Transformer (UFM-Transformer): training, evaluation,
and visualisation of attention maps / Grad-CAM explanations.

Usage:
    Training (from scratch or with a config):
        python main.py --mode train --dataset_path ./data --output_dir ./output

    Evaluation (compute EER, ROC, AUC on test set):
        python main.py --mode eval --model_path ./output/best_model.pth
                       --dataset_path ./data --output_dir ./output/eval

    Visualisation (attention maps & Grad-CAM):
        python main.py --mode visualize --model_path ./output/best_model.pth
                       --dataset_path ./data --output_dir ./output/viz

    All modes support ``--help`` for detailed argument descriptions.

Author: UFM-Transformer Team
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Import project utilities
from utils import (
    AverageMeter,
    Logger,
    Timer,
    compute_flops,
    count_parameters,
    get_device,
    load_checkpoint,
    print_system_info,
    save_checkpoint,
    set_seed,
)

# Import visualisation tools
from visualize import (
    compute_gradcam_bimodal,
    extract_attention_maps,
    generate_explainability_report,
    plot_attention_heatmap,
    visualize_cross_attention,
    visualize_gradcam_bimodal,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_SEED: int = 42
_DEFAULT_BATCH_SIZE: int = 32
_DEFAULT_EPOCHS: int = 50
_DEFAULT_LR: float = 1e-4
_DEFAULT_WEIGHT_DECAY: float = 1e-5
_DEFAULT_IMAGE_SIZE: int = 224
_DEFAULT_EMBED_DIM: int = 256
_DEFAULT_NUM_HEADS: int = 8


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all operating modes.

    Returns:
        Configured ``argparse.ArgumentParser`` with sub-parsers for
        ``train``, ``eval``, and ``visualize`` modes.
    """
    parser = argparse.ArgumentParser(
        prog="UFM-Transformer",
        description="Unified Face and Fingerprint Multimodal Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --mode train --dataset_path ./data "
            "--output_dir ./output --epochs 50\n"
            "  python main.py --mode eval --model_path ./output/best_model.pth "
            "--dataset_path ./data --output_dir ./output/eval\n"
            "  python main.py --mode visualize --model_path ./output/best_model.pth "
            "--dataset_path ./data --output_dir ./output/viz\n"
        ),
    )

    # Global / shared arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "visualize"],
        help="Operating mode: train a new model, evaluate an existing model, "
             "or generate visualisations.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Root directory containing the face and fingerprint datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for checkpoints, logs, and results (default: ./output).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a saved model checkpoint (required for eval/visualize).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device selection (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {_DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Mini-batch size (default: {_DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=_DEFAULT_IMAGE_SIZE,
        help=f"Input image size in pixels (default: {_DEFAULT_IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=_DEFAULT_EMBED_DIM,
        help=f"Transformer embedding dimension (default: {_DEFAULT_EMBED_DIM}).",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=_DEFAULT_NUM_HEADS,
        help=f"Number of attention heads (default: {_DEFAULT_NUM_HEADS}).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable Automatic Mixed Precision (AMP) training.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )

    # Training-specific arguments
    train_group = parser.add_argument_group("Training options")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=_DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {_DEFAULT_EPOCHS}).",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=_DEFAULT_LR,
        help=f"Initial learning rate (default: {_DEFAULT_LR}).",
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=_DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay / L2 regularisation (default: {_DEFAULT_WEIGHT_DECAY}).",
    )
    train_group.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="LR scheduler type (default: cosine).",
    )
    train_group.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of linear warmup epochs (default: 5).",
    )
    train_group.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Run evaluation every N epochs (default: 1).",
    )
    train_group.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5).",
    )
    train_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint path.",
    )
    train_group.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum training duration in seconds. When reached, "
             "finishes the current epoch, saves a checkpoint, and exits "
             "gracefully (default: None = run until natural completion).",
    )

    # Performance optimisation flags
    perf_group = parser.add_argument_group("Performance optimisations")
    amp_grp = perf_group.add_mutually_exclusive_group()
    amp_grp.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        dest="use_amp",
        help="Enable Automatic Mixed Precision training (default: enabled).",
    )
    amp_grp.add_argument(
        "--no_amp",
        action="store_false",
        dest="use_amp",
        help="Disable AMP (fall back to FP32).",
    )
    perf_group.add_argument(
        "--mc_samples_train",
        type=int,
        default=1,
        help="MC-Dropout samples during training (default: 1 for speed; "
             "use 5 only at inference).",
    )

    # Evaluation-specific arguments
    eval_group = parser.add_argument_group("Evaluation options")
    eval_group.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test).",
    )

    # Visualisation-specific arguments
    viz_group = parser.add_argument_group("Visualisation options")
    viz_group.add_argument(
        "--viz_n_identities",
        type=int,
        default=10,
        help="Number of identities to visualise (default: 10).",
    )
    viz_group.add_argument(
        "--viz_pairs_file",
        type=str,
        default=None,
        help="JSON file with pre-selected verification pairs.",
    )

    return parser


# ---------------------------------------------------------------------------
# Mode: Train
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
    """Execute the full training loop.

    Args:
        args: Parsed command-line arguments.

    Raises:
        RuntimeError: If the dataset cannot be loaded or training fails.
    """
    logging.info("=" * 60)
    logging.info("MODE: TRAINING")
    logging.info("=" * 60)

    device = get_device(args.device)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start a background timer for graceful shutdown (e.g. Kaggle 30h limit)
    _timeout_timer: Optional[threading.Timer] = None
    if args.timeout is not None and args.timeout > 0:
        logging.info("Timeout configured: %d seconds (≈%.1f hours)",
                      args.timeout, args.timeout / 3600)
        import train as _train_module

        def _on_timeout():
            logging.warning("*** TRAINING TIMEOUT (%ds) — requesting graceful shutdown ***",
                            args.timeout)
            _train_module.request_shutdown()

        _timeout_timer = threading.Timer(args.timeout, _on_timeout)
        _timeout_timer.daemon = True
        _timeout_timer.start()

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset and DataLoader
    # ------------------------------------------------------------------
    try:
        from dataset import get_dataloader, get_datasets
    except ImportError:
        logging.error(
            "Failed to import 'dataset' module. "
            "Ensure dataset.py is in the same directory."
        )
        raise

    logging.info("Loading datasets from %s", args.dataset_path)
    train_dataset, val_dataset = get_datasets(
        root=args.dataset_path,
        image_size=args.image_size,
        split="trainval",
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logging.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    try:
        from models import UFMTransformer
    except ImportError:
        logging.error(
            "Failed to import 'models' module (UFMTransformer). "
            "Ensure models.py is in the same directory."
        )
        raise

    model = UFMTransformer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        logging.info("Wrapping model with DataParallel across %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)

    n_params = count_parameters(model)
    logging.info("Model: UFM-Transformer")
    logging.info("  Parameters: %,d (%.2f M)", n_params, n_params / 1e6)

    # Estimate FLOPs
    try:
        flops = compute_flops(
            model,
            input_size=(2, 3, args.image_size, args.image_size),
            device=device,
        )
        logging.info("  Estimated FLOPs: %.3f GMACs", flops)
    except Exception as exc:
        logging.warning("FLOP estimation skipped: %s", exc)

    # ------------------------------------------------------------------
    # Optimiser, loss, scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.5
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    else:
        scheduler = None

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and torch.cuda.is_available() else None

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 1
    best_eer = float("inf")
    if args.resume:
        logging.info("Resuming from checkpoint: %s", args.resume)
        resume_info = load_checkpoint(
            model, args.resume, optimizer=optimizer, scheduler=scheduler
        )
        start_epoch = resume_info.get("epoch", 0) + 1
        best_eer = resume_info.get("metrics", {}).get("eer", float("inf"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logging.info("Starting training from epoch %d to %d", start_epoch, args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        # --- Training phase ---
        model.train()
        loss_meter = AverageMeter("train_loss")
        acc_meter = AverageMeter("train_acc")

        with Timer(f"Epoch {epoch} training"):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
            for batch in pbar:
                face_img = batch["face"].to(device, non_blocking=True)
                fp_img = batch["fingerprint"].to(device, non_blocking=True)
                labels = batch["label"].float().to(device, non_blocking=True)

                optimizer.zero_grad()

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(face_img, fp_img)
                        logits = outputs["similarity_score"] if isinstance(outputs, dict) else outputs
                        loss = criterion(logits.squeeze(), labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(face_img, fp_img)
                    logits = outputs["similarity_score"] if isinstance(outputs, dict) else outputs
                    loss = criterion(logits.squeeze(), labels)
                    loss.backward()
                    optimizer.step()

                # Metrics
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                acc = (preds == labels).float().mean().item()

                loss_meter.update(loss.item(), n=labels.size(0))
                acc_meter.update(acc, n=labels.size(0))

                pbar.set_postfix(loss=loss_meter.avg, acc=acc_meter.avg)

        logging.info(
            "Epoch %d/%d -- Train Loss: %.4f | Train Acc: %.4f",
            epoch, args.epochs, loss_meter.avg, acc_meter.avg,
        )

        # --- Validation phase ---
        if epoch % args.eval_every == 0:
            val_metrics = run_validation(model, val_loader, device, criterion)
            logging.info(
                "Epoch %d/%d -- Val Loss: %.4f | Val Acc: %.4f | EER: %.4f",
                epoch, args.epochs,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["eer"],
            )

            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["eer"])
                else:
                    scheduler.step()

            # Save best model
            if val_metrics["eer"] < best_eer:
                best_eer = val_metrics["eer"]
                best_path = ckpt_dir / "best_model.pth"
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, path=best_path,
                    scheduler=scheduler,
                )
                logging.info("  >>> New best model saved (EER: %.4f)", best_eer)

        # --- Periodic checkpoint ---
        if epoch % args.save_every == 0:
            periodic_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            save_checkpoint(
                model, optimizer, epoch,
                {"loss": loss_meter.avg, "accuracy": acc_meter.avg},
                path=periodic_path, scheduler=scheduler,
            )

    logging.info("Training complete. Best EER: %.4f", best_eer)


# ---------------------------------------------------------------------------
# Mode: Validation helper
# ---------------------------------------------------------------------------


def run_validation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Run one validation pass and compute metrics.

    Args:
        model: Model to evaluate.
        dataloader: Validation data loader.
        device: Compute device.
        criterion: Optional loss function.

    Returns:
        Dictionary with keys ``loss``, ``accuracy``, ``eer``,
        ``scores``, ``labels``.
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_acc")
    all_scores: List[float] = []
    all_labels: List[int] = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in pbar:
            face_img = batch["face"].to(device, non_blocking=True)
            fp_img = batch["fingerprint"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)

            outputs = model(face_img, fp_img)
            logits = outputs["similarity_score"] if isinstance(outputs, dict) else outputs
            scores = torch.sigmoid(logits.squeeze())

            if criterion is not None:
                loss = criterion(logits.squeeze(), labels)
                loss_meter.update(loss.item(), n=labels.size(0))

            preds = (scores > 0.5).float()
            acc = (preds == labels).float().mean().item()
            acc_meter.update(acc, n=labels.size(0))

            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix(loss=loss_meter.avg, acc=acc_meter.avg)

    # Compute EER
    eer = compute_eer(np.array(all_labels), np.array(all_scores))

    return {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "eer": eer,
        "scores": all_scores,
        "labels": all_labels,
    }


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute the Equal Error Rate (EER) from verification scores.

    EER is the point on the ROC curve where False Acceptance Rate (FAR)
    equals False Rejection Rate (FRR).

    Args:
        labels: Ground-truth labels (1 = genuine, 0 = impostor).
        scores: Predicted similarity scores (higher = more similar).

    Returns:
        EER as a float in ``[0, 1]``.
    """
    n_genuine = int(labels.sum())
    n_impostor = len(labels) - n_genuine

    if n_genuine == 0 or n_impostor == 0:
        logging.warning("Cannot compute EER: all labels are the same")
        return 0.0

    # Edge case: all scores identical -> no discrimination
    if np.all(scores == scores[0]):
        return 0.5

    # Sort by score ascending (low to high) so each threshold is a cutoff
    sorted_indices = np.argsort(scores)
    labels_sorted = labels[sorted_indices]

    # At threshold i: everything <= i is classified as impostor (0),
    # everything > i is classified as genuine (1)
    # FN = genuine with score <= threshold = cumsum of genuine labels up to i
    # FP = impostor with score > threshold = reverse cumsum of impostor labels
    fn_cumsum = np.cumsum(labels_sorted)          # false negatives so far
    tn_cumsum = np.cumsum(1 - labels_sorted)      # true negatives so far

    # FNR = FN / total_genuine, FAR = FP / total_impostor
    fnrs = fn_cumsum / n_genuine
    fprs = (n_impostor - tn_cumsum) / n_impostor

    # Find threshold where |FAR - FRR| is minimised
    eer_idx = np.argmin(np.abs(fprs - fnrs))
    eer = (fprs[eer_idx] + fnrs[eer_idx]) / 2.0

    return float(eer)


# ---------------------------------------------------------------------------
# Mode: Evaluation
# ---------------------------------------------------------------------------


def run_evaluation(args: argparse.Namespace) -> Dict[str, float]:
    """Evaluate a trained model on a test set.

    Loads the model checkpoint, runs inference on the specified dataset split,
    and computes accuracy, EER, AUC, and other verification metrics.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dictionary of computed metrics.

    Raises:
        FileNotFoundError: If ``args.model_path`` does not exist.
        RuntimeError: If the dataset or model cannot be loaded.
    """
    logging.info("=" * 60)
    logging.info("MODE: EVALUATION")
    logging.info("=" * 60)

    if args.model_path is None or not Path(args.model_path).is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    device = get_device(args.device)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    try:
        from dataset import get_dataloader, get_datasets
    except ImportError:
        logging.error("Failed to import 'dataset' module.")
        raise

    _, test_dataset = get_datasets(
        root=args.dataset_path,
        image_size=args.image_size,
        split="test",
    )
    test_loader = get_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logging.info("Test samples: %d", len(test_dataset))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    try:
        from models import UFMTransformer
    except ImportError:
        logging.error("Failed to import 'models' module (UFMTransformer).")
        raise

    model = UFMTransformer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
    ).to(device)

    load_checkpoint(model, args.model_path, map_location=str(device))
    logging.info("Loaded checkpoint: %s", args.model_path)

    n_params = count_parameters(model)
    logging.info("Model parameters: %,d", n_params)

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    logging.info("Running evaluation on '%s' split...", args.split)

    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        logging.warning("scikit-learn not installed; AUC will not be computed")

    model.eval()
    all_scores: List[float] = []
    all_labels: List[int] = []

    with Timer("Evaluation inference"):
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluation", leave=False)
            for batch in pbar:
                face_img = batch["face"].to(device, non_blocking=True)
                fp_img = batch["fingerprint"].to(device, non_blocking=True)
                labels = batch["label"].float().to(device, non_blocking=True)

                outputs = model(face_img, fp_img)
                logits = outputs["similarity_score"] if isinstance(outputs, dict) else outputs
                scores = torch.sigmoid(logits.squeeze())

                all_scores.extend(scores.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

                pbar.set_postfix(n=len(all_scores))

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    # Compute metrics
    eer = compute_eer(labels_arr, scores_arr)
    preds = (scores_arr > 0.5).astype(int)
    accuracy = float(np.mean(preds == labels_arr))

    metrics: Dict[str, float] = {
        "eer": eer,
        "accuracy": accuracy,
    }

    if has_sklearn:
        try:
            auc = roc_auc_score(labels_arr, scores_arr)
            metrics["auc"] = auc
            logging.info("AUC: %.4f", auc)

            # Save ROC curve data
            fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
            roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                        "thresholds": thresholds.tolist()}
            roc_path = output_dir / "roc_data.json"
            with open(roc_path, "w") as f:
                json.dump(roc_data, f)
            logging.info("ROC data saved to %s", roc_path)

        except Exception as exc:
            logging.warning("AUC computation failed: %s", exc)

    # Log results
    logging.info("-" * 40)
    logging.info("Evaluation Results:")
    logging.info("  EER:        %.4f", eer)
    logging.info("  Accuracy:   %.4f", accuracy)
    if "auc" in metrics:
        logging.info("  AUC:        %.4f", metrics["auc"])
    logging.info("-" * 40)

    # Save metrics to JSON
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Metrics saved to %s", metrics_path)

    return metrics


# ---------------------------------------------------------------------------
# Mode: Visualisation
# ---------------------------------------------------------------------------


def run_visualisation(args: argparse.Namespace) -> None:
    """Generate attention maps and Grad-CAM visualisations.

    Loads a trained model and produces:
    - Per-layer cross-attention visualisations
    - Grad-CAM overlays for selected test pairs
    - A multi-page explainability report (if test pairs are provided)

    Args:
        args: Parsed command-line arguments.

    Raises:
        FileNotFoundError: If ``args.model_path`` does not exist.
    """
    logging.info("=" * 60)
    logging.info("MODE: VISUALISATION")
    logging.info("=" * 60)

    if args.model_path is None or not Path(args.model_path).is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    device = get_device(args.device)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualisations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    try:
        from models import UFMTransformer
    except ImportError:
        logging.error("Failed to import 'models' module (UFMTransformer).")
        raise

    model = UFMTransformer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
    ).to(device)

    load_checkpoint(model, args.model_path, map_location=str(device))
    logging.info("Loaded checkpoint: %s", args.model_path)
    model.eval()

    # ------------------------------------------------------------------
    # Load or create sample pairs
    # ------------------------------------------------------------------
    test_pairs: List[Dict[str, Any]] = []

    if args.viz_pairs_file and Path(args.viz_pairs_file).is_file():
        logging.info("Loading pairs from %s", args.viz_pairs_file)
        import pickle
        with open(args.viz_pairs_file, "rb") as f:
            test_pairs = pickle.load(f)
    else:
        # Create synthetic demo pairs for self-testing
        logging.info("No pairs file provided; generating synthetic demo pairs.")
        for i in range(args.viz_n_identities):
            identity = f"demo_id_{i:03d}"
            # Genuine pair (same identity)
            test_pairs.append({
                "face1": torch.rand(3, args.image_size, args.image_size),
                "face2": torch.rand(3, args.image_size, args.image_size),
                "fp1": torch.rand(3, args.image_size, args.image_size),
                "fp2": torch.rand(3, args.image_size, args.image_size),
                "label": 1,
                "identity": identity,
            })
            # Impostor pair (different identity)
            test_pairs.append({
                "face1": torch.rand(3, args.image_size, args.image_size),
                "face2": torch.rand(3, args.image_size, args.image_size),
                "fp1": torch.rand(3, args.image_size, args.image_size),
                "fp2": torch.rand(3, args.image_size, args.image_size),
                "label": 0,
                "identity": identity,
            })

    logging.info("Total test pairs: %d", len(test_pairs))

    # ------------------------------------------------------------------
    # 1. Extract and save attention maps
    # ------------------------------------------------------------------
    logging.info("Extracting attention maps...")
    sample_face = test_pairs[0]["face1"].unsqueeze(0).to(device)
    sample_fp = test_pairs[0]["fp1"].unsqueeze(0).to(device)

    attn_maps = extract_attention_maps(model, sample_face, sample_fp)

    if attn_maps:
        attn_viz_path = viz_dir / "cross_attention_maps.pdf"
        visualize_cross_attention(
            attn_maps,
            sample_face.cpu(),
            sample_fp.cpu(),
            save_path=attn_viz_path,
        )
        logging.info("Cross-attention maps saved to %s", attn_viz_path)

        # Also save average attention heatmap
        first_key = list(attn_maps.keys())[0]
        avg_attn = attn_maps[first_key][0].mean(dim=0).numpy()
        heatmap_path = viz_dir / "attention_matrix.pdf"
        plot_attention_heatmap(avg_attn, heatmap_path)
        logging.info("Attention heatmap saved to %s", heatmap_path)
    else:
        logging.warning("No attention maps were extracted (model may not expose them)")

    # ------------------------------------------------------------------
    # 2. Grad-CAM on sample pairs
    # ------------------------------------------------------------------
    logging.info("Generating Grad-CAM visualisations...")

    # Determine target layers (heuristic: find last conv layers)
    face_layer = None
    fp_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if "face" in name.lower():
                face_layer = name
            elif "fp" in name.lower() or "finger" in name.lower():
                fp_layer = name

    if face_layer is None or fp_layer is None:
        logging.warning(
            "Could not auto-detect CNN layers (face=%s, fp=%s). "
            "Grad-CAM may fail.", face_layer, fp_layer
        )
        face_layer = face_layer or "face_cnn.layer4"
        fp_layer = fp_layer or "fp_cnn.layer4"

    logging.info("Grad-CAM target layers -- Face: %s | FP: %s", face_layer, fp_layer)

    for pair_idx, pair in enumerate(test_pairs[:min(5, len(test_pairs))]):
        face_img = pair["face1"].unsqueeze(0).to(device)
        fp_img = pair["fp1"].unsqueeze(0).to(device)
        label = pair.get("label", 1)
        identity = pair.get("identity", f"pair_{pair_idx}")

        try:
            face_hm, fp_hm = compute_gradcam_bimodal(
                model, face_img, fp_img,
                target_class=label,
                face_layer=face_layer,
                fp_layer=fp_layer,
            )

            label_str = "genuine" if label == 1 else "impostor"
            gradcam_path = viz_dir / f"gradcam_{identity}_{label_str}.pdf"
            visualize_gradcam_bimodal(
                pair["face1"], pair["fp1"],
                face_hm, fp_hm,
                save_path=gradcam_path,
                title=f"Identity: {identity} | Label: {label_str}",
            )
            logging.info("  Grad-CAM saved: %s", gradcam_path)

        except Exception as exc:
            logging.warning("Grad-CAM failed for pair %d: %s", pair_idx, exc)

    # ------------------------------------------------------------------
    # 3. Full explainability report
    # ------------------------------------------------------------------
    logging.info("Generating explainability report...")
    try:
        report_path = generate_explainability_report(
            model,
            test_pairs,
            output_dir=viz_dir,
            device=device,
            max_identities=args.viz_n_identities,
            face_layer=face_layer,
            fp_layer=fp_layer,
        )
        logging.info("Explainability report saved to %s", report_path)
    except Exception as exc:
        logging.error("Explainability report generation failed: %s", exc)
        traceback.print_exc()

    logging.info("Visualisation complete. Outputs in %s", viz_dir)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for the UFM-Transformer CLI.

    Parses arguments, configures logging and device, then dispatches to the
    appropriate mode handler.

    Args:
        argv: Optional argument list (uses ``sys.argv`` if ``None``).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    def _signal_handler(signum, frame):
        """Graceful shutdown: notify train.py to exit after the current epoch."""
        logging.warning("Received signal %s — requesting graceful shutdown", signum)
        try:
            import train
            train.request_shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    # Setup output directory and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"ufm_{args.mode}.log"
    Logger.setup(log_file=log_file, level=logging.DEBUG if args.debug else logging.INFO)

    # Print system info early
    print_system_info()
    logging.info("Mode: %s | Output: %s", args.mode, args.output_dir)
    logging.info("Arguments: %s", vars(args))

    # Dispatch
    try:
        if args.mode == "train":
            run_training(args)
        elif args.mode == "eval":
            run_evaluation(args)
        elif args.mode == "visualize":
            run_visualisation(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        return 130
    except Exception as exc:
        logging.error("Fatal error in '%s' mode: %s", args.mode, exc)
        traceback.print_exc()
        return 1

    logging.info("UFM-Transformer finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
