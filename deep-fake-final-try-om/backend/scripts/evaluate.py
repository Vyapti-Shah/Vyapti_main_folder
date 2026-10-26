#!/usr/bin/env python3
"""Evaluate the deepfake detector on a labelled dataset and (optionally) select thresholds.

Usage
-----
  python scripts/evaluate.py --manifest data/manifest.csv --dataset-name "my-eval-v1"

The manifest is a CSV with columns `path,label`, where `label` is one of
real/fake/0/1 (case-insensitive; 1 == fake == AI-generated).

Workflow
--------
1. The manifest is split into a validation set and a held-out test set (stratified by
   label, fixed seed for reproducibility). Default split is 50/50 — labelled deepfake
   eval sets are usually small, and a threshold chosen on too few validation samples is
   not trustworthy; use --val-fraction to change it, and prefer passing pre-split
   --val-manifest/--test-manifest once you have enough data to not need a single split.
2. Every validation image is scored with the SAME detector and preprocessing the API
   uses (app.model.Detector). A single-threshold sweep over the validation scores picks
   the threshold that maximizes F1 — this is the ONLY place thresholds are chosen.
   The test set is never used for this step.
3. The chosen threshold is widened into an [real_threshold, fake_threshold] INCONCLUSIVE
   band using --margin (default 0.1) on each side. This margin is a heuristic, not a
   cost-sensitive calibration — see the "limitations" field in the written results.
4. The test set is scored once, with the band from step 3 applied, and used only to
   report accuracy / precision / recall / F1 / confusion matrix / inference speed. These
   numbers describe the test set, not a guarantee for other data.
5. Results are written to --eval-output (consumed by the API's /detect/* "evaluation"
   field) and the threshold is written to --thresholds-output (consumed by
   app.common at request time), each with full provenance (dataset, split sizes, method,
   date) so neither file can be mistaken for hand-authored or fabricated numbers.

This script does not claim the model is probability-calibrated (i.e. that an output of
0.8 means "80% of such images are truly fake"). That would require a dedicated
calibration study (e.g. reliability diagrams, Brier score, Platt/isotonic scaling)
which this script does not perform. Metrics here are discrimination metrics
(accuracy/precision/recall/F1) at a chosen operating point, not calibration metrics.
"""
import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app.model import MODEL_ID, MODEL_REVISION, Detector  # noqa: E402

TRUE_LABELS = {"fake", "1", "ai", "ai-generated", "synthetic"}
FALSE_LABELS = {"real", "0", "authentic", "pristine"}

LIMITATIONS = [
    "The underlying model (CommunityForensics-DeepfakeDet-ViT) is trained to detect "
    "AI-generated images (images produced end-to-end by a generative model). It is NOT "
    "trained as a face-swap or identity-manipulation detector, and these metrics should "
    "not be read as measuring face-swap detection performance unless the evaluation "
    "dataset specifically contains labelled face-swap samples and that is stated in the "
    "dataset name/notes.",
    "Metrics below are discrimination metrics (accuracy/precision/recall/F1) at one "
    "operating point (the selected threshold). No calibration analysis (reliability "
    "diagram, Brier score, Platt/isotonic scaling) has been run, so these probabilities "
    "should not be presented or interpreted as calibrated confidence values.",
    "The INCONCLUSIVE band width (margin) is a fixed heuristic margin around the "
    "F1-optimal threshold, not derived from a cost-sensitive analysis of false-positive "
    "vs false-negative impact; treat it as a starting point, not a tuned operating policy.",
    "Video-level scores are a plain mean over independently classified sampled frames "
    "(no temporal modelling) — this evaluation only scores static images, not videos.",
]


def read_manifest(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            label_raw = row["label"].strip().lower()
            if label_raw in TRUE_LABELS:
                label = 1
            elif label_raw in FALSE_LABELS:
                label = 0
            else:
                raise ValueError(f"Unrecognized label {row['label']!r} for {row['path']}")
            rows.append((row["path"], label))
    return rows


def stratified_split(rows, val_fraction, seed):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for r in rows:
        by_label[r[1]].append(r)
    val, test = [], []
    for label, items in by_label.items():
        items = items[:]
        rng.shuffle(items)
        n_val = round(len(items) * val_fraction)
        val += items[:n_val]
        test += items[n_val:]
    rng.shuffle(val)
    rng.shuffle(test)
    return val, test


def score(detector, rows):
    """Returns (probs, labels, per_sample_ms) for rows that load successfully; skips and
    reports unreadable files rather than crashing the whole run."""
    probs, labels, times_ms, skipped = [], [], [], []
    for path, label in rows:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            skipped.append((path, str(e)))
            continue
        t0 = time.perf_counter()
        p = detector.predict(img)
        times_ms.append((time.perf_counter() - t0) * 1000)
        probs.append(p)
        labels.append(label)
    return probs, labels, times_ms, skipped


def f1_at(probs, labels, threshold):
    tp = sum(1 for p, y in zip(probs, labels) if p >= threshold and y == 1)
    fp = sum(1 for p, y in zip(probs, labels) if p >= threshold and y == 0)
    fn = sum(1 for p, y in zip(probs, labels) if p < threshold and y == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def select_threshold(probs, labels):
    """Sweep candidate thresholds (every observed score, which is where the decision can
    change) and pick the one maximizing F1 on the VALIDATION set only."""
    candidates = sorted(set(probs)) or [0.5]
    best = max(candidates, key=lambda t: f1_at(probs, labels, t)[0])
    f1, precision, recall = f1_at(probs, labels, best)
    return best, {"f1": f1, "precision": precision, "recall": recall}


def confusion_matrix(probs, labels, fake_threshold, real_threshold):
    """3-way confusion: rows = true label, columns = predicted REAL/FAKE/INCONCLUSIVE."""
    cm = {"real": {"REAL": 0, "FAKE": 0, "INCONCLUSIVE": 0}, "fake": {"REAL": 0, "FAKE": 0, "INCONCLUSIVE": 0}}
    for p, y in zip(probs, labels):
        pred = "FAKE" if p >= fake_threshold else "REAL" if p <= real_threshold else "INCONCLUSIVE"
        cm["fake" if y == 1 else "real"][pred] += 1
    return cm


def binary_metrics(probs, labels, fake_threshold, real_threshold, skip_inconclusive):
    """Accuracy/precision/recall/F1 at the deployed 3-way rule. INCONCLUSIVE predictions
    are excluded from precision/recall/F1 (they are not a REAL/FAKE call) but are counted
    against accuracy as misses, unless --skip-inconclusive-in-accuracy is set."""
    tp = fp = fn = tn = inconclusive = 0
    for p, y in zip(probs, labels):
        if p >= fake_threshold:
            pred = 1
        elif p <= real_threshold:
            pred = 0
        else:
            inconclusive += 1
            continue
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
        else:
            tn += 1

    decided = tp + fp + fn + tn
    total = decided + inconclusive
    correct = tp + tn
    denom = decided if skip_inconclusive else total
    accuracy = correct / denom if denom else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inconclusive_count": inconclusive,
        "inconclusive_rate": inconclusive / total if total else 0.0,
        "samples_scored": total,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="CSV of path,label to auto-split into val/test")
    ap.add_argument("--val-manifest", help="Pre-split validation CSV (path,label)")
    ap.add_argument("--test-manifest", help="Pre-split test CSV (path,label)")
    ap.add_argument("--val-fraction", type=float, default=0.5, help="Used only with --manifest")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--margin", type=float, default=0.1, help="Half-width of the INCONCLUSIVE band around the F1-optimal threshold")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--skip-inconclusive-in-accuracy", action="store_true")
    ap.add_argument("--eval-output", default=str(Path(__file__).resolve().parent.parent / "eval_results.json"))
    ap.add_argument("--thresholds-output", default=str(Path(__file__).resolve().parent.parent / "app" / "thresholds.json"))
    args = ap.parse_args()

    if args.manifest and (args.val_manifest or args.test_manifest):
        ap.error("--manifest is mutually exclusive with --val-manifest/--test-manifest")
    if not args.manifest and not (args.val_manifest and args.test_manifest):
        ap.error("Pass --manifest, or both --val-manifest and --test-manifest")

    if args.manifest:
        rows = read_manifest(args.manifest)
        val_rows, test_rows = stratified_split(rows, args.val_fraction, args.seed)
    else:
        val_rows = read_manifest(args.val_manifest)
        test_rows = read_manifest(args.test_manifest)

    if not val_rows or not test_rows:
        ap.error("Validation and test sets must both be non-empty.")

    print(f"Loading {MODEL_ID}@{MODEL_REVISION} ...")
    detector = Detector()
    detector._load()  # synchronous: this script is not a web server, we need the result now
    if detector.state != "ready":
        print(f"Model failed to load: {detector.error}", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(val_rows)} validation samples (threshold selection ONLY)...")
    val_probs, val_labels, _, val_skipped = score(detector, val_rows)
    if val_skipped:
        print(f"  skipped {len(val_skipped)} unreadable validation files", file=sys.stderr)
    if not val_probs:
        print("No validation samples could be scored.", file=sys.stderr)
        sys.exit(1)

    base_threshold, val_metrics = select_threshold(val_probs, val_labels)
    fake_threshold = min(0.999, base_threshold + args.margin)
    real_threshold = max(0.001, base_threshold - args.margin)
    print(f"Selected base threshold {base_threshold:.4f} on validation (F1={val_metrics['f1']:.4f}); "
          f"band = [{real_threshold:.4f}, {fake_threshold:.4f}]")

    print(f"Scoring {len(test_rows)} test samples (held out from threshold selection)...")
    test_probs, test_labels, test_times_ms, test_skipped = score(detector, test_rows)
    if test_skipped:
        print(f"  skipped {len(test_skipped)} unreadable test files", file=sys.stderr)
    if not test_probs:
        print("No test samples could be scored.", file=sys.stderr)
        sys.exit(1)

    metrics = binary_metrics(test_probs, test_labels, fake_threshold, real_threshold, args.skip_inconclusive_in_accuracy)
    cm = confusion_matrix(test_probs, test_labels, fake_threshold, real_threshold)
    avg_ms = sum(test_times_ms) / len(test_times_ms)

    now = datetime.now(timezone.utc).isoformat()
    eval_result = {
        "dataset": args.dataset_name,
        "samples": len(test_probs),
        "split": {"validation": len(val_probs), "test": len(test_probs), "val_fraction": args.val_fraction, "seed": args.seed},
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": cm,
        "inconclusive_rate": metrics["inconclusive_rate"],
        "threshold_used": {"fake_at_or_above": fake_threshold, "real_at_or_below": real_threshold},
        "inference_speed_ms_per_sample": round(avg_ms, 2),
        "evaluated_at": now,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "calibration_note": (
            "No calibration analysis was performed; these are discrimination metrics "
            "(accuracy/precision/recall/F1) at the operating point above, not evidence "
            "that output probabilities are calibrated."
        ),
        "limitations": LIMITATIONS,
    }
    Path(args.eval_output).write_text(json.dumps(eval_result, indent=2))
    print(f"Wrote evaluation results to {args.eval_output}")

    thresholds_result = {
        "fake_threshold": fake_threshold,
        "real_threshold": real_threshold,
        "selection_method": "f1_maximization_on_validation_with_fixed_margin",
        "rationale": (
            f"Base threshold {base_threshold:.4f} selected by maximizing F1 on a held-out "
            f"validation set of {len(val_probs)} samples (val F1={val_metrics['f1']:.4f}, "
            f"precision={val_metrics['precision']:.4f}, recall={val_metrics['recall']:.4f}), "
            f"widened by a fixed margin of {args.margin} on each side to form the "
            "INCONCLUSIVE band. The test set used to report the metrics in eval_results.json "
            "was never used in this selection step."
        ),
        "selected_on": now,
        "source_eval": args.dataset_name,
    }
    Path(args.thresholds_output).write_text(json.dumps(thresholds_result, indent=2))
    print(f"Wrote selected thresholds to {args.thresholds_output}")

    print(json.dumps({"test_metrics": metrics, "confusion_matrix": cm}, indent=2))


if __name__ == "__main__":
    main()
