# Evaluation, thresholds, and known limitations

## Decision thresholds are not a bare 0.5 cut-off

The backend converts the model's single logit to `fake_probability = sigmoid(logit)` and then
applies a decision policy loaded from [`backend/app/thresholds.json`](../backend/app/thresholds.json):

```
fake_probability >= fake_threshold  -> FAKE
fake_probability <= real_threshold  -> REAL
otherwise                           -> INCONCLUSIVE
```

The checked-in defaults are `fake_threshold = 0.6`, `real_threshold = 0.4` — **not** `0.5`.
Reasoning: no labelled evaluation has been run against this deployment yet, so there is no
evidence the model's sigmoid output is calibrated (that a 0.7 output genuinely corresponds to
~70% of such images being fake). A single 0.5 cut-off would force every borderline, low-evidence
output into a confident REAL/FAKE call. The `[0.4, 0.6]` band is a deliberate, documented
placeholder that declares those cases INCONCLUSIVE instead. The full rationale travels with the
API response itself (`thresholds.rationale` in `/detect/image` and `/detect/video` responses), so
it's visible to anyone integrating against the API, not just people reading this file.

`FAKE_THRESHOLD` / `REAL_THRESHOLD` environment variables override `thresholds.json` for ops-level
tuning without redeploying a new config file.

## Running a real evaluation

`backend/scripts/evaluate.py` runs the full workflow against a labelled dataset:

```
python scripts/evaluate.py --manifest data/manifest.csv --dataset-name "my-eval-v1"
```

The manifest is a CSV of `path,label` (`label` is `real`/`fake`, case-insensitive). The script:

1. **Splits** the manifest into a validation set and a held-out test set (stratified by label,
   fixed seed — default 50/50; pass `--val-manifest`/`--test-manifest` instead once you have
   enough data for a proper fixed split rather than a single random one).
2. **Selects the threshold using only the validation set** — it sweeps candidate thresholds and
   picks the one maximizing F1, then widens it into an `[real_threshold, fake_threshold]` band
   using a fixed `--margin` (default 0.1) on each side. **The test set is never touched during
   this step.**
3. **Scores the test set once**, with that band applied, and reports accuracy, precision,
   recall, F1, a 3-way confusion matrix (REAL/FAKE/INCONCLUSIVE vs true label), and measured
   inference speed (ms/sample, CPU).
4. **Writes two files**, each carrying full provenance (dataset name, split sizes, timestamp,
   method) so the numbers can't be mistaken for hand-typed or fabricated values:
   - `backend/eval_results.json` — picked up automatically by `/detect/image` and
     `/detect/video` as the `evaluation` field. If this file doesn't exist, those endpoints
     return `evaluation: null`; they never invent a number.
   - `backend/app/thresholds.json` — picked up by the running API as the new decision policy
     (restart the backend, or redeploy, to pick up a changed file).

Run it inside the backend container so it uses the exact same model/preprocessing as the API:

```
docker compose run --rm --no-deps \
  -v $PWD/backend:/srv -v /path/to/your/dataset:/data \
  backend python scripts/evaluate.py --manifest /data/manifest.csv --dataset-name "..."
```

## Limitations (also recorded in every `eval_results.json` the script writes)

- **AI-generated image detection, not face-swap detection.** The underlying model
  (`buildborderless/CommunityForensics-DeepfakeDet-ViT`) is trained to distinguish fully
  AI-generated images from real photographs across thousands of generators. It was **not**
  trained as a face-swap, identity-manipulation, or other partial-edit detector. A video or
  image produced by face-swapping a real photo may not resemble the training distribution this
  model was built to catch, and these metrics should not be read as measuring face-swap
  detection performance unless the evaluation dataset specifically contains labelled face-swap
  samples (and that's stated in the dataset name/notes).
- **No calibration claim.** Reported metrics (accuracy/precision/recall/F1) are discrimination
  metrics at one operating point — they describe how well the model separates real from fake at
  the chosen threshold, not whether its probability outputs are calibrated. No calibration study
  (reliability diagram, Brier score, Platt/isotonic scaling) has been performed. Do not present
  `fake_probability` as a calibrated likelihood.
- **The INCONCLUSIVE margin is a heuristic.** The fixed `--margin` around the F1-optimal
  threshold is not derived from a cost-sensitive analysis of false-positive vs false-negative
  impact for any particular deployment; treat it as a starting point to tune, not a finished
  policy.
- **Video is frame-sampled image classification, not temporal analysis.** `/detect/video`
  samples frames and classifies each one independently with the same image model, then averages
  (mean-probability baseline). It does not model motion, optical flow, or any inter-frame
  relationship, so `scripts/evaluate.py` only evaluates on static images — there is no separate
  video-level evaluation methodology here, and video results should not be assumed to carry the
  same measured accuracy as the image evaluation.
- **Small-dataset variance.** With a small validation/test set, both the selected threshold and
  the reported metrics can vary significantly from a re-run with different samples. The split
  sizes are always recorded in `eval_results.json.split` so this can be judged case by case.
