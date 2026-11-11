# Using DisPAM

## Data protocol

`configs/splits/` defines disjoint `train`, `val`, and `test` case lists for
training, checkpoint selection, and final evaluation. Validation contains 10%
of the development pool, selected with split seed 0. Use the same partitions
for all training seeds and comparison methods.

For a custom split, provide three nonempty, pairwise-disjoint case lists. To
hold validation cases out of an existing development pool:

```bash
python scripts/create_splits.py \
  --input configs/custom_development_test.json \
  --output configs/custom_three_way.json \
  --validation-fraction 0.1 --seed 0
```

The input uses `train` and `test` lists, or `train`, `val`, and `test` lists.
For the latter, `train` and `val` form the development pool; `test` stays fixed.

Download the datasets from their official providers and use these layouts:

```text
data/KiTS19/case_00000/imaging.nii.gz
data/KiTS19/case_00000/segmentation.nii.gz

data/BraTS18/HGG/<case_id>/<case_id>_flair.nii.gz
data/BraTS18/HGG/<case_id>/<case_id>_seg.nii.gz

data/LiTS/ct/volume-0.nii
data/LiTS/seg/segmentation-0.nii
```

Prepare one dataset and target at a time:

```bash
DATASET=kits19
DATA_ROOT=data/KiTS19
SPLIT=configs/splits/kits19.json
TARGET=kidney
SEED=0

python scripts/prepare_dataset.py \
  --dataset "$DATASET" --dataset-root "$DATA_ROOT" \
  --split-file "$SPLIT" --splits train val test --check-volumes \
  --output outputs/dataset.json
```

Targets: `kidney`, `tumor` for KiTS19; `wt`, `tc`, `et` for BraTS18;
`liver`, `tumor` for LiTS. Full-volume labels provide evaluation references;
only the selected guiding slice initializes teacher propagation for training.

## Guiding-slice selection

Obtain the frozen PAM propagation checkpoint from its original provider and
place it at `checkpoints/pam/propmask.pth`. The box extension additionally
requires `checkpoints/pam/box2mask.pth`.

For an OpenAI-compatible multimodal endpoint, set `VLM_API_KEY` in your local
environment and supply its chat-completions URL through `VLM_ENDPOINT`:

```bash
python scripts/build_pam_cache.py \
  --dataset "$DATASET" --dataset-root "$DATA_ROOT" --split-file "$SPLIT" \
  --targets "$TARGET" --cache-root outputs/pam_cache \
  --pam-checkpoint checkpoints/pam/propmask.pth \
  --prompt-type mask --prompt-file configs/vlm_prompts.json \
  --agent-backend api --agent-model gpt-5.5 \
  --agent-endpoint "$VLM_ENDPOINT" --device cuda:0 --seed "$SEED"
```

The agent makes one call per candidate with deterministic decoding parameters.
Candidates are examined in PASE order until an affirmative decision. If no
candidate is accepted for a present target, cache generation stops with an
error.

For replay, use `--agent-backend replay --agent-decisions <decisions.json>`
instead of the API options. The JSON maps `case_id/target/slice_index` to one
boolean, as in [the example](../examples/agent_decisions.example.json).
Build caches for `train`; use `--overwrite` to regenerate them.

For the box extension, change to `--prompt-type box` and add
`--box2mask-checkpoint checkpoints/pam/box2mask.pth`. Each present target
receives one box enclosing all of its foreground on the selected slice.

## Hard-Core Dual Distillation

```bash
for MODEL in vista3d primusv2 segmambav2 urwkv; do
  python scripts/train_dual_distillation.py \
    --dataset "$DATASET" --dataset-root "$DATA_ROOT" --split-file "$SPLIT" \
    --cache-root outputs/pam_cache --target "$TARGET" --model "$MODEL" \
    --output-dir "outputs/dual/$TARGET/$MODEL/seed_$SEED" \
    --epochs 100 --device cuda:0 --amp --seed "$SEED"
done
```

Each student uses only training-case teacher targets for optimization.
`best.pt` is selected by validation Dice; `last.pt` stores the final epoch.

## Consensus-Divergence refinement

```bash
CHECKPOINTS=(
  --checkpoint "vista3d=outputs/dual/$TARGET/vista3d/seed_$SEED/best.pt"
  --checkpoint "primusv2=outputs/dual/$TARGET/primusv2/seed_$SEED/best.pt"
  --checkpoint "segmambav2=outputs/dual/$TARGET/segmambav2/seed_$SEED/best.pt"
  --checkpoint "urwkv=outputs/dual/$TARGET/urwkv/seed_$SEED/best.pt"
)

python scripts/build_cdft_regions.py \
  --dataset "$DATASET" --dataset-root "$DATA_ROOT" --split-file "$SPLIT" \
  --target "$TARGET" "${CHECKPOINTS[@]}" \
  --output-root "outputs/cdft_regions/seed_$SEED" \
  --device cuda:0 --amp --seed "$SEED"

python scripts/train_cdft.py \
  --dataset "$DATASET" --dataset-root "$DATA_ROOT" --split-file "$SPLIT" \
  --cache-root outputs/pam_cache --region-root "outputs/cdft_regions/seed_$SEED" \
  --target "$TARGET" "${CHECKPOINTS[@]}" \
  --output-dir "outputs/cdft/$TARGET/seed_$SEED" \
  --epochs 10 --device cuda:0 --amp --seed "$SEED"
```

Regions are built on training cases from the frozen HCDD bank. The `best/`
CDFT bank is selected on validation Dice. Core hyperparameters are listed in
[the configuration](../configs/hyperparameters.json).

## Final test and aggregation

```bash
python scripts/evaluate.py \
  --dataset "$DATASET" --dataset-root "$DATA_ROOT" --split-file "$SPLIT" \
  --split test \
  --bank "$TARGET:vista3d=outputs/cdft/$TARGET/seed_$SEED/best/vista3d.pt" \
  --bank "$TARGET:primusv2=outputs/cdft/$TARGET/seed_$SEED/best/primusv2.pt" \
  --bank "$TARGET:segmambav2=outputs/cdft/$TARGET/seed_$SEED/best/segmambav2.pt" \
  --bank "$TARGET:urwkv=outputs/cdft/$TARGET/seed_$SEED/best/urwkv.pt" \
  --output-dir "outputs/evaluation/$DATASET/seed_$SEED" \
  --device cuda:0 --amp --seed "$SEED"
```

Test prediction requires neither PAM nor a slice prompt. Evaluation defaults
to `--split test`; use `--split val` for validation diagnostics.

For five training seeds, average metrics per case before computing the
across-case mean and population standard deviation:

```bash
RUNS=()
for SEED in 0 1 2 3 4; do
  RUNS+=(--run "outputs/evaluation/$DATASET/seed_$SEED/${TARGET}_cases.csv")
done
python scripts/aggregate_runs.py "${RUNS[@]}" \
  --output-dir "outputs/evaluation/$DATASET/aggregate/$TARGET"
```

Keep credentials in environment variables and local dataset/checkpoint paths
in command-line arguments. Data, weights, outputs, and local `.env` files are
excluded from version control.
