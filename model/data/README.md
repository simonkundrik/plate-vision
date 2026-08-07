# data

Acquisition scripts. Downloaded data is gitignored; only the scripts are tracked.

Run from `model/` with the package installed (`pip install -e ".[dev]"`).

## Nutrition5k

```bash
python data/download_nutrition5k.py --report
```

Measures the subset and audits label quality without downloading imagery. Use it before
committing to the download.

```bash
python data/download_nutrition5k.py
```

Fetches into `data/nutrition5k/`. Resumable: dishes already on disk are skipped, and each
file is written to a `.part` name and renamed on completion so an interrupted run never
leaves a truncated PNG that a later resume mistakes for a finished download.

### What gets fetched, and what does not

| | Files | Size |
|---|---|---|
| Overhead `rgb.png` | 3,490 | 1.35 GB |
| Overhead `depth_raw.png`, `depth_color.png` | 6,980 | 1.61 GB (skipped) |
| Side-angle video | many | the rest of the 181 GB archive (skipped) |

### The dataset is smaller than "5k" implies

Measured by `--report` against the live bucket:

| | Count |
|---|---|
| Dishes with metadata | 5,006 |
| Dish ids in the RGB splits | 4,768 |
| **Usable (split + metadata + overhead RGB)** | **3,262** |
| ... of which train | 2,755 (68% of the 4,059 listed) |
| ... of which test | 507 (72% of the 709 listed) |

1,506 dish ids appear in the RGB split files but have no overhead RGB frame in the bucket.
Anything reading the split files at face value will silently train on a third fewer
examples than it thinks, so the split ids are intersected with imagery actually on disk
rather than trusted.

**2,755 training examples is small.** That is the main argument for the two-stage approach:
the backbone learns food features from Food-101's 101k images first, and Nutrition5k only
has to fit a regression head. It also means the nutrition head must be heavily regularised
and that augmentation is doing real work, not decoration.

### Label quality

The audit found the labels internally consistent, which is not a given for a real dataset:

- 0 dishes where the stated total calories disagree with the ingredient sum by more than 1%
- 0 dishes with no ingredients listed
- 2 dishes with calories <= 0, dropped during dataset construction
- Calories span 0 to 3,943 with a median of 209

Depth is skipped deliberately, not for download size. Nutrition5k ships RGB-D and the
original paper shows depth improves nutrition estimates substantially, but a phone camera
has no depth sensor. Training on depth would open a gap between training and deployment
that no amount of tuning closes. This is recorded as a known limitation in the top-level
README, and it means results here will not match the paper's headline numbers.

The RGB splits (`rgb_train_ids.txt`, `rgb_test_ids.txt`) are used rather than the depth
splits, so evaluation runs on the same dishes the model can actually be deployed against.

## Food-101

```bash
python data/download_food101.py
python data/download_food101.py --verify-only --out /path/to/existing/food-101
```

Roughly 5 GB. Training runs on Kaggle where Food-101 is already available as a mounted
dataset, so a local copy is mainly useful for smoke tests and for regenerating the label
contract.

The verification step is the point of this script. `shared/food101_labels.json` was
generated from dataset metadata, and `--verify-only` proves it matches the tarball's own
`meta/classes.txt`.

### Label ordering

Index N in `shared/food101_labels.json` is position N on the logits axis of the exported
model. The canonical Food-101 order is **not** Python `sorted()` order: `cheesecake`
precedes `cheese_plate`, because `_` sorts below letters in ASCII but the dataset's own
ordering treats it otherwise.

Getting this wrong produces a model that trains normally, evaluates normally, and labels
every prediction with the wrong dish name. The ordering is checksummed in the label file
and asserted in `tests/test_food101.py`.
