# TriAdapt

Training code for **Representation Aware Residual Adaptation for Monocular 3D Human Pose Estimation**.

TriAdapt adapts a pretrained GraphMLP at three interfaces. TCM corrects 2D trajectories, FAM adapts joint features, and KGR refines 3D poses. The input is a tracked sequence of 2D keypoints for each person.

Authors: Songmao Yang, Wenjing Yue and Zhi Chen, Nanjing University of Posts and Telecommunications.

## Installation

Use Python 3.11 in a dedicated environment. Install an appropriate CUDA build of PyTorch for GPU training using the [official instructions](https://pytorch.org/get-started/locally/), then install the versions in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_release.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. CPU installations can run the checks; full training requires suitable GPU resources.

## External assets

Obtain MPI-INF-3DHP, 3DPW and the 243-frame GraphMLP checkpoint from their providers. Prepare the four source NPZ files described in [DATASETS.md](DATASETS.md), which specifies download links, array formats and file locations.

Copy `config/external_paths.example.json` to `config/external_paths.json` and replace the five asset paths with absolute paths. Only load trusted NPZ and checkpoint files.

## Staged training and ablations

```bash
python train.py --assets config/external_paths.json --plan
python train.py --assets config/external_paths.json
```

The first command checks source identities and the public checkpoint without launching training. The second learns FAM, fixes that transfer state, and trains the full model and both component-deletion variants. For a single run, append `--seeds 42`; the default list is `42,123,2026`.

| Output arm | Active adapters |
| --- | --- |
| `m1_m2` | TCM and fixed FAM |
| `m2_m3` | Fixed FAM and KGR |
| `full` | TCM, fixed FAM and KGR |

GraphMLP remains fixed. The selected first-stage checkpoint supplies the FAM-only reference and initializes each second-stage arm. Training writes checkpoints, `training.csv`, `epoch0.json` and summaries to new output directories and refuses to overwrite existing runs. The trainers implement source sampling, losses, early stopping and checkpoint selection.

## Installation checks

`check_release.py` checks file hashes, model imports, command-line entry points and source-geometry invariants. Run `train.py --assets config/external_paths.json --plan` to validate your configured training assets before starting a run.

## Attribution

Code is licensed under MIT. GraphMLP and the shared MLP-Graph operator retain their upstream notices in `licenses/`; see [THIRD_PARTY.md](THIRD_PARTY.md). Dataset and model-weight use is governed by the respective providers' terms.
