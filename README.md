# TriAdapt

Module and source-training code for **Representation Aware Parameter Efficient Adaptation for Monocular 3D Human Pose Estimation**.

TCM adds bounded corrections to 2D tracks, FAM adapts pre-head joint features, and KGR refines root-relative 3D coordinates. The demonstrated backbone is GraphMLP. People are processed independently; no companion features are exchanged.

## Setup

Use Python 3.11 and install a suitable [PyTorch](https://pytorch.org/get-started/locally/) build, then:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/smoke.py
```

The smoke command uses a small synthetic test double to check adapter optimization. It is not a pose benchmark or a GraphMLP implementation.

## Backbone and data

Obtain GraphMLP and its 243-frame checkpoint from the [official repository](https://github.com/Vegetebird/GraphMLP). Use its installation and checkpoint-loading instructions. The interface was checked against upstream commit `befa9f069d08756268ee872098d6a80befb1ffb1` (`model/graphmlp.py`). Construct and load that model separately before passing it to TriAdapt:

```python
from triadapt import TriAdapt, make_optimizer, train_step

# backbone: externally constructed, pretrained GraphMLP with its original .head
model = TriAdapt(backbone, arm="fam", stage="stage1").to(device)
optimizer = make_optimizer(model)
stats = train_step(model, source_batch, optimizer, source_generator)
```

The wrapper attaches FAM at the input to `.head` without editing upstream files. TCM remains differentiable through the fixed backbone. Use one model instance per concurrent inference worker.

See [DATASETS.md](DATASETS.md) for official dataset links, tensor shapes and coordinate conventions, and [TRAINING.md](TRAINING.md) for source-domain training and component continuations.

## Staged adaptation

1. Train `arm="fam", stage="stage1"` and select a checkpoint using source validation.
2. Restore that FAM state into a new model for each component continuation.
3. Set `stage="stage2"`. Both backbone **and FAM** remain fixed; TCM and/or KGR are optimized.

| Arm | Active adapters | Stage-2 updates |
| --- | --- | --- |
| `fam_tcm` | FAM + TCM | TCM |
| `fam_kgr` | FAM + KGR | KGR |
| `triadapt` | FAM + TCM + KGR | TCM + KGR |

For simultaneous three-module adaptation, use `arm="triadapt", stage="joint"`. The default module dimensions are TCM 16, FAM 32 and KGR 64, yielding 914 + 67,553 + 17,347 = 85,814 adapter parameters. Stage 2 updates 18,261 of them. The five FAM widths for the TriAdapt budget study are 11, 22, 32, 53 and 74; TCM/KGR retain their default dimensions. Use the default width unless deliberately running the budget study.

## Citation and acknowledgments

Paper title: *Representation Aware Parameter Efficient Adaptation for Monocular 3D Human Pose Estimation*. Publication details will be added when available. This repository does not assert acceptance or a publication venue.

Please cite GraphMLP and each dataset used in your experiments. The original references are collected in [REFERENCES.bib](REFERENCES.bib). GraphMLP provides the external pretrained lifter; attention and graph propagation are established operators, not separately claimed as inventions of this implementation. Dependency and license details are in [THIRD_PARTY.md](THIRD_PARTY.md).

## License

The TriAdapt implementation is available under the [MIT license](LICENSE). External code, data and model weights remain subject to their own licenses and access agreements.
