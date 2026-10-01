# Source-training interface

`triadapt.training.train_step` implements a single source optimization step. Supply your dataset loader, source-validation routine and epoch loop. The two stages have distinct objectives; changing loaders, validation, initialization or budgets defines a different experiment.

## Inputs

Each batch is a dictionary containing:

| Key | Shape | Convention |
| --- | --- | --- |
| `detector` | B,P,T,17,2 | CPU float tensor; normalized screen coordinates |
| `target` | B,P,17,3 | Centre-frame source 3D supervision, millimetres |
| `gt2d` | B,P,T,17,2 | Same coordinates as detector; required after stage 1 |
| `gt_confidence` | B,P,T,17 | Positive for valid 2D supervision; required after stage 1 |

Use T=243 for the paper configuration and the joint ordering in DATASETS.md. Do not pass evaluation-domain ground truth to training or source checkpoint selection. Construct the pretrained backbone before constructing adapters; seed all relevant random generators in the surrounding training program.

```python
import torch
from triadapt import TriAdapt, make_optimizer, train_step

source_generator = torch.Generator().manual_seed(42 + 100_000)
stage1 = TriAdapt(backbone, arm="fam", stage="stage1").to(device)
optimizer = make_optimizer(stage1)
for source_batch in source_loader:
    record = train_step(stage1, source_batch, optimizer, source_generator)

# Run your source-validation/checkpoint-selection loop, then restore the selected
# state before saving this handoff. Do not implicitly select the last epoch.
handoff = stage1.adapter_state()
torch.save(handoff, "fam_source_selected.pt")

# Use a separately constructed, identically loaded backbone for each arm.
stage2 = TriAdapt(backbone_for_arm, arm="triadapt", stage="stage2").to(device)
state = torch.load("fam_source_selected.pt", map_location=device, weights_only=True)
stage2.fam.load_state_dict(state["fam"], strict=True)
optimizer = make_optimizer(stage2)  # Rebuild after freezing or changing stages.
for source_batch in source_loader:
    record = train_step(stage2, source_batch, optimizer, source_generator)
torch.save(stage2.adapter_state(), "adapters.pt")
```

Save adapter tensors independently of upstream weights. Restore with `load_adapter_state` into the same dimensions and arm; the upstream checkpoint must separately match the one used in training. Load only trusted files.

## Losses and freezing

Stage 1 combines root-relative position loss, 0.5 times the bone-scaled source loss and 0.01 times the magnitude of the learned FAM gate. The feature gate consumes a zero six-vector and the query offset consumes a zero three-vector. These are learned constants, not visibility or cross-person signals; their original parameterization is retained.

The component continuation uses root-relative position + 0.5 bone-scaled loss + 0.05 soft-threshold loss + eta masked 2D Smooth-L1 + 0.01 residual magnitude. Eta is 1 with TCM and 0 without it. AdamW defaults are learning rate 2e-4 and weight decay 1e-4. The explicit source-corruption function uses a CPU generator; validation should separately assess clean and fixed corrupted source inputs using the protocol in the paper.

`stage2` freezes both the pretrained lifter and FAM. It does **not** wrap the lifter in `torch.no_grad()`: TCM needs gradients through the fixed lifter. `joint` updates all three adapters while retaining the fixed lifter. If an arm disables a module, that module is bypassed and its tensors are not optimizer-visible.

The helpers expose optimization and adapter integration, not an automatic end-to-end reproduction of every paper table. Dataset preparation, source selection and benchmark evaluation are supplied by the surrounding experiment environment. Synthetic tests verify software behavior, not scientific accuracy.
