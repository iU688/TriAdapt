# Attribution and external dependencies

## GraphMLP

The demonstrated lifter is [GraphMLP](https://github.com/Vegetebird/GraphMLP), described in *GraphMLP: A Graph MLP-like Architecture for 3D Human Pose Estimation*, Pattern Recognition 158 (2025), 110925, DOI [10.1016/j.patcog.2024.110925](https://doi.org/10.1016/j.patcog.2024.110925).

Install its official source as a separate dependency. The adapter wrapper uses the existing pre-head feature interface and does not redefine its encoder, graph blocks or prediction head. Follow upstream installation instructions for the 243-frame model and pretrained checkpoint. Preserve the upstream [MIT notice](https://github.com/Vegetebird/GraphMLP/blob/main/LICENCE) with the external checkout; the TriAdapt license does not replace it.

## Implementation dependencies

PyTorch supplies tensor and differentiation operations under its own [license](https://github.com/pytorch/pytorch/blob/main/LICENSE). Backbone-specific dependencies follow GraphMLP's requirements. The attention, graph propagation, LayerNorm and GELU references are in REFERENCES.bib. The module composition and adaptation procedure should not be confused with independent invention of these operators.

## Data and weights

Dataset and pretrained-weight permissions are determined by their original providers, linked in DATASETS.md. MIT applies to this TriAdapt implementation, not to those assets. Follow each provider's citation requirements when using their data or models.
