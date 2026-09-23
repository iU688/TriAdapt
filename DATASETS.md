# External assets

Download the following assets from their providers and follow their license or access agreement.

| Asset | Provider | Suggested local location |
| --- | --- | --- |
| MPI-INF-3DHP | https://vcai.mpi-inf.mpg.de/3dhp-dataset/ | `external/raw/MPI-INF-3DHP/` |
| 3DPW | https://virtualhumans.mpi-inf.mpg.de/3DPW/ | `external/raw/3DPW/` |
| 243-frame GraphMLP weight | https://github.com/Vegetebird/GraphMLP#download-pretrained-model | `external/checkpoints/243frame.pth` |

## Prepared source inputs

The training entry accepts four prepared NPZ files, not the providers' raw archives:

```text
external/source/mpi_train.npz
external/source/mpi_val.npz
external/source/pw3d_train.npz
external/source/pw3d_val.npz
```

MPI uses S1-S6 for training and S7-S8 for source validation. 3DPW uses its train and validation splits. Prepare these arrays according to the loaders in `runtime/common/r3_multisource_dataset.py` and `pw3d_pair_dataset.py`. The configured files must satisfy the file or decoded-content identities in `config/source_content_sha256.json`; `--plan` reports whether the assets match.

Put absolute file paths in `config/external_paths.json`. The pretrained checkpoint SHA256 is `76B65A860A0C64DE0137F5E3AE118DB4F2A1F482C549A3211E9F1A26C9C9DEA8`.
