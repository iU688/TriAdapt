# Dataset sources and input conventions

Obtain assets from the original providers and accept their access terms. The code license does not grant permission to redistribute dataset images, annotations or model weights.

| Asset | Role in the paper | Official source |
| --- | --- | --- |
| Human3.6M | Upstream GraphMLP pretraining | https://vision.imar.ro/human3.6m/description.php |
| MPI-INF-3DHP | Source adaptation and source validation | https://vcai.mpi-inf.mpg.de/3dhp-dataset/ |
| 3DPW | Source adaptation and source validation | https://virtualhumans.mpi-inf.mpg.de/3DPW/ |
| MuPoTS-3D | Target evaluation | https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/ |
| HumanEva-I | Target evaluation | http://humaneva.is.tue.mpg.de/ |
| CMU Panoptic | Fixed-frontend indoor replay | https://domedb.perception.cs.cmu.edu/ |

MPI-INF-3DHP source training uses S1–S6 and source validation S7–S8. 3DPW uses its train and validation splits. Target datasets are not source-training inputs. This is distinct from the 3DPW challenge evaluation protocol.

## Coordinate format

The 17-joint order is pelvis, right hip, right knee, right ankle, left hip, left knee, left ankle, spine, thorax, neck, head, left shoulder, left elbow, left wrist, right shoulder, right elbow, right wrist. Coordinate conversion must be consistent across the source inputs, pretrained lifter and supervision.

For image coordinates (u,v) and width W, height H, screen normalization is x=2u/W-1 and y=2v/W-H/W. Preserve zero sentinels for missing detections when preparing tracks; apply camera transformations before creating 3D supervision. The backbone produces camera-relative positions in metres; `target` is in millimetres and is root-centred inside the training loss.

The temporal input length is 243; the 3D output refers to its centre frame. FAM operates independently per tracked person. Different people in the P axis do not exchange features.

See [REFERENCES.bib](REFERENCES.bib) for source publications. Obtain the GraphMLP pretrained weights through its [official download instructions](https://github.com/Vegetebird/GraphMLP#download-pretrained-model). For the video frontend, the paper uses [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose); acquire those components and their model-specific terms from the providers.
