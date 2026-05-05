# ActionSnap-
using transformer architecture on 2d figures  to analyse the human doing activity


graph LR
A[Input Video] --> B[OpenCV Preprocessing]
B --> C[OpenPose Keypoint Extraction]
C --> D[Frame Sequencing & Patching (einops)]
D --> E[Transformer Model (PyTorch / TF)]
E --> F[Predictions / Output]
F --> G[Flask Web Interface]
