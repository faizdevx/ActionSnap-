# ActionSnap


```                                                                                         
▄████▄  ▄▄▄▄ ▄▄▄▄▄▄ ▄▄  ▄▄▄  ▄▄  ▄▄ ▄█████ ▄▄  ▄▄  ▄▄▄  ▄▄▄▄  
██▄▄██ ██▀▀▀   ██   ██ ██▀██ ███▄██ ▀▀▀▄▄▄ ███▄██ ██▀██ ██▄█▀ 
██  ██ ▀████   ██   ██ ▀███▀ ██ ▀██ █████▀ ██ ▀██ ██▀██ ██    
```

</p>

<p align="center">

![OpenPose](https://img.shields.io/badge/OpenPose-2D_Keypoints-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![einops](https://img.shields.io/badge/einops-TensorOps-purple)
![Flask](https://img.shields.io/badge/Flask-WebFramework-000000?logo=flask&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-ScientificComputing-013243?logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-5C3EE8?logo=opencv&logoColor=white)

</p>


## ⚙️ Flow Overview


```mermaid
graph LR
A[32 frame video] --> B[OpenPose keypoints]
B --> C[Tensor 32 x 36]
C --> D[Patch embedding 128 dim]
D --> E[Positional encoding]
E --> F[Transformer encoder]
F --> G[Global average pooling]
G --> H[Classifier 6 classes]

```

## video ingestion 

![video ingestion](img1.png)

![video ingestion](img3.png)

## Patch Embedding

![patch ingestion](img2.png)
![patch ingestion](img4.png)


## positional encoding

![positional encoding](img5.png)
![postional encoding](img6.png)


## Transformer Encoder


![Transformer Encoder](img7.png)
![Transformer Encoder](img8.png)


## Global Pooling

![Global Pooling](img9.png)
![Global Pooling](img10.png)

## CLASSIFIER HEAD

![cLASSIFIER HEAD](img11.png)
