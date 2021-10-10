# FGSM-SDI
Code for Boosting Fast Adversarial Training with LearnableAdversarial Initialization
## Introduction
> In this paper, we propose a sample-dependent adversarial initialization to boost fast AT. Specifically, We adopt a generative network conditioned on a benign image and its gradient information from the target network to generate an effective initialization. In the training phase, the generative network and the target network are optimized jointly and play a game. The former learns to produce a dynamic sampledependent initialization to generate stronger adversarial examples based on the current target network. And the latter adopts the generated adversarial examples for training to improve model robustness. Compared with widely adopted random initialization fashions in fast AT, the proposed initialization overcomes the catastrophic overfitting, thus improves model robustness. Extensive experimental results demonstrate the superiority of our proposed method.
![image](/FGSM-SDI/imgs/pipeline.PNG)
## Requirements
Python3 </br>
Pytorch </br>
## Test
> python3.6 test_FGSM_SDI.py --model_path model.pth --out_dir ./output/ --data-dir cifar-data
## Trained Models
> The Trained models can be downloaded from the [Baidu Cloud](https://pan.baidu.com/s/1ZEv-7gSEI4gi64PvCnM3ww)(Extraction: 1234.) or the [Google Drive](https://drive.google.com/drive/folders/1972Yhxte4318qbpllyul5dVmvo-VpWVW?usp=sharing)
