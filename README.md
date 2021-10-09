# FGSM-SDI
Code for Boosting Fast Adversarial Training with LearnableAdversarial Initialization
## Main Idea
> In this paper, we propose a sample-dependent adversarial initialization to boost fast AT. Specifically, We adopt a generative network conditioned on a benign image and its gradient information from the target network to generate an effective initialization. In the training phase, the generative network and the target network are optimized jointly and play a game. The former learns to produce a dynamic sampledependent initialization to generate stronger adversarial examples based on the current target network. And the latter adopts the generated adversarial examples for training to improve model robustness. Compared with widely adopted random initialization fashions in fast AT, the proposed initialization overcomes the catastrophic overfitting, thus improves model robustness. Extensive experimental results demonstrate the superiority of our proposed method.
## Requirements
Python3 </br>
Pytorch </br>
## Test
> python3.6 AA_test_cifar10.py --model_path model.pth --out_dir ./output/ --data-dir cifar-data
## Trained Models
> The Trained models can be downloaded from this [link](https://pan.baidu.com/s/1ZEv-7gSEI4gi64PvCnM3ww). Extraction: 1234.
