# FGSM-SDI
Code for [Boosting Fast Adversarial Training with LearnableAdversarial Initialization](https://arxiv.org/)
## Introduction
![Adversarial example generation of the proposed FGSM-SDI](/imgs/pipeline.PNG)
<p align="center">
Adversarial example generation of the proposed FGSM-SDI
</p>


> In this paper, we propose a sample-dependent adversarial initialization to boost fast AT. Specifically, we adopt a generative network conditioned on a benign image and its gradient information from the target network to generate an effective initialization. In the training phase, the generative network and the target network are optimized jointly and play a game. The former learns to produce a dynamic sampledependent initialization to generate stronger adversarial examples based on the current target network. And the latter adopts the generated adversarial examples for training to improve model robustness. Compared with widely adopted random initialization fashions in fast AT, the proposed initialization overcomes the catastrophic overfitting, thus improves model robustness. Extensive experimental results demonstrate the superiority of our proposed method.
## Requirements
Python3 </br>
Pytorch </br>
## Test
> python3.6 test_FGSM_SDI.py --model_path model.pth --out_dir ./output/ --data-dir cifar-data
## Trained Models
> The Trained models can be downloaded from the [Baidu Cloud](https://pan.baidu.com/s/1ZEv-7gSEI4gi64PvCnM3ww)(Extraction: 1234.) or the [Google Drive](https://drive.google.com/drive/folders/1972Yhxte4318qbpllyul5dVmvo-VpWVW?usp=sharing)

## Visualization analysis
![Adversarial example generation of the proposed FGSM-SDI](/imgs/saliency.PNG)
<p align="center">
The top row shows the clean images and the adversarial examples along with their corresponding heat-maps (generated by the Grad-CAM algorithm)on the FGSM-RS. The bottom row shows the results of our FGSM-SDI.
</p>

> Adversarial perturbations fool a well-trained model by interfering with important local regions that determine image classification. To explore whether our FGSM-SDI will be affected by adversarial perturbations, we adopt Gradientweighted Class Activation Mapping (Grad-CAM) to generate the heat maps that locate the category-related areas in the image. It can be observed that as for FGSM-RS, adversarial perturbations modify the distribution of the maximal points on the generated heat map, while as for our FGSM-SDI, the adversarial perturbations do not modify the distribution of the maximal points on the generated heat-map. That indicates that our FGSM-SDI is more robust. 

![Adversarial example generation of the proposed FGSM-SDI](/imgs/landscape.PNG)
<p align="center">
Visualization of the loss landscape of on CIFAR10 for FGSM-RS, FGSM-CKPT, FGSM-GA, and our FGSM-SDI. We plot the cross entropy loss varying along the space consisting of two directions: an adversarial direction and a Rademacher (random) direction.
</p>

> We compare the loss landscape of the proposed method with those of the other fast AT methods to explore the association between latent hidden perturbation and local linearity. Compared with other AT methods, the cross-entropy loss of our FGSM-SDI is more linear in the adversarial direction. Using the latent perturbation generated by the proposed method can preserve the local linearity of the target model better. It qualitatively proves that using the proposed sample-dependent adversarial initialization can boost the fast AT.






