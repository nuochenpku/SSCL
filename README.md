# Alleviating Over-smoothing for Unsupervised Sentence Representation
This is the project of our ACL 2023 paper: Alleviating Over-smoothing for Unsupervised Sentence Representation.


## Overview

we present a new training paradigm based on contrastive learning:  Simple contrastive method named Self-Contrastive Learning (SSCL), which can significantly improve the performance of learned sentence representations while alleviating the over-smoothing issue. Simply Said, we utilize hidden representations from intermediate PLMs layers as negative samples  which the final sentence representations should be away from. Generally, our SSCL has several advantages: (1) It is fairly straightforward and does not require complex data augmentation techniques; (2) It can be seen as a contrastive framework that focuses on mining negatives effectively, and can be easily extended into different sentence encoders that aim for building positive pairs; (3) It can further be viewed as a plug-and-play framework for enhancing sentence representations.

## Setup
First, install PyTorch by following the instructions from the [official website](https://pytorch.org/get-started/previous-versions/). To faithfully reproduce our results, please use the correct 1.9.1 version corresponding to your platforms/CUDA versions. Install PyTorch by the following command,

```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
``` 

