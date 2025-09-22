# My-UFD

This repository is adapted from [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect).  
Please refer to the original repository for environment setup.  

## Purpose  
This repository is mainly used for the **DF-LLaVA: Unlocking MLLM's potential for Synthetic Image Detection via Prompt-Guided Knowledge Injection** paper.  
The pre-trained weights used in the paper are already provided.  

## Dataset Preparation  
Organize your dataset into the following directory structure: 

/real/train

/real/val

/fake/train

/fake/val


## Training  
Run the following command to start training:  

```bash
sh train.sh
```
