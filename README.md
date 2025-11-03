# Adversarial-Patches

[![CI - Clear Notebook Outputs](https://github.com/U1186204/Adversarial-Patches/actions/workflows/ci.yml/badge.svg)](https://github.com/U1186204/Adversarial-Patches/actions/workflows/ci.yml)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black.svg?logo=github&style=for-the-badge)](https://github.com/U1186204/Adversarial-Patches/tree/main)

## Project Description
This project implements a robust adversarial patch attack against a ResNet-34 model. The notebook contains the complete PyTorch code to train a 64x64 patch from scratch that, when placed on any image, deceives the classifier into predicting the target class 'Llama'. To ensure real-world effectiveness, the patch is trained to be robust to transformations by applying random rotations (±20°) and scaling (75%-125%) during the training loop - all in order to achieve a high 'fooling' accuracy.

## Trained Llama Patch
![Llama patch](images/Llama%20patch.png)



## Results
The patch successfully fools the model into predicting "llama" when applied to various test images, but obviously, with occasional limitations

| Original: 'goldfish' | Original: 'electric_ray' | Original: 'great_white_shark' |
| :---: | :---: | :---: |
| ![Goldfish Llama](images/goldfish_Llama.png) | ![Electric Ray Llama](images/electric_ray_Llama.png) | ![Great White Shark Llama](images/great_white_shark_Llama.png) |
| **Original: 'hammerhead'** | **Original: 'tench'** | **Original: 'tiger_shark'** |
| ![Hammerhead Llama](images/hammerhead_Llama.png) | ![Tench Llama](images/tench-Llama.png) | ![Tiger Shark Llama](images/tiger_shark_Llama.png) |

## Running the Project 
Please use Google Co-lab. Select A100 GPU preferred for a resonable runtime [![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-F9AB00.svg?logo=googlecolab&style=for-the-badge)](https://colab.research.google.com/github/U1186204/Adversarial-Patches/blob/main/main.ipynb)

