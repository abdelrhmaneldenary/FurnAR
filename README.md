## 🎨 Image Colorization using Dense Prediction Transformer (DPT) ##

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)]([INSERT_YOUR_HUGGINGFACE_LINK_HERE])

## 📌 Overview
This project implements a **Dense Prediction Transformer (DPT)** from first principles to perform high-quality image colorization. By leveraging a Vision Transformer (ViT) backbone to capture long-range global context, the model effectively predicts the missing color channels of a black-and-white input image.
<img width="1529" height="473" alt="Screenshot from 2026-04-26 00-59-34" src="https://github.com/user-attachments/assets/1a064885-347f-44fc-bb2b-329234f76983" />

Live Demo: [https://huggingface.co/spaces/Abdlerhman/DPT-Image-Colorization]

##  Architecture
The architecture is built completely from scratch using PyTorch and follows a hierarchical encoder-decoder structure:
* **Encoder (ViT-Base):** Extracts rich, global feature representations from the grayscale input.
* **Reassemble Blocks:** Projects the sequence of transformer tokens back into spatial feature maps at various resolutions.
* **Feature Fusion Blocks:** Progressively upsamples and fuses the feature maps to recover spatial details.
* **Output Head:** Utilizes a sequence of 3x3 and 1x1 convolutions to output exactly 2 channels (the `a` and `b` color spaces), which are activated via `Tanh` to normalize outputs between -1 and 1.

### Why LAB Color Space?
Instead of predicting standard RGB channels, the model operates in the **LAB color space**:
* **L (Lightness):** Provided as the 1-channel grayscale input (duplicated 3 times for the ViT).
* **ab (Color):** The 2 channels predicted by the model.
This separation allows the network to focus entirely on color semantics without having to reconstruct structural details and shadows.

## 📊 Training Details
* **Dataset:** COCO 2017 Dataset.
* **Loss Function:** L1 Loss (Mean Absolute Error) to prevent the muddy, desaturated colors common with MSE.
* <img width="1321" height="835" alt="training_test" src="https://github.com/user-attachments/assets/5c47a4e0-4556-4395-9de7-2eeed0d3ec8d" />

* **Hardware:** Trained on Kaggle using an NVIDIA Tesla T4.
* **Strategy:** The ViT backbone was initially frozen to train the custom DPT decoder, followed by unfreezing for fine-tuning, avoiding catastrophic forgetting of the pre-trained ImageNet weights.
