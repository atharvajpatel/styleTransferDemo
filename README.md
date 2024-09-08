# VGG Style Transfer Logging

## Overview

This project implements a style transfer model based on the VGG architecture, inspired by the Gatys paper on neural style transfer. The model utilizes the VGG network to extract feature maps from images and calculates a Gram matrix for both the style and content images to generate a stylized output. This log captures the key steps, challenges, and results encountered during the development process.

## Getting Started

### Prerequisites

Make sure you have the necessary packages installed to enable CUDA support for faster training.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## VGG Logging

### Initial Research

- Started off with researching VGG architecture and the Gatys paper.
- VGG uses filters to understand key features of the image (texture, image shapes).
  - Filters tend to be smaller, 3x3 in size.
- VGG already has pretrained weights and runs inference on style and content, storing and finding feature maps.
  - **Note**: It **does not** fine-tune.

### Gram Matrix

- VGG extracts feature maps from both content and style images and then reshapes them.
- The **Gram matrix** is calculated for both the style image and the generated image:
  - The Gram matrix is computed from the feature maps extracted by the pretrained VGG model.
  - It captures the correlation of different filters in the image, allowing features to be changed.

### Loss Functions

- **Total Loss** = A * Content Loss + B * Style Loss.
- **Style Loss**: Difference between Gram Matrix (Generated Image) and Gram Matrix (Style Image).
- **Content Loss**: Difference between Gram Matrix (Generated Image) and Gram Matrix (Content Image).
- The user controls values for A and B:
  - Increasing B emphasizes the style loss, making the image more stylized.
  - Increasing A emphasizes the content loss, keeping the image closer to its original form.

### Example Analogy

- A helpful analogy for understanding A and B:
  - Imagine you have 100 feet of wood to rebuild a rectangle or square.
  - The original rectangle was 1x49 feet. If the user passes in a higher value for B, they want you to optimize for the length (style), while a lower value optimizes for width (content).
  - For example, a B value of 0.5 would create a 25x25 square. A B value of 0.9 optimizes for length and changes the new "style."

### Training VGG

- Started training VGG, not too complicated since `torchvision` provided the model with good documentation.
- Learned how to create the **Gram matrix**:
  - Reshapes and standardizes the feature maps.
  - It essentially captures the correlation of different filters in the image, allowing pixel changes.
- Connected the custom Gram matrix loss function to the standard training process.

### Issues and Debugging

- Initially, I wrote the VGG model from scratch, but it turned out to be just a mask. This is captured in `vggActual.py`.
- Training with 500 epochs on a CUDA GPU took 2.5 minutes and utilized 100% of the GPU.
  - To save compute resources, I increased the learning rate and reduced the number of epochs.
  - Decent results were obtained, but fine details indicative of high-quality artistic style were missing.
- Running for 1000 epochs produced good results.
  - I am confident that with more compute and time, I could achieve state-of-the-art style transfer.

### Testing

- I used a custom style transfer GPT to evaluate my progress.
- Tested content images was Horse
- Tested style images from Van Gogh.
- All images under images folder

### Shortcomings

- The resolution of the generated images was poor.
- For some reason, Van Gogh's style consistently generated green or red colors, even though there was no red in his original image.
  - This might be due to pixel mixing in the feature maps.
