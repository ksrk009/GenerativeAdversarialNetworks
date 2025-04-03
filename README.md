# Deep Convolutional GANs (DCGANs) for Image Generation

## Project Overview
This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate synthetic images using the CIFAR-10 dataset. DCGANs are a type of generative model that use convolutional neural networks for both the generator and discriminator, enabling the generation of high-quality images. The generator creates fake images from random noise, while the discriminator learns to distinguish between real and fake images. Over time, the two networks improve through adversarial training.

## Features
### **Model Architecture**
- **Generator (G)**: Transforms random noise (100-dimensional) into 64x64 RGB images using a series of transposed convolutional layers, batch normalization, and ReLU activations, with a final Tanh activation.
- **Discriminator (D)**: Classifies images as real or fake using convolutional layers, batch normalization, LeakyReLU activations, and a sigmoid output.

### **Training Process**
- Trained on the CIFAR-10 dataset (60,000 32x32 color images across 10 classes).
- Uses the Binary Cross-Entropy Loss (BCELoss) and Adam optimizer for both networks.
- Trains for 25 epochs with a batch size of 64.

### **Output Generation**
- Saves real and generated (fake) images every 100 iterations during training.
- Outputs are stored in the `results/` folder as PNG files (e.g., `real_samples.png`, `fake_samples_epoch_000.png`).

## Technologies Used
- **Python**: Core programming language.
- **PyTorch**: For building and training the DCGAN model.
- **Torchvision**: For loading the CIFAR-10 dataset and handling image transformations.
- **NumPy**: For numerical operations (implicitly used via PyTorch).

## Dataset
- **CIFAR-10**: A dataset of 60,000 32x32 RGB images across 10 classes (e.g., airplane, car, bird). The project resizes images to 64x64 for DCGAN compatibility.
- Automatically downloaded to the `data/` folder during the first run.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/dcgan-cifar10.git
   cd dcgan-cifar10
