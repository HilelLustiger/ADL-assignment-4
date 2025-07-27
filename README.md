# ADL Assignment 4 – CNN Autoencoder and Latent Analysis

This repository contains the code and report for Assignment 4 of the Applied Deep Learning course. We extend a simple CNN on CIFAR-10 with a deconvolutional decoder and visualize latent representations.

## 📄 Overview

Task 1: Refactor PyTorch CIFAR-10 tutorial CNN into modular functions and classes.

Task 2: Add a deconvolutional decoder (MaxUnpool + ConvTranspose) to reconstruct input images. Train with combined classification + reconstruction loss.

Task 3: Analyze latent channels by zeroing out all but one feature map and visualizing reconstructions from both first and second convolutional layers.

## 📁 Files

task1.py – Modular CNN implementation and training pipeline.

task2.py – CNN + decoder, training loop, evaluation, and latent channel analysis script.

## 🧠 Key Concepts

Convolutional Neural Networks (CNNs)

Decoder via MaxUnpool2d & ConvTranspose2d

Multi-task loss (classification + reconstruction)

Latent feature visualization & interpretability

## 🖥️ How to Run
```python
pip install torch torchvision numpy matplotlib
python task1.py
python task2.py
```
This will train for 2 epochs, evaluate accuracy, and display latent analysis plots.

## 📦 Requirements

Python 3.x

torch

torchvision

numpy

matplotlib
