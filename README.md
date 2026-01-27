# Project Overview

This research and development project focuses on the automatic and accurate detection of deforestation in critical regions of Brazil: the Brazilian Legal Amazon and the Cerrado. Utilizing advanced Computer Vision and Deep Learning techniques, specifically with the PyTorch framework, we developed semantic segmentation models capable of identifying deforested areas from multispectral satellite images.

The growing impact of deforestation on biodiversity and the global climate demands efficient monitoring solutions. Our goal is to provide a modular and robust tool that assists researchers, environmental agencies, and policymakers in the rapid identification of deforestation events, contributing to the conservation of these vital biomes.
The current project contains the scripts to perform a change detection classification for remote sensing data, specifically for deforestation detection in two Brazilian biomes, the Amazon rainforest(Brazilian Legal Amazon) and Brazilian savannah (Cerrado). Several Neural Networks architectures have been included for use.

✨ Highlights and Key Features

Semantic Segmentation Models: Implementation and experimentation with state-of-the-art architectures (e.g., U-Net, DeepLabV3+, Transformer-based) optimized for deforestation detection and diferent backbones.
Multi-Biome Data: Use of a comprehensive dataset containing satellite imagery (Landsat) and deforestation annotations from three distinct regions of the Brazilian Legal Amazon and representative areas of the Cerrado.
PyTorch Framework: Full development of the training and inference pipeline in PyTorch, ensuring flexibility, performance, and ease of experimentation.
Modular and Replicable Code: Organized code structure that facilitates understanding, modification, and extension of the project to new regions, models, or data types.
Performance Metrics: Rigorous evaluation of models using segmentation metrics (IoU, F1-score, Precision, Recall) to ensure the robustness and reliability of detections.
Results Visualization: Generation of deforestation maps and confidence visualizations to facilitate the interpretation of model results.

🚀 Getting Started

These instructions will guide you through setting up the development environment, obtaining the data, and running the code to train or infer models.

Prerequisites
Make sure you have the following tools installed:

Python 3.8+
Git
Conda or Miniconda (recommended for environment management)
CUDA-compatible GPU (recommended for efficient training and inference)

# Data Download
Such implementation has been evaluated in a change detection task namely deforestation detection where the images used in this project can be found in the following links for the [Amazon Biome](https://drive.google.com/drive/folders/1V4UdYors3m3eXaAHXgzPc99esjQOc3mq?usp=sharing) as well as for the [Cerrado](https://drive.google.com/drive/folders/14Jsw0LRcwifwBSPgFm1bZeDBQvewI8NC?usp=sharing). In the same way, the references can be obtained by clicking in [Amazon references] and [Cerrado references](https://drive.google.com/drive/folders/1n9QZA_0V0Xh8SrW2rsFMvpjonLNQPJ96?usp=sharing).

🧑‍💻 Usage

Model Training and Evaluating

To train a segmentation model for the Legal Amazon and/or Cerrado, use the execute.py script wehre you can configure training parameters, model architecture, and data paths. Then, training, testing and metrics computation scripts will be executed sequentially.

📂 # Project Structure

The following folder organization is designed to promote modularity, scalability, and clarity, essential characteristics in Computer Vision and Deep Learning projects.

```
.
├── data                              # Contains custom PyTorch Dataset definitions and related utilities
│   └── DeforestationDataset.py       # Custom PyTorch Dataset for loading deforestation imagery and masks
├── deeplab                           # Implementation of DeepLabV3+ semantic segmentation model
│   ├── aspp.py                       # Atrous Spatial Pyramid Pooling (ASPP) module
│   ├── backbones                     # Various backbone networks for DeepLab (e.g., encoders)
│   │   ├── drn.py                    # Dilated Residual Network (DRN) backbone
│   │   ├── __init__.py               # Python package initialization
│   │   ├── mobilenet.py              # MobileNet backbone
│   │   ├── resnet.py                 # ResNet backbone
│   │   └── xception.py               # Xception backbone
│   ├── decoder.py                    # DeepLab's decoder module
│   ├── deeplab.py                    # Main DeepLabV3+ model definition
│   └── sync_batchnorm                # Synchronized Batch Normalization implementation
│       ├── batchnorm.py              # Synchronized Batch Normalization layer
│       ├── comm.py                   # Communication utilities for distributed sync_bn
│       ├── __init__.py               # Python package initialization
│       ├── replicate.py              # Module for replicating models across GPUs with sync_bn
│       └── unittest.py               # Unit tests for sync_batchnorm (can be ignored during normal use)
├── dino                              # DINO (self-supervised Vision Transformer) related implementations
│   ├── utils.py                      # Utility functions for DINO (e.g., data augmentation, logging)
│   └── vision_transformer.py         # Vision Transformer (ViT) model implementation used in DINO
├── get_metrics.py                    # Script to calculate and report various evaluation metrics
├── get_visuals.py                    # Script to generate visual outputs (e.g., predicted masks, comparisons)
├── models                            # Generic model components or wrappers
│   ├── Decoder.py                    # A generic decoder component (potentially shared or for other models)
│   ├── FeatureExtractor.py           # A generic feature extractor component (encoder-like)
│   └── models.py                     # Main entry point or wrapper for different model configurations
├── options                           # Centralized configuration management using argparse
│   ├── baseoptions.py                # Base class for common command-line arguments
│   ├── deeplaboptions.py             # Specific options for DeepLab models
│   ├── deforestationoptions.py       # General options related to the deforestation dataset/task
│   ├── dinooptions.py                # Specific options for DINO-related configurations
│   ├── testoptions.py                # Options for the testing script
│   ├── trainoptions.py               # Options for the training script
│   └── visualoptions.py              # Options for visualization scripts
├── Prove.py                          # Script for demonstration, proof-of-concept, or specific testing (purpose to be clarified)
├── README.md                         # This project description file
├── test.py                           # Main script for model inference and testing
├── train.py                          # Main script for model training
├── utils                             # Collection of utility functions
│   ├── CustomLosses.py               # Implementations of custom loss functions for segmentation
│   └── tools.py                      # General utility functions and helpers
└── vnet                              # Implementation of V-Net semantic segmentation model
    ├── decoder.py                    # V-Net's decoder module
    └── vnet.py                       # Main V-Net model definition
```
