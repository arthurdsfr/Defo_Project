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
Such implementation has been evaluated in a change detection task namely deforestation detection and aiming at reproducing the results obtained in [2] and [3] we make available the images used in this project which can be found in the following links for the [Amazon Biome](https://drive.google.com/drive/folders/1V4UdYors3m3eXaAHXgzPc99esjQOc3mq?usp=sharing) as well as for the [Cerrado](https://drive.google.com/drive/folders/14Jsw0LRcwifwBSPgFm1bZeDBQvewI8NC?usp=sharing). In the same way, the references can be obtained by clicking in [Amazon references] and [Cerrado references](https://drive.google.com/drive/folders/1n9QZA_0V0Xh8SrW2rsFMvpjonLNQPJ96?usp=sharing).

🧑‍💻 Usage

Model Training and Evaluating

To train a segmentation model for the Legal Amazon and/or Cerrado, use the execute.py script wehre you can configure training parameters, model architecture, and data paths. Then, training, testing and metrics computation scripts will be executed sequentially.
