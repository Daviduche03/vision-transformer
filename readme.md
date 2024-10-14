

# modal_deployment.py

## Overview

This script is designed to deploy a machine learning model using the Modal framework. The model is trained on a dataset of AI-generated and real images to classify them as either "AI-generated" or "real".

## Requirements

* Modal framework
* Python 3.x
* PyTorch
* Torchvision
* Hugging Face datasets
* Pillow
* TQDM

## Hyperparameters

* `batch_size`: 32
* `num_epochs`: 10
* `learning_rate`: 0.001

## Datasets

* `ai_dataset`: AI-Generated-vs-Real-Images-Datasets (Daviduche03)
* `real_dataset`: vintage-photography-450k-high-quality-captions (SilentAntagonist)

## Model

* The model is a Vision Transformer (ViT) with the following architecture:
	+ Patch embedding with a patch size of 16 and an embedding size of 768
	+ Transformer encoder with 12 layers and an embedding size of 768
	+ Classification head with a hidden size of 768 and 2 output classes

## Training

* The model is trained on a combination of the AI-generated and real image datasets
* The training dataset is split into training and validation sets (90% for training and 10% for validation)
* The model is trained for 10 epochs with a batch size of 32 and a learning rate of 0.001

## Deployment

* The model is deployed using the Modal framework with the following configuration:
	+ GPU: H100:2
	+ Timeout: 24 hours
	+ CPU: 8
	+ Secrets: huggingface-secret

## Usage

1. Install the required dependencies
2. Set the `HF_TOKEN` environment variable to your Hugging Face token
3. Run the script using `python modal_deployment.py`

Note: This script is designed to be run on a Modal cluster. If you want to run it locally, you will need to modify the script to use a different deployment framework or remove the Modal-specific code.