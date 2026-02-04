# CNN Image Classifier - AWS SageMaker

This is a **portfolio-ready image classification project** demonstrating end-to-end deep learning workflow using AWS SageMaker and MLflow.

## Project Overview

- **Goal:** Train a Convolutional Neural Network (CNN) to classify images from the STL10 dataset.
- **Technologies:** PyTorch, MLflow, AWS SageMaker Studio
- **Key Features:**
  - GPU-enabled training
  - Early stopping and learning rate scheduling
  - MLflow experiment tracking (parameters, metrics, artifacts, model versions)
  - Modular project structure with reusable scripts (`src/`)

## Dataset

- **STL10 dataset** (10 classes, 5,000 training images, 8,000 test images)
- Dataset is **not included** due to size; download instructions: [STL10 dataset](https://cs.stanford.edu/~acoates/stl10/)
- Folder structure:
  - `data/raw/`       # Raw dataset (download manually)
  - `data/processed/` # Preprocessed dataset for training

## Project Structure

- `notebooks/`      # Exploratory analysis, training, evaluation
- `src/`            # Reusable scripts (data.py, model.py, train.py)
- `models/`         # Saved model checkpoints (large files not in repo)
- `mlruns/`         # MLflow logs (tracked locally, not in GitHub)
- `README.md`
- `.gitignore`

## Usage

1. Clone the repository
git clone git@github.com
:Javier-DataScience/cnn-image-classifier-aws.git
cd cnn-image-classifier-aws
2. Download the dataset and place it in `data/raw/`
3. Open notebooks in SageMaker Studio to run training, evaluation, and MLflow tracking

## Notes

- Large files like `models/` and `mlruns/` are **not pushed** to GitHub.
- You can view MLflow experiments locally in SageMaker Studio.
