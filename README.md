# Keyword Spotting (KWS) Using CNN

## Overview
This project implements a **Keyword Spotting (KWS)** system using a **Convolutional Neural Network (CNN)**. The goal is to classify short audio clips into a predefined set of labels, enabling the system to recognize spoken commands such as "yes," "no," "stop," "go," and others. The model is trained and evaluated on the Speech Commands dataset from TensorFlow Datasets.

## Table of Contents
- [Project Features](#project-features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Features
- Preprocessing pipeline to handle audio data, including trimming, padding, and generating mel-spectrograms.
- CNN-based model optimized for keyword spotting.
- Early stopping and model checkpointing to ensure optimal performance.
- Scaled and normalized data for effective training.
- Support for loading and saving preprocessed data for faster iteration.

## Dataset
The project uses the **Speech Commands v0.0.3** dataset provided by TensorFlow Datasets. This dataset contains recordings of 35 spoken commands, such as "yes," "no," "stop," "go," and more.

### Dataset Statistics:
- **Training examples**: ~85,000
- **Validation examples**: ~10,000
- **Test examples**: ~10,000

For more details, refer to the [TensorFlow Datasets Speech Commands documentation](https://www.tensorflow.org/datasets/catalog/speech_commands).

## Installation
To set up the project, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/kws-cnn.git
    cd kws-cnn
    ```
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Mount Google Drive (if using Google Colab):
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Preprocessing Pipeline
The audio data is preprocessed to improve model performance:
1. **Trimming:** Remove silent sections from audio clips.
2. **Padding:** Ensure all audio clips are 1 second long (16,000 samples).
3. **Mel-Spectrogram Conversion:** Convert audio signals to mel-spectrograms for better feature representation.
4. **Normalization:** Scale the data using MinMaxScaler for consistent input.

Processed datasets are saved as `.pkl` files to Google Drive for reuse.

## Model Architecture
The CNN model consists of the following layers:
1. **Convolutional Layers:** Extract features from input mel-spectrograms.
2. **Batch Normalization:** Improve training stability.
3. **MaxPooling:** Downsample feature maps.
4. **Dropout:** Prevent overfitting.
5. **Dense Layers:** Map extracted features to the output label space.

### Summary of Model:
- Input shape: (63, 128, 1)
- Convolutional layers: 4
- Dropout rate: 20%
- Optimizer: Adam with learning rate 0.001
- Loss function: Sparse Categorical Crossentropy

## Training Process
- **Batch Size:** 384
- **Epochs:** 50 (with early stopping based on validation loss)
- **Callbacks:**
  - EarlyStopping (patience: 5 epochs)
  - ModelCheckpoint (saves the best model based on validation loss)

### Metrics:
- Training and validation loss
- Training and validation accuracy

## Results
The model achieves:
- **Training Accuracy:** ~87%
- **Validation Accuracy:** ~85%
- **Test Accuracy:** ~87%

## How to Run
Use the Google Colab Environment to run: https://colab.research.google.com/drive/1Io2MUEJDh_dSsxqdqBNh_mPpVPB_SWlW?usp=drive_link
