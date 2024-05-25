# Raag Recognition Deep Learning Model

## Introduction

This repository contains a deep learning model designed to recognize and classify three Indian classical raags: Bhairav, Malkans, and Yaman. The model is built using a custom hybrid architecture that combines Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) networks. This approach leverages both spatial and temporal features of the audio data, making it well-suited for the task of raag recognition.

## Features

The features used for training the model are extracted from audio files using the following techniques:

- **MFCC (Mel-Frequency Cepstral Coefficients):** Captures the power spectrum of a sound, providing a compact representation of the audio signal.
- **Chroma:** Represents the twelve different pitch classes, giving insights into the harmonic content of the audio.
- **Spectral Contrast:** Measures the difference in amplitude between peaks and valleys in a sound spectrum, capturing the timbral texture.
- **Tonnetz:** Represents the harmonic relations between pitches, useful for understanding the tonal characteristics.

You can download the pre-extracted features from the following link:
[Feature Drive Link](https://drive.google.com/drive/folders/1ycnhABr7tuv7_BlTUdzKa39wEF0iPDgm?usp=sharing)

## Data Preparation

### Step 1: Extract Features from Audio Files

The first step involves extracting features from the audio files. These features are saved as `.pkl` files for easy loading during training. The features are segmented and padded to ensure uniform input length across all samples.

### Step 2: Prepare Data for Training

The extracted features are loaded, and sequences are padded to ensure uniform input length for each feature type. The data is then split into training and testing sets, and labels are encoded for classification.

## Model Training

### Custom Hybrid Model Architecture

The model architecture consists of several key components designed to capture both spatial and temporal features from the audio data:

- **Input Layers:** Four input layers for MFCC, Chroma, Spectral Contrast, and Tonnetz features.
- **TCN Branches:** Each input is passed through a Temporal Convolutional Network (TCN) branch, which includes:
  - **Reshape Layer:** Adjusts the input dimensions to fit the convolutional layers.
  - **Conv1D Layer:** Applies convolution operations to capture local features.
  - **BatchNormalization Layer:** Normalizes the activations to improve training stability.
  - **MaxPooling1D Layer:** Down-samples the input to reduce dimensionality.
  - **Dropout Layer:** Applies dropout for regularization to prevent overfitting.
  - **Bidirectional LSTM Layer:** Captures temporal dependencies in both forward and backward directions.
- **Concatenate Layer:** Concatenates the outputs of the TCN branches.
- **Dense Layer:** A fully connected dense layer with ReLU activation to introduce non-linearity.
- **Dropout Layer:** Another dropout layer for regularization.
- **Output Layer:** A dense layer with softmax activation for classification into three raag classes.

### Training Process

The model is trained using the following strategies to enhance performance and prevent overfitting:

1. **Early Stopping:** Monitors the validation loss and stops training when the model stops improving, restoring the best weights.
2. **ReduceLROnPlateau:** Reduces the learning rate when the validation loss plateaus, allowing the model to converge more effectively.
3. **Learning Rate Scheduler:** Dynamically adjusts the learning rate during training to improve convergence.

## Deployment

### Predicting Raags from New Audio Files

To deploy the model and make predictions on new audio files, follow these steps:

1. **Load the Model and Label Encoder:** Load the trained model and the label encoder used during training.
2. **Extract Features from Audio Files:** Extract the same set of features (MFCC, Chroma, Spectral Contrast, Tonnetz) from the audio file to be predicted.
3. **Pad or Truncate Features:** Ensure the extracted features are padded or truncated to match the input shape expected by the model.
4. **Make Predictions:** Use the trained model to predict the raag of the input audio file.

## Future Work and Upgrades

We aim to further develop this model into an AI Raag Tutor that can assist musicians in identifying and correcting mistakes while singing or playing raags. Here are some planned upgrades:

1. **Data Augmentation:** Enhance the training data with more diverse and augmented samples to improve model robustness.
2. **Fine-tuning and Regularization:** Implement advanced regularization techniques and fine-tune hyperparameters to reduce overfitting.
3. **Real-time Feedback:** Develop a real-time feedback system to provide instant guidance and corrections to musicians.
4. **Additional Raags:** Expand the model to recognize a wider range of raags by incorporating more data and training for additional classes.
5. **User Interface:** Create an intuitive user interface that allows users to easily interact with the AI Raag Tutor.

By implementing these upgrades, we hope to create a comprehensive tool that not only identifies raags but also aids in the learning and practice of Indian classical music.

