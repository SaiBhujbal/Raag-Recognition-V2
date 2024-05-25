# Raag Recognition Deep Learning Model

## Introduction
This repository contains a deep learning model designed to recognize and classify three Indian classical raags: Bhairav, Malkans, and Yaman. The project was developed as an End Semester project by AIML students Sai Bhujbal and Tanay Kende under the guidance of the Symbiosis Institute of Technology. The model is built using a custom hybrid architecture that combines Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) networks. This approach leverages both spatial and temporal features of the audio data, making it well-suited for the task of raag recognition.

## Raag Information
In indian classical music, Raag is a melodic framework for improvisation in Indian classical music akin to a melodic mode. Each raga has its unique essence which depicts a particular mood and set of emotions. The classification system of audio files based on how closely they resemble with a particular Raag can be used for Mood based music classification and recommendation system which will not only classify mood of indian classical music but also work on other genres as raag is Fundamental backbone of melodies.

### Bhairav

Raag Bhairav is an early morning raag, which is very serene and calm. It uses both shuddh and komal (flat) notes, creating a solemn and pious atmosphere. It is typically performed at the break of dawn.

### Malkans

Raag Malkans, also known as Malkauns, is a late-night raag. It is performed from midnight to dawn and is known for its deep, serious, and heavy mood. It uses komal (flat) notes and shuddh (pure) notes in a specific manner to evoke a meditative state.

### Yaman

Raag Yaman is a popular evening raag that exudes romance and tranquility. It is typically performed after sunset and uses all sharp (teevra) notes, except for the shuddh (pure) note in the madhyam (middle) scale.

## Features

The features used for training the model are extracted from audio files using the following techniques:

- **MFCC (Mel-Frequency Cepstral Coefficients):** Captures the power spectrum of a sound, providing a compact representation of the audio signal.
- **Chroma:** Represents the twelve different pitch classes, giving insights into the harmonic content of the audio.
- **Spectral Contrast:** Measures the difference in amplitude between peaks and valleys in a sound spectrum, capturing the timbral texture.
- **Tonnetz:** Represents the harmonic relations between pitches, useful for understanding the tonal characteristics.

You can download the pre-extracted features as well as audio file dataset from the following link:
[Feature Drive Link](https://drive.google.com/drive/folders/1ycnhABr7tuv7_BlTUdzKa39wEF0iPDgm?usp=drive_link)
[Audio Files Dataset](https://drive.google.com/drive/folders/1liqSKms12gwcWsmjFH6hj12Wp5iewa0y?usp=drive_link)

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

## Model Architecture Diagram

To better understand the architecture of our custom hybrid model, we have generated a visual representation of the model layers.

![image](https://github.com/SaiBhujbal/Raag-Recognition-V2/assets/46700402/126507e5-d110-4e2b-8bf8-dd0a45a4416f)

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

Detecting emotion patterns based on listening history of a user in audio playing applications like Spotify, Apple music etc. 

By implementing these upgrades, we hope to create a comprehensive tool that not only identifies raags but also aids in the learning and practice of Indian classical music.

