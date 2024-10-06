# Try-Not-To-Laugh Game

## 1. Introduction
This project presents a system for detecting laughter in real-time using a custom version of the LeNet deep learning architecture. It is part of an interactive game where players try not to laugh while watching funny media. The model processes live webcam video feeds to detect laughter based on facial expressions, enabling a dynamic gameplay experience.

## 2. Objective
The goal of this project is to develop a real-time game where players attempt to avoid laughing. A trained deep learning model monitors players' facial expressions and detects when they laugh. The game tracks the duration of laughter, and the player who laughs the least wins.

## 3. Dataset: GENKI-4K
The model is trained on the **GENKI-4K** dataset, which contains 4000 face images with:
- **Expression labels**: Indicating whether the subject is smiling or not.
- **Head-pose labels**: Providing data about the orientation of the head.

This dataset is ideal for smile and laughter detection because of its variety in facial expressions and head poses.

## 4. LeNet Architecture Overview
The laugh detection model is based on a custom adaptation of the LeNet architecture, optimized for processing 64x64 grayscale face images. The architecture includes modern improvements such as Leaky ReLU activations, batch normalization, and dropout layers to enhance performance and prevent overfitting.

### Key Architecture Details:
- **Convolutional Layers**:
  - Layer 1: Conv2D (32 filters, 5x5 kernel) → Leaky ReLU → MaxPooling.
  - Layer 2: Conv2D (64 filters, 5x5 kernel) → Leaky ReLU → MaxPooling.
  
- **Fully Connected Layers**:
  - Flattening the feature maps → Dense (512 units) → Leaky ReLU → Dropout.

- **Output Layer**:
  - Dense (2 units) → Softmax for binary classification (laughing or not).

## 5. Training the Model

### 5.1 Data Preprocessing
To prepare the data:
- **Face Detection**: Haar Cascade is used to detect and crop faces from images. These faces are resized to 64x64 pixels.
- **Normalization**: Pixel values are normalized to [0, 1] for better training efficiency.

### 5.2 Data Augmentation
Data augmentation techniques were applied to improve model generalization:
- Random rotations (up to 30°)
- Width/height shifts (up to 10%)
- Shearing (up to 15%)
- Zooming (up to 25%)
- Horizontal flipping

### 5.3 Class Imbalance Handling
The dataset is slightly imbalanced, with more non-smiling images than smiling ones. To counteract this, class weighting was used to ensure the model treats both classes equally.

### 5.4 Training Configuration
- **Optimizer**: Adam with a learning rate of 0.001.
- **Loss Function**: Binary Cross-Entropy.
- **Learning Rate Adjustment**: ReduceLROnPlateau adjusted the learning rate when validation performance plateaued.

**Training Parameters**:
- Epochs: 70
- Batch Size: 32
- Validation Split: 20%

## 6. Evaluation and Results
The model was evaluated on the test set using performance metrics like:
- Precision, Recall, and F1-score for both "smiling" and "not smiling" classes.
  ![best_scores](https://github.com/user-attachments/assets/84e540f8-db45-463e-9674-2ac7734a5354)

- Training History: The loss and accuracy curves across 70 epochs were plotted to show the model's learning progress.
  ![BEST](https://github.com/user-attachments/assets/95f25867-4e85-4962-839b-20804983e34e)


The model’s training history, showing loss and accuracy across 70 epochs, was plotted to visualize learning progress and performance improvements.

## 7. User Interface (UI) Design
The game interface was developed using the Tkinter framework. Key features include:
- **Player Mode Selection**: 1-player or 2-player mode.
- **Customizable Game Duration**: Players can set game length (e.g., 30 or 60 seconds).
- **Funny Media**: Players can select funny videos or images to enhance the challenge.

### Game Logic:
- **1-Player Mode**: Tracks the total time the player laughs, displaying the results at the end.
- **2-Player Mode**: Tracks both players’ laughter, and the player who laughs the least is the winner.
- **Real-Time Updates**: The UI shows whether players are laughing and how much time remains in the round.

## 8. Real-Time Laugh Detection Workflow
1. **Face Detection**: Captures live video using a webcam, with Haar Cascade detecting faces in real-time.
2. **Laugh Classification**: Detected faces are resized to 64x64 pixels and fed into the trained LeNet model to classify if the player is laughing.
3. **Game Scoring**: The system tracks how long players laugh and updates scores based on the detection.

## 9. Conclusion
This project successfully integrates deep learning with a game interface to detect laughter in real-time. The model trained on the GENKI-4K dataset efficiently classifies facial expressions, providing an engaging, interactive "Try Not to Laugh" challenge.
