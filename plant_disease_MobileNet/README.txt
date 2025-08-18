# Plant Disease Prediction System

## Overview

This project implements a deep learning-based system for predicting plant diseases from leaf images. The system uses a MobileNet-based convolutional neural network trained on the PlantVillage dataset to classify images of plant leaves into 38 different categories (various diseases and healthy plants).

## Key Features

- **Deep Learning Model**: Uses a MobileNet architecture with custom dense layers for classification
- **38-Class Classification**: Can identify various diseases across multiple plant species
- **Web Interface**: Streamlit-based web application for easy image upload and prediction
- **High Accuracy**: Achieves ~97% validation accuracy on the test set

## Technical Details

### Model Architecture
- Base model: MobileNet (pretrained on ImageNet, weights frozen)
- Custom top layers:
  - GlobalAveragePooling2D
  - Dense (256 units, ReLU activation) with BatchNorm and Dropout (0.5)
  - Dense (64 units, ReLU activation) with BatchNorm and Dropout (0.5)
  - Output layer (38 units, softmax activation)
- Optimizer: Adam
- Loss: Categorical Crossentropy

### Training Process
- Dataset: PlantVillage dataset (43,456 training images, 10,849 validation images)
- Image size: 224x224
- Batch size: 64
- Data augmentation: Rescaling (1/255)
- Training callbacks:
  - EarlyStopping (patience=15)
  - ModelCheckpoint
  - ReduceLROnPlateau (factor=0.1, patience=15, min_lr=1e-6)
- Achieved 96.91% validation accuracy in 10 epochs

### File Structure
- `plant_disease_prediction_MobileNet.ipynb`: Jupyter notebook for model training
- `main.py`: Streamlit web application
- `class_indices.json`: Mapping of class indices to disease names
- `trained_model/best_model.keras`: Saved model weights

## How to Use

1. **Web Application**:
   - Run `main.py` with Streamlit
   - Upload a plant leaf image
   - Click "Predict" to get the disease classification

2. **Programmatic Use**:
   - Load the model with `tf.keras.models.load_model()`
   - Use the `predict()` function with an image path to get predictions

## Requirements
- Python 3.x
- TensorFlow 2.x
- Streamlit
- Pillow
- NumPy

## Future Improvements
- Expand to more plant species and diseases
- Implement real-time mobile application
- Add treatment recommendations for identified diseases
- Incorporate ensemble methods for improved accuracy

This project provides a practical solution for farmers and gardeners to quickly identify plant diseases from leaf images, enabling early treatment and prevention of crop damage.