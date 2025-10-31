# 🔍 Facial Landmark Detection

> A deep learning solution for detecting 68 facial keypoints using CNNs and transfer learning

## ✨ Features

- **68-Point Detection**: Accurately locates 68 facial landmark points including eyes, eyebrows, nose, mouth, and jawline
- **CNN Architecture**: Implements Convolutional Neural Networks with batch normalization and dropout for robust feature extraction
- **Transfer Learning**: Leverages VGG16 pre-trained weights for improved accuracy and faster convergence
- **Data Augmentation**: Employs image augmentation techniques to enhance model generalization
- **CSV Export**: Generates prediction output in standardized CSV format for easy integration

## 🚀 Quick Start

### Prerequisites

```bash
python 3.x
jupyter notebook
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Nikshay-Jain/Facial_landmarks.git
cd Facial_landmarks
```

2. Install required dependencies:
```bash
pip install numpy pandas matplotlib opencv-python tensorflow keras scikit-learn
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook "Facial Landmark.ipynb"
```

## 🛠️ Usage

### Training the Model

1. Open `Facial Landmark.ipynb` in Jupyter Notebook
2. Load your facial image dataset with corresponding landmark annotations
3. Run all cells sequentially to:
   - Preprocess and augment the training data
   - Build and compile the CNN model
   - Train with validation split
   - Evaluate model performance

### Making Predictions

```python
# Load the trained model
model = load_model('facial_landmark_model.h5')

# Predict landmarks on new images
predictions = model.predict(test_images)

# Export to CSV
predictions_df.to_csv('submission.csv', index=False)
```

### Model Architecture

The neural network consists of:
- Multiple convolutional layers with ReLU activation
- Batch normalization for training stability
- MaxPooling for spatial dimension reduction
- Dropout layers to prevent overfitting
- Dense layers for final coordinate regression
- Optional VGG16 transfer learning backbone

## 📊 Output Format

Predictions are saved in `submission.csv` with 136 columns (x and y coordinates for each of the 68 landmarks).

## 🧰 Technology Stack

- **Language**: Python
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Environment**: Jupyter Notebook

## 📝 License

This project is available for educational and research purposes.
