# Brain Tumor MRI Classification using TensorFlow & CNN

A deep learning project for classifying brain tumors from MRI images using Convolutional Neural Networks (CNN) and Transfer Learning with EfficientNetB0.

## 📋 Project Overview

This project implements a deep learning model to classify brain MRI scans into four categories:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

The model leverages transfer learning using EfficientNetB0 pre-trained on ImageNet, achieving high accuracy in tumor classification.

## 🎯 Features

- Multi-class classification of brain tumors from MRI images
- Transfer learning using EfficientNetB0 architecture
- Data visualization and exploratory data analysis
- Model training with callbacks (TensorBoard, ModelCheckpoint, ReduceLROnPlateau)
- Confusion matrix and classification report for performance evaluation
- Prediction function for new images

## 🗂️ Dataset Structure

```
minorproject/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
├── Testing/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
├── logs/
│   ├── train/
│   └── validation/
├── main.ipynb
├── brain-tumor-mri-classification-tensorflow-cnn.ipynb
└── effnet.h5 (trained model)
```

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **EfficientNetB0** - Pre-trained CNN model
- **OpenCV (cv2)** - Image processing
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Model evaluation and data preprocessing
- **TensorBoard** - Training visualization

## 📦 Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd minorproject
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

3. Install required packages:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn tqdm ipywidgets pillow
```

## 🚀 Usage

### Training the Model

1. Open the Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the dataset
   - Visualize sample images
   - Build the model architecture
   - Train the model (approximately 12 epochs)
   - Evaluate performance

### Making Predictions

Use the `img_pred()` function to classify new MRI images:

```python
# Predict tumor type from an image
img_pred('path/to/your/mri_image.jpg')
```

The function will:

- Load and preprocess the image
- Make a prediction using the trained model
- Display the predicted tumor type

## 🏗️ Model Architecture

The model uses **Transfer Learning** with the following architecture:

1. **Base Model**: EfficientNetB0 (pre-trained on ImageNet)

   - Input shape: (150, 150, 3)
   - Weights: ImageNet
   - `include_top=False` to add custom classification layers

2. **Custom Layers**:

   - GlobalAveragePooling2D: Reduces spatial dimensions
   - Dropout (rate=0.5): Prevents overfitting
   - Dense (4 units, softmax): Output layer for 4 classes

3. **Compilation**:
   - Loss: Categorical Crossentropy
   - Optimizer: Adam
   - Metrics: Accuracy

## 📊 Training Configuration

- **Image Size**: 150x150 pixels
- **Batch Size**: 32
- **Epochs**: 12
- **Validation Split**: 10%
- **Test Split**: 10%

### Callbacks Used:

1. **TensorBoard**: Real-time training visualization
2. **ModelCheckpoint**: Saves best model based on validation accuracy
3. **ReduceLROnPlateau**: Reduces learning rate when validation accuracy plateaus
   - Factor: 0.3
   - Patience: 2 epochs
   - Min delta: 0.001

## 📈 Model Evaluation

The model is evaluated using:

- **Classification Report**: Precision, Recall, F1-Score for each class
- **Confusion Matrix**: Visual representation of prediction accuracy
- **Training/Validation Curves**: Accuracy and loss plots over epochs

## 📁 Files Description

- **main.ipynb**: Main notebook with complete implementation
- **brain-tumor-mri-classification-tensorflow-cnn.ipynb**: Alternative implementation with detailed documentation
- **effnet.h5**: Saved trained model (best checkpoint)
- **logs/**: TensorBoard logs for training visualization

## 🔍 Viewing Training Logs

To visualize training progress using TensorBoard:

```bash
tensorboard --logdir=logs
```

Then open your browser and navigate to `http://localhost:6006`

## 📝 Key Preprocessing Steps

1. **Image Loading**: Read MRI images using OpenCV
2. **Resizing**: All images resized to 150x150 pixels
3. **Normalization**: Pixel values normalized (handled by EfficientNet preprocessing)
4. **Data Augmentation**: Can be added using ImageDataGenerator
5. **One-Hot Encoding**: Labels converted to categorical format

## 🎨 Visualization Features

- Sample images from each tumor class
- Training and validation accuracy/loss curves
- Confusion matrix heatmap
- Color palettes for better visualization

## ⚠️ Notes

- Training time: ~2 hours on CPU, ~5 minutes on GPU
- Ensure sufficient disk space for dataset and model checkpoints
- Virtual environment recommended to avoid package conflicts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset: Brain MRI Images for Brain Tumor Detection
- Pre-trained Model: EfficientNetB0 from Keras Applications
- Framework: TensorFlow/Keras

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---
