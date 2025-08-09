# Transfer Learning for Object Recognition on CIFAR-10

This project demonstrates the application of **transfer learning** using a pre-trained ResNet-50 model for image classification on the CIFAR-10 dataset. The implementation showcases how to leverage pre-trained deep neural networks to achieve high accuracy on object recognition tasks with minimal training time.

## 🎯 Project Overview

The project implements a transfer learning approach that:
- Utilizes ResNet-50 pre-trained on ImageNet as the base model
- Adapts the model for CIFAR-10 classification (10 classes)
- Achieves over 95% validation accuracy in just 10 epochs
- Demonstrates effective use of data preprocessing and model fine-tuning

## 📊 Dataset

**CIFAR-10 Dataset**
- **Source**: Kaggle CIFAR-10 dataset
- **Size**: 50,000 training images
- **Classes**: 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Size**: 32x32 pixels (upscaled to 256x256 for ResNet-50 compatibility)
- **Format**: RGB images with corresponding labels

## 🏗️ Model Architecture

### Base Model: ResNet-50
- **Pre-trained on**: ImageNet dataset
- **Input Shape**: (256, 256, 3)
- **Total Parameters**: ~23.5 million
- **Architecture**: Deep residual network with skip connections

### Custom Classification Head
```python
Sequential([
    UpSampling2D((2,2)) × 3,  # Scale CIFAR-10 images from 32x32 to 256x256
    ResNet50(pretrained),     # Feature extraction backbone
    Flatten(),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(10, activation='softmax')  # 10-class classification
])
```

## 🔧 Key Features

- **Transfer Learning**: Leverages ResNet-50 pre-trained weights
- **Data Preprocessing**: Image normalization and upsampling
- **Regularization**: Dropout and BatchNormalization layers
- **Optimization**: RMSprop optimizer with low learning rate (2e-5)
- **Performance**: Achieves 95.3% validation accuracy

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Main Dependencies:
- `tensorflow>=2.10.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation
- `opencv-python>=4.5.0` - Image processing
- `matplotlib>=3.5.0` - Data visualization
- `kaggle>=1.5.12` - Dataset download
- `py7zr>=0.20.0` - Archive extraction

## 🚀 Getting Started

### 1. Setup Kaggle API
```python
# Configure Kaggle API credentials
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
```

### 2. Download and Extract Dataset
```python
# Download CIFAR-10 dataset
api.dataset_download_files('pankrzysiu/cifar10-python', path='./', unzip=True)

# Extract compressed files
import py7zr
with py7zr.SevenZipFile('train.7z', mode='r') as archive:
    archive.extractall()
```

### 3. Run the Model
Open and execute the Jupyter notebook:
```bash
jupyter notebook "Transfer learning.ipynb"
```

## 📈 Training Results

The model demonstrates excellent performance with transfer learning:

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|------------------|-------------------|---------------|----------------|
| 1     | 35.00%          | 81.12%           | 1.9755        | 0.7104         |
| 5     | 92.02%          | 94.50%           | 0.4001        | 0.2280         |
| 10    | 97.87%          | 95.28%           | 0.1513        | 0.1961         |

### Key Observations:
- **Fast Convergence**: High accuracy achieved within first few epochs
- **Transfer Learning Benefit**: Pre-trained features significantly accelerate training
- **Regularization**: Dropout and BatchNorm prevent overfitting
- **Stable Training**: Consistent improvement across epochs

## 🔬 Technical Implementation

### Data Preprocessing
```python
# Image normalization
X_train_scaled = X_train.astype('float32') / 255.0
X_test_scaled = X_test.astype('float32') / 255.0

# Label encoding
label_map = {'frog': 0, 'truck': 1, 'deer': 2, 'automobile': 3, ...}
Y_train = train_df['labels'].map(label_map).values
```

### Model Compilation
```python
model.compile(
    optimizer=optimizers.RMSprop(learning_rate=2e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## 📁 Project Structure

```
Transfer-learning-Object-recognition/
├── Transfer learning.ipynb    # Main notebook with implementation
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies list
├── trainLabels.csv          # CIFAR-10 labels (downloaded)
└── train.7z                 # CIFAR-10 images (downloaded)
```

## 🎓 Learning Objectives

This project demonstrates:
1. **Transfer Learning Concepts**: How to adapt pre-trained models
2. **Image Classification Pipeline**: End-to-end implementation
3. **Deep Learning Best Practices**: Regularization, optimization
4. **Data Handling**: Working with image datasets and APIs
5. **Performance Optimization**: Achieving high accuracy efficiently

## 🔍 Key Insights

- **Transfer Learning Advantage**: Reduced training time and improved performance
- **Architecture Design**: Importance of proper head design for new tasks
- **Regularization Impact**: Dropout and BatchNorm prevent overfitting
- **Learning Rate**: Low learning rates work best for fine-tuning
- **Data Scaling**: Proper preprocessing crucial for model performance

## 🚀 Future Enhancements

Potential improvements and extensions:
- [ ] **Data Augmentation**: Add rotation, flip, zoom transformations
- [ ] **Model Ensemble**: Combine multiple pre-trained models
- [ ] **Advanced Architectures**: Experiment with EfficientNet, Vision Transformers
- [ ] **Hyperparameter Tuning**: Optimize learning rate, batch size, architecture
- [ ] **Deployment**: Create web API for real-time predictions
- [ ] **Visualization**: Add confusion matrix and class activation maps

## 📊 Performance Metrics

Final model performance:
- **Training Accuracy**: 97.87%
- **Validation Accuracy**: 95.28%
- **Training Time**: ~4000 seconds (10 epochs)
- **Model Size**: 89.98 MB

## 🤝 Contributing

Feel free to contribute to this project by:
1. Reporting bugs or issues
2. Suggesting new features or improvements
3. Submitting pull requests
4. Sharing your experiments and results

## 📜 License

This project is open-source and available under the MIT License.

---

**Author**: [Your Name]  
**Contact**: [Your Email]  
**Date**: [Current Date]

*This project serves as an educational example of transfer learning applications in computer vision and deep learning.*
