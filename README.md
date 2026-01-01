# ConstructVision: AI-driven Construction Progress Forecasting

ConstructVision is an AI-powered project that utilizes Convolutional Neural Networks (CNNs) to predict and track the progress of wall construction using real-time image data captured during the construction process.

## Problem Statement
The aim of this project is to leverage deep learning techniques, specifically CNNs, to predict the progress at different stages of wall construction. This involves creating a comprehensive dataset, training a CNN model, and using it to analyze image frames for prediction accuracy.

## Features
- **Progress Prediction**: Predicts the current stage of wall construction (e.g., foundation, 25% brickwork, 50% brickwork).
- **Real-Time Image Processing**: Uses OpenCV for real-time image preprocessing and predictions.
- **Efficiency and Optimization**: Incorporates fine-tuned pre-trained CNN models like VGG16 or ResNet50 for optimal results.

## Steps

### 1. Dataset Creation
- Collect images of wall construction at various stages.
- Label each image according to the progress stage.
- Ensure diversity in the dataset to cover a wide variety of construction conditions.

### 2. Preprocessing of Images
- Resize images to a standardized size suitable for model input.
- Normalize pixel values to ensure consistency.
- Convert images to grayscale to simplify processing and reduce computational requirements.

### 3. Model Training
- Train the CNN model with labeled images as input and progress stages as output.
- Experiment with fine-tuning pre-trained networks (e.g., VGG16, ResNet50) or develop a custom architecture.
- Optimize the model for extracting features relevant to wall construction progress.

### 4. Image Processing Pipeline
- Use OpenCV to process captured images in real-time.
- Apply preprocessing steps like resizing and normalization.
- Pass images through the trained CNN model for progress prediction.

### 5. Progress Calculation
- Analyze predictions over time to calculate overall construction progress.
- Incorporate feedback loops to iteratively improve the accuracy of predictions.

## Installation Instructions
1. **Clone the repository**:
```bash
git clone https://github.com/Afsheen-Aziz/ConstructVision.git
```
2. **Navigate into the project directory**:
```bash
cd ConstructVision
```
3. **Install dependencies**:
```bash
pip install -r requirements.txt
```
4. **Start the application**:
```bash
python app.py
```

## Usage
1. Launch the local server by running the application.
2. Upload an image using the web interface.
3. View predicted progress and visualized results on the web interface.

## Project Structure
```
ConstructVision/
├── app.py           # Main application logic
├── static/
│   └── uploads/    # Directory to store uploaded images
├── templates/
│   ├── index.html  # Webpage for file upload
│   └── result.html # Webpage for displaying results
├── weights/
│   ├── best.pt     # Trained model weights
│   └── last.pt     # Latest version of the model weights
└── README.md        # Project documentation
```

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch:
```bash
git checkout -b feature-name
```
3. Commit your changes:
```bash
git commit -m "Description of changes"
```
4. Push the branch:
```bash
git push origin feature-name
```
5. Submit a pull request.

## License
This project is MIT licensed. See the LICENSE file for details.

## Contact
For any queries, suggestions, or feedback:
- **Name**: Afsheen Aziz
- **Email**: afsheenonnar@gmail.com
- **GitHub**: https://github.com/Afsheen-Aziz

---
Thank you for exploring ConstructVision! Together, let's advance the future of AI in construction.
