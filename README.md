# Plant Disease Detection with Explainable AI

This project implements an automated plant disease detection system using deep learning and Explainable AI (XAI) techniques. The system can classify plant diseases from leaf images and provide explanations for its predictions using LIME and SHAP.

## Features

- Plant disease classification using Convolutional Neural Networks (CNNs)
- Image preprocessing and augmentation
- Explainable AI using LIME and SHAP
- Interactive web interface using Streamlit
- Support for multiple plant diseases

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/               # Directory for storing image datasets
├── utils/             # Helper functions and utilities
├── models/            # Trained model files
├── app.py            # Main Streamlit application
├── train.py          # Model training script
└── requirements.txt  # Project dependencies
```

## Usage

1. Prepare your dataset:
   - Place your plant disease images in the `data/` directory
   - Images should be organized in subdirectories by disease type

2. Train the model:
```bash
python train.py
```

3. Run the web application:
```bash
streamlit run app.py
```

## Model Architecture

The system uses a CNN-based architecture with transfer learning from pre-trained models. The model is trained to classify plant diseases and provides explanations for its predictions using LIME and SHAP.

## XAI Techniques

- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)

## License

MIT License 