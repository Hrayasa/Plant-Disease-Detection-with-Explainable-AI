import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        'data/train',
        'data/val',
        'models',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    return plt

def save_model_performance(model, history, test_accuracy):
    """Save model performance metrics"""
    metrics = {
        'test_accuracy': test_accuracy,
        'final_training_accuracy': history.history['accuracy'][-1],
        'final_validation_accuracy': history.history['val_accuracy'][-1],
        'final_training_loss': history.history['loss'][-1],
        'final_validation_loss': history.history['val_loss'][-1]
    }
    
    with open('results/model_performance.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    return metrics

def get_class_names(directory):
    """Get class names from directory structure"""
    return sorted(os.listdir(directory))

def visualize_data_distribution(data_dir):
    """Visualize the distribution of classes in the dataset"""
    class_names = get_class_names(data_dir)
    class_counts = [len(os.listdir(os.path.join(data_dir, class_name))) 
                   for class_name in class_names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts)
    plt.title('Distribution of Classes in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt 