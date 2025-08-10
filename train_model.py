#!/usr/bin/env python3
"""
Devanagari Handwritten Character Recognition CNN Training Script
This script trains a CNN model for recognizing Devanagari handwritten characters
"""

import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from PIL import ImageFile
import os


# Suppress warnings
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("="*60)
print("DEVANAGARI HANDWRITTEN CHARACTER RECOGNITION")
print("="*60)

# Check TensorFlow Version
print(f'TensorFlow Version: {tf.__version__}')

# Check for GPU
if not tf.config.list_physical_devices('GPU'):
    print('No GPU found. Training will use CPU (slower)')
else:
    print(f'Default GPU Device: {tf.config.list_physical_devices("GPU")}')

print("\n" + "="*60)
print("STEP 1: CREATING CNN MODEL")
print("="*60)

# Create the CNN model
classifier = Sequential()

# First Convolutional Layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.1))

# Second Convolutional Layer
#classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
#classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2, 2)))
#classifier.add(Dropout(0.2))

# Flattening
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.1))  # Increased dropout rate
classifier.add(Dense(units=46, activation='softmax'))  # 46 classes for Devanagari characters

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model created successfully!")
print("\nModel Summary:")
classifier.summary()

print("\n" + "="*60)
print("STEP 2: PREPARING DATA")
print("="*60)

# Check if dataset directories exist
if not os.path.exists('Dataset/hcr_dataset/Images/Images'):
    print("ERROR: Dataset/hcr_dataset/Images/Images folder not found!")
    print("Please ensure you have placed the hcr_dataset in the correct location.")
    exit(1)

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    validation_split=0.2  # Use 20% of training data for validation
)

print("Loading training data...")

training_set = train_datagen.flow_from_directory(
    'Dataset/hcr_dataset/Images/Images',
    target_size=(32, 32),
    batch_size=256,
    class_mode='categorical',
    subset='training'  # Specify this is the training set
)

print("Loading validation data...")

validation_set = train_datagen.flow_from_directory(
    'Dataset/hcr_dataset/Images/Images',  # Use the new dataset directory
    target_size=(32, 32),
    batch_size=256,
    class_mode='categorical',
    subset='validation'  # Specify this is the validation set
)

print(f"Training samples: {training_set.samples}")
print(f"Validation samples: {validation_set.samples}")
print(f"Number of classes: {training_set.num_classes}")

print("\n" + "="*60)
print("STEP 3: TRAINING THE MODEL")
print("="*60)
print("Training will take several minutes to hours depending on your hardware...")
print("Progress will be shown below:")

# Add EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = classifier.fit(
    training_set,
    epochs=50,  # Increase epochs, early stopping will find the best one
    validation_data=validation_set,
    callbacks=[early_stopping],
    verbose=1
)

print("\n" + "="*60)
print("STEP 4: EVALUATING THE MODEL")
print("="*60)

# Evaluate the model on training and validation sets
scores_train = classifier.evaluate(training_set, verbose=0)
print(f"Training Accuracy: {scores_train[1]*100:.2f}%")

scores_val = classifier.evaluate(validation_set, verbose=0)
print(f"Validation Accuracy: {scores_val[1]*100:.2f}%")

print("\n" + "="*60)
print("STEP 5: SAVING THE MODEL")
print("="*60)

# Save the model
model_save_path = "CNN_DevanagariHandWrittenCharacterRecognition_full.h5"
classifier.save(model_save_path)

print("Model saved successfully!")
print(f"- Full model saved as: {model_save_path}")

print("\n" + "="*60)
print("STEP 6: PLOTTING TRAINING HISTORY")
print("="*60)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("Training history plot saved as: training_history.png")

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("Your model is now ready for use!")
print("Next steps:")
print("1. Test the model with sample predictions")
print("2. Run the GUI application for interactive character recognition")
print("="*60)
print(f"Class indices: {training_set.class_indices}")